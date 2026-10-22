"""Host/native small acceptance with process-specific resource samples."""

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/results/local_bge_20261014"
RUNTIME = ROOT / "experiments/runtime/local_bge"


def main():
    pid = json.loads((RUNTIME / "service.json").read_text())["pid"]
    stop = threading.Event()
    samples = []

    def monitor():
        while not stop.is_set():
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True
            )
            memory = [int(row.split(",")[1]) for row in output.splitlines() if row.split(",")[0].strip() == str(pid)]
            status = dict(
                line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line
            )
            samples.append(
                {
                    "time": time.time(),
                    "gpu_used_mib": sum(memory),
                    "rss_kib": int(status["VmRSS"].split()[0]),
                    "hwm_kib": int(status["VmHWM"].split()[0]),
                }
            )
            stop.wait(0.2)

    observer = threading.Thread(target=monitor)
    observer.start()
    commands = [
        ([sys.executable, "-m", "scripts.local_bge.probe", "--output", str(OUT / "host")], "host-probe.log"),
        ([sys.executable, "-m", "scripts.local_bge.native_probe"], "native-probe.log"),
    ]
    results = []
    try:
        for command, log in commands:
            start = time.monotonic()
            with (OUT / log).open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
            results.append(
                {
                    "command": command,
                    "cwd": str(ROOT),
                    "exit_code": result.returncode,
                    "seconds": time.monotonic() - start,
                    "log": str(OUT / log),
                }
            )
            if result.returncode:
                raise RuntimeError(f"Acceptance failed: {log}")
    finally:
        stop.set()
        observer.join()
        (OUT / "resource-samples.json").write_text(json.dumps(samples, indent=2))
        (OUT / "acceptance-commands.json").write_text(json.dumps(results, indent=2))

    def size(path):
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())

    peak = max(row["gpu_used_mib"] for row in samples)
    host = json.loads((OUT / "host/result.json").read_text())
    native = json.loads((OUT / "native/output/result.json").read_text())
    resources = {
        "model_bytes": size(RUNTIME / "model"),
        "deployment_bytes": size(RUNTIME),
        "deployment_disk_usage_bytes": int(
            subprocess.check_output(["du", "-s", "--block-size=1", str(RUNTIME)], text=True).split()[0]
        ),
        "install_cache_bytes": size(RUNTIME / "install-cache"),
        "sampled_gpu_peak_mib": peak,
        "rss_peak_bytes": max(row["hwm_kib"] for row in samples) * 1024,
        "cuda_allocator_peak_bytes": native["health"]["cuda_peak_allocated_bytes"],
        "cuda_reserved_peak_bytes": native["health"]["cuda_peak_reserved_bytes"],
        "load_seconds": host["health"]["load_seconds"],
        "disk_budget_bytes": 20 * 1024**3,
        "gpu_budget_mib": 12 * 1024,
        "external_embedding_api_cost_usd": 0,
        "hardware_cost": "unpriced",
    }
    resources["budget_passed"] = (
        resources["deployment_bytes"] <= resources["disk_budget_bytes"] and peak <= resources["gpu_budget_mib"]
    )
    (OUT / "resources.json").write_text(json.dumps(resources, indent=2))
    assert resources["budget_passed"], resources
    print(json.dumps({"status": "passed", "resources": resources}))


if __name__ == "__main__":
    main()
