"""Run the synthetic probe inside the existing native filesystem boundary."""

import asyncio
import json
from pathlib import Path
import shutil
import sys

from repoforge.evaluation.experiment import freeze_source, write_json, sha
from repoforge.evaluation.experiment_native import source_directory
from repoforge.evaluation.native_execution import NativeExecutor, NativeMount


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "experiments/results/local_bge_20261014/native"
    output.mkdir(exist_ok=False)
    source = freeze_source(root, output)
    frozen_directory = source_directory(output, {"source": source})
    inputs = output / "inputs"
    inputs.mkdir()
    shutil.copy2(root / "scripts/local_bge/probe.py", inputs / "probe.py")
    results = output / "output"
    results.mkdir()
    framework = Path(sys.prefix).resolve()
    base = Path(sys.base_prefix).resolve()
    runner = NativeExecutor(
        [
            NativeMount(frozen_directory, "/agent"),
            NativeMount(framework, str(framework)),
            NativeMount(base / "bin", str(base / "bin")),
            NativeMount(base / "lib", str(base / "lib")),
            NativeMount(inputs, "/input"),
            NativeMount(results, "/out", True),
        ],
        environment={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "/agent",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        network=True,
    )
    # Loopback HTTP is reachable through the shared host network namespace.
    # No cloud keys, model directory, service venv or NVIDIA devices are mounted.
    command = [str(framework / "bin/python"), "/input/probe.py", "--output", "/out"]
    result = asyncio.run(runner.run(command, cwd="/agent", timeout=180))
    (output / "worker.log").write_text(result.output)
    write_json(
        output / "execution.json",
        {
            "command": command,
            "cwd": "/agent",
            "return_code": result.return_code,
            "seconds": result.seconds,
            "timed_out": result.timed_out,
            "mounts": [{"source": str(m.source), "target": m.target} for m in runner.mounts],
            "source": source,
            "input_script_sha256": sha(inputs / "probe.py"),
            "environment_keys": list(runner.environment()),
            "network": "host shared; loopback reachable",
        },
    )
    print(json.dumps({"return_code": result.return_code, "seconds": result.seconds, "output": str(output)}))
    return result.return_code


if __name__ == "__main__":
    raise SystemExit(main())
