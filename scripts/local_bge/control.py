"""Start/stop/status for the independent loopback service, run from repo root."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "experiments/runtime/local_bge"
PIDFILE = RUNTIME / "service.json"


def identity(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()
        return None if state[2] == "Z" else state[21]
    except FileNotFoundError:
        return None


def alive():
    if not PIDFILE.exists():
        return None
    state = json.loads(PIDFILE.read_text())
    return state if identity(state["pid"]) == state["start_ticks"] else None


def health():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8787/health", timeout=5) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    action = parser.parse_args().action
    state = alive()
    if action == "stop":
        if state:
            os.kill(state["pid"], signal.SIGTERM)
            for _ in range(100):
                if identity(state["pid"]) != state["start_ticks"]:
                    break
                time.sleep(0.1)
            if alive():
                raise TimeoutError("Service is still stopping; PID record preserved")
        PIDFILE.unlink(missing_ok=True)
        print("stopped")
        return
    if action == "status":
        print(json.dumps(health(), indent=2))
        return
    if state:
        print(json.dumps(health(), indent=2))
        return
    # Refuse to mistake another listener for this managed process.
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8787))
    env = {k: v for k, v in os.environ.items() if not ("KEY" in k or "TOKEN" in k)}
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        BGE_MODEL_DIR=str(RUNTIME / "model"),
        HF_HOME=str(RUNTIME / "hf-cache"),
        TORCH_HOME=str(RUNTIME / "torch-cache"),
        XDG_CACHE_HOME=str(RUNTIME / "cache"),
    )
    with (RUNTIME / "service.log").open("a") as log:
        process = subprocess.Popen(
            [str(RUNTIME / "venv/bin/python"), "-m", "scripts.local_bge.server"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    PIDFILE.write_text(json.dumps({"pid": process.pid, "start_ticks": identity(process.pid)}))
    print(json.dumps({"pid": process.pid, "log": str(RUNTIME / "service.log")}))
    for _ in range(240):
        if process.poll() is not None:
            raise RuntimeError(f"Service exited {process.returncode}; see {RUNTIME / 'service.log'}")
        try:
            print(json.dumps(health(), indent=2))
            return
        except (OSError, ValueError):
            time.sleep(0.5)
    raise TimeoutError("Service is still starting; inspect status/log")


if __name__ == "__main__":
    main()
