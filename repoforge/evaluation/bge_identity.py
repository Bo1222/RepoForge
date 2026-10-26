"""Host-only identity checks for the separately deployed BGE service."""

import json
from pathlib import Path
import subprocess
import urllib.request

from repoforge.evaluation.experiment import sha


SPEC = {"status": "ok", "model": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "dimensions": 1024, "max_input_tokens": 2048,
        "batch_size": 8, "precision": "float16"}


def check_health(base_url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base_url.rstrip("/").removesuffix("/v1") + "/health", timeout=5) as response:
        health = json.load(response)
    if any(health.get(key) != value for key, value in SPEC.items()):
        raise ValueError("BGE service identity differs from the frozen specification")
    return health


def capture_identity(root, base_url):
    root = Path(root)
    runtime = root / "experiments/runtime/local_bge"
    model = json.loads((runtime / "model-manifest.json").read_text())
    if model["model"] != SPEC["model"] or model["revision"] != SPEC["revision"]:
        raise ValueError("BGE model manifest differs from the service specification")
    files = {str((runtime / "model" / name).relative_to(root)): item["sha256"]
             for name, item in model["files"].items()}
    for name in ("scripts/local_bge/requirements.lock", "scripts/local_bge/server.py",
                 "experiments/runtime/local_bge/model-manifest.json"):
        files[name] = sha(root / name)
    for name, expected in files.items():
        if sha(root / name) != expected:
            raise ValueError(f"BGE deployment file changed: {name}")
    dependencies = subprocess.check_output([str(runtime / "venv/bin/python"), "-c",
        "import importlib.metadata,json; print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}))"], text=True)
    dependencies = json.loads(dependencies)
    for line in (root / "scripts/local_bge/requirements.lock").read_text().splitlines():
        name, version = line.split("==")
        if dependencies.get(name) != version:
            raise ValueError(f"BGE service dependency differs from lock: {name}")
    return {"specification": SPEC, "files": files, "dependencies": dependencies,
            "health_at_freeze": check_health(base_url), "external_api_cost_usd": 0,
            "hardware_cost": "unpriced"}


def validate_identity(root, manifest):
    expected = manifest.get("embedding_service")
    if expected is None:
        raise ValueError("Local BGE execution requires a new manifest with deployment identity")
    current = capture_identity(root, manifest["embedding_settings"]["embed_base_url"])
    for key in ("specification", "files", "dependencies"):
        if current[key] != expected[key]:
            raise ValueError(f"BGE service {key} changed after candidate freeze")
    return current["health_at_freeze"]
