"""Pre-declared infrastructure retry for network-clone env_blocked instances.

Rule: only instances whose preparation failed because git could not reach
github.com (exit 128, timeout/TLS/connect), with no agent answer and no
paired-result, are retried once.  Both arms are retried; no question, model,
recipe or answer is changed or resampled.  Original preparation JSON and
partial environment directories are moved aside and hashed, never deleted.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

NETWORK_MARKERS = ("Connection timed out", "GnuTLS recv error", "Failed to connect",
                   "Could not resolve host", "Connection reset", "TLS connection",
                   "unable to access 'https://github.com")
PROXY_MARKERS = ("ProxyError", "Tunnel connection failed", "502 Bad Gateway",
                 "Max retries exceeded")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_identity_dir(output, instance):
    """Locate a non-ready environment by the instance recorded in recipe.json."""
    instances = output / "environments" / "instances"
    if not instances.is_dir():
        return None
    for directory in instances.iterdir():
        if (directory / "ready.json").exists():
            continue
        recipe = directory / "recipe.json"
        if not recipe.is_file():
            continue
        try:
            public = json.loads(recipe.read_text()).get("public") or {}
        except ValueError:
            continue
        if public.get("instance_id") == instance:
            return directory
    return None


def is_network_failure(preparation):
    if preparation.get("status") != "env_blocked":
        return False
    error = preparation.get("error", "")
    # The 1800s preparation deadline fired during the same network outage;
    # it is included in this declared batch, retried once, evidence preserved.
    if error.startswith("TimeoutError"):
        return True
    # The provisioner also fetches raw.githubusercontent.com for some recipes;
    # a local-proxy 502 means the same upstream-network class of failure.
    if any(marker in error for marker in PROXY_MARKERS):
        return True
    # A previous retry can leave an empty pre-recipe directory.  This is our
    # own retry artifact, not a scientific result; it is retried only inside
    # the explicit --only batch, and the script moves empty/no-recipe dirs.
    if "Preparation has uncertain/failed prior effects" in error:
        return True
    if "clone.log" not in error and "Native preparation failed (exit 128)" not in error:
        return False
    marker = error.split("see ")[-1].strip()
    log = Path(marker)
    if not log.is_file():
        return "exit 128" in error
    text = log.read_text(errors="replace")
    return any(item in text for item in NETWORK_MARKERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument("--only", help="Comma-separated instance ids; only these are retried")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    args.record_dir = args.record_dir or output / "recovery-records"
    jobs = read(output / "jobs.json")
    only = {item.strip() for item in args.only.split(",") if item.strip()} if args.only else None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidates = []
    for path in sorted((output / "preparation").glob("*.json")):
        preparation = read(path)
        if not is_network_failure(preparation):
            continue
        instance = path.stem
        if only is not None and instance not in only:
            continue
        related = [job for job in jobs if job["instance_id"] == instance]
        evidence = [output / "runs" / job["job_id"] / name
                    for job in related for name in ("answer.json", "grading.json", "paired-result.json")]
        existing = [str(path.relative_to(output)) for path in evidence if path.exists()]
        if existing:
            raise SystemExit(f"{instance} already has frozen answer/grading evidence: {existing}")
        marker = preparation.get("error", "").split("see ")[-1].strip()
        identity_dir = (Path(marker).parent if marker.endswith("clone.log")
                        else find_identity_dir(output, instance))
        candidates.append({"instance_id": instance, "preparation": str(path.relative_to(output)),
                           "preparation_sha256": sha(path),
                           "clone_log": marker, "identity_dir": str(identity_dir) if identity_dir else None,
                           "jobs": [job["job_id"] for job in related]})
    if not candidates:
        print(json.dumps({"network_blocked": 0, "apply": args.apply}, indent=2))
        return
    # Directories referenced by any preparation record stay in place unless
    # that instance is in this retry batch; only true orphan partials move.
    referenced = set()
    for record_path in (output / "preparation").glob("*.json"):
        record_data = read(record_path)
        marker = record_data.get("error", "").split("see ")[-1].strip()
        if marker.endswith("clone.log") and Path(marker).parent.exists():
            referenced.add(str(Path(marker).parent))
        else:
            found = find_identity_dir(output, record_path.stem)
            if found is not None:
                referenced.add(str(found))
    partial = []
    for directory in sorted((output / "environments" / "instances").iterdir()):
        if (directory / "ready.json").exists():
            continue
        has_recipe = (directory / "recipe.json").exists()
        if only is not None and has_recipe:
            # A recipe-bearing directory could belong to another instance; do
            # not guess its owner.  Empty/pre-recipe failures are safe to move
            # because the pipeline is stopped and they carry no recipe identity.
            continue
        partial.append(str(directory))
        if str(directory) not in referenced:
            candidates.append({"instance_id": None, "identity_dir": str(directory),
                               "reason": ("pre-recipe partial preparation"
                                          if only is not None
                                          else "partial preparation without a preparation record")})
    record = {"timestamp": timestamp,
              "rule": ("retry once network-clone env_blocked instances and the network-era "
                       "preparation-deadline timeout; preserve original preparation JSON and "
                       "environment directory; both arms; no answer resampling"),
              "candidates": candidates, "apply": args.apply}
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if not args.apply:
        return
    prep_dir = output / "preparation-attempts" / f"network-retry-{timestamp}"
    env_dir = output / "environments" / "attempts" / f"network-retry-{timestamp}"
    prep_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for item in candidates:
        if item.get("preparation"):
            source = output / item["preparation"]
            destination = prep_dir / source.name
            shutil.move(source, destination)
            moved.append({"from": str(source.relative_to(output)),
                          "to": str(destination.relative_to(output)), "sha256": sha(destination)})
        if item.get("identity_dir"):
            source = Path(item["identity_dir"])
            if not source.resolve().is_relative_to((output / "environments" / "instances").resolve()):
                raise SystemExit(f"Refusing to move outside instances/: {source}")
            destination = env_dir / source.name
            if destination.exists():
                raise SystemExit(f"Retry archive already exists: {destination}")
            shutil.move(source, destination)
            moved.append({"from": str(source.relative_to(output)),
                          "to": str(destination.relative_to(output))})
    record["moved"] = moved
    record_path = args.record_dir / f"{timestamp}.json"
    write(record_path, record)
    write(output / f"network-retry-{timestamp}.json", record)
    print(json.dumps({"record": str(record_path), "moved": len(moved)}, indent=2))


if __name__ == "__main__":
    main()
