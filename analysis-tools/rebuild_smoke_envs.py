#!/usr/bin/env python3
"""Rebuild the five deleted T7 smoke environments from frozen receipts.

The native gate re-verifies ``native-gate-provision.json`` (smoke IDs) before
every holdout job.  The user-directed dev cleanup removed those five conda
prefixes and baseline repositories.  This tool replays the frozen ``recipe.json``
commands at the original path strings, restores the original evidence files,
and verifies the frozen receipt byte for byte.  It never rewrites the gate
snapshot or any receipt.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from moatless.evaluation.experiment import load_manifest, now, read_json, write_json
from moatless.evaluation.native_provision import NativeProvisioner

CONDA = "/mnt/data2/home/geqb/anaconda3/bin/conda"
OUTPUT = "smoke-env-rebuild.json"


def run(lines, cwd, log, env):
    script = "#!/bin/bash\nset -euo pipefail\n" + "\n".join(lines) + "\n"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        stream.write(f"\n===== {now()} =====\n".encode())
        stream.write(script.encode())
        stream.flush()
        result = subprocess.run(["/bin/bash", "-c", script], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed (exit {result.returncode}); see {log}")
    return log.read_text(errors="replace")


def clone(repo_url, commit, destination, log, env):
    destination.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, 5):
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        subprocess.run(["git", "init", str(destination)], check=True,
                       env=env, capture_output=True, text=True)
        command = ["git", "-C", str(destination), "-c", "http.version=HTTP/1.1",
                   "fetch", "--depth=1", repo_url, commit]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        with log.open("a") as stream:
            stream.write(f"clone attempt {attempt}: {' '.join(command)}\n")
            stream.write(result.stdout + result.stderr + "\n")
        if result.returncode == 0:
            last = None
            break
        last = result.stderr or result.stdout
        time.sleep(5)
    if last is not None:
        # Fallback: fetch the repository's default branch without blobs, then
        # check out the exact base commit.
        if destination.exists():
            shutil.rmtree(destination)
        result = subprocess.run(
            ["git", "-c", "http.version=HTTP/1.1", "clone", "--filter=blob:none",
             "--no-checkout", repo_url, str(destination)],
            env=env, capture_output=True, text=True)
        with log.open("a") as stream:
            stream.write("clone fallback\n" + result.stdout + result.stderr + "\n")
        if result.returncode:
            raise RuntimeError(f"git clone failed: {last}")
    result = subprocess.run(["git", "-C", str(destination), "checkout", "--detach", commit],
                            env=env, capture_output=True, text=True)
    with log.open("a") as stream:
        stream.write("checkout\n" + result.stdout + result.stderr + "\n")
    if result.returncode:
        raise RuntimeError(f"git checkout failed: {result.stderr}")
    remote = subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", repo_url],
                            env=env, capture_output=True, text=True)


def prune_stale_conda_records(parent, receipt, log, env):
    """Drop conda-meta records for packages already replaced by pip.

    The frozen receipt records the post-pip environment.  A conda package whose
    files were uninstalled and replaced by a pypi entry is no longer installed
    by conda, so its stale conda-meta record must not be presented as current.
    Only records absent from the receipt and superseded by a pypi entry are
    removed; any other extra record is an error.
    """
    prefix = parent / "conda"
    expected = set(receipt["conda_metadata"])
    # prepare() records dependencies with this exact call; conda also drops
    # stale conda-meta records for packages whose files pip replaced.
    result = subprocess.run([CONDA, "list", "--prefix", str(prefix), "--json"],
                            env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"conda list failed: {result.stderr}")
    dependencies = json.loads(result.stdout)
    pypi = {row["name"] for row in dependencies if row.get("platform") == "pypi"}
    extras = {p.name: p for p in (prefix / "conda-meta").glob("*.json")
              if p.name not in expected}
    if not extras:
        return []
    removed = []
    for name, path in sorted(extras.items()):
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record.get("name") not in pypi:
            raise ValueError(f"unexpected extra conda record: {name}")
        path.unlink()
        removed.append(name)
    with log.open("a") as stream:
        stream.write("pruned stale conda records: " + ", ".join(removed) + "\n")
    return removed


def rebuild(candidate, manifest, gate, identity, prepared):
    receipt = prepared["environment"]
    parent = Path(receipt["repo"]).parent
    root = parent.parent.parent
    entry = {"instance_id": identity, "parent": str(parent), "timestamp": now()}
    try:
        NativeProvisioner.verify(receipt)
        entry["status"] = "already_valid"
        return entry
    except Exception as exc:
        entry["before"] = f"{type(exc).__name__}: {exc}"

    recipe = read_json(parent / "recipe.json")
    if not recipe:
        entry["status"] = "failed"
        entry["error"] = "recipe.json is missing; cannot replay the frozen preparation"
        return entry
    backup_root = root / "rebuild-backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f"{parent.name}-{int(time.time())}"
    if backup.exists():
        entry["status"] = "failed"
        entry["error"] = f"backup already exists: {backup}"
        return entry
    for name in ("conda", "baseline"):
        if (parent / name).exists():
            shutil.rmtree(parent / name)
    parent.rename(backup)
    parent.mkdir(parents=True)
    try:
        # Preserve every frozen evidence file before replaying the recipe.
        restored = 0
        for name in receipt.get("evidence", {}):
            source = backup / name
            if not source.exists():
                raise FileNotFoundError(f"original evidence file is missing: {source}")
            shutil.copy2(source, parent / name)
            restored += 1
        shutil.copy2(backup / "ready.json", parent / "ready.json")

        logs = root / "rebuild-logs" / identity
        env = {
            "HOME": str(Path.home()),
            "PATH": f"{Path(CONDA).parent}:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "CONDA_PKGS_DIRS": str(root / "conda-packages"),
            "CONDA_ENVS_PATH": str(root / "named-environments"),
            "PIP_CACHE_DIR": str(root / "pip-cache"),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
                     "https_proxy", "http_proxy", "all_proxy", "no_proxy",
                     "SSL_CERT_FILE", "PIP_INDEX_URL"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        run(recipe["commands"], parent, logs / "environment.log", env)
        clone(f"https://github.com/{recipe['public']['repo']}.git",
              recipe["public"]["base_commit"], parent / "baseline",
              logs / "clone.log", env)
        setup_patch = backup / "setup.patch"
        if setup_patch.exists() and setup_patch.stat().st_size:
            patch = subprocess.run(
                ["git", "-C", str(parent / "baseline"), "apply", "--binary", str(setup_patch)],
                env=env, capture_output=True, text=True)
            with (logs / "clone.log").open("a") as stream:
                stream.write("setup.patch\n" + patch.stdout + patch.stderr + "\n")
            if patch.returncode:
                raise RuntimeError(f"setup.patch failed: {patch.stderr}")
        run(recipe["install_commands"], parent, logs / "install.log", env)
        pruned = prune_stale_conda_records(parent, receipt, logs / "install.log", env)
        NativeProvisioner.verify(receipt)
        entry.update({"status": "rebuilt", "restored": restored,
                      "pruned_stale_conda_records": pruned,
                      "recipe_identity": receipt.get("recipe_identity"),
                      "backup": str(backup)})
    except Exception as exc:
        failed = parent.with_name(f"{parent.name}.rebuild-failed-{int(time.time())}")
        if parent.exists():
            parent.rename(failed)
        backup.rename(parent)
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["failed_artifacts"] = str(failed)
        entry["backup"] = str(parent)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--only", action="append", default=None)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    manifest = load_manifest(candidate)
    gate = read_json(candidate / "native-gate-provision.json")
    selected = args.only or list(gate)
    results = []
    for identity in selected:
        if identity not in gate:
            raise SystemExit(f"{identity} is not in the frozen smoke gate")
        result = rebuild(candidate, manifest, gate, identity, gate[identity])
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        write_json(candidate / OUTPUT, {
            "manifest_hash": manifest["manifest_hash"], "timestamp": now(),
            "gate_snapshot_sha256": hashlib.sha256(
                (candidate / "native-gate-provision.json").read_bytes()).hexdigest(),
            "rows": results,
            "verified": all(row["status"] in ("already_valid", "rebuilt") for row in results),
        })
    if len(results) != len(gate):
        print(json.dumps({"partial": True, "rows": len(results)}), flush=True)
        return
    if not all(row["status"] in ("already_valid", "rebuilt") for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
