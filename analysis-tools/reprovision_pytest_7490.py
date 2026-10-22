#!/usr/bin/env python3
"""Re-provision pytest-7490 with the current native provisioner.

Option A: create a new environment root and a fresh pytest instance.  The
current provisioner exports ``SETUPTOOLS_SCM_PRETEND_VERSION=<instance.version>``
for pytest, so the generated ``src/_pytest/_version.py`` satisfies
``pyproject.toml``'s ``minversion = "2.0"``.  The old conda/pip package caches
and the verified public source are reused through the provisioner's own
``verified_cached_source`` path; historical experiment directories are never
modified.

The script writes:
  <output>/environment/instances/<new-identity>/...  the new environment
  <output>/pytest-reprovision.json                   receipt plus version evidence
  <output>/provision-override.json                   mergeable provision receipt
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from moatless.evaluation.experiment import now, read_json, sha, write_json
from moatless.evaluation.native_provision import NativeProvisioner

INSTANCE = "pytest-dev__pytest-7490"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="T7 reference directory containing provision-native.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    output.mkdir(parents=True)
    receipts = read_json(reference / "provision-native.json")
    old = receipts[INSTANCE]
    old_environment = old["environment"]
    old_repo = Path(old_environment["repo"])
    old_prefix = Path(old_environment["prefix"])
    old_instance = old_repo.parent
    old_root = old_instance.parent.parent
    ready = read_json(old_instance / "ready.json")
    recipe = read_json(old_instance / "recipe.json")
    public = ready["public"]
    if public["instance_id"] != INSTANCE:
        raise SystemExit("reference receipt is not pytest-7490")
    environment_root = output / "environment"
    environment_root.mkdir()
    instances = environment_root / "instances"
    instances.mkdir()
    # A verified cache receipt lets NativeProvisioner reuse the public source
    # without touching the historical instance or fetching from GitHub.
    cache = instances / f"cache-{INSTANCE}"
    cache.mkdir()
    shutil.copy2(old_instance / "ready.json", cache / "ready.json")
    for name in ("conda-packages", "pip-cache", "named-environments"):
        source = old_root / name
        if source.exists():
            (environment_root / name).symlink_to(source, target_is_directory=True)
    provisioner = NativeProvisioner(environment_root)
    receipt = asyncio.run(provisioner.prepare(
        public, environment_setup_commit=recipe.get("environment_setup_commit")))
    version_file = Path(receipt["repo"]) / "src/_pytest/_version.py"
    evidence = {
        "version": 1,
        "created": now(),
        "instance_id": INSTANCE,
        "old_environment": {"repo": str(old_repo), "prefix": str(old_prefix),
                            "ready_sha256": sha(old_instance / "ready.json"),
                            "install_commands": recipe.get("install_commands")},
        "new_environment": receipt,
        "source_environment_root": str(environment_root),
        "version_file": str(version_file),
        "version_file_sha256": sha(version_file),
        "version_file_content": version_file.read_text()[:4000],
        "provisioner_adaptation": "SETUPTOOLS_SCM_PRETEND_VERSION=" + str(public.get("version")),
    }
    write_json(output / "pytest-reprovision.json", evidence)
    write_json(output / "provision-override.json", {
        INSTANCE: {"environment": receipt, "public": receipt["public"], "source_sha256": None}})
    print(json.dumps({
        "instance_id": INSTANCE,
        "new_repo": receipt["repo"],
        "new_prefix": receipt["prefix"],
        "version_file": str(version_file),
        "version_line": next((line for line in version_file.read_text().splitlines()
                              if line.startswith("__version__")), ""),
        "receipt": str(output / "pytest-reprovision.json"),
    }, indent=1))


if __name__ == "__main__":
    main()
