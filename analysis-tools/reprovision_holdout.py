#!/usr/bin/env python3
"""Re-provision missing/failed holdout environments with a private libmamba conda.

Read-only for the frozen Agent source.  It preserves failed preparation
directories, writes verified receipts into provision-native.json, and records
solver/recipe deviations.  It never runs a holdout job and never changes the
manifest, lock, job IDs or answers.
"""

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from repoforge.evaluation.experiment import content_hash, now, public_instance, read_json, sha, write_json, append_json
from repoforge.evaluation.native_provision import (
    ASTROPY_COMPATIBILITY_BOOTSTRAP, PYTHON36_BOOTSTRAP, SCIKIT_CYTHON_BOOTSTRAP,
    SCIKIT_MODERN_BOOTSTRAP, SCIKIT_PYTEST_BOOTSTRAP, NativeProvisioner,
)
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

DEFAULT_CONDA = REPO_ROOT / "experiments/runtime/miniforge-t7/bin/conda"


class FixProvisioner(NativeProvisioner):
    """Same provisioner, private conda and an explicit longer env-solve timeout."""

    def __init__(self, directory, *, conda, timeout=1800, extra_environment=None, **kwargs):
        super().__init__(directory, conda=conda, **kwargs)
        self.fix_timeout = timeout
        self.started = time.monotonic()
        self.fix_environment = extra_environment or {}

    async def _run(self, argv, cwd, log, *, timeout=1800, extra_environment=None):
        remaining = self.fix_timeout - (time.monotonic() - self.started)
        if remaining <= 0:
            raise TimeoutError(
                f"environment preparation exceeded the {self.fix_timeout}-second deadline")
        text = " ".join(str(part) for part in argv)
        if any(marker in text for marker in ("prepare-env.sh", "install-repo.sh", "conda")):
            timeout = min(self.fix_timeout, remaining)
        else:
            timeout = min(timeout, remaining)
        environment = {**self.fix_environment, **(extra_environment or {})}
        return await super()._run(argv, cwd, log, timeout=timeout,
                                  extra_environment=environment)


def identity_for(manifest, public, setup_commit):
    """Mirror NativeProvisioner.prepare's environment identity."""
    specs = MAP_REPO_VERSION_TO_SPECS[public["repo"]][public["version"]]
    parts = [public, setup_commit, specs, importlib.metadata.version("swebench")]
    if str(specs["python"]) == "3.6":
        parts.append({"python36_bootstrap": PYTHON36_BOOTSTRAP})
    if public["repo"] == "scikit-learn/scikit-learn" and public["version"] == "0.22":
        parts.append({"cython_bootstrap": SCIKIT_CYTHON_BOOTSTRAP})
        parts.append({"pytest_bootstrap": SCIKIT_PYTEST_BOOTSTRAP})
    if public["repo"] == "scikit-learn/scikit-learn" and public["version"] == "1.3":
        parts.append({"modern_bootstrap": SCIKIT_MODERN_BOOTSTRAP})
    if public["repo"] == "astropy/astropy":
        parts.append({"compatibility_bootstrap": ASTROPY_COMPATIBILITY_BOOTSTRAP})
    if public["repo"] == "matplotlib/matplotlib":
        parts.append({"public_tools": manifest["native"]["public_tools"]["digest"],
                      "adaptation": "private-media-tex-sources-v1"})
    return content_hash(parts)


def move_aside(environment_root, target, reason):
    archive = environment_root / "rejected-provision" / (
        f"{target.name}-{reason}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(archive))
    return archive


def complete_psf_receipt(target):
    """The frozen module map lacks psf/requests; finish its receipt externally."""
    target = Path(target)
    recipe = read_json(target / "recipe.json")
    if not recipe or not (target / "conda").is_dir() or not (target / "dependencies.json").exists():
        raise RuntimeError("psf preparation did not reach the import-probe stage")
    public, prefix, repo = recipe["public"], target / "conda", target / "baseline"
    probe = ("import importlib,json,sys; from pathlib import Path; "
             "m=importlib.import_module('requests'); p=Path(m.__file__).resolve(); "
             f"assert Path({str(repo)!r}).resolve() in p.parents, str(p); "
             "print(json.dumps({'python':sys.version, 'executable':sys.executable, 'source':str(p)}))")
    environment = {"PATH": f"{prefix / 'bin'}:/usr/bin:/bin", "LANG": "C.UTF-8",
                   "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    with (target / "import-probe.json").open("wb") as stream:
        result = subprocess.run([str(prefix / "bin/python"), "-c", probe], cwd=repo,
                                env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("psf requests import probe failed")
    with (target / "system-libraries.log").open("wb") as stream:
        subprocess.run(["ldd", str(prefix / "bin/python")], cwd=repo, stdout=stream,
                       stderr=subprocess.STDOUT)
    setup = subprocess.run(["git", "-C", str(repo), "diff", "--binary", public["base_commit"]],
                           capture_output=True)
    if setup.returncode:
        raise RuntimeError("psf setup.patch generation failed")
    (target / "setup.patch").write_bytes(setup.stdout)
    receipt = {"version": 1, "public": public, "recipe_identity": target.name,
               "repo": str(repo), "prefix": str(prefix), "python": recipe["specs"]["python"],
               "dependencies": content_hash(read_json(target / "dependencies.json")),
               "conda_metadata": NativeProvisioner.metadata(prefix),
               "package_metadata": NativeProvisioner.package_metadata(prefix),
               "setup_patch_sha256": sha(target / "setup.patch"),
               "setup_has_tracked_changes": bool(setup.stdout.strip()),
               "created": now(), "host": {"os": platform.platform(), "machine": platform.machine()},
               "evidence": {p.name: sha(p) for p in target.iterdir() if p.is_file()},
               "execution_backend": "native", "official_harness": "not_run",
               "provider_calls": 0, "provision_solver": "reused_classic"}
    write_json(target / "ready.json", receipt)
    NativeProvisioner.verify(receipt)
    return receipt


def instance_dirs(environment_root, public, identity):
    """All preparation directories for one instance, including spec-mutated identities."""
    directories = []
    target = environment_root / "instances" / identity
    if target.exists():
        directories.append(target)
    for recipe in environment_root.glob("instances/*/recipe.json"):
        try:
            recorded = read_json(recipe)
        except Exception:
            continue
        if not recorded:
            continue
        if recorded.get("public") == public and recipe.parent not in directories:
            directories.append(recipe.parent)
    ready = [directory for directory in directories if (directory / "ready.json").exists()]
    return directories, ready


def provision_home(environment_root):
    """Private HOME so conda ignores the account-level channels/solver config."""
    home = Path(environment_root) / "conda-home"
    home.mkdir(parents=True, exist_ok=True)
    path = home / ".condarc"
    if not path.exists():
        path.write_text(
            "channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/\n"
            "channels:\n"
            "  - conda-forge\n"
            "channel_priority: strict\n"
            "solver: libmamba\n"
            "ssl_verify: false\n"
            "show_channel_urls: true\n")
    return home


def prepare_one(manifest, environment_root, public, setup_commit, *, conda, timeout, only=None):
    repo = public["repo"]
    identity = identity_for(manifest, public, setup_commit)
    target = environment_root / "instances" / identity
    directories, ready = instance_dirs(environment_root, public, identity)
    if ready:
        receipt = read_json(ready[0] / "ready.json")
        NativeProvisioner.verify(receipt)
        return receipt, {"action": "already_ready", "identity": receipt["recipe_identity"],
                         "target": str(ready[0])}
    if repo == "psf/requests":
        for directory in directories:
            if (directory / "recipe.json").exists() and (directory / "conda").is_dir():
                receipt = complete_psf_receipt(directory)
                return receipt, {"action": "manual_psf_receipt",
                                 "identity": receipt["recipe_identity"], "target": str(directory)}
    archives = [str(move_aside(environment_root, directory, "prior-failure"))
                for directory in directories]
    extra = {"PATH": "/usr/bin:/bin", "HOME": str(provision_home(environment_root))}
    specs = MAP_REPO_VERSION_TO_SPECS[repo][public["version"]]
    original_pre_install = specs.get("pre_install")
    original_pip_packages = specs.get("pip_packages")
    skipped_pre_install = False
    pinned_packages = None
    if repo == "matplotlib/matplotlib" and original_pip_packages and any(
            package == "pikepdf" for package in original_pip_packages):
        # The official environment leaves pikepdf unpinned; pip backtracks
        # through many releases against the Tsinghua mirror.  Pin the version
        # resolved by matplotlib 3.5 (8.4.1) for the remaining instances.
        specs["pip_packages"] = [
            "pikepdf==8.4.1" if package == "pikepdf" else package
            for package in original_pip_packages]
        pinned_packages = "pikepdf==8.4.1"
    if repo == "django/django" and public["instance_id"] == "django__django-7530":
        # The host already provides en_US.utf8; do not run apt-get / edit /etc.
        specs["pre_install"] = []
        skipped_pre_install = True
        # extend_environment() requires PATH when extra_environment is nonempty.
        extra.update({"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"})
    if public["instance_id"] == "matplotlib__matplotlib-13989":
        # matplotlib 3.0 builds ft2font from the public freetype sources copied
        # into repo/build; the official image supplies libfreetype6-dev via
        # apt-get.  Point the build at those public headers and the host runtime
        # library instead of changing host packages.
        build_include = target / "baseline/build/freetype-2.6.1/include"
        fix_root = environment_root / "fix-libs" / "freetype"
        libdir = fix_root / "lib"
        pkgconfig = libdir / "pkgconfig"
        pkgconfig.mkdir(parents=True, exist_ok=True)
        host_lib = Path("/usr/lib/x86_64-linux-gnu/libfreetype.so.6")
        link = libdir / "libfreetype.so"
        if host_lib.exists() and not link.exists():
            link.symlink_to(host_lib)
        (pkgconfig / "freetype2.pc").write_text(
            "prefix={}\n".format(build_include.parent.parent) +
            "includedir={}\n".format(build_include) +
            "libdir={}\n".format(libdir) +
            "Name: freetype2\nVersion: 2.6.1\n"
            "Cflags: -I${includedir}\nLibs: -L${libdir} -lfreetype\n")
        extra.update({"PATH": "/usr/bin:/bin",
                      "CPPFLAGS": f"-I{build_include}", "CFLAGS": f"-I{build_include}",
                      "LDFLAGS": f"-L{libdir}", "PKG_CONFIG_PATH": str(pkgconfig),
                      "PKG_CONFIG_LIBDIR": str(pkgconfig)})
        skipped_pre_install = True  # audit marker: controlled header adaptation
    provisioner = FixProvisioner(environment_root, conda=conda, timeout=timeout,
                                 extra_environment=extra,
                                 public_tools=manifest["native"].get("public_tools"))
    try:
        receipt = asyncio.run(provisioner.prepare(
            public, environment_setup_commit=setup_commit))
    finally:
        if skipped_pre_install:
            specs["pre_install"] = original_pre_install
        if pinned_packages:
            specs["pip_packages"] = original_pip_packages
    return receipt, {"action": "reprovisioned", "identity": identity,
                     "target": str(environment_root / "instances" / receipt["recipe_identity"]),
                     "archives": archives,
                     "skipped_pre_install": skipped_pre_install,
                     "pinned_packages": pinned_packages,
                     "solver": "libmamba"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Total per-instance environment deadline (seconds); default 30 minutes")
    parser.add_argument("--only", help="Comma-separated instance IDs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    manifest = read_json(candidate / "manifest.json")
    environment_root = Path(manifest["native"]["environment_root"]).resolve()
    public_rows = read_json(candidate / "public/instances.json")
    recipes = read_json(candidate / "host-install-recipes.json")
    receipt_path = candidate / "provision-native.json"
    receipts = read_json(receipt_path) if receipt_path.exists() else {}
    holdout_ids = list(manifest["split"]["holdout"])
    only = {item.strip() for item in args.only.split(",")} if args.only else None
    missing = [identity for identity in holdout_ids if identity not in receipts
               and (not only or identity in only)]
    before = len([identity for identity in holdout_ids if identity in receipts])
    print(json.dumps({"holdout": len(holdout_ids), "ready_before": before,
                      "missing": len(missing), "conda": str(args.conda),
                      "timeout": args.timeout, "dry_run": args.dry_run}, indent=2))
    if args.dry_run:
        return
    if not args.conda.is_file():
        raise SystemExit("Private conda missing: " + str(args.conda))
    solver = subprocess.run([str(args.conda), "config", "--show", "solver"],
                            capture_output=True, text=True)
    print("private solver:", solver.stdout.strip())
    audit = {"timestamp": now(), "manifest_hash": manifest["manifest_hash"],
             "case": str(candidate), "private_conda": str(args.conda),
             "private_conda_version": subprocess.run(
                 [str(args.conda), "--version"], capture_output=True, text=True).stdout.strip(),
             "solver": solver.stdout.strip(), "timeout": args.timeout,
             "ready_before": before, "rows": [], "failures": []}
    for identity in missing:
        public = public_rows[identity]
        setup = recipes.get(identity, {}).get("environment_setup_commit")
        print(f"=== {identity} ===", flush=True)
        try:
            receipt, detail = prepare_one(manifest, environment_root, public, setup,
                                          conda=args.conda, timeout=args.timeout)
            NativeProvisioner.verify(receipt)
            receipts[identity] = {"environment": receipt,
                                  "source_sha256": manifest["source"]["sha256"],
                                  "public": receipt["public"]}
            write_json(receipt_path, receipts)
            append_json(candidate / "holdout-provision.jsonl",
                        {"instance_id": identity, "status": "ready", "timestamp": now(),
                         "repair": detail.get("action"), "solver": detail.get("solver")})
            audit["rows"].append({"instance_id": identity, **detail})
            print({"instance_id": identity, "action": detail["action"],
                   "identity": receipt["recipe_identity"]}, flush=True)
        except Exception as exc:
            audit["failures"].append({"instance_id": identity,
                                      "type": type(exc).__name__, "message": str(exc)})
            append_json(candidate / "holdout-provision.jsonl",
                        {"instance_id": identity, "status": "failed",
                         "type": type(exc).__name__, "message": str(exc),
                         "timestamp": now(), "repair": "skipped_after_timeout"})
            print({"instance_id": identity, "failed": type(exc).__name__,
                   "message": str(exc)[:300]}, flush=True)
    ready_after = len([identity for identity in holdout_ids if identity in receipts])
    audit["ready_after"] = ready_after
    audit["missing_after"] = [identity for identity in holdout_ids if identity not in receipts]
    write_json(candidate / "provision-fix-run.json", audit)
    write_json(candidate / "holdout-provision-summary.json", {
        "manifest_hash": manifest["manifest_hash"], "selected": len(holdout_ids),
        "ready": ready_after, "failed": audit["missing_after"],
        "solver": "mixed: classic for pre-existing ready; libmamba for repaired",
        "provider_calls": 0, "timestamp": now()})
    print(json.dumps({"ready_after": ready_after, "missing": audit["missing_after"]}, indent=2))
    # The user-directed rule is to skip environments that cannot finish in the
    # per-instance deadline; missing instances are recorded and handled by
    # scan-holdout / adjudication, not by failing the reprovision command.


if __name__ == "__main__":
    main()
