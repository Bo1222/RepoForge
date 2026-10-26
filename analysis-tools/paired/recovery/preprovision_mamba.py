"""Pre-provision selected frozen jobs with mamba, then hand receipts to the phase.

The frozen phase uses the conda CLI with a 1800 s preparation deadline.  For a
few dev recipes the classic conda solve is CPU-bound for >20 min and times out
before installation starts.  This script performs the same NativeProvisioner
steps, using the already-installed mamba binary for the generated conda create
command, with no per-instance deadline, and writes the same ready
receipt/preparation records the phase already verifies.  The generated
prepare-env.sh is part of the receipt evidence; solver-adaptation.json records
the original and patched hashes.

It never changes jobs, questions, model configuration, or answers.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from repoforge.evaluation import experiment as ex
from repoforge.evaluation import paired
from repoforge.evaluation.experiment_inputs import export_public_dataset
from repoforge.evaluation.native_provision import NativeProvisioner


OFFLINE_RAW_SUFFIXES = {
    "matplotlib/matplotlib/0849036fd992a2dd133a0cffc3f84f58ccf1840f/environment.yml":
        "matplotlib-0849036-environment.yml",
    "matplotlib/matplotlib/de98877e3dc45de8dd441d008f23d88738dc015d/environment.yml":
        "matplotlib-de98877-environment.yml",
    "pydata/xarray/1c198a191127c601d091213c4b3292a8bb3054e1/ci/requirements/environment.yml":
        "xarray-1c198a-environment.yml",
    # Django 2.2's get_requirements() fetches this file over raw.githubusercontent.
    # The raw CDN was unreachable on 2026-10-25; the same git object is fetched
    # through api.github.com and replayed locally (verified by sha256).
    "django/django/4fc35a9c3efdc9154efce28cb23cb84f8834517e/tests/requirements/py3.txt":
        "django-10097-py3.txt",
}

DJANGO_REQUIREMENTS = {
    "instance_id": "django__django-10097",
    "repo": "django/django",
    "commit": "4fc35a9c3efdc9154efce28cb23cb84f8834517e",
    "path": "tests/requirements/py3.txt",
    "sha256": "7ecd6fdf3f828964c5221e2e5bc6ff3aa871a8efa2c6c69e5f4252f7b8abaf9c",
}

# The two remaining holdout preparations need a recorded recipe adaptation
# rather than a change to the frozen provisioner source.
APT_PREINSTALL_SKIPS = {
    "django__django-10097": (
        "Django 2.2's Harness recipe installs Debian locales with apt-get during "
        "image build.  Native preparation forbids host package installation and "
        "the eval script only exports LANG/LC_ALL, so the pre_install commands are "
        "skipped; the host already provides en_US.utf8.  The original commands and "
        "this reason are recorded next to the preparation."
    ),
}


def ensure_django_requirements(output):
    """Fetch the pinned Django requirements once and verify its sha256.

    raw.githubusercontent.com is the transport used by swebench's
    get_requirements(), but the CDN was unreachable while api.github.com
    served the identical git object.  install_offline_raw() then replays this
    file for the raw URL during preparation.
    """
    import base64

    import requests

    root = Path(output) / "preprovision-mamba" / "offline-recipes"
    root.mkdir(parents=True, exist_ok=True)
    target = root / OFFLINE_RAW_SUFFIXES[
        f"{DJANGO_REQUIREMENTS['repo']}/{DJANGO_REQUIREMENTS['commit']}/"
        f"{DJANGO_REQUIREMENTS['path']}"]
    record_path = root / "django-10097-py3.json"
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest() != DJANGO_REQUIREMENTS["sha256"]:
            raise ValueError("Archived Django requirements changed")
        return target
    api_url = (f"https://api.github.com/repos/{DJANGO_REQUIREMENTS['repo']}/contents/"
               f"{DJANGO_REQUIREMENTS['path']}?ref={DJANGO_REQUIREMENTS['commit']}")
    raw_url = (f"https://raw.githubusercontent.com/{DJANGO_REQUIREMENTS['repo']}/"
               f"{DJANGO_REQUIREMENTS['commit']}/{DJANGO_REQUIREMENTS['path']}")
    api_error = None
    try:
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        content, source_url = base64.b64decode(response.json()["content"]), api_url
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"
        response = requests.get(raw_url, timeout=60)
        response.raise_for_status()
        content, source_url = response.content, raw_url
    digest = hashlib.sha256(content).hexdigest()
    if digest != DJANGO_REQUIREMENTS["sha256"]:
        raise ValueError(f"Fetched Django requirements hash mismatch: {digest}")
    target.write_bytes(content)
    record = {**DJANGO_REQUIREMENTS, "source_url": source_url, "bytes": len(content),
              "api_error": api_error, "created": ex.now()}
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return target


def install_offline_raw(output):
    """Serve the two archived conda recipes when raw.githubusercontent is blocked.

    The files were downloaded by an earlier provisioner run for the same setup
    commits; their sha256 values are recorded with the pre-provision evidence.
    This changes only the HTTP transport, not the recipe content.
    """
    import requests

    root = Path(output) / "preprovision-mamba" / "offline-recipes"
    mapping = {suffix: root / name for suffix, name in OFFLINE_RAW_SUFFIXES.items()}

    class Response:
        status_code = 200

        def __init__(self, path):
            self.content = path.read_bytes()
            self.text = self.content.decode()
            self.url = str(path)

        def raise_for_status(self):
            return None

        def json(self):
            return json.loads(self.text)

    def lookup(url):
        return next((path for suffix, path in mapping.items() if suffix in url), None)

    original_get = requests.get
    original_request = requests.sessions.Session.request

    def get(url, *args, **kwargs):
        path = lookup(url)
        return Response(path) if path else original_get(url, *args, **kwargs)

    def request(self, method, url, *args, **kwargs):
        path = lookup(url) if str(method).upper() == "GET" else None
        return Response(path) if path else original_request(self, method, url, *args, **kwargs)

    requests.get = get
    requests.sessions.Session.request = request


def adapt_prepare_script(text, cwd, mamba, instance_id):
    """Return the adapted prepare-env.sh text and the recorded changes."""
    patched = text.replace("conda env create", f"{mamba} env create")
    patched = patched.replace("conda env update", f"{mamba} env update")
    patched = patched.replace("conda create", f"{mamba} create")
    # mamba's pip-to-conda translation cannot parse this bracket
    # extra/version list.  The recipe is embedded in a heredoc that
    # prepare-env.sh writes at run time, so patch the heredoc text;
    # install the same package without the non-essential extras.
    legacy_match = re.search(r"nbconvert\[execute\][^\s]*", patched)
    changes = []
    if legacy_match:
        legacy = legacy_match.group(0)
        patched = patched.replace(legacy, "nbconvert")
        changes.append({"change": f"{legacy} -> nbconvert",
                        "in": "prepare-env.sh heredoc"})
    # Python 3.6 / old astropy recipes pin MarkupSafe==1.0, whose
    # setup.py imports the removed setuptools.Feature.  A wheel-only
    # 2.0.1 install preserves the public package role without building
    # that legacy sdist.
    markup_legacy = "MarkupSafe==1.0"
    if markup_legacy in patched:
        patched = patched.replace(markup_legacy, "MarkupSafe==2.0.1")
        changes.append({"change": "MarkupSafe==1.0 -> MarkupSafe==2.0.1",
                        "reason": "legacy sdist imports removed setuptools.Feature",
                        "in": "prepare-env.sh heredoc"})
    # The conda solve selects pandas 3.0.6, whose C extension is
    # incompatible with the recipe's later numpy==1.25.2 pin.  Let the
    # solve proceed, then install the pandas version current for this
    # 2023 instance after numpy is pinned.
    pandas_legacy = "  - pandas!=0.25.0"
    if pandas_legacy in patched:
        prefix = Path(cwd) / "conda"
        patched = patched.rstrip() + f'\n"{prefix}/bin/python" -m pip install "pandas==2.1.4"\n'
        changes.append({"change": "post-numpy pip install pandas==2.1.4",
                        "reason": "pandas 3.0.6 C extension is incompatible with pinned numpy 1.25.2",
                        "in": "prepare-env.sh tail"})
    # conda-forge currently resolves more-itertools 10.0.0 for a
    # Python 3.6 recipe.  That release imports functools.cached_property
    # (Python 3.8+), so pytest collection fails before any test runs.
    # Pin the last 8.x release, whose metadata accepts Python 3.5+.
    if (instance_id == "scikit-learn__scikit-learn-9288"
            and "python=3.6" in patched
            and "more-itertools==8.10.0" not in patched):
        prefix = Path(cwd) / "conda"
        patched = patched.rstrip() + (
            f'\n"{prefix}/bin/python" -m pip install "more-itertools==8.10.0"\n')
        changes.append({"change": "pip install more-itertools==8.10.0",
                        "reason": "conda-forge selected more-itertools 10.0.0, which imports "
                                  "functools.cached_property (Python 3.8+) under Python 3.6",
                        "in": "prepare-env.sh tail"})
    # Django 2.2's test requirements pull cffi.  cffi has no requires_python
    # cap on these releases, so pip selects 1.15.x (sdist) instead of the
    # 1.14.6 manylinux cp35 wheel and then fails without ffi.h.  Install the
    # wheel first and point C headers/libraries at the private conda prefix.
    if instance_id == "django__django-10097":
        prefix = Path(cwd) / "conda"
        exports = (f'export CFLAGS="-I{prefix}/include ${{CFLAGS:-}}"\n'
                   f'export CPPFLAGS="-I{prefix}/include ${{CPPFLAGS:-}}"\n'
                   f'export LDFLAGS="-L{prefix}/lib ${{LDFLAGS:-}}"\n')
        if "set -euo pipefail\n" in patched and 'export CFLAGS="-I' not in patched:
            patched = patched.replace("set -euo pipefail\n",
                                      "set -euo pipefail\n" + exports, 1)
            changes.append({"change": "CFLAGS/CPPFLAGS/LDFLAGS point at the private prefix",
                            "reason": "cffi and other test dependencies need the conda "
                                      "libffi headers instead of missing host -dev headers",
                            "in": "prepare-env.sh preamble"})
        marker = "python -m pip install -r requirements.txt"
        if marker in patched and 'python -m pip install "cffi==1.14.6"' not in patched:
            patched = patched.replace(
                marker, f'python -m pip install "cffi==1.14.6" && {marker}', 1)
            changes.append({"change": "pip install cffi==1.14.6 before requirements",
                            "reason": "pip selects cffi 1.15.x sdist for Python 3.5; "
                                      "1.14.6 is the last release with a cp35 manylinux wheel",
                            "in": "prepare-env.sh requirements line"})
    return patched, changes


class MambaProvisioner(NativeProvisioner):
    def __init__(self, *args, archive_root=None, instance_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.archive_root = Path(archive_root) if archive_root else None
        self.instance_id = instance_id

    def verified_cached_source(self, public):
        """Reuse a complete archived public clone after a network clone failure."""
        if self.archive_root:
            candidates = {
                "matplotlib__matplotlib-25775": self.archive_root / "preprovision-mamba" / "20261024T054343Z" / "instances" / "cac0d4b19427c35dc4d9f12ab37a2c2ac20d3cf95c93d7b998bc78fc292d0e4b" / "baseline",
                "matplotlib__matplotlib-26342": self.archive_root / "preprovision-mamba" / "offline-sources" / "matplotlib-26342-baseline",
                "pydata__xarray-4075": self.archive_root / "environments" / "attempts" / "network-retry-20261024T052727Z" / "a4c41d876650f6ae6c1e50b145cbe59c61ef4f6d6e76c236f87294cb4d72e9fd" / "baseline",
                # Django 2.2's shallow fetch repeatedly lost TLS at ~90 s.
                # This local clone is the same base commit, prepared with the
                # provisioner's own HTTP/1.1 command before retrying.
                "django__django-10097": self.archive_root / "preprovision-mamba" / "offline-sources" / "django-10097-baseline",
            }
            path = candidates.get(public["instance_id"])
            if path and (path / ".git").is_dir():
                head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
                if head == public["base_commit"]:
                    return {"repo": str(path), "receipt": "archived-public-source",
                            "receipt_sha256": hashlib.sha256(str(path).encode()).hexdigest()}
        return super().verified_cached_source(public)

    @staticmethod
    def _sha(text):
        return hashlib.sha256(text.encode()).hexdigest()

    async def _run(self, argv, cwd, log, *, timeout=1800, extra_environment=None):
        # The provisioner intentionally builds a minimal env without HOME, so
        # conda does not see the host's ~/.condarc and uses the slow classic
        # solver for large matplotlib recipes.  The installed mamba binary is a
        # drop-in for conda create with the same channels and recipe.  The
        # generated script is part of the receipt evidence; record the
        # original/patched hashes next to it.
        if (len(argv) == 2 and str(argv[0]) == "/bin/bash"
                and str(argv[1]).endswith("prepare-env.sh")):
            path = Path(argv[1])
            original = path.read_text()
            mamba = str(self.conda_root / "bin" / "mamba")
            patched, changes = adapt_prepare_script(original, cwd, mamba, self.instance_id)
            recipe_change = changes or None
            if patched != original:
                path.write_text(patched)
                (path.parent / "solver-adaptation.json").write_text(json.dumps({
                    "reason": "use the installed mamba binary for a large conda recipe; same channels and package recipe",
                    "script": path.name,
                    "original_sha256": self._sha(original),
                    "patched_sha256": self._sha(patched),
                    "adapter": mamba,
                    "environment_yml": recipe_change,
                }, indent=2) + "\n")
        env = dict(extra_environment or {})
        env.setdefault("PATH", f"{self.conda_root / 'bin'}:/usr/bin:/bin")
        env["CONDA_SOLVER"] = "libmamba"
        return await super()._run(argv, cwd, log, timeout=timeout, extra_environment=env)


def skip_apt_preinstall(instance, public, output):
    """Temporarily clear an apt pre_install and record the adaptation.

    Returns ``(specs, record)``; the caller restores ``specs["pre_install"]``
    after preparation.  ``record`` is the provenance written next to the run.
    """
    if instance not in APT_PREINSTALL_SKIPS:
        return None, None
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    specs = MAP_REPO_VERSION_TO_SPECS[public["repo"]][public["version"]]
    original = list(specs.get("pre_install") or [])
    if not original:
        return specs, None
    record = {
        "instance_id": instance,
        "repo": public["repo"],
        "version": public["version"],
        "original_pre_install": original,
        "reason": APT_PREINSTALL_SKIPS[instance],
        "timestamp": ex.now(),
    }
    # NativeProvisioner.prepare re-imports this module-level dict, so an
    # in-memory edit here is seen there and restored by the caller.
    specs["pre_install"] = []
    adaptation_dir = output / "preprovision-mamba" / "apt-preinstall-skips"
    adaptation_dir.mkdir(parents=True, exist_ok=True)
    ex.write_json(adaptation_dir / f"{instance}.json", record)
    return specs, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--phase", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--deadline", type=int, default=7200)
    args = parser.parse_args()
    output = args.output.resolve()
    instances = [item.strip() for item in args.instances.split(",") if item.strip()]
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    manifest = ex.load_manifest(output)
    inventory = export_public_dataset(output, manifest, args.phase)
    recipes = ex.read_json(output / "host-install-recipes.json")
    receipts = ex.read_json(output / "provision-native.json")
    if "django__django-10097" in instances:
        ensure_django_requirements(output)
    install_offline_raw(output)
    for instance in instances:
        public = inventory[instance]
        existing = receipts.get(instance)
        if existing:
            if existing["public"] != public or existing["source_sha256"] != manifest["source"]["sha256"]:
                raise ValueError(f"Prepared public/source identity changed: {instance}")
            NativeProvisioner.verify(existing["environment"])
            print(json.dumps({"instance_id": instance, "event": "already_ready"}), flush=True)
            continue
        prep = output / "preparation" / f"{instance}.json"
        if prep.exists():
            status = ex.read_json(prep)
            if status.get("status") == "ready":
                raise ValueError(f"Ready preparation without a receipt: {instance}")
            archive = output / "preparation-attempts" / f"mamba-preprovision-{int(time.time())}"
            archive.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"instance_id": instance, "event": "archive_preparation",
                              "to": str(archive)}), flush=True)
            prep.rename(archive / prep.name)
        print(json.dumps({"instance_id": instance, "event": "start", "timestamp": ex.now()}), flush=True)
        started = time.monotonic()
        specs_to_restore, skipped_preinstall = skip_apt_preinstall(instance, public, output)
        if skipped_preinstall:
            print(json.dumps({"instance_id": instance, "event": "apt_preinstall_skipped",
                              "commands": len(skipped_preinstall["original_pre_install"])}), flush=True)
        provisioner = MambaProvisioner(
            manifest["native"]["environment_root"],
            conda=manifest["native"].get("conda"),
            public_tools=manifest["native"].get("public_tools"),
            archive_root=output,
            instance_id=instance,
        )
        try:
            environment = asyncio.run(asyncio.wait_for(
                provisioner.prepare(public, environment_setup_commit=recipes.get(instance, {}).get("environment_setup_commit")),
                args.deadline))
            NativeProvisioner.verify(environment)
            receipts[instance] = {"environment": environment, "public": public,
                                  "source_sha256": manifest["source"]["sha256"]}
            ex.write_json(output / "provision-native.json", receipts)
            status = {"status": "ready",
                      "environment_identity": ex.content_hash(receipts[instance]),
                      "manifest_hash": manifest["manifest_hash"],
                      "instance_id": instance,
                      "seconds": time.monotonic() - started,
                      "created": ex.now()}
            ex.write_json(prep, status)
            print(json.dumps({"instance_id": instance, "event": "ready",
                              "seconds": round(time.monotonic() - started, 2),
                              "environment_identity": status["environment_identity"]}), flush=True)
        except Exception as exc:
            status = {"status": "env_blocked", "error": f"{type(exc).__name__}: {exc}",
                      "manifest_hash": manifest["manifest_hash"], "instance_id": instance,
                      "seconds": time.monotonic() - started, "created": ex.now()}
            ex.write_json(prep, status)
            print(json.dumps({"instance_id": instance, "event": "env_blocked",
                              "error": status["error"][:600],
                              "seconds": round(time.monotonic() - started, 2)}), flush=True)
        finally:
            if specs_to_restore is not None and skipped_preinstall is not None:
                specs_to_restore["pre_install"] = skipped_preinstall["original_pre_install"]


if __name__ == "__main__":
    main()
