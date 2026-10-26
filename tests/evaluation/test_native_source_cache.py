import subprocess

import pytest

from repoforge.evaluation.experiment import write_json
from repoforge.evaluation.native_provision import NativeProvisioner, SCIKIT_CYTHON_BOOTSTRAP


def cached_fixture(tmp_path):
    root = tmp_path / "preparation"
    target = root / "instances/old"
    repo, prefix = target / "baseline", target / "conda"
    repo.mkdir(parents=True)
    prefix.mkdir()
    (repo / "public.py").write_text("public_source = True\n")
    for argv in (["git", "init", "-q"], ["git", "add", "."],
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "public fixture"]):
        subprocess.run(argv, cwd=repo, check=True, capture_output=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    public = {"instance_id": "pytest-dev__pytest-7490", "repo": "pytest-dev/pytest",
              "version": "6.0", "base_commit": base, "problem_statement": "public problem"}
    receipt = {"public": public, "repo": str(repo), "prefix": str(prefix),
               "conda_metadata": {}, "package_metadata": {}, "evidence": {}, "setup_has_tracked_changes": False}
    write_json(target / "ready.json", receipt)
    return NativeProvisioner(root, conda="/bin/true"), public, receipt, target / "ready.json"


def test_only_exact_public_task_and_verified_base_can_supply_cached_source(tmp_path):
    provisioner, public, receipt, path = cached_fixture(tmp_path)
    assert provisioner.verified_cached_source(public)["repo"] == receipt["repo"]
    assert provisioner.verified_cached_source({**public, "instance_id": "other-task"}) is None
    receipt["setup_has_tracked_changes"] = True
    write_json(path, receipt)
    assert provisioner.verified_cached_source(public) is None
    receipt["setup_has_tracked_changes"] = False
    receipt["public"] = {**public, "base_commit": "0" * 40}
    write_json(path, receipt)
    with pytest.raises(ValueError, match="identity changed"):
        provisioner.verified_cached_source(receipt["public"])


def test_offline_bootstrap_failure_stops_before_pip_install(tmp_path):
    # The old "remove && install" line let a failed remove bypass set -e.
    script = SCIKIT_CYTHON_BOOTSTRAP.replace("conda remove --offline --force --yes --prefix {prefix} cython", "false")
    marker = tmp_path / "pip-ran"
    script = script.replace("python -m pip install Cython==0.29.36", f"touch {marker}")
    result = subprocess.run(["/bin/bash", "-ec", script])
    assert result.returncode == 1
    assert not marker.exists()
