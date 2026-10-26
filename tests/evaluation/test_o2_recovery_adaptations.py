"""Exercise the holdout recovery adaptations without touching a candidate.

Both scripts only manipulate synthetic text/trees here: no model call, no
benchmark execution and no frozen experiment artifact is read or written.
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "analysis-tools/paired/recovery"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_script_pins_more_itertools_only_for_sklearn(tmp_path):
    module = load("preprovision_mamba")
    text = "#!/bin/bash\nmamba create --prefix /x/conda python=3.6 numpy -y\n"

    patched, changes = module.adapt_prepare_script(
        text, tmp_path, "/mamba", "scikit-learn__scikit-learn-9288")
    assert "more-itertools==8.10.0" in patched
    assert any("more-itertools" in change["change"] for change in changes)
    assert patched.count("more-itertools==8.10.0") == 1

    again, changes_again = module.adapt_prepare_script(
        patched, tmp_path, "/mamba", "scikit-learn__scikit-learn-9288")
    assert again == patched and changes_again == []

    other, _ = module.adapt_prepare_script(text, tmp_path, "/mamba", "django__django-10097")
    assert "more-itertools" not in other


def test_apt_skip_is_recorded_and_restorable(tmp_path):
    module = load("preprovision_mamba")
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    public = {"repo": "django/django", "version": "2.2"}
    specs = MAP_REPO_VERSION_TO_SPECS["django/django"]["2.2"]
    original = list(specs.get("pre_install") or [])
    try:
        restored, record = module.skip_apt_preinstall(
            "django__django-10097", public, tmp_path)
        assert restored is specs
        assert record["original_pre_install"] == original
        assert specs["pre_install"] == []
        record_path = (tmp_path / "preprovision-mamba/apt-preinstall-skips/"
                       "django__django-10097.json")
        assert json.loads(record_path.read_text()) == record
    finally:
        specs["pre_install"] = original

    assert module.skip_apt_preinstall("other-instance", public, tmp_path) == (None, None)


def test_django_prepare_script_pins_cffi_and_headers(tmp_path):
    module = load("preprovision_mamba")
    text = ("#!/bin/bash\nset -euo pipefail\n"
            "python -m pip install -r requirements.txt\n")

    patched, changes = module.adapt_prepare_script(
        text, tmp_path, "/mamba", "django__django-10097")
    assert ('python -m pip install "cffi==1.14.6" && '
            "python -m pip install -r requirements.txt") in patched
    assert 'export CFLAGS="-I' in patched
    assert any("cffi==1.14.6" in change["change"] for change in changes)

    again, changes_again = module.adapt_prepare_script(
        patched, tmp_path, "/mamba", "django__django-10097")
    assert again == patched and changes_again == []

    other, _ = module.adapt_prepare_script(text, tmp_path, "/mamba", "other-instance")
    assert "cffi" not in other
    assert "CFLAGS" not in other


def test_django_requirements_replay_is_local(tmp_path):
    module = load("preprovision_mamba")
    import requests

    root = tmp_path / "preprovision-mamba/offline-recipes"
    root.mkdir(parents=True)
    suffix = ("django/django/4fc35a9c3efdc9154efce28cb23cb84f8834517e/"
              "tests/requirements/py3.txt")
    (root / module.OFFLINE_RAW_SUFFIXES[suffix]).write_text("argon2-cffi >= 16.1.0\n")
    original_get = requests.get
    original_request = requests.sessions.Session.request
    try:
        module.install_offline_raw(tmp_path)
        response = requests.get("https://raw.githubusercontent.com/" + suffix)
        assert response.status_code == 200
        assert response.text.startswith("argon2-cffi")
    finally:
        requests.get = original_get
        requests.sessions.Session.request = original_request


def test_collection_error_adjudication_rewrites_and_archives(tmp_path):
    instance = "pylint-dev__pylint-4551"
    job_id = "a" * 24
    f2p = [f"tests/x.py::t{index}" for index in range(3)]
    run = tmp_path / "runs" / job_id
    baseline_dir = tmp_path / "grading-native/baselines/deadbeef"
    run.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)
    (tmp_path / "preparation").mkdir()
    (tmp_path / "jobs.json").write_text(json.dumps([{
        "phase": "holdout", "variant": "a5", "repeat": 0, "instance_id": instance,
        "manifest_hash": "m", "entrypoint": "batch", "job_id": job_id}]))
    (run / "original-baseline.json").write_text(json.dumps({
        "FAIL_TO_PASS": f2p,
        "evidence": {
            "grading-native/baselines/deadbeef/case.json": "x",
            "grading-native/baselines/deadbeef/compatible-report.json": "y",
        }}))
    (run / "grading.json").write_text(json.dumps({"local_grading": "environment_error"}))
    (tmp_path / "preparation" / f"{instance}.json").write_text(
        json.dumps({"status": "env_blocked", "error": "baseline"}))
    (baseline_dir / "case.json").write_text(json.dumps({
        "status": "environment_error", "patch_applicable": True, "local_resolved": None,
        "test_statuses": {"tests/x.py": "ERROR"},
        "checks": [{"return_code": 0}, {"return_code": 2, "timed_out": False}]}))
    (baseline_dir / "compatible-report.json").write_text(json.dumps({
        instance: {"tests_status": {"FAIL_TO_PASS": {"success": [], "failure": f2p}}}}))

    script = TOOLS / "adjudicate_baseline_collection_error.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path), "--instance", instance,
         "--apply", "--record-dir", str(tmp_path / "records")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    case = json.loads((baseline_dir / "case.json").read_text())
    assert case["status"] == "completed"
    assert all(case["test_statuses"][name] == "ERROR" for name in f2p)
    assert case["adjudication"]["original_status"] == "environment_error"
    assert (run / "grading.json").exists() is False
    assert (run / "original-baseline.json").exists() is False
    assert not (tmp_path / "preparation" / f"{instance}.json").exists()
    record = json.loads(next((tmp_path / "records").glob("*.json")).read_text())
    assert len(record["archived_grading"]) == 2
    assert record["archived_preparation"]["sha256"]
    assert list(baseline_dir.glob("adjudication-*/case.raw.json"))

    # A second apply is a no-op because the adjudicated case is already present.
    second = subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path), "--instance", instance,
         "--apply", "--record-dir", str(tmp_path / "records")],
        capture_output=True, text=True)
    assert second.returncode == 0
    assert "already_adjudicated" in second.stdout
