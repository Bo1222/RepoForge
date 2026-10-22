"""Native result/gate boundaries without API calls, hidden data or Docker."""

from types import SimpleNamespace
from pathlib import Path
import subprocess

import pytest

from moatless.evaluation.experiment import aggregate, freeze_answer, prepare, validate_gate, write_json
from moatless.evaluation.native_grading import NativeGrader
ROOT = Path(__file__).resolve().parents[2]


def test_local_service_identity_rejects_live_revision_and_dimension_drift(tmp_path, monkeypatch):
    import io
    import json
    from moatless.evaluation.experiment_inputs import validate_embedding_service

    manifest, _ = prepare(ROOT, tmp_path / 'local', execution_backend='native')
    execution = manifest['embedding_execution']
    expected = {'status': 'ok', 'model': 'BAAI/bge-m3', 'revision': execution['model_revision'],
                'dimensions': 1024, 'max_input_tokens': 2048, 'precision': 'float16', 'batch_size': 8}
    current = dict(expected)

    class Opener:
        def open(self, url, timeout):
            assert url == 'http://127.0.0.1:8787/health'
            return io.BytesIO(json.dumps(current).encode())

    monkeypatch.setattr('urllib.request.build_opener', lambda *args: Opener())
    assert validate_embedding_service(manifest) == expected
    for field, value in (('revision', 'different'), ('dimensions', 1536), ('max_input_tokens', 8192)):
        current.clear()
        current.update(expected, **{field: value})
        with pytest.raises(ValueError, match='frozen identity'):
            validate_embedding_service(manifest)


def frozen_answer(output, job, *, generation_status="completed", error_type=None, cost_complete=True):
    directory = output / "runs" / job["job_id"]
    directory.mkdir(parents=True)
    for name in ("patch.diff", "events.jsonl", "sdk.jsonl"):
        (directory / name).write_text("")
    for name in ("trajectory.json", "agent-tests.json", "test-requests.json"):
        write_json(directory / name, [])
    write_json(directory / "usage.json", {"reference_cost": 0.1, "cost_complete": cost_complete})
    write_json(directory / "generation.json", {"job_id": job["job_id"], "manifest_hash": job["manifest_hash"],
        "selected_node_id": 0, "generation_status": generation_status, "error_type": error_type,
        "official_harness": "not_run"})
    return directory, freeze_answer(directory, job)


@pytest.mark.parametrize("generation_status,error_type,cost_complete,message", [
    ("failed", "IsADirectoryError", True, "Generation infrastructure failure"),
    ("completed", None, False, "Unknown generation usage/cost"),
])
def test_invalid_frozen_answer_stops_before_any_provider_check(
    tmp_path, monkeypatch, generation_status, error_type, cost_complete, message
):
    from moatless.evaluation.experiment_native import run_job

    output = tmp_path / "native"
    manifest, jobs = prepare(ROOT, output, execution_backend="native")
    job = jobs[0]
    frozen_answer(output, job, generation_status=generation_status, error_type=error_type, cost_complete=cost_complete)
    monkeypatch.setattr("moatless.evaluation.experiment_native.validate_embedding_service",
                        lambda *_: pytest.fail("No service request may run for an invalid frozen answer"))
    with pytest.raises(ValueError, match=message):
        run_job(output, manifest, job)


def test_native_identity_preserves_110_jobs_and_never_reports_official_score(tmp_path):
    output = tmp_path / "native"
    manifest, jobs = prepare(ROOT, output, execution_backend="native")
    assert manifest["execution_backend"] == manifest["grading_backend"] == "native"
    assert len(jobs) == 110
    assert sum(job["phase"] == "smoke" for job in jobs) == 15
    assert sum(job["phase"] == "dev" for job in jobs) == 90
    report = aggregate(output)
    assert report["groups"]["a0"]["local_resolved_over_selected"] is None
    job = next(job for job in jobs if job["phase"] == "dev")
    directory, answer = frozen_answer(output, job)
    assert aggregate(output)["groups"]["a0"]["local_resolved_over_selected"] is None  # No independent grade yet.
    write_json(directory / "grading.json", {"answer_hash": answer["answer_hash"], "local_grading": "completed",
        "local_resolved": True, "official_resolved": None, "official_harness": "not_run"})
    group = aggregate(output)["groups"]["a0"]
    assert group["local_resolved_over_selected"] == pytest.approx(1 / 90)
    assert group["graded"] == 1 and group["official_resolved_over_selected"] is None


def test_native_gate_cannot_be_replaced_by_pass_flags(tmp_path):
    output = tmp_path / "native"
    manifest, _ = prepare(ROOT, output, execution_backend="native")
    write_json(output / "native-gate.json", {"manifest_hash": manifest["manifest_hash"], "execution_backend": "native",
        "passed": True, "local_contracts_passed": True, "real_recovery_passed": True, "cli_verified": True})
    with pytest.raises(ValueError, match="actual contracts"):
        validate_gate(output, manifest)


def test_grader_refuses_private_material_before_frozen_answer(tmp_path, monkeypatch):
    output = tmp_path / "native"
    manifest, jobs = prepare(ROOT, output, execution_backend="native")
    grader = NativeGrader(output, manifest)
    def private_read(_):
        pytest.fail("Private material was accessed before answer freeze")
    monkeypatch.setattr(grader, "_private_instance", private_read)
    with pytest.raises(FileNotFoundError):
        grader.grade(jobs[0])


def test_compatible_script_keeps_real_failure_despite_successful_cleanup(tmp_path):
    from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
    spec = SimpleNamespace(eval_script_list=[f"echo '{START_TEST_OUTPUT}'", "false", f"echo '{END_TEST_OUTPUT}'", "true"])
    script = tmp_path / "eval.sh"
    script.write_text(NativeGrader.eval_script(spec, tmp_path))
    result = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True)
    assert result.returncode == 1
    assert START_TEST_OUTPUT in result.stdout and END_TEST_OUTPUT in result.stdout


def test_configuration_gate_accepts_public_finish_schema_and_rejects_embedding_drift(tmp_path):
    from moatless.evaluation.native_gate import configuration_evidence
    from moatless.message_history.policy import ContextPolicy

    output = tmp_path / "native"
    manifest, jobs = prepare(ROOT, output, execution_backend="native")
    config = manifest["cli_config"]
    selected = [job for job in jobs if job["phase"] in {"smoke", "cli_smoke"}]
    for job in selected:
        managed = job["entrypoint"] == "cli" or manifest["profiles"][job["variant"]]["managed"]
        graph = job["entrypoint"] == "cli" or manifest["profiles"][job["variant"]]["flow"] == "graph"
        loaded = {"manifest_hash": manifest["manifest_hash"], "job_id": job["job_id"],
                  "flow_type": "LangGraphFlow" if graph else "AgenticLoop",
                  "model": config["model"], "model_base_url": config["base_url"],
                  "key_env_matches": True, "model_parameters": config["params"],
                  "memory_type": "ManagedMessageHistoryGenerator" if managed else "MessageHistoryGenerator",
                  "tool_names": [], "schema_names": [],
                  "tool_isolation": "native-offline-v1", "index_settings": manifest["embedding_settings"],
                  "policy": ContextPolicy(**config["context_policy"]).model_dump(mode="json")}
        from moatless.evaluation.configuration_bindings import tool_bindings
        tools = tool_bindings(manifest, job)
        loaded["tool_names"], loaded["schema_names"] = list(tools), list(tools.values())
        loaded.update({key: config[key] for key in
                       ("max_tokens", "temperature", "max_iterations", "max_cost", "model_timeout")})
        write_json(output / "runs" / job["job_id"] / "loaded-config.json", loaded)
    assert configuration_evidence(output, manifest)["passed"]
    batch = next(job for job in selected if job["entrypoint"] != "cli")
    path = output / "runs" / batch["job_id"] / "loaded-config.json"
    from moatless.evaluation.experiment import read_json
    loaded = read_json(path)
    loaded["index_settings"]["embed_model"] = "text-embedding-3-small"
    write_json(path, loaded)
    result = configuration_evidence(output, manifest)
    assert not result["passed"]
    assert not next(row for row in result["rows"] if row["job_id"] == batch["job_id"])["checks"]["index"]


def test_gate_binds_raw_baseline_logs_not_only_candidate_reports(tmp_path):
    # This fixture tests evidence integrity only, never claims real acceptance.
    from moatless.evaluation.experiment import sha
    from moatless.evaluation.native_gate import COMPONENTS, validate_native_gate

    output = tmp_path / "native"
    manifest, jobs = prepare(ROOT, output, execution_backend="native")
    components, evidence = {}, {}
    for name in COMPONENTS:
        path = output / (name + ".json")
        component = {"passed": True, "manifest_hash": manifest["manifest_hash"]}
        if name == "recovery":
            component["cases"] = [{"case": mode + "-" + fault, "passed": True, "first_exit_code": 88}
                                  for mode in ("shadow", "disk")
                                  for fault in ("after_generation", "before_tool", "after_effect", "after_commit")]
        write_json(path, component)
        components[name] = path.name
        evidence[path.name] = sha(path)
    write_json(output / "provision-native.json", {})
    evidence["provision-native.json"] = sha(output / "provision-native.json")
    for job in jobs:
        if job["phase"] not in {"smoke", "cli_smoke"}:
            continue
        directory, answer = frozen_answer(output, job)
        raw = directory / "baseline-output.txt"
        raw.write_text("original failure fixture")
        relative = str(raw.relative_to(output))
        baseline = {"original_failure_reproduced": True, "evidence": {relative: sha(raw)}}
        write_json(directory / "original-baseline.json", baseline)
        write_json(directory / "grading.json", {"answer_hash": answer["answer_hash"],
            "grading_backend": "native", "evaluation_protocol": manifest["evaluation_protocol"],
            "grader_source_sha256": manifest["source"]["files"]["moatless/evaluation/native_grading.py"],
            "local_grading": "completed", "local_resolved": False, "original_baseline": baseline,
            "official_harness": "not_run", "official_resolved": None})
        write_json(directory / "own-tests.json", {"answer_hash": answer["answer_hash"],
            "hidden_tests": False, "status": "not_run"})
    for path in (output / "runs").rglob("*"):
        if path.is_file():
            evidence[str(path.relative_to(output))] = sha(path)
    gate = {"manifest_hash": manifest["manifest_hash"], "execution_backend": "native",
            "components": components, "evidence": evidence}
    write_json(output / "native-gate.json", gate)
    validate_native_gate(output, manifest)
    raw_path = next(name for name in evidence if name.endswith("baseline-output.txt"))
    del gate["evidence"][raw_path]
    write_json(output / "native-gate.json", gate)
    with pytest.raises(ValueError, match="raw original-failure"):
        validate_native_gate(output, manifest)
    gate["evidence"][raw_path] = sha(output / raw_path)
    write_json(output / "native-gate.json", gate)
    (output / raw_path).write_text("changed fixture")
    with pytest.raises(ValueError, match="evidence changed"):
        validate_native_gate(output, manifest)


def test_public_workspace_copy_keeps_symlinks_and_clean_status(tmp_path):
    # Dereferenced symlinks become tracked typechanges, so git status no longer
    # matches the base commit and the CLI worktree entrypoint refuses to run.
    from moatless.evaluation.experiment_native import copy_public_workspace

    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n")
    (source / "nested" / "target.txt").write_text("payload\n")
    (source / "link.txt").symlink_to(Path("nested") / "target.txt")
    (source / "broken-link.txt").symlink_to("missing.txt")
    git = ["git", "-C", str(source), "-c", "user.name=NativeCopy", "-c", "user.email=copy@example.invalid",
           "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "fixture"], check=True)
    # Untracked caches like in the prepared public repositories.
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "module.cpython-310.pyc").write_text("stale")

    destination = tmp_path / "destination"
    copy_public_workspace(source, destination)

    assert (destination / "module.py").read_text() == "VALUE = 1\n"
    assert (destination / "link.txt").is_symlink()
    assert (destination / "link.txt").readlink() == Path("nested") / "target.txt"
    assert (destination / "broken-link.txt").is_symlink()
    assert not (destination / "__pycache__").exists()
    status = subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=all"], text=True)
    assert status == ""


def accepted_job(output, job, *, grade_hash=None, own_hash=None):
    directory, answer = frozen_answer(output, job)
    write_json(directory / "grading.json", {"answer_hash": grade_hash or answer["answer_hash"],
                                            "local_grading": "completed", "local_resolved": True})
    write_json(directory / "own-tests.json", {"answer_hash": own_hash or answer["answer_hash"],
                                              "hidden_tests": False, "status": "executed"})
    return directory, answer


def test_release_deletes_checkpoints_only_after_acceptance(tmp_path):
    import json
    from moatless.evaluation.experiment import verify_answer
    from moatless.evaluation.experiment_native import release_job_workspace

    output = tmp_path / "native"
    manifest, jobs = prepare(ROOT, output, execution_backend="native")
    job = next(j for j in jobs if j["phase"] == "smoke")
    directory, answer = accepted_job(output, job)
    (directory / "checkpoint.sqlite").write_bytes(b"c" * 100)
    (directory / "checkpoint.sqlite-wal").write_bytes(b"w" * 10)
    nested = directory / "sessions" / "0123456789abcdef"
    nested.mkdir(parents=True)
    (nested / "checkpoint.sqlite.outcomes.sqlite").write_bytes(b"o")
    (output / "workspaces" / job["job_id"]).mkdir(parents=True)
    (output / "runtime-prefixes" / job["job_id"]).mkdir(parents=True)

    release_job_workspace(output, manifest, job)

    assert not (directory / "checkpoint.sqlite").exists()
    assert not (directory / "checkpoint.sqlite-wal").exists()
    assert not (nested / "checkpoint.sqlite.outcomes.sqlite").exists()
    assert not (output / "workspaces" / job["job_id"]).exists()
    assert not (output / "runtime-prefixes" / job["job_id"]).exists()
    records = [json.loads(line) for line in (directory / "disposable-release.jsonl").read_text().splitlines()]
    summary = [row for row in records if row["path"] == "checkpoint.sqlite*"]
    assert summary and summary[0]["files"] == 3 and summary[0]["bytes"] == 111 and summary[0]["status"] == "released"
    assert (directory / "patch.diff").exists() and (directory / "usage.json").exists()
    verify_answer(directory, job)


def test_release_refuses_without_matching_acceptance_and_keeps_checkpoints(tmp_path):
    from moatless.evaluation.experiment_native import release_job_workspace

    output = tmp_path / "native"
    manifest, jobs = prepare(ROOT, output, execution_backend="native")
    job = next(j for j in jobs if j["phase"] == "smoke")
    directory, _ = accepted_job(output, job, grade_hash="different-hash")
    (directory / "checkpoint.sqlite").write_bytes(b"c" * 50)
    with pytest.raises(ValueError, match="required before release"):
        release_job_workspace(output, manifest, job)
    assert (directory / "checkpoint.sqlite").exists()


def test_testbed_setup_repairs_generated_pytest_version_without_answer_edit(tmp_path):
    from moatless.evaluation.experiment_native import apply_testbed_setup
    (tmp_path / "src/_pytest").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\nminversion = "2.0"\n')
    version = tmp_path / "src/_pytest/_version.py"
    version.write_text(
        "__version__ = version = '0.1.dev1+g7f7a36478'\n"
        "__version_tuple__ = version_tuple = (0, 1, 'dev1', 'g7f7a36478')\n"
        "__commit_id__ = commit_id = 'g7f7a36478'\n")
    record = apply_testbed_setup(tmp_path, {"public": {"repo": "pytest-dev/pytest", "version": "6.0"}})
    assert record["before"] == "0.1.dev1+g7f7a36478"
    assert record["after"] == "6.0" and record["minversion"] == "2.0"
    assert record["in_answer_patch"] is False
    assert "6.0" in version.read_text()
    assert apply_testbed_setup(tmp_path, {"public": {"repo": "pytest-dev/pytest", "version": "6.0"}}) is None
    assert apply_testbed_setup(tmp_path, {"public": {"repo": "django/django", "version": "3.0"}}) is None
