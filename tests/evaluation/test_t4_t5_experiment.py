import json
from pathlib import Path

import pytest

from repoforge.evaluation.experiment import (
    aggregate, freeze_answer, load_manifest, prepare, public_instance, validate_gate, verify_answer, write_json,
)
from repoforge.evaluation.experiment_docker import bundle_for
from repoforge.evaluation.experiment_usage import SDKRecorder


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = {"instance_id": "pytest-dev__pytest-7490", "repo": "pytest-dev/pytest", "base_commit": "base",
          "version": "6.0", "problem_statement": "issue"}


def freeze_fake(directory, job, cost=.1):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "patch.diff").write_text("complete patch including tests")
    (directory / "events.jsonl").write_text("")
    (directory / "sdk.jsonl").write_text("")
    for name in ("trajectory.json", "agent-tests.json", "test-requests.json"):
        write_json(directory / name, [])
    write_json(directory / "usage.json", {"reference_cost": cost})
    write_json(directory / "generation.json", {"job_id": job["job_id"], "manifest_hash": job["manifest_hash"],
                                                "selected_node_id": 1, "generation_status": "completed"})
    return freeze_answer(directory, job)


def test_new_manifest_separates_15_smoke_90_dev_5_cli_and_never_overwrites(tmp_path):
    output = tmp_path / "experiment"
    manifest, jobs = prepare(ROOT, output)
    assert sum(job["phase"] == "smoke" for job in jobs) == 15
    assert sum(job["phase"] == "dev" for job in jobs) == 90
    assert sum(job["phase"] == "cli_smoke" for job in jobs) == 5
    assert not set(manifest["split"]["dev"]) & set(manifest["split"]["holdout"])
    assert manifest["cli_config"]["context_policy"]["enable_task_state"] is False
    assert manifest["cli_config"]["max_iterations"] == 40
    assert manifest["cli_config"]["max_cost"] == 1.0
    for job in jobs:
        bundle = bundle_for(manifest, job, PUBLIC)
        assert bundle["cli_config"]["max_iterations"] == 40
        assert bundle["cli_config"]["max_cost"] == 1.0
    assert "Tests run automatically after each code change" not in manifest["agent_configs"]["a0"]["system_prompt"]
    assert "Tests run automatically after each code change" in manifest["agent_configs"]["graph_legacy"]["system_prompt"]
    assert load_manifest(output)["manifest_hash"] == manifest["manifest_hash"]
    assert any(name.startswith("repoforge/utils/_static/tiktoken_cache/") for name in manifest["source"]["files"])
    from repoforge.agent.agent import ActionAgent
    loaded = ActionAgent.from_dict(manifest["agent_configs"]["a0"])
    assert type(loaded.completion_model).__name__ == "ToolCallCompletionModel"
    assert all(type(action.completion_model).__name__ == "JsonCompletionModel" for action in loaded.actions
               if getattr(action, "completion_model", None))
    with pytest.raises(ValueError, match="new experiment"):
        prepare(ROOT, output)
    report = aggregate(output)
    assert report["groups"]["a0"]["selected"] == 90
    assert report["groups"]["a0"]["graded"] == 0
    assert report["groups"]["a0"]["cost_per_resolved"] is None
    write_json(output / "jobs.json", jobs[1:])
    with pytest.raises(ValueError, match="job selection changed"):
        load_manifest(output)


def test_worker_bundle_whitelists_fields_and_has_no_holdout_or_grading(tmp_path):
    manifest, jobs = prepare(ROOT, tmp_path / "e")
    row = {**PUBLIC, "patch": "GOLD", "hints_text": "HINT", "test_patch": "HIDDEN", "FAIL_TO_PASS": ["secret"]}
    bundle = bundle_for(manifest, jobs[0], row)
    assert bundle["instance"] == PUBLIC
    assert "GOLD" not in json.dumps(bundle) and "HIDDEN" not in json.dumps(bundle)
    assert "holdout" not in bundle
    with pytest.raises(ValueError, match="Missing"):
        public_instance({"instance_id": "wrong"})


def test_frozen_answer_and_grading_identity_are_checked(tmp_path):
    output = tmp_path / "e"
    manifest, jobs = prepare(ROOT, output)
    job = next(job for job in jobs if job["phase"] == "dev")
    directory = output / "runs" / job["job_id"]
    answer = freeze_fake(directory, job)
    write_json(directory / "grading.json", {"answer_hash": answer["answer_hash"], "official_harness": "completed", "resolved": True})
    report = aggregate(output)
    assert report["groups"]["a0"]["resolved_over_selected"] == pytest.approx(1 / 90)
    assert report["groups"]["a0"]["cost_complete"] is False
    (directory / "patch.diff").write_text("mutated answer")
    with pytest.raises(ValueError, match="changed"):
        verify_answer(directory, job)


def test_dev_gate_requires_actual_frozen_smoke_even_when_flags_say_pass(tmp_path):
    output = tmp_path / "e"
    manifest, jobs = prepare(ROOT, output)
    write_json(output / "t4-gate.json", {"manifest_hash": manifest["manifest_hash"],
        "local_contracts_passed": True, "real_recovery_passed": True, "cli_verified": True, "evidence": {}})
    with pytest.raises(ValueError, match="actual evidence"):
        validate_gate(output, manifest)
    (output / "manifest.json").write_text("{}")
    with pytest.raises((KeyError, ValueError)):
        load_manifest(output)


@pytest.mark.asyncio
async def test_sdk_ledger_counts_auxiliary_and_retries_and_stops_shared_budget(tmp_path, monkeypatch):
    from litellm.types.utils import ModelResponse
    async def sdk(**kwargs):
        return ModelResponse(model="deepseek-flash", usage={"prompt_tokens": 1000, "completion_tokens": 100},
                             choices=[{"message": {"role": "assistant", "content": "ok"}}])
    monkeypatch.setattr("litellm.acompletion", sdk)
    recorder = SDKRecorder(tmp_path, {"embedding": {}}, max_cost=.0008)
    with recorder.capture():
        import litellm
        for _ in range(2):
            await litellm.acompletion(model="openai/deepseek-flash", messages=[], extra_body={"thinking": {"type": "disabled"}})
        with pytest.raises(Exception, match="budget exhausted"):
            await litellm.acompletion(model="openai/deepseek-flash", messages=[], extra_body={"thinking": {"type": "disabled"}})
    assert recorder.totals()["sdk_calls"] == 2
    assert recorder.totals()["reference_cost"] == pytest.approx(.00084)
    assert SDKRecorder(tmp_path, {"embedding": {}}, max_cost=1).totals()["sdk_calls"] == 2


def test_unconfirmed_and_unknown_sdk_usage_are_not_free_or_replayed(tmp_path):
    recorder = SDKRecorder(tmp_path, {"embedding": {}})
    identity, started = recorder.begin("embedding", {"model": "unknown", "input": "query"})
    assert recorder.totals()["cost_complete"] is False
    assert recorder.totals()["reference_cost"] is None
    with pytest.raises(Exception, match="Unconfirmed"):
        SDKRecorder(tmp_path, {"embedding": {}})
    recorder.end(identity, started, "embedding", "unknown", response={"usage": {"prompt_tokens": 3}})
    assert recorder.totals()["reference_cost"] is None
    with pytest.raises(Exception, match="Unknown"):
        recorder.begin("embedding", {"model": "unknown"})


@pytest.mark.asyncio
async def test_ledger_rejects_accidental_thinking_before_provider_call(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    original = AsyncMock()
    monkeypatch.setattr("litellm.acompletion", original)
    recorder = SDKRecorder(tmp_path, {"embedding": {}})
    with recorder.capture():
        import litellm
        with pytest.raises(Exception, match="thinking disabled"):
            await litellm.acompletion(model="openai/deepseek-flash", messages=[])
    assert not original.called


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
async def test_same_source_loop_graph_fixed_replay(managed):
    from scripts.migration_smoke import replay
    from repoforge.flow import AgenticLoop, LangGraphFlow
    loop, _ = await replay(AgenticLoop, managed=managed)
    graph, _ = await replay(LangGraphFlow, managed=managed)
    # Documented migration metadata improvement: Graph associates Agent events
    # with project/trajectory, where Loop's context-free direct replay leaves null.
    for left, right in zip(loop["events"], graph["events"]):
        assert left["project_id"] in (None, "migration")
        assert right["project_id"] == "migration"
        assert left["trajectory_id"] in (None, "coding")
        assert right["trajectory_id"] == "coding"
        left.pop("project_id")
        right.pop("project_id")
        left.pop("trajectory_id")
        right.pop("trajectory_id")
    assert loop == graph


def test_docker_driver_io_freezes_and_reuses_answer_and_own_tests_exclude_grading(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import repoforge.evaluation.experiment_docker as driver
    output = tmp_path / "e"
    manifest, jobs = prepare(ROOT, output)
    job = next(job for job in jobs if job["phase"] == "cli_smoke")
    write_json(output / "provision.json", {job["instance_id"]: {
        "image_id": "image", "source_sha256": manifest["source"]["sha256"], "public": PUBLIC}})
    client, container = MagicMock(), MagicMock()
    container.id, container.image.id = "container", "image"
    client.containers.run.return_value = container
    monkeypatch.setattr(driver, "docker_client", lambda: client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-probe-key")
    def worker(*args, **kwargs):
        freeze_fake(output / "runs" / job["job_id"], job)
        return 0
    monkeypatch.setattr(driver, "exec_worker", worker)
    answer = driver.run_job(output, manifest, job)
    assert container.stop.called
    assert driver.run_job(output, manifest, job) == answer
    assert client.containers.run.call_count == 1
    write_json(output / "runs" / job["job_id"] / "grading.json", {"secret_official_labels": True})
    def own_worker(container, command, log, **kwargs):
        write_json(Path(log).parent / "own-test-validation.json", {"status": "executed", "results": []})
        return 0
    monkeypatch.setattr(driver, "exec_worker", own_worker)
    driver.validate_agent_tests(output, manifest, job)
    volumes = client.containers.run.call_args.kwargs["volumes"]
    own_input = next(Path(path) for path, binding in volumes.items() if binding["bind"] == "/answer")
    assert {path.name for path in own_input.iterdir()} == {"patch.diff", "test-requests.json"}
    assert client.containers.run.call_args.kwargs["environment"] == {}


def test_baseline_diagnostic_is_post_freeze_isolated_and_never_applies_product_patch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import repoforge.evaluation.experiment_docker as driver
    output = tmp_path / "e"
    manifest, jobs = prepare(ROOT, output)
    job = jobs[0]
    directory = output / "runs" / job["job_id"]
    answer = freeze_fake(directory, job)
    write_json(output / "dataset-receipt.json", {"fingerprint": "frozen"})
    write_json(output / "provision.json", {job["instance_id"]: {"baseline_image_id": "base-image", "public": PUBLIC}})
    row = {**PUBLIC, "patch": "GOLD_PRODUCT_PATCH", "test_patch": "HIDDEN_TEST_PATCH"}
    class Dataset(list):
        _fingerprint = "frozen"
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: Dataset([row]))
    spec = SimpleNamespace(repo="pytest-dev/pytest", eval_script="official test script", FAIL_TO_PASS=["test_bug"], PASS_TO_PASS=["test_existing"])
    monkeypatch.setattr("swebench.harness.test_spec.test_spec.make_test_spec", lambda data: spec)
    client, container = MagicMock(), MagicMock()
    client.containers.run.return_value = container
    container.id, container.image.id = "diagnostic", "base-image"
    container.exec_run.return_value = SimpleNamespace(exit_code=0, output=b"base\n")
    monkeypatch.setattr(driver, "docker_client", lambda: client)
    copies = []
    monkeypatch.setattr("swebench.harness.docker_utils.copy_to_container", lambda c, src, dst: copies.append(src.read_text()))
    monkeypatch.setattr("swebench.harness.docker_utils.exec_run_with_timeout", lambda *a: ("actual test output", False, 1.2))
    monkeypatch.setattr("swebench.harness.grading.get_logs_eval", lambda *a: ({"test_bug": "FAILED", "test_existing": "PASSED"}, True))
    result = driver.baseline_job(output, manifest, job)
    assert result["failure_reproduced"] is True
    assert result["tests_status"]["PASS_TO_PASS"]["success"] == ["test_existing"]
    assert result["product_patch_applied"] is False
    assert result["official_harness"] == "not_run"
    assert copies == ["official test script"]
    assert client.containers.run.call_args.kwargs["volumes"] == {}
    assert client.containers.run.call_args.kwargs["environment"] == {}
    assert container.exec_run.call_count == 1  # clean-base check only, no git apply
    assert verify_answer(directory, job) == answer
    assert driver.baseline_job(output, manifest, job) == result
    assert client.containers.run.call_count == 1


def test_dataset_fingerprint_change_refuses_public_export(tmp_path, monkeypatch):
    import repoforge.evaluation.experiment_docker as driver
    manifest, _ = prepare(ROOT, tmp_path / "e")
    output = tmp_path / "e"
    write_json(output / "dataset-receipt.json", {"fingerprint": "original"})
    class Dataset(list):
        _fingerprint = "changed"
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: Dataset([PUBLIC]))
    with pytest.raises(ValueError, match="fingerprint changed"):
        driver.export_public_dataset(output, manifest, "smoke")
    assert not (output / "public/instances.json").exists()


def test_algorithm_selection_requires_complete_labels_and_freezes_analysis(tmp_path):
    from repoforge.evaluation.experiment import freeze_hypothesis
    output = tmp_path / "e"
    _, jobs = prepare(ROOT, output)
    dev = [job for job in jobs if job["phase"] == "dev"]
    answers = {}
    for index, job in enumerate(dev):
        directory = output / "runs" / job["job_id"]
        answer = freeze_fake(directory, job)
        answers[job["job_id"]] = answer
        write_json(directory / "grading.json", {"answer_hash": answer["answer_hash"], "official_harness": "completed",
                                               "resolved": index >= 2})
    assert aggregate(output)["selected_direction"] is None  # unlabeled failures cannot select A/B
    for job, category in zip(dev[:2], ("test_timing", "context_omission")):
        write_json(output / "runs" / job["job_id"] / "labels.json", {
            "answer_hash": answers[job["job_id"]]["answer_hash"],
            "labels": [{"category": category, "primary": True, "evidence": ["trajectory node 1:0"]}]})
    assert aggregate(output)["selected_direction"] == "A"  # explicit tie rule
    hypothesis = freeze_hypothesis(output)
    from repoforge.evaluation.experiment import content_hash
    assert hypothesis["analysis_snapshot_hash"] == content_hash(json.loads((output / "hypothesis-analysis.json").read_text()))
    with pytest.raises(ValueError, match="already frozen"):
        freeze_hypothesis(output)
    first = dev[0]
    write_json(output / "runs" / first["job_id"] / "labels.json", {
        "answer_hash": answers[first["job_id"]]["answer_hash"],
        "labels": [{"category": "unknown", "primary": True, "evidence": ["missing failure explanation"]}]})
    assert aggregate(output)["selected_direction"] is None  # unknown evidence dominates intervention


def test_repeat_failure_tie_prefers_action_over_context(tmp_path):
    output = tmp_path / "tie"
    _, jobs = prepare(ROOT, output)
    first = next(job for job in jobs if job["phase"] == "dev")
    repeats = [job for job in jobs if job["phase"] == "dev" and job["instance_id"] == first["instance_id"]]
    for job, category in zip(repeats[:2], ["context_omission", "unverified_finish"]):
        directory = output / "runs" / job["job_id"]
        answer = freeze_fake(directory, job)
        write_json(directory / "grading.json", {"answer_hash": answer["answer_hash"],
            "official_harness": "completed", "resolved": False})
        write_json(directory / "labels.json", {"answer_hash": answer["answer_hash"], "labels": [
            {"category": category, "primary": True, "evidence": ["trajectory.json"]}]})
    report = aggregate(output)
    assert report["primary_failure_instances"] == {"unverified_finish": 1}
    assert report["selected_direction"] is None  # Remaining 88 jobs have not been graded.
