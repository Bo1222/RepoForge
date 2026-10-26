"""Independent experiment identities, phase ordering and evidence; no provider calls."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from repoforge.evaluation import experiment as ex
from repoforge.evaluation import paired

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def candidate(tmp_path):
    output = tmp_path / "paired"
    manifest, jobs = paired.prepare(ROOT, output)
    return output, manifest, jobs


def test_frozen_matrix_and_alternating_pair_order(candidate):
    output, manifest, jobs = candidate
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 170
    assert {p: sum(j["phase"] == p for j in jobs) for p in paired.PHASES} == {"smoke": 10, "dev": 60, "holdout": 100}
    assert [j["variant"] for j in jobs[:4]] == ["a0", "a5", "a5", "a0"]
    assert manifest["profiles"]["a0"]["action_policy"]["variant"] == "off"
    assert all(p["submit_product_only"] for p in manifest["profiles"].values())
    assert ex.load_manifest(output) == manifest
    jobs.reverse()
    ex.write_json(output / "jobs.json", jobs)
    with pytest.raises(ValueError, match="job selection"):
        ex.load_manifest(output)


@pytest.mark.parametrize("mutation", ["prompt", "export", "split", "policy", "historical_protocol"])
def test_rejects_confounded_or_changed_design(candidate, mutation):
    _, manifest, _ = candidate
    m = deepcopy(manifest)
    if mutation == "prompt":
        m["agent_configs"]["a0"]["system_prompt"] += "different"
    elif mutation == "export":
        m["profiles"]["a0"]["submit_product_only"] = False
    elif mutation == "split":
        m["split"]["dev"].reverse()
    elif mutation == "policy":
        m["profiles"]["a0"]["action_policy"]["variant"] = "a0"
    else:
        m["t7"] = {"protocol": "t7-policy-ablation-v1"}
    with pytest.raises(ValueError):
        paired.make_jobs(m)


def test_no_grade_or_attempt_is_not_a_zero_percent_result(candidate):
    output, manifest, _ = candidate
    report = paired.report(output, manifest, "dev")
    assert not report["terminal"]
    assert report["paired"]["pairable"] == 0
    assert len(report["paired"]["missing_pairs"]) == 30
    for group in report["groups"].values():
        assert group["resolved_over_selected"] is None
        assert group["known_reference_cost"] is None
        assert group["unknown"] == 30


def synthetic_answer(output, manifest, job, *, resolved=True, cost_complete=True, policy_override=None):
    from repoforge.evaluation.configuration_bindings import tool_bindings
    from repoforge.message_history.policy import ContextPolicy
    directory = output / "runs" / job["job_id"]
    directory.mkdir(parents=True, exist_ok=True)
    config = manifest["cli_config"]
    tools = tool_bindings(manifest, job)
    policy = policy_override or manifest["profiles"][job["variant"]]["action_policy"]
    loaded = {"job_id": job["job_id"], "manifest_hash": manifest["manifest_hash"],
        "flow_type": "LangGraphFlow", "memory_type": "ManagedMessageHistoryGenerator",
        "policy": ContextPolicy(**config["context_policy"]).model_dump(mode="json"), "action_policy": policy,
        "model": config["model"], "model_base_url": config["base_url"], "model_parameters": config["params"],
        "key_env_matches": True, "tool_names": list(tools), "schema_names": list(tools.values()),
        "index_settings": manifest["embedding_settings"], "tool_isolation": "native-offline-v1",
        **{k: config[k] for k in ("temperature", "max_tokens", "max_iterations", "max_cost", "model_timeout")}}
    generation = {"job_id": job["job_id"], "manifest_hash": manifest["manifest_hash"],
        "generation_status": "completed", "selected_node_id": 0, "submit_product_only": True,
        "submitted_verified_patch": False, "policy_metrics": {"actual_tool_steps": 1}}
    for name in ("patch.diff", "events.jsonl", "sdk.jsonl"):
        (directory / name).write_text("")
    for name in ("trajectory.json", "agent-tests.json", "test-requests.json"):
        ex.write_json(directory / name, [])
    ex.write_json(directory / "generation.json", generation)
    ex.write_json(directory / "usage.json", {"reference_cost": .1, "cost_complete": cost_complete})
    ex.write_json(directory / "loaded-config.json", loaded)
    ex.write_json(directory / "runtime-start.json", {"index_digest": "same-index"})
    ex.write_json(directory / "policy-decisions.json", [] if job["variant"] == "a0" else [{"config": policy}])
    answer = ex.freeze_answer(directory, job)
    ex.write_json(directory / "original-baseline.json", {"original_failure_reproduced": True})
    ex.write_json(directory / "grading.json", {"answer_hash": answer["answer_hash"],
        "local_grading": "completed", "local_resolved": resolved,
        "original_baseline": {"original_failure_reproduced": True}})
    ex.write_json(directory / "own-tests.json", {"answer_hash": answer["answer_hash"]})
    return directory


def test_off_cannot_silently_enable_policy(candidate):
    output, manifest, jobs = candidate
    synthetic_answer(output, manifest, jobs[0], policy_override=manifest["profiles"]["a5"]["action_policy"])
    with pytest.raises(ValueError, match="configuration mismatch"):
        paired.job_result(output, manifest, jobs[0])


def test_unknown_cost_and_corrupt_frozen_answer_are_rejected(candidate):
    output, manifest, jobs = candidate
    directory = synthetic_answer(output, manifest, jobs[0], cost_complete=False)
    with pytest.raises(ValueError, match="Unknown generation cost"):
        paired.job_result(output, manifest, jobs[0])
    (directory / "patch.diff").write_text("modified after freeze")
    with pytest.raises(ValueError, match="evidence changed"):
        paired.job_result(output, manifest, jobs[0])


def test_smoke_then_dev_lock_and_evidence_are_required(candidate):
    output, manifest, jobs = candidate
    with pytest.raises(FileNotFoundError):
        paired.require_previous(output, manifest, "dev")
    for job in (j for j in jobs if j["phase"] == "smoke"):
        synthetic_answer(output, manifest, job)
    paired.freeze_phase(output, manifest, "smoke")
    paired.require_previous(output, manifest, "dev")
    with pytest.raises(FileNotFoundError):
        paired.require_previous(output, manifest, "holdout")
    first = output / "runs" / jobs[0]["job_id"] / "loaded-config.json"
    first.write_text("{}")
    with pytest.raises(ValueError, match="Evidence changed"):
        paired.require_previous(output, manifest, "dev")


def test_environment_blocked_pairs_keep_formal_denominator(candidate):
    output, manifest, jobs = candidate
    identity = manifest["split"]["dev"][0]
    ex.write_json(output / "preparation" / f"{identity}.json", {"status": "env_blocked", "error": "deadline"})
    for job in (j for j in jobs if j["phase"] == "dev" and j["instance_id"] != identity):
        synthetic_answer(output, manifest, job)
    result = paired.freeze_phase(output, manifest, "dev")
    assert result["terminal"] and result["paired"]["missing_pairs"] == [identity]
    for group in result["groups"].values():
        assert group["selected"] == 30 and group["resolved"] == 29 and group["unknown"] == 1
        assert group["resolved_over_selected"] == pytest.approx(29 / 30)


def test_statistics_are_paired_and_reproducible_with_missing_results():
    outcomes = {"win": (False, True), "loss": (True, False), "same": (True, True), "unknown": (None, False)}
    rows = [{"instance_id": identity, "variant": arm, "resolved": result}
            for identity, pair in outcomes.items() for arm, result in zip(paired.ARMS, pair)]
    a = paired.paired_statistics(rows, list(outcomes), samples=100)
    assert a == paired.paired_statistics(rows, list(outcomes), samples=100)
    assert a["fail_to_success"] == a["success_to_fail"] == 1
    assert a["pairable"] == 3 and a["delta_pairable"] == 0
    assert a["missing_pairs"] == ["unknown"]


def test_preparation_timeout_is_not_silently_retried(candidate, monkeypatch):
    output, manifest, _ = candidate
    calls = []
    from repoforge.evaluation.native_provision import NativeProvisioner
    async def prepare(self, public, **kwargs):
        calls.append(public)
        raise TimeoutError("deadline reached")
    monkeypatch.setattr(NativeProvisioner, "prepare", prepare)
    public = {"instance_id": "test", "repo": "test/test", "base_commit": "a" * 40,
              "problem_statement": "public task", "version": "1"}
    assert paired.provision_one(output, manifest, public, {}) is False
    assert paired.provision_one(output, manifest, public, {}) is False
    assert len(calls) == 1


def test_controller_freezes_answers_before_grading_and_does_not_resample(candidate, monkeypatch):
    from repoforge.evaluation import experiment_native as native
    output, manifest, jobs = candidate
    calls = []
    def inventory(*args):
        ex.write_json(output / "dataset-receipt.json", {"fingerprint": manifest["paired"]["dataset_fingerprint"]})
        ex.write_json(output / "host-install-recipes.json", {})
        return {identity: {"instance_id": identity} for identity in manifest["smoke_ids"]}
    def generate(output, manifest, job, **kwargs):
        calls.append((job["job_id"], "generate"))
        directory = synthetic_answer(output, manifest, job)
        (directory / "grading.json").unlink()
        (directory / "own-tests.json").unlink()
    def own(output, manifest, job):
        calls.append((job["job_id"], "own"))
        answer = ex.verify_answer(output / "runs" / job["job_id"], job)
        ex.write_json(output / "runs" / job["job_id"] / "own-tests.json", {"answer_hash": answer["answer_hash"]})
    def grade(output, manifest, job):
        calls.append((job["job_id"], "grade"))
        answer = ex.verify_answer(output / "runs" / job["job_id"], job)
        ex.write_json(output / "runs" / job["job_id"] / "grading.json", {
            "answer_hash": answer["answer_hash"], "local_grading": "completed", "local_resolved": True,
            "original_baseline": {"original_failure_reproduced": True}})
    monkeypatch.setattr("repoforge.evaluation.experiment_inputs.export_public_dataset", inventory)
    monkeypatch.setattr(paired, "provision_one", lambda *a: True)
    monkeypatch.setattr(native, "run_job", generate)
    monkeypatch.setattr(native, "validate_agent_tests", own)
    monkeypatch.setattr(native, "grade_job", grade)
    monkeypatch.setattr(native, "release_job_workspace", lambda *a: None)
    result = paired.run_phase(output, manifest, "smoke")
    assert result["terminal"] and len(calls) == 30
    assert [stage for _, stage in calls[:3]] == ["generate", "own", "grade"]
    assert (output / "smoke-lock.json").exists()
    assert paired.run_phase(output, manifest, "smoke") == result
    assert len(calls) == 30  # Reopening a frozen phase makes no provider/backend calls.
