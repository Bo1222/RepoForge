"""T7 frozen matrix, lock, acceptance and analysis contracts."""

from copy import deepcopy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from repoforge.evaluation import t7
from repoforge.evaluation.experiment import (
    PROFILES, SMOKE_IDS, SMOKE_VARIANTS, T7_DEV_VARIANTS, T7_HOLDOUT_VARIANTS,
    T7_PROFILES, T7_PROTOCOL, content_hash, freeze_answer, make_jobs, sha, write_json,
)
from repoforge.evaluation.experiment_inputs import export_public_dataset
from repoforge.evaluation.generation import validate_generation_health
from repoforge.message_history.policy import content_hash as digest

ROOT = Path(__file__).resolve().parents[2]


def manifest():
    dev = [*SMOKE_IDS, *[f"dev-{index}" for index in range(25)]]
    holdout = [f"holdout-{index}" for index in range(100)]
    return {
        "manifest_hash": "frozen", "version": 2,
        "split": {"dev": dev, "holdout": holdout},
        "smoke_ids": list(SMOKE_IDS),
        "profiles": {**deepcopy(PROFILES), **deepcopy(T7_PROFILES)},
        "holdout_state": "locked", "official_harness": "not_run",
        "t7": {"protocol": T7_PROTOCOL, "smoke_variants": list(SMOKE_VARIANTS),
               "cli_variants": ["a0"], "dev_variants": list(T7_DEV_VARIANTS),
               "dev_repeats": 1, "dev_order": "smoke-first-v1",
               "holdout_variants": list(T7_HOLDOUT_VARIANTS), "holdout_repeats": 1,
               "selection_rule": "pre-registered", "falsification_rule": "pre-registered",
               "analysis_plan_sha256": "plan", "small_sample_statement": "n=30 exploratory"},
    }


def test_t7_matrix_is_670_with_frozen_composition():
    jobs = make_jobs(manifest())
    counts = {phase: sum(job["phase"] == phase for job in jobs)
              for phase in ("smoke", "cli_smoke", "dev", "holdout")}
    assert counts == {"smoke": 15, "cli_smoke": 5, "dev": 150, "holdout": 500}
    assert len({job["job_id"] for job in jobs}) == 670
    assert {job["variant"] for job in jobs if job["phase"] == "dev"} == set(T7_DEV_VARIANTS)
    assert {job["variant"] for job in jobs if job["phase"] == "holdout"} == set(T7_HOLDOUT_VARIANTS)
    assert all(job["variant"] == "a0" and job["entrypoint"] == "cli"
               for job in jobs if job["phase"] == "cli_smoke")


def test_t7_dev_order_is_smoke_first_per_variant():
    jobs = make_jobs(manifest())
    ordered = [*SMOKE_IDS, *[f"dev-{index}" for index in range(25)]]
    for variant in T7_DEV_VARIANTS:
        rows = [job for job in jobs if job["phase"] == "dev" and job["variant"] == variant]
        assert [row["instance_id"] for row in rows] == ordered
        assert {row["repeat"] for row in rows} == {0}


def test_legacy_matrix_unchanged():
    legacy = manifest()
    del legacy["t7"]
    legacy["profiles"] = {"loop_legacy": {}, "graph_legacy": {}, "a0": {}}
    jobs = make_jobs(legacy)
    assert sum(job["phase"] == "smoke" for job in jobs) == 15
    assert sum(job["phase"] == "cli_smoke" for job in jobs) == 5
    assert sum(job["phase"] == "dev" for job in jobs) == 90


@pytest.mark.parametrize("change", [
    lambda value: value["t7"].update(protocol="other"),
    lambda value: value["t7"].update(dev_repeats=3),
    lambda value: value["t7"].update(dev_variants=["a0"]),
    lambda value: value["t7"].update(holdout_variants=["a0"]),
    lambda value: value["t7"].update(smoke_variants=["a0"]),
    lambda value: value.update(holdout_state="unlocked"),
    lambda value: value["t7"].update(analysis_plan_sha256=""),
    lambda value: value["profiles"].pop("loop_legacy"),
])
def test_t7_rejects_changed_protocol_or_variants(change):
    value = manifest()
    change(value)
    with pytest.raises(ValueError):
        make_jobs(value)


def _loadable_experiment(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()
    (output / "source.tar").write_bytes(b"frozen-source")
    value = manifest()
    value["source"] = {"path": "source.tar", "sha256": sha(output / "source.tar"), "files": {}}
    value["dataset"] = "princeton-nlp/SWE-bench_Verified"
    value["evaluation_protocol"] = "native-swebench-compatible-v1"
    value["grading_backend"] = "native"
    value.pop("manifest_hash", None)
    value["manifest_hash"] = content_hash(value)
    write_json(output / "manifest.json", value)
    write_json(output / "jobs.json", make_jobs(value))
    return output, value


def test_manifest_and_job_tamper_rejected(tmp_path):
    from repoforge.evaluation.experiment import load_manifest
    output, value = _loadable_experiment(tmp_path)
    assert load_manifest(output)["manifest_hash"] == value["manifest_hash"]
    jobs = json.loads((output / "jobs.json").read_text())
    jobs[0]["variant"] = "a4"
    write_json(output / "jobs.json", jobs)
    with pytest.raises(ValueError, match="job selection"):
        load_manifest(output)


class _Dataset(list):
    _fingerprint = "frozen-fingerprint"


def test_export_public_dataset_holdout_selects_100(monkeypatch, tmp_path):
    value = manifest()
    value["dataset"] = "fake/dataset"
    rows = []
    for identity in [*value["split"]["dev"], *value["split"]["holdout"]]:
        rows.append({"instance_id": identity, "repo": "owner/repo",
                     "base_commit": "a" * 40, "problem_statement": "public problem",
                     "version": "1.0", "environment_setup_commit": "b" * 40})
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: _Dataset(rows))
    inventory = export_public_dataset(tmp_path, value, "holdout")
    assert set(inventory) == set(value["split"]["holdout"])
    stored = json.loads((tmp_path / "public/instances.json").read_text())
    assert len(stored) == 100
    assert not set(stored) & set(value["split"]["dev"])


def _lock_experiment(tmp_path, winner="a1"):
    output = tmp_path / "locked"
    output.mkdir()
    value = manifest()
    value["t7"]["analysis_plan_sha256"] = "plan"
    report = output / "report-dev.json"
    write_json(report, {"manifest_hash": value["manifest_hash"]})
    analysis = output / "analysis/t7"
    analysis.mkdir(parents=True)
    for name in ("labels.jsonl", "stats.json", "paired.json", "recommendation.json", "report.md"):
        (analysis / name).write_text(name)
    lock = {"version": 1, "protocol": T7_PROTOCOL, "manifest_hash": value["manifest_hash"],
            "winner": winner, "control": "a0", "dev_report_sha256": sha(report),
            "analysis": {name: sha(analysis / name) for name in (
                "labels.jsonl", "stats.json", "paired.json", "recommendation.json", "report.md")},
            "analysis_plan_sha256": "plan",
            "selection": {"winner": winner}, "falsification": {"conclusion": "recorded"},
            "holdout_runs_at_lock": 0, "holdout_state": "locked",
            "user_confirmation": {"confirmed": True, "confirmed_at": "now"}}
    lock["lock_hash"] = t7._lock_hash(lock)
    write_json(output / "t7-lock.json", lock)
    return output, value, lock


def test_lock_validation_binds_report_analysis_and_user_confirmation(tmp_path):
    output, value, lock = _lock_experiment(tmp_path)
    assert t7.validate_lock(output, value)["lock_hash"] == lock["lock_hash"]
    (output / "analysis/t7/report.md").write_text("tampered")
    with pytest.raises(ValueError, match="analysis changed"):
        t7.validate_lock(output, value)


def test_lock_is_immutable_and_allows_only_two_arms(tmp_path):
    output, value, _ = _lock_experiment(tmp_path, winner="a1")
    with pytest.raises(ValueError, match="immutable"):
        t7.create_lock(output, value, winner="a1", confirm_user=True)
    jobs = make_jobs(value)
    a1 = next(job for job in jobs if job["phase"] == "holdout" and job["variant"] == "a1")
    a0 = next(job for job in jobs if job["phase"] == "holdout" and job["variant"] == "a0")
    a2 = next(job for job in jobs if job["phase"] == "holdout" and job["variant"] == "a2")
    assert t7.assert_holdout_job_allowed(output, value, a1)["winner"] == "a1"
    assert t7.assert_holdout_job_allowed(output, value, a0)["control"] == "a0"
    with pytest.raises(ValueError, match="not locked"):
        t7.assert_holdout_job_allowed(output, value, a2)


def test_lock_rejects_holdout_effects_before_locking(tmp_path):
    output, value, _ = _lock_experiment(tmp_path, winner="a1")
    (output / "t7-lock.json").unlink()
    job = next(job for job in make_jobs(value)
               if job["phase"] == "holdout" and job["variant"] == "a2")
    write_json(output / "runs" / job["job_id"] / "generation.json", {"job_id": job["job_id"]})
    assert t7._holdout_effects(output, make_jobs(value)) == [job["job_id"]]
    with pytest.raises(ValueError, match="before locking"):
        t7.create_lock(output, value, winner="a1", confirm_user=True)


def test_aggregate_reports_variant_scoped_metrics_and_unlock(tmp_path):
    from repoforge.evaluation.experiment import aggregate
    output, value = _loadable_experiment(tmp_path)
    for job in make_jobs(value):
        if job["phase"] != "dev":
            continue
        if not (output / "runs" / job["job_id"]).exists() and job["variant"] in {"a0", "a1"}:
            write_json(output / "runs" / job["job_id"] / "generation.json", {
                "job_id": job["job_id"], "manifest_hash": value["manifest_hash"],
                "generation_status": "completed",
                "policy_metrics": {"decisions": 2, "interventions": 1}})
            write_json(output / "runs" / job["job_id"] / "usage.json", {
                "reference_cost": 0.1, "known_reference_cost": 0.1, "cost_complete": True})
    report = aggregate(output, "dev", holdout_unlock={"winner": "a1"})
    assert report["holdout_state"] == "unlocked_by_t7_lock"
    assert report["lock_sha256"]
    assert report["policy_metrics_by_variant"]["a1"]["decisions"] == 60
    assert report["policy_metrics_by_variant"]["a1"]["interventions"] == 30
    assert report["policy_metrics_by_variant"]["a0"]["decisions"] == 60
    assert report["groups"]["a1"]["selected"] == 30


def test_policy_stop_is_a_legal_business_result(tmp_path):
    directory = tmp_path / "run"
    write_json(directory / "generation.json", {
        "generation_status": "completed", "finish_reason": "policy_stop"})
    write_json(directory / "usage.json", {"cost_complete": True})
    assert validate_generation_health(directory)["finish_reason"] == "policy_stop"


def _accepted_job(output, value, job, *, local_grading, local_resolved,
                  reproduced, finish_reason="terminal"):
    directory = output / "runs" / job["job_id"]
    write_json(directory / "generation.json", {
        "job_id": job["job_id"], "manifest_hash": value["manifest_hash"],
        "generation_status": "completed", "finish_reason": finish_reason,
        "selected_node_id": 1, "policy_metrics": {}})
    write_json(directory / "usage.json", {"reference_cost": 0.1,
                                          "known_reference_cost": 0.1, "cost_complete": True})
    for name, body in (("patch.diff", "diff --git a/x b/x\n"),
                       ("trajectory.json", {}), ("events.jsonl", ""), ("sdk.jsonl", ""),
                       ("agent-tests.json", []), ("test-requests.json", [])):
        (directory / name).write_text(body if isinstance(body, str) else json.dumps(body))
    answer = freeze_answer(directory, job)
    from repoforge.evaluation.configuration_bindings import tool_bindings
    from repoforge.message_history.policy import ContextPolicy
    from repoforge.policy import ActionPolicyConfig
    profile = value["profiles"][job["variant"]]
    managed = job["entrypoint"] == "cli" or profile["managed"]
    flow = "LangGraphFlow" if job["entrypoint"] == "cli" or profile["flow"] == "graph" else "AgenticLoop"
    tools = tool_bindings(value, job)
    write_json(directory / "loaded-config.json", {
        "manifest_hash": value["manifest_hash"], "job_id": job["job_id"],
        "entrypoint": job["entrypoint"], "flow_type": flow,
        "memory_type": "ManagedMessageHistoryGenerator" if managed else "MessageHistoryGenerator",
        "policy": ContextPolicy(**value["cli_config"]["context_policy"]).model_dump(mode="json")
        if managed else None,
        "action_policy": (ActionPolicyConfig(**profile.get("action_policy", {"variant": "off"})).model_dump(mode="json")
                          if flow == "LangGraphFlow" else {"variant": "off"}),
        "tool_names": list(tools), "schema_names": list(tools.values()),
        "max_iterations": 40, "max_cost": 1.0, "tool_isolation": "native-offline-v1"})
    evidence = output / "grading-native/case.json"
    write_json(evidence, {"case": "grading"})
    write_json(directory / "grading.json", {
        "answer_hash": answer["answer_hash"], "grading_backend": "native",
        "evaluation_protocol": value["evaluation_protocol"],
        "grader_source_sha256": sha(ROOT / "repoforge/evaluation/native_grading.py"),
        "local_grading": local_grading, "local_resolved": local_resolved,
        "original_baseline": {"original_failure_reproduced": reproduced},
        "official_harness": "not_run", "official_resolved": None,
        "evidence": {"grading-native/case.json": sha(evidence)}})
    own = output / "own-tests-native/result.json"
    write_json(own, {"result": "executed"})
    write_json(directory / "own-tests.json", {
        "answer_hash": answer["answer_hash"], "hidden_tests": False,
        "status": "executed", "evidence": {"own-tests-native/result.json": sha(own)}})
    return answer


def _acceptance_fixture(tmp_path, name):
    output = tmp_path / name
    output.mkdir()
    value = manifest()
    value["manifest_hash"] = "frozen"
    value["evaluation_protocol"] = "native-swebench-compatible-v1"
    value["source"] = {"files": {
        "repoforge/evaluation/native_grading.py": sha(ROOT / "repoforge/evaluation/native_grading.py")}}
    value["cli_config"] = {"max_iterations": 40, "max_cost": 1.0,
                           "context_policy": {"version": 1, "context_window": 8000,
                                              "max_input_tokens": 4096,
                                              "max_observation_tokens": 4000,
                                              "recent_interactions": 2, "enable_task_state": False},
                           "api_key_env": "DEEPSEEK_API_KEY"}
    value["agent_configs"] = {variant: {"actions": [
        {"action_class": "repoforge.actions.finish.Finish"}]}
        for variant in value["profiles"]}
    job = next(item for item in make_jobs(value)
               if item["phase"] == "dev" and item["variant"] == "a0")
    write_json(output / "env-blocked-jobs.json", {job["instance_id"]: {
        "record": "docs/2026-10-18_T7_留出环境预审登记.md", "phase": "dev"}})
    return output, value, job


def test_env_blocked_accepts_environment_error_or_normal_completion(tmp_path):
    output, value, job = _acceptance_fixture(tmp_path, "blocked")
    _accepted_job(output, value, job, local_grading="environment_error",
                  local_resolved=None, reproduced=False)
    assert t7.accept_job(output, value, job)["env_blocked"]
    _accepted_job(output, value, job, local_grading="completed",
                  local_resolved=True, reproduced=True)
    assert t7.accept_job(output, value, job)["local_resolved"] is True


def test_policy_stop_job_is_accepted(tmp_path):
    output, value, job = _acceptance_fixture(tmp_path, "policy-stop")
    _accepted_job(output, value, job, local_grading="completed", local_resolved=True,
                  reproduced=True, finish_reason="policy_stop")
    assert t7.accept_job(output, value, job)["finish_reason"] == "policy_stop"


def test_legacy_smoke_acceptance_uses_actual_flow(tmp_path):
    output, value, _ = _acceptance_fixture(tmp_path, "legacy-smoke")
    job = next(item for item in make_jobs(value)
               if item["phase"] == "smoke" and item["variant"] == "loop_legacy")
    _accepted_job(output, value, job, local_grading="completed", local_resolved=True,
                  reproduced=True)
    row = t7.accept_job(output, value, job)
    assert row["checks"]["flow"] and row["checks"]["memory"]
    assert row["checks"]["context"] and row["checks"]["action_policy"]


def test_scan_classification_is_deterministic(tmp_path):
    output = tmp_path / "scan"
    output.mkdir()
    ftp = ["module::test_case"]
    assert t7.classify_scan({"provision_failed": True}, output) == "provision_failed"
    assert t7.classify_scan({"case": {"status": "completed"}, "FAIL_TO_PASS": []}, output) \
        == "empty_or_corrupt_ftp"
    assert t7.classify_scan({"case": {"status": "environment_error"}, "FAIL_TO_PASS": ftp}, output) \
        == "environment_error"
    assert t7.classify_scan({"case": {"status": "completed"}, "FAIL_TO_PASS": ftp,
                             "original_failure_reproduced": True}, output) == "clean"
    raw = output / "holdout-scan" / "case" / "test-output.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("FAILED module::test_case\n")
    assert t7.classify_scan({"case": {"status": "completed"}, "FAIL_TO_PASS": ftp,
                             "observed_ftp_failures": [], "test_statuses": {},
                             "original_failure_reproduced": False,
                             "evidence": {"holdout-scan/case/test-output.txt": "x"}}, output) \
        == "parse_suspect"
    raw.write_text("ModuleNotFoundError")
    assert t7.classify_scan({"case": {"status": "completed"}, "FAIL_TO_PASS": ftp,
                             "observed_ftp_failures": ["module::other"], "test_statuses": {
                                 "module::other": "FAILED"},
                             "original_failure_reproduced": False,
                             "evidence": {"holdout-scan/case/test-output.txt": "x"}}, output) \
        == "ftp_not_reproduced"
    raw.write_text("No matching distribution found for demo")
    assert t7.classify_scan({"case": {"status": "environment_error"}, "FAIL_TO_PASS": ftp,
                             "evidence": {"holdout-scan/case/test-output.txt": "x"}}, output) \
        == "build_wheel_gap"


def test_paired_bootstrap_is_deterministic():
    from scripts.t7_analysis import _pairs
    rows = []
    for index in range(30):
        resolved_a0 = index % 2 == 0
        resolved_a1 = index % 3 == 0
        rows.append({"instance_id": f"i-{index}", "variant": "a0",
                     "local_resolved": resolved_a0})
        rows.append({"instance_id": f"i-{index}", "variant": "a1",
                     "local_resolved": resolved_a1})
    first = _pairs(rows, "dev")
    second = _pairs(rows, "dev")
    assert first == second
    a1 = first["a1"]
    assert a1["bootstrap"]["seed"] == 20261018
    assert a1["bootstrap"]["resamples"] == 10000
    assert len(a1["bootstrap"]["ci95"]) == 2
    assert a1["paired"] == 30


def test_label_rules_are_deterministic_and_evidence_bound():
    from scripts.t7_analysis import label_facts
    base = {"job_id": "job", "evidence": {"grading.json": "runs/job/grading.json"},
            "local_grading": "completed", "generation_status": "completed",
            "finish_reason": "terminal", "local_resolved": False, "has_product_patch": True,
            "test_files": [], "tests": {"base_public_requests": 0,
                                        "effective_verification": True},
            "progress": {"actual_steps": 5, "no_gain_steps": 0, "no_gain_rate": 0.0,
                         "max_no_gain_streak": 0},
            "policy_stop_reasons": [], "patch_applicable": True,
            "original_failure_reproduced": True,
            "tests_status": {"FAIL_TO_PASS": {"module::test": "FAILED"},
                             "PASS_TO_PASS": {}}}

    def categories(**changes):
        return [item["category"] for item in label_facts({**base, **changes})]

    assert categories(has_product_patch=False) == ["context_omission"]
    assert categories(finish_reason="max_iterations") == ["budget"]
    assert categories(local_grading="environment_error") == ["environment"]
    assert categories(test_files=["tests/test_x.py"],
                      tests={"base_public_requests": 0,
                             "effective_verification": True}) == ["test_timing"]
    assert categories(progress={"actual_steps": 5, "no_gain_steps": 3,
                                "no_gain_rate": 0.6, "max_no_gain_streak": 3}) == ["invalid_repeat"]
    assert categories(tests={"base_public_requests": 1,
                             "effective_verification": False}) == ["unverified_finish"]
    assert categories(tests_status={"FAIL_TO_PASS": {"module::test": "FAILED"},
                                    "PASS_TO_PASS": {"module::old": "FAILED"}}) == ["edit_error"]
    for item in label_facts({**base, "has_product_patch": False}):
        assert item["primary"] is True and item["evidence"]
