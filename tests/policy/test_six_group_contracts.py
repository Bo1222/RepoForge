"""Behavioral acceptance cases for the six gaps, using public-only executions."""

from copy import deepcopy
import sys

import pytest

from moatless.actions.check_syntax import CheckSyntaxArgs
from moatless.actions.create_file import CreateFileArgs
from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.execution_memory import test_records as records
from moatless.policy.repair import assess
from moatless.policy.replanning import update_repair_control, accept_replan, evidence_rows
from moatless.verification_inputs import compare_runs, test_identity as case_identity
from tests.policy.test_acceptance_repair import fixture, plan, edit
from moatless.actions.register_acceptance_plan import RegisterAcceptancePlanArgs
from tests.memory.test_repository_context import execute


@pytest.mark.asyncio
async def test_old_related_failure_survives_edit_and_unrelated_green_until_actual_rerun(tmp_path):
    agent, root, workspace, runtime = await fixture(
        tmp_path,
        {
            "test_extra.py": "from calc import value\ndef test_extra(): assert value() == 2\n",
            "test_unrelated.py": "def test_other(): assert True\n",
        },
    )
    node = await execute(
        agent,
        root,
        RegisterAcceptancePlanArgs(plan=plan()),
        edit(new="return 3"),
        RunTestsArgs(test_files=["test_extra.py"], test_source="base_public"),
    )
    assert assess(node, workspace).failures
    node = await execute(
        agent,
        node,
        edit("return 3", "return 2"),
        RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"),
        RunTestsArgs(test_files=["test_calc.py"], test_source="base_public", product_source="base"),
        RunTestsArgs(test_files=["test_unrelated.py"]),
    )
    pending = assess(node, workspace)
    assert pending.state != "ready"
    assert pending.failure_obligations[0]["targets"] == ["test_extra.py"]
    assert pending.next_check["test_files"] == ["test_extra.py"]
    node = await execute(agent, node, RunTestsArgs(test_files=["test_extra.py"], test_source="base_public"))
    assert assess(node, workspace).state == "ready"
    assert records(node, workspace)[0]["status"] == "failed"  # immutable history


@pytest.mark.asyncio
async def test_changing_failed_test_cannot_erase_old_failure(tmp_path):
    agent, root, workspace, _ = await fixture(tmp_path)
    node = await execute(
        agent,
        root,
        RegisterAcceptancePlanArgs(plan=plan()),
        edit(new="return 3"),
        RunTestsArgs(test_files=["test_calc.py"]),
    )
    node = await execute(
        agent,
        node,
        StringReplaceArgs(path="test_calc.py", old_str="== 2", new_str="== 3"),
        RunTestsArgs(test_files=["test_calc.py"]),
    )
    assert assess(node, workspace).failure_obligations
    assert assess(node, workspace).state != "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,old,new",
    [
        ("conftest.py", "FLAG = 1", "FLAG = 2"),
        ("tests/helpers.py", "FLAG = 1", "FLAG = 2"),
        ("pytest.ini", "addopts = -q", "addopts = -v"),
        ("tests/data.txt", "one", "two"),
        ("assets/custom_cases.json", '{"value":1}', '{"value":2}'),
    ],
)
async def test_workspace_auxiliary_edits_invalidate_receipts_and_require_same_baseline_bundle(tmp_path, path, old, new):
    # Seed nested files before constructing the public runner/base.
    (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / path).write_text(old + "\n")
    agent, root, workspace, runtime = await fixture(tmp_path)
    runtime.files[path] = old + "\n"
    if path.startswith("assets/"):
        runtime.verification_files = [path]
    node = await execute(
        agent,
        root,
        edit(),
        RunTestsArgs(test_files=["test_calc.py"]),
        RunTestsArgs(test_files=["test_calc.py"], product_source="base"),
    )
    first = records(node, workspace)
    assert first[0]["comparison"]["status"] == "known"
    raw = deepcopy(node.action_steps[-2].observation.properties["test_run"])
    node = await execute(agent, node, StringReplaceArgs(path=path, old_str=old, new_str=new))
    assert records(node, workspace)[0]["effective_status"] == "stale"
    node = await execute(agent, node, RunTestsArgs(test_files=["test_calc.py"]))
    assert records(node, workspace)[-1]["comparison"]["status"] == "unknown"
    node = await execute(agent, node, RunTestsArgs(test_files=["test_calc.py"], product_source="base"))
    latest = records(node, workspace)
    assert latest[-2]["comparison"]["status"] == "known"
    assert path in latest[-2]["verification_inputs"]
    assert new in runtime.calls[-1]["patch"]
    assert raw["input_identity"] == first[0]["input_identity"]


@pytest.mark.asyncio
async def test_cache_outputs_preserve_receipt_but_data_add_delete_runner_and_env_stale_it(tmp_path, monkeypatch):
    agent, root, workspace, runtime = await fixture(tmp_path)
    node = await execute(agent, root, edit(), RunTestsArgs(test_files=["test_calc.py"]))
    for name in ["report.xml", "junit.xml", "result.log", "__pycache__/calc.pyc", ".pytest_cache/v/cache/nodeids"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("new output")
    assert records(node, workspace)[-1]["effective_status"] == "passed"
    data = tmp_path / "tests" / "dataset.csv"
    data.parent.mkdir()
    data.write_text("1,2\n")
    assert records(node, workspace)[-1]["effective_status"] == "stale"
    data.unlink()
    assert records(node, workspace)[-1]["effective_status"] == "passed"
    original_command = runtime.test_command
    runtime.test_command = "pytest --new-runner-option"
    assert records(node, workspace)[-1]["effective_status"] == "stale"
    runtime.test_command = original_command
    assert records(node, workspace)[-1]["effective_status"] == "passed"
    monkeypatch.setenv("TASK_FEATURE_FLAG", "new")
    assert records(node, workspace)[-1]["effective_status"] == "stale"


def pair():
    common = {
        "recipe_version": 3,
        "complete": True,
        "return_code": 1,
        "counts": {"passed": 1, "failed": 2},
        "runner_fingerprint": "runner",
        "base_identity": {"commit": "base"},
        "test_bundle_hash": "bundle",
        "environment_before": "env",
        "environment_after": "env",
        "test_source": "workspace",
        "targets": ["test_calc.py"],
        "input_identity": "stable",
        "input_identity_after": "stable",
    }

    def row(name, status):
        return {"file_path": "test_calc.py", "name": name, "status": status}

    baseline = {
        **common,
        "source_id": "base",
        "product_source": "base",
        "results": [
            row("ClassA::test_value[1]", "PASSED"),
            row("ClassB::test_value[1]", "FAILED"),
            row("test_value[2]", "FAILED"),
        ],
    }
    candidate = {
        **common,
        "source_id": "current",
        "product_source": "current",
        "results": [
            row("ClassA::test_value[1]", "FAILED"),
            row("ClassB::test_value[1]", "PASSED"),
            row("test_value[2]", "FAILED"),
        ],
    }
    return candidate, baseline


def test_public_baseline_comparison_reports_regressions_preexisting_and_fixed_by_full_id():
    candidate, baseline = pair()
    comparison = compare_runs(candidate, baseline)
    assert comparison["status"] == "known"
    assert comparison["introduced_failures"] == ["test_calc.py::ClassA::test_value[1]"]
    assert comparison["fixed_tests"] == ["test_calc.py::ClassB::test_value[1]"]
    assert comparison["preexisting_failures"] == ["test_calc.py::test_value[2]"]
    assert case_identity({"file_path": "x.py", "name": "ClassA", "method": "test_a[p]"}) == "x.py::ClassA::test_a[p]"


@pytest.mark.parametrize(
    "mutation",
    [
        "absent",
        "partial",
        "timeout",
        "missing_name",
        "duplicate",
        "collection",
        "skipped",
        "different_bundle",
        "different_env",
        "different_runner",
        "different_base",
        "different_ids",
        "during_edit",
        "legacy",
    ],
)
def test_untrustworthy_comparison_is_unknown_never_empty_regression_set(mutation):
    candidate, baseline = pair()
    if mutation == "absent":
        baseline = None
    elif mutation == "partial":
        baseline["complete"] = False
    elif mutation == "timeout":
        baseline["timed_out"] = True
    elif mutation == "missing_name":
        baseline["results"][0]["name"] = None
    elif mutation == "duplicate":
        baseline["results"].append(baseline["results"][0])
    elif mutation == "collection":
        baseline["counts"]["error"] = 1
    elif mutation == "skipped":
        baseline["results"][0]["status"] = "SKIPPED"
    elif mutation == "different_bundle":
        baseline["test_bundle_hash"] = "different"
    elif mutation == "different_env":
        baseline["environment_after"] = "different"
    elif mutation == "different_runner":
        baseline["runner_fingerprint"] = "different"
    elif mutation == "different_base":
        baseline["base_identity"] = {"commit": "different"}
    elif mutation == "different_ids":
        baseline["results"][0]["name"] += "other"
    elif mutation == "during_edit":
        baseline["input_identity_after"] = "different"
    elif mutation == "legacy":
        baseline["recipe_version"] = 2
    comparison = compare_runs(candidate, baseline)
    assert comparison["status"] == "unknown" and comparison["introduced_failures"] is None


def failed_record(source, patch, targets=None):
    return {
        "source_id": source,
        "product_source": "current",
        "status": "failed",
        "product_patch_hash": patch,
        "targets": targets or ["test_calc.py"],
        "product_files": ["calc.py"],
        "results": [
            {
                "status": "FAILED",
                "file_path": "test_calc.py",
                "name": "test_value",
                "failure_output": "AssertionError: wrong value",
            }
        ],
    }


@pytest.mark.asyncio
async def test_repair_revisit_target_change_and_new_evidence_replan_are_durable(tmp_path):
    agent, root, workspace, _ = await fixture(tmp_path)
    # Real read observations are the only way to discharge the replan gate.
    read = ReadFile()
    await read.initialize(workspace)
    agent._action_map[ReadFileArgs] = read
    node = await execute(agent, root, ReadFileArgs(file_path="calc.py"))
    a, b, a_again = [
        failed_record(f"{node.node_id}:{i}", patch, targets)
        for i, patch, targets in [
            (0, "A", ["test_calc.py"]),
            (1, "B", ["test_calc.py", "test_extra.py"]),
            (2, "A", ["test_extra.py"]),
        ]
    ]
    # The source IDs must correspond to actual trajectory entries for chronology.
    from moatless.node import ActionStep
    from moatless.actions.schema import Observation

    node.action_steps.extend(
        [
            ActionStep(action=RunTestsArgs(test_files=["test_calc.py"]), observation=Observation.create("failure"))
            for _ in range(2)
        ]
    )
    saved = {}
    assert update_repair_control(node, [a, b, a_again], saved, 3) is None
    family, gate = next(iter(saved["replans"].items()))
    assert saved["failure_families"][family]["repair_attempts"] == 2 and gate["status"] == "required"
    proposal = {
        "family_id": family,
        "hypothesis": "The empty input follows an unhandled branch",
        "next_steps": "Read the missing branch and repair it",
        "evidence_sources": [f"{node.node_id}:0"],
    }
    assert not accept_replan(node, proposal, saved)[0]  # before trigger
    repeated = await execute(agent, node, ReadFileArgs(file_path="calc.py"))
    proposal["evidence_sources"] = [f"{repeated.node_id}:0"]
    assert not accept_replan(repeated, proposal, saved)[0]  # same content/range
    fresh = await execute(agent, repeated, ReadFileArgs(file_path="test_calc.py"))
    # Background evidence from before the trigger and a repository path (not
    # a source ID) may be cited alongside the new evidence; the path resolves
    # to the post-trigger read that actually contained it.
    proposal["evidence_sources"] = [f"{node.node_id}:0", "test_calc.py"]
    assert accept_replan(fresh, proposal, saved)[0]
    saved = deepcopy(saved)  # same serialized structure used by graph journal
    assert saved["replans"][family]["status"] == "accepted"
    # Testing the unchanged candidate does not consume another repair trial.
    same = failed_record(f"{fresh.node_id}:0", "A", ["test_calc.py"])
    assert update_repair_control(fresh, [a, b, a_again, same], saved, 3) is None
    final = failed_record(f"{fresh.node_id}:0", "C", ["test_calc.py"])
    saved["counted_failure_receipts"].remove(final["source_id"])
    assert update_repair_control(fresh, [final], saved, 3) == "repair_exhausted"


@pytest.mark.asyncio
async def test_base_public_executes_original_helpers_and_new_workspace_helper_does_not_stale_it(tmp_path):
    (tmp_path / "conftest.py").write_text("FLAG = 1\n")
    agent, root, workspace, runtime = await fixture(tmp_path)
    runtime.files["conftest.py"] = "FLAG = 1\n"
    node = await execute(agent, root, edit(), RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"))
    before = records(node, workspace)[-1]
    node = await execute(
        agent,
        node,
        StringReplaceArgs(path="conftest.py", old_str="FLAG = 1", new_str="FLAG = 2"),
        CreateFileArgs(path="tests/new_helper.py", file_text="FLAG = 9\n"),
        RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"),
    )
    after = records(node, workspace)
    assert after[0]["effective_status"] == "passed"
    assert before["test_bundle_hash"] == after[-1]["test_bundle_hash"]
    assert "conftest.py" not in runtime.calls[-1]["patch"] and "new_helper" not in runtime.calls[-1]["patch"]


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["a6", "a7"])
async def test_old_assertion_conflict_requires_public_quote_and_execution_and_only_partially_delivers(
        tmp_path, variant):
    from moatless.actions.explain_test_conflict import ExplainTestConflict, ExplainTestConflictArgs

    agent, root, workspace, _ = await fixture(
        tmp_path, {"test_old.py": "from calc import value\ndef test_old(): assert value() == 1\n"}
    )
    action = ExplainTestConflict()
    await action.initialize(workspace)
    agent._action_map[ExplainTestConflictArgs] = action
    node = await execute(
        agent,
        root,
        RegisterAcceptancePlanArgs(plan=plan()),
        edit(),
        RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"),
        RunTestsArgs(test_files=["test_calc.py"], test_source="base_public", product_source="base"),
        RunTestsArgs(test_files=["test_old.py"], test_source="base_public"),
        RunTestsArgs(test_files=["test_old.py"], test_source="base_public", product_source="base"),
    )
    assert assess(node, workspace).state == "repair"
    old = records(node, workspace)[-2]
    args = ExplainTestConflictArgs(
        criterion_id="behavior",
        candidate_source=old["source_id"],
        failing_case="test_old.py::test_old",
        public_quote="empty input",
        original_assertion="assert value() == 1",
        explanation="The public issue requires the new value, while this old assertion expects the old value",
    )
    bad = args.model_copy(update={"original_assertion": "assert True"})
    denied = await execute(agent, node, bad)
    assert denied.observation.properties["fail_reason"] == "invalid_args"
    node = await execute(agent, node, args)
    assessment = assess(node, workspace)
    assert assessment.state == "partial_ready", assessment
    assert assessment.explained_failures and not assessment.failures
    assert records(node, workspace)[-2]["status"] == "failed"
    from moatless.flow.langgraph_flow import LangGraphFlow
    from moatless.actions.finish import FinishArgs
    from scripts.migration_smoke import response

    agent.completion_model._responses = [response(FinishArgs(finish_reason="review public conflict"))]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=12, action_policy={"variant": variant})
    last = await flow.run(workspace=workspace, node_id=node.node_id)
    assert last.completion_status["status"] == "partially_verified"
    assert "verified_checkpoint" not in flow.graph_state["repair_state"]
    assert flow.graph_state["repair_state"]["partial_checkpoint"]["patch_hash"] == assessment.patch_hash


@pytest.mark.asyncio
async def test_timeout_partial_pass_is_not_success_and_cleanup_is_explicit(tmp_path):
    import asyncio
    from moatless.actions.run_tests import RunTests
    from moatless.testing.schema import TestResult, TestStatus

    agent, root, workspace, runtime = await fixture(tmp_path)

    async def partial(**kwargs):
        runtime.last_test_results = [TestResult(name="test_value", file_path="test_calc.py", status=TestStatus.PASSED)]
        try:
            await asyncio.Event().wait()
        finally:
            runtime.last_test_cleanup_confirmed = True

    runtime.run_tests = partial
    for action in agent.actions:
        if isinstance(action, RunTests):
            action.timeout = 0.02
    node = await execute(agent, root, edit(), RunTestsArgs(test_files=["test_calc.py"]))
    record = records(node, workspace)[-1]
    assert record["counts"] == {"passed": 1} and record["timed_out"] and record["partial"]
    assert record["status"] == "error" and record["complete"] is False
    assert node.observation.properties["continuation_safe"] is True


def test_exit_code_and_parsed_counts_must_agree_for_comparison():
    candidate, baseline = pair()
    baseline["return_code"] = 0
    assert compare_runs(candidate, baseline)["status"] == "unknown"
    baseline["return_code"] = 1
    baseline["counts"] = {"passed": 999, "failed": 2}
    assert compare_runs(candidate, baseline)["introduced_failures"] is None


@pytest.mark.asyncio
async def test_fixture_log_is_input_even_though_ordinary_logs_are_output(tmp_path):
    source = tmp_path / "tests" / "fixture.log"
    source.parent.mkdir()
    source.write_text("original fixture\n")
    agent, root, workspace, _ = await fixture(tmp_path)
    node = await execute(agent, root, edit(), RunTestsArgs(test_files=["test_calc.py"]))
    assert records(node, workspace)[-1]["effective_status"] == "passed"
    source.write_text("changed fixture\n")
    assert records(node, workspace)[-1]["effective_status"] == "stale"


@pytest.mark.asyncio
async def test_removing_import_from_unregistered_failed_test_cannot_erase_its_original_relation(tmp_path):
    agent, root, workspace, _ = await fixture(
        tmp_path, {"test_extra.py": "from calc import value\ndef test_extra(): assert value() == 2\n"}
    )
    node = await execute(
        agent,
        root,
        RegisterAcceptancePlanArgs(plan=plan()),
        edit(new="return 3"),
        RunTestsArgs(test_files=["test_extra.py"]),
    )
    node = await execute(
        agent,
        node,
        StringReplaceArgs(path="test_extra.py", old_str="from calc import value", new_str="value = lambda: 2"),
        RunTestsArgs(test_files=["test_extra.py"]),
    )
    pending = assess(node, workspace).failure_obligations
    assert pending and pending[0]["targets"] == ["test_extra.py"]


@pytest.mark.asyncio
async def test_read_cache_cannot_hide_changed_auxiliary_disk_input(tmp_path):
    (tmp_path / "conftest.py").write_text("FLAG = 1\n")
    agent, root, workspace, runtime = await fixture(tmp_path)
    runtime.files["conftest.py"] = "FLAG = 1\n"
    root.file_context.add_file("conftest.py", show_all_spans=True)
    assert root.file_context._files["conftest.py"].content == "FLAG = 1\n"
    node = await execute(agent, root, edit(), RunTestsArgs(test_files=["test_calc.py"]))
    assert records(node, workspace)[-1]["effective_status"] == "passed"
    (tmp_path / "conftest.py").write_text("FLAG = 2\n")
    assert records(node, workspace)[-1]["effective_status"] == "stale"


@pytest.mark.asyncio
async def test_internal_symlink_input_is_bound_and_external_input_is_unavailable(tmp_path):
    data = tmp_path / "fixture.txt"
    data.write_text("one\n")
    (tmp_path / "fixture_link.txt").symlink_to("fixture.txt")
    agent, root, workspace, _ = await fixture(tmp_path)
    node = await execute(agent, root, edit(), RunTestsArgs(test_files=["test_calc.py"]))
    assert records(node, workspace)[-1]["effective_status"] == "passed"
    data.write_text("two\n")
    assert records(node, workspace)[-1]["effective_status"] == "stale"
    (tmp_path / "external_link.txt").symlink_to("/usr/bin/python3")
    node = await execute(agent, node, RunTestsArgs(test_files=["test_calc.py"]))
    assert node.observation.properties["fail_reason"] == "environment_error"
    assert records(node, workspace)[-1]["status"] != "passed"


@pytest.mark.asyncio
async def test_new_repro_can_reprove_unchanged_old_failure_with_real_new_bundle_baseline(tmp_path):
    agent, root, workspace, _ = await fixture(
        tmp_path, {"test_extra.py": "from calc import value\ndef test_extra(): assert value() == 2\n"}
    )
    node = await execute(
        agent,
        root,
        RegisterAcceptancePlanArgs(plan=plan()),
        edit(new="return 3"),
        RunTestsArgs(test_files=["test_extra.py"]),
    )
    node = await execute(
        agent,
        node,
        CreateFileArgs(path="test_new.py", file_text="from calc import value\ndef test_new(): assert value() == 2\n"),
        edit("return 3", "return 2"),
        RunTestsArgs(test_files=["test_extra.py"]),
    )
    assert assess(node, workspace).failure_obligations  # new bundle alone is insufficient
    node = await execute(agent, node, RunTestsArgs(test_files=["test_extra.py"], product_source="base"))
    assert not assess(node, workspace).failure_obligations  # same old case + real comparable fail->pass


@pytest.mark.asyncio
async def test_added_product_file_or_changed_exception_cannot_reset_same_behavior_repair_cap(tmp_path):
    agent, root, workspace, _ = await fixture(tmp_path)
    from moatless.node import ActionStep
    from moatless.actions.schema import Observation

    node = root.create_child()
    node.action_steps = [
        ActionStep(action=RunTestsArgs(test_files=["test_calc.py"]), observation=Observation.create("failure"))
        for _ in range(4)
    ]
    rows = [failed_record(f"{node.node_id}:{i}", patch) for i, patch in enumerate(["A", "B", "A", "C"])]
    rows[1]["product_files"].append("new_product.py")
    rows[2]["results"][0]["failure_output"] = "ValueError: different symptom of the same failed case"
    saved = {}
    assert update_repair_control(node, rows[:3], saved, 3) is None
    assert len(saved["failure_families"]) == 1
    assert next(iter(saved["failure_families"].values()))["repair_attempts"] == 2
    assert update_repair_control(node, rows, saved, 3) == "repair_exhausted"
