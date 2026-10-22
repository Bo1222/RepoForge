"""Offline regression tests for public acceptance, execution truth and recovery."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import pytest
import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def wakeup_for_restricted_selector():
    # This container does not deliver cross-thread socketpair wakeups. A timer
    # keeps the real asyncio/aiosqlite engine moving; no executor is mocked.
    async def tick():
        while True:
            await asyncio.sleep(0.01)
    task = asyncio.create_task(tick())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

from moatless.actions.create_file import CreateFile, CreateFileArgs
from moatless.actions.check_syntax import CheckSyntax
from moatless.actions.replan_repair import ReplanRepair, ReplanRepairArgs
from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.verified_finish import VerifiedFinishArgs
from moatless.actions.register_acceptance_plan import RegisterAcceptancePlan, RegisterAcceptancePlanArgs
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.actions.test_targets import discover_test_targets
from moatless.execution_memory import test_records as records
from moatless.flow.langgraph_flow import LangGraphFlow
from moatless.actions.view_code import CodeSpan, ViewCodeArgs
from moatless.actions.schema import Observation
from moatless.file_context import FileContext
from moatless.node import ActionStep, Node
from moatless.policy.replanning import failure_detail, update_no_gain
from moatless.repository.repository import InMemRepository
from moatless.message_history.policy import content_hash
from moatless.policy.repair import AcceptancePlan, assess, registered_plan
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.testing.schema import TestResult as Result, TestStatus as Status
from scripts.migration_smoke import response
from tests.memory.test_repository_context import setup, execute


FILES = {"calc.py": "def value():\n    return 1\n",
         "test_calc.py": "from calc import value\ndef test_value():\n    assert value() == 2\n"}


class PublicPytest(RuntimeEnvironment):
    """Real pytest on supplied public files and patch; no model/provider call."""
    def __init__(self, files):
        self.files, self.calls = files, []
        self.python_executable = sys.executable
        self.test_command = "python -m pytest -q -c /dev/null --junitxml"
        self.last_test_execution, self.last_test_results = [], []

    def recovery_state(self):
        return {"kind": "offline-pytest", "base": content_hash(self.files)}

    async def run_tests(self, patch=None, test_files=None):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name, content in self.files.items():
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            if patch:
                applied = subprocess.run(["git", "apply", "-"], input=patch, text=True,
                                         cwd=tmp, capture_output=True, timeout=5)
                assert applied.returncode == 0, applied.stderr
            report = directory / "report.xml"
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-c", "/dev/null",
                "--junitxml", str(report), *test_files], cwd=tmp, text=True, capture_output=True, timeout=10,
                env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": tmp})
            results = []
            for item in ET.parse(report).iter("testcase"):
                failed, error = item.find("failure"), item.find("error")
                status = Status.FAILED if failed is not None else Status.ERROR if error is not None else Status.PASSED
                module = (item.attrib.get("classname", "").split(".")[0] + ".py") if item.attrib.get("classname") else test_files[0]
                results.append(Result(status=status, name=item.attrib.get("name"), file_path=module,
                    failure_output=failed.text if failed is not None else error.text if error is not None else None))
            self.last_test_execution = [{"target": path, "return_code": proc.returncode, "command": "public pytest"}
                                        for path in test_files]
            self.last_test_results = results
            self.calls.append({"patch": patch, "targets": test_files, "return_code": proc.returncode})
            return results


def plan(extra=False):
    criteria = [{"id": "behavior", "behavior": "value returns 2 for the issue input", "public_quote": "empty input"}]
    checks = [{"id": "repro", "criteria": ["behavior"], "targets": ["test_calc.py"]}]
    if extra:
        criteria.append({"id": "preserved", "behavior": "existing behavior still works", "public_quote": "empty input", "kind": "regression"})
        checks.append({"id": "regression", "criteria": ["preserved"], "targets": ["test_other.py"], "proof": "regression"})
    return AcceptancePlan(criteria=criteria, checks=checks)


async def fixture(tmp_path, extra=None):
    files = {**FILES, **(extra or {})}
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    runtime = PublicPytest(files)
    agent, root, workspace = await setup(tmp_path, shadow=True, runtime=runtime,
        actions=[RegisterAcceptancePlan(), CheckSyntax(), ReplanRepair(), StringReplace(), CreateFile(), RunTests(timeout=3), Finish()])
    return agent, root, workspace, runtime


def edit(old="return 1", new="return 2"):
    return StringReplaceArgs(path="calc.py", old_str=old, new_str=new)


@pytest.mark.asyncio
async def test_graph_partial_pass_runs_remaining_check_then_finalizes_without_model_finish(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path, {"test_other.py": "def test_other():\n    assert 2 + 2 == 4\n"})
    agent.completion_model._responses = [response(RegisterAcceptancePlanArgs(plan=plan(True)), edit(),
        RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"), edit("return 2", "return 999"))]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=5, action_policy={"variant": "a6"})
    last = await asyncio.wait_for(flow.run(workspace=workspace), timeout=15)
    assert flow.is_finished() == "acceptance_satisfied"
    assert last.completion_status["status"] == "verified"
    assert agent.completion_model._calls == 1
    assert last.action_steps[3].observation.properties["skip_reason"] == "test_observation_barrier"
    assert [c["return_code"] for c in runtime.calls] == [0, 1, 0, 0]
    assert runtime.calls[-1]["targets"] == ["test_other.py"]
    checkpoint = flow.graph_state["repair_state"]["verified_checkpoint"]
    assert checkpoint["product_patch"] == last.file_context.generate_git_patch(ignore_tests=True)
    assert checkpoint["patch_hash"] == content_hash(checkpoint["product_patch"])
    assert last.completion_status["resolved"] is None
    assert flow.graph_state["repair_state"]["failure_families"] == {}


@pytest.mark.asyncio
async def test_false_finish_and_no_plan_cannot_bypass_gate(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    agent.completion_model._responses = [response(edit(), FinishArgs(finish_reason="passed")),
                                          response(FinishArgs(finish_reason="correct"))]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=3, action_policy={"variant": "a6"})
    last = await flow.run(workspace=workspace)
    assert not runtime.calls
    assert not last.file_context.generate_git_patch(ignore_tests=True)
    assert last.completion_status["status"] == "unverified"


@pytest.mark.asyncio
async def test_repeated_edit_failures_require_read_or_replan(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    bad = edit(old="not_present_anywhere")
    agent.completion_model._responses = [response(RegisterAcceptancePlanArgs(plan=plan()), bad, bad), response(bad)]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=3, action_policy={"variant": "a6"})
    last = await flow.run(workspace=workspace)
    assert last.action_steps[0].observation.properties["skip_reason"] == "edit_loop"
    assert not runtime.calls and last.completion_status["status"] == "unverified"
    assert "Read the current file/span" in last.policy_feedback


@pytest.mark.asyncio
async def test_pass_without_failing_public_reproduction_is_insufficient(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    node = await execute(agent, root, RegisterAcceptancePlanArgs(plan=plan()), edit(),
                         RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"))
    assessment = assess(node, workspace)
    assert assessment.state == "partial"
    assert "repro:baseline_comparison" in assessment.outstanding
    assert assessment.next_check["product_source"] == "base"
    node = await execute(agent, node, RunTestsArgs(test_files=["test_calc.py"], test_source="base_public", product_source="base"))
    assert assess(node, workspace).state == "ready"
    changed = await execute(agent, node, edit("return 2", "return 3"))
    assert assess(changed, workspace).state != "ready"
    assert any(r["effective_status"] == "stale" for r in records(changed, workspace))


@pytest.mark.asyncio
async def test_workspace_modified_tests_are_executed_and_full_patch_is_valid_identity(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    node = await execute(agent, root, RegisterAcceptancePlanArgs(plan=plan()), edit(),
        StringReplaceArgs(path="test_calc.py", old_str="assert value() == 2", new_str="assert value() == 2  # repro"),
        RunTestsArgs(test_files=["test_calc.py"]))
    record = records(node, workspace)[-1]
    assert "# repro" in runtime.calls[-1]["patch"]
    assert record["test_source"] == "workspace" and record["status"] == "passed"
    assert record["product_patch_hash"] != record["tested_patch_hash"]
    assert assess(node, workspace).state != "ready"


@pytest.mark.asyncio
async def test_new_repro_baseline_uses_identical_test_bundle(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    node = await execute(agent, root, edit(), CreateFileArgs(path="test_new.py", file_text=FILES["test_calc.py"]),
        RunTestsArgs(test_files=["test_new.py"]), RunTestsArgs(test_files=["test_new.py"], product_source="base"))
    current, baseline = records(node, workspace)[-2:]
    assert current["test_bundle_hash"] == baseline["test_bundle_hash"]
    assert [r["status"] for r in (current, baseline)] == ["passed", "failed"]
    assert "test_new.py" in runtime.calls[-1]["patch"] and "calc.py\n" not in runtime.calls[-1]["patch"]


@pytest.mark.asyncio
async def test_timeout_never_reuses_cached_pass_but_preserves_historical_complete_pass(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    node = await execute(agent, root, edit(), RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"))
    async def timeout(**kwargs):
        raise asyncio.TimeoutError()
    runtime.run_tests = timeout
    node = await execute(agent, node, RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"))
    current = records(node, workspace)
    assert current[-1]["status"] == "error" and current[-1]["counts"] == {}
    assert current[-1]["timed_out"] and current[-1]["complete"] is False
    assert current[-2]["effective_status"] == "passed" and not current[-2]["superseded"]


@pytest.mark.asyncio
async def test_related_failure_survives_later_unrelated_green(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path, {"test_other.py": "def test_other(): assert True\n"})
    node = await execute(agent, root, RegisterAcceptancePlanArgs(plan=plan()), edit(new="return 3"),
        RunTestsArgs(test_files=["test_calc.py"]), RunTestsArgs(test_files=["test_other.py"]))
    assert assess(node, workspace).state == "repair"
    assert assess(node, workspace).failures[0]["tests"] == ["test_calc.py"]


@pytest.mark.asyncio
async def test_plan_revision_can_replace_unexecutable_check_target(tmp_path):
    agent, root, workspace, _ = await fixture(tmp_path)
    bad = plan()
    bad.checks[0].targets = ["missing_test.py"]
    node = await execute(agent, root, RegisterAcceptancePlanArgs(plan=bad))
    assert node.observation.properties.get("fail_reason") is None
    good = plan()
    good.revision = 2
    good.checks[0].targets = ["test_calc.py"]
    node = await execute(agent, node, RegisterAcceptancePlanArgs(plan=good))
    assert node.observation.properties.get("fail_reason") is None, node.observation.message
    assert registered_plan(node).revision == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["return True\n", "break\n", "nonlocal missing\n"])
async def test_create_rejects_compile_errors_before_write(tmp_path, content):
    agent, root, workspace, _ = await fixture(tmp_path)
    node = await execute(agent, root, CreateFileArgs(path="bad.py", file_text=content))
    assert node.observation.properties["fail_reason"] == "syntax_error"
    assert not node.file_context.file_exists("bad.py")


@pytest.mark.asyncio
async def test_plan_revision_cannot_delete_weaken_or_self_certify(tmp_path):
    agent, root, workspace, _ = await fixture(tmp_path)
    node = await execute(agent, root, RegisterAcceptancePlanArgs(plan=plan(True)))
    replacement = plan()
    replacement.revision = 2
    rejected = await execute(agent, node, RegisterAcceptancePlanArgs(plan=replacement))
    assert rejected.observation.properties["fail_reason"] == "invalid_args"
    assert registered_plan(rejected).revision == 1
    with pytest.raises(ValueError):
        AcceptancePlan.model_validate({**plan().model_dump(), "passed": True})


def test_discovery_covers_disk_and_preserves_rank(tmp_path):
    from moatless.repository.file import FileRepository
    (tmp_path / "aaa_test.py").write_text("def test_unrelated(): assert True\n")
    (tmp_path / "test_calc.py").write_text(FILES["test_calc.py"])
    targets = discover_test_targets(FileRepository(repo_path=str(tmp_path)), ["calc.py"])
    assert targets[0] == "test_calc.py" and "aaa_test.py" in targets


@pytest.mark.asyncio
async def test_a7_mandatory_check_is_dispatched_with_check_id_and_finalizes(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    agent.completion_model._responses = [
        response(RegisterAcceptancePlanArgs(plan=plan()), edit(),
                 RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"))]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=20, action_policy={"variant": "a7"})
    last = await asyncio.wait_for(flow.run(workspace=workspace), timeout=20)
    assert flow.is_finished() == "acceptance_satisfied"
    assert last.completion_status["status"] == "verified"
    checkpoint = flow.graph_state["repair_state"]["verified_checkpoint"]
    assert checkpoint["patch_hash"] == content_hash(last.file_context.generate_git_patch(ignore_tests=True))
    policy_checks = [step.action for node in last.get_trajectory() for step in node.action_steps
                     if step.origin == "policy" and step.action.name == "RunTests"]
    assert policy_checks and policy_checks[-1].check_id == "repro"
    assert policy_checks[-1].product_source == "base"
    assert [call["return_code"] for call in runtime.calls] == [0, 1]
    decisions = [decision for node in last.get_trajectory() for decision in (node.policy_decisions or [])]
    assert decisions and all(item["config"]["variant"] == "a7" for item in decisions)


@pytest.mark.asyncio
async def test_a7_no_gain_requires_replan_repair_before_writes(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    read = ReadFile()
    await read.initialize(workspace)
    agent._action_map[ReadFileArgs] = read
    agent.completion_model._responses = [
        response(RegisterAcceptancePlanArgs(plan=plan()),
                 ReadFileArgs(file_path="calc.py"), ReadFileArgs(file_path="calc.py"),
                 ReadFileArgs(file_path="calc.py")),
        # The first proposal after the repeated reads is replanned and opens
        # the evidence gate; this read is therefore skipped.
        response(ReadFileArgs(file_path="test_calc.py"),
                 ReplanRepairArgs(family_id="no_progress",
                                  hypothesis="The repeated read cannot reveal the missing branch",
                                  evidence_sources=["2:0"],
                                  next_steps="Read the public test and repair the return value")),
        response(ReadFileArgs(file_path="test_calc.py"),
                 ReplanRepairArgs(family_id="no_progress",
                                  hypothesis="The repeated read cannot reveal the missing branch",
                                  evidence_sources=["3:0"],
                                  next_steps="Read the public test and repair the return value")),
        response(edit(), RunTestsArgs(test_files=["test_calc.py"], test_source="base_public")),
    ]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=20,
                               action_policy={"variant": "a7", "stagnation_limit": 2})
    last = await asyncio.wait_for(flow.run(workspace=workspace), timeout=30)
    skipped = [step for node in last.get_trajectory() for step in node.action_steps
               if (step.observation.properties or {}).get("skip_reason") == "no_progress" or
               (step.observation.properties or {}).get("skip_reason") == "evidence_based_replan_required"]
    assert skipped
    saved = flow.graph_state["repair_state"]
    assert saved["replans"]["no_progress"]["status"] == "accepted"
    assert saved["replans"]["no_progress"]["evidence_sources"] == ["3:0"]
    assert flow.is_finished() == "acceptance_satisfied"
    assert last.completion_status["status"] == "verified"


@pytest.mark.asyncio
async def test_a7_policy_verification_appends_structured_check(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    read = ReadFile()
    await read.initialize(workspace)
    agent._action_map[ReadFileArgs] = read
    agent.completion_model._responses = [
        response(RegisterAcceptancePlanArgs(plan=plan()),
                 edit("return 1", "return 2"), edit("return 2", "return 3"),
                 edit("return 3", "return 2"), ReadFileArgs(file_path="calc.py")),
    ]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=20,
                                action_policy={"variant": "a7"})
    last = await asyncio.wait_for(flow.run(workspace=workspace), timeout=20)
    assert flow.is_finished() == "acceptance_satisfied"
    assert last.completion_status["status"] == "verified"
    policy_tests = [step.action for node in last.get_trajectory() for step in node.action_steps
                    if step.origin == "policy" and step.action.name == "RunTests"]
    # The policy schedules the registered check before the base comparison.
    assert policy_tests[0].check_id == "repro" and policy_tests[0].product_source == "current"
    skipped = [step for node in last.get_trajectory() for step in node.action_steps
               if (step.observation.properties or {}).get("skip_reason") == "verification_due"]
    assert skipped and skipped[0].action.name == "ReadFile"


@pytest.mark.asyncio
async def test_a7_pending_repair_gate_rejects_writes_and_stops_after_two_rejections(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    failing = RunTestsArgs(test_files=["test_calc.py"], test_source="base_public")
    agent.completion_model._responses = [
        response(RegisterAcceptancePlanArgs(plan=plan()), edit(new="return 3"), failing),
        response(edit("return 3", "return 4"), failing),
        response(edit("return 4", "return 5"), failing),
        response(edit("return 5", "return 6")),
        response(edit("return 6", "return 7")),
    ]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=20,
                                action_policy={"variant": "a7"})
    last = await asyncio.wait_for(flow.run(workspace=workspace), timeout=30)
    assert flow.is_finished() == "policy_stop"
    assert last.completion_status["status"] == "unverified"
    assert last.completion_status["reason"] == "replan_exhausted"
    gates = [gate for gate in flow.graph_state["repair_state"]["replans"].values()
             if gate.get("status") == "required"]
    assert gates and gates[0]["rejections"] >= 2
    blocked = [step for node in last.get_trajectory() for step in node.action_steps
               if (step.observation.properties or {}).get("skip_reason") == "evidence_based_replan_required"]
    assert blocked


@pytest.mark.asyncio
async def test_a7_verification_budget_stops_through_policy_reason(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    agent.completion_model._responses = [
        response(RegisterAcceptancePlanArgs(plan=plan()), edit(),
                 RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"),
                 RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"))]
    flow = LangGraphFlow.create(
        root=root, agent=agent, max_iterations=20,
        action_policy={"variant": "a7", "max_verification_attempts": 1})
    last = await asyncio.wait_for(flow.run(workspace=workspace), timeout=20)
    assert flow.is_finished() == "policy_stop"
    assert last.completion_status["status"] == "unverified"
    assert last.completion_status["reason"] == "verification_budget_exhausted"
    assert len(runtime.calls) == 1
    decisions = [decision for node in last.get_trajectory() for decision in (node.policy_decisions or [])]
    assert decisions[-1]["decision"]["kind"] == "stop"


@pytest.mark.asyncio
async def test_a7_product_edit_requires_acceptance_plan_before_execution(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    agent.completion_model._responses = [response(edit())]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=2,
                                action_policy={"variant": "a7"})
    last = await flow.run(workspace=workspace)
    assert not runtime.calls and not last.file_context.generate_git_patch(ignore_tests=True)
    skipped = [step for node in last.get_trajectory() for step in node.action_steps
               if (step.observation.properties or {}).get("skip_reason") == "acceptance_plan_required"]
    assert skipped and skipped[0].action.name == "StringReplace"
    assert last.completion_status["status"] == "unverified"


@pytest.mark.asyncio
async def test_a7_finish_without_plan_cannot_be_verified(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    finish = VerifiedFinishArgs(finish_reason="done", test_verification="nothing was executed")
    agent.completion_model._responses = [response(finish), response(finish)]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=3, action_policy={"variant": "a7"})
    last = await flow.run(workspace=workspace)
    assert not runtime.calls and not last.file_context.generate_git_patch(ignore_tests=True)
    assert last.completion_status["status"] == "unverified"
    assert last.completion_status["reason"]


def test_a7_receipt_validity_predicate_is_shared(tmp_path):
    from moatless.verification_inputs import receipt_complete
    good = {"recipe_version": 3, "complete": True, "timed_out": False, "patch_stale": False,
            "input_identity": "i", "input_identity_after": "i",
            "environment_before": "e", "environment_after": "e"}
    assert receipt_complete(good)
    assert not receipt_complete({**good, "patch_stale": True})
    assert not receipt_complete({**good, "input_identity_after": "changed"})
    assert not receipt_complete({**good, "recipe_version": 2})


@pytest.mark.asyncio
async def test_a7_raw_runner_output_enters_policy_feedback(tmp_path):
    from moatless.policy.action import ActionPolicy, ActionPolicyConfig, BudgetSnapshot, extract_state
    test_body = ("from calc import value\n"
                 "def test_value():\n"
                 "    assert value() == 3, 'minversion mismatch: expected _version.py'\n")
    agent, root, workspace, runtime = await fixture(tmp_path, {"test_calc.py": test_body})
    node = await execute(agent, root, edit(new="return 2"), RunTestsArgs(test_files=["test_calc.py"]))
    features, _ = extract_state(node, 0, workspace, variant="a7")
    assert "minversion" in features.last_test_output
    assert "_version.py" in features.failure_excerpt
    decision = ActionPolicy(ActionPolicyConfig(variant="a7")).decide(
        features, ActionStep(action=ReadFileArgs(file_path="calc.py")),
        BudgetSnapshot(remaining_rounds=20))
    feedback = LangGraphFlow._a7_feedback(features, decision)
    assert "minversion" in feedback and "return_code=1" in feedback


@pytest.mark.asyncio
async def test_a6_checkpoint_cannot_resume_as_a7(tmp_path):
    from moatless.flow.durable import RecoveryConflict
    directory = tmp_path / "repo"
    directory.mkdir()
    agent, root, workspace, runtime = await fixture(directory)
    agent.completion_model._responses = [
        response(RegisterAcceptancePlanArgs(plan=plan()), edit(),
                 RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"))]
    path = str(tmp_path / "checkpoint.sqlite")
    kwargs = dict(root=root, agent=agent, max_iterations=20, checkpoint_path=path,
                  trajectory_id="protocol-switch")
    a6 = LangGraphFlow.create(**kwargs, action_policy={"variant": "a6"})
    await a6.run(workspace=workspace)
    a7 = LangGraphFlow.create(**kwargs, action_policy={"variant": "a7"})
    with pytest.raises(RecoveryConflict, match="configuration changed"):
        await a7.run(workspace=workspace)


@pytest.mark.asyncio
async def test_a7_durable_resume_does_not_replay_completed_effects(tmp_path):
    directory = tmp_path / "repo"
    directory.mkdir()
    agent, root, workspace, runtime = await fixture(directory)
    agent.completion_model._responses = [
        response(RegisterAcceptancePlanArgs(plan=plan()), edit(),
                 RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"))]
    path = str(tmp_path / "checkpoint.sqlite")
    kwargs = dict(root=root, agent=agent, max_iterations=20, checkpoint_path=path,
                  trajectory_id="a7-resume", action_policy={"variant": "a7"})
    flow = LangGraphFlow.create(**kwargs)
    class Crash(BaseException):
        pass
    crashed = False
    original = flow._journal_node
    def wrapped(name, fn):
        call = original(name, fn)
        async def run(state, runtime):
            nonlocal crashed
            result = await call(state, runtime)
            if name == "assess" and not crashed:
                node = flow.get_node_by_id(result["current_node_id"])
                if len(records(node, workspace)) == 2:
                    crashed = True
                    raise Crash()
            return result
        return run
    flow._journal_node = wrapped
    with pytest.raises(Crash):
        await flow.run(workspace=workspace)
    calls = len(runtime.calls)
    assert calls == 2
    restored = LangGraphFlow.create(**kwargs)
    last = await restored.run(workspace=workspace)
    assert restored.is_finished() == "acceptance_satisfied"
    assert last.completion_status["status"] == "verified"
    assert len(runtime.calls) == calls and agent.completion_model._calls == 1
    assert restored.graph_state["repair_state"]["verified_checkpoint"]["patch_hash"] == \
        content_hash(last.file_context.generate_git_patch(ignore_tests=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["record_observation", "assess", "finalize", "commit_iteration"])
async def test_actual_graph_durable_resume_never_replays_completed_effects(tmp_path, phase):
    directory = tmp_path / "repo"
    directory.mkdir()
    agent, root, workspace, runtime = await fixture(directory)
    agent.completion_model._responses = [response(RegisterAcceptancePlanArgs(plan=plan()), edit(),
        RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"))]
    path = str(tmp_path / "checkpoint.sqlite")
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=5, checkpoint_path=path, trajectory_id="a6-test",
                               action_policy={"variant": "a6"})
    class Crash(BaseException):
        pass
    crashed = False
    original = flow._journal_node
    def wrapped(name, fn):
        call = original(name, fn)
        async def run(state, runtime):
            nonlocal crashed
            result = await call(state, runtime)
            if name == phase and not crashed:
                # Crash only after the final receipt is durable. This covers
                # receipt -> assessment -> checkpoint -> finalized transitions.
                node = flow.get_node_by_id(result["current_node_id"])
                if len(records(node, workspace)) == 2:
                    crashed = True
                    raise Crash()
            return result
        return run
    flow._journal_node = wrapped
    with pytest.raises(Crash):
        await flow.run(workspace=workspace)
    calls = len(runtime.calls)
    assert assess(flow.root.get_last_node(), workspace).state == "ready", assess(flow.root.get_last_node(), workspace)
    flow._journal_node = original
    # A new flow object reads the journal; this is not a method-level replay.
    restored = LangGraphFlow.create(root=root, agent=agent, max_iterations=5, checkpoint_path=path, trajectory_id="a6-test",
                                   action_policy={"variant": "a6"})
    last = await restored.run(workspace=workspace)
    assert restored.is_finished() == "acceptance_satisfied" and last.terminal
    assert len(runtime.calls) == calls == 2 and agent.completion_model._calls == 1


def test_a6_no_gain_gate_opens_replan_after_repeated_view():
    repository = InMemRepository({"test_calc.py": "def test_value():\n    assert True\n"})
    context = FileContext(repo=repository)
    step = ActionStep(
        action=ViewCodeArgs(files=[CodeSpan(file_path="test_calc.py")]),
        observation=Observation.create(
            "same view",
            properties={"code_view": {"consistent": True,
                                      "files": {"test_calc.py": "same"},
                                      "ranges_by_file": {"test_calc.py": [[1, 2]]},
                                      "start_line": 1, "end_line": 2}},
        ),
    )
    node = Node(node_id=1, file_context=context)
    node.action_steps = [step]
    saved = {"version": 1, "attempts": {}, "failure_families": {}}
    messages = []
    for _ in range(4):
        message, _ = update_no_gain(saved, node, 0, step.observation.properties,
                                    stagnation_limit=4, repeat_limit=2)
        messages.append(message)
    assert saved["replans"]["no_progress"]["status"] == "required"
    assert saved["replans"]["no_progress"]["read_attempts"] == 0
    assert "ReplanRepair" in messages[-1]
    assert saved["no_gain"]["repeat_count"] >= 2


def test_a6_failure_detail_preserves_command_and_exit_code():
    detail = failure_detail({"commands": ["pytest -rA testing/test_skipping.py"],
                             "return_code": 4, "counts": {"unknown": 1},
                             "targets": ["testing/test_skipping.py"]})
    assert "pytest -rA" in detail
    assert "return_code=4" in detail
    assert "unknown" in detail
