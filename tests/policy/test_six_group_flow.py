"""Exercise the actual graph/journal for bounded recovery, replan and budgets."""

import asyncio
import sqlite3
from copy import deepcopy

import pytest
from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.actions.replan_repair import ReplanRepairArgs
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.actions.register_acceptance_plan import RegisterAcceptancePlanArgs
from moatless.exceptions import WorkspaceConsistencyError
from moatless.flow.durable import OutcomeJournal, RecoveryConflict
from moatless.flow.langgraph_flow import LangGraphFlow
from moatless.message_history.policy import content_hash
from moatless.policy.obligations import failure_family
from scripts.migration_smoke import response
from tests.policy.test_acceptance_repair import fixture, plan, edit


@pytest.mark.asyncio
async def test_model_proposed_tests_share_the_global_verification_budget(tmp_path):
    agent, root, workspace, runtime = await fixture(tmp_path)
    agent.completion_model._responses = [
        response(
            RegisterAcceptancePlanArgs(plan=plan()), edit(new="return 3"), RunTestsArgs(test_files=["test_calc.py"])
        ),
        response(edit("return 3", "return 4"), RunTestsArgs(test_files=["test_calc.py"])),
    ]
    flow = LangGraphFlow.create(
        root=root, agent=agent, max_iterations=5, action_policy={"variant": "a6", "max_verification_attempts": 1}
    )
    last = await flow.run(workspace=workspace)
    assert flow.is_finished() == "verification_budget_exhausted"
    assert len(runtime.calls) == 1 and last.completion_status["status"] == "unverified"


@pytest.mark.asyncio
async def test_real_replan_gate_survives_durable_resume_and_one_final_failed_trial_exits(tmp_path):
    directory = tmp_path / "repo"
    directory.mkdir()
    agent, root, workspace, runtime = await fixture(directory)
    read = ReadFile()
    await read.initialize(workspace)
    agent._action_map[ReadFileArgs] = read
    family = failure_family({"product_files": ["calc.py"]}, {"failure_output": "AssertionError"})
    agent.completion_model._responses = [
        response(
            RegisterAcceptancePlanArgs(plan=plan()), edit(new="return 3"), RunTestsArgs(test_files=["test_calc.py"])
        ),
        response(edit("return 3", "return 4"), RunTestsArgs(test_files=["test_calc.py"])),
        response(edit("return 4", "return 3"), RunTestsArgs(test_files=["test_calc.py"])),
        response(
            ReadFileArgs(file_path="test_calc.py"),
            ReplanRepairArgs(
                family_id=family,
                hypothesis="The assertion exposes the wrong return branch",
                evidence_sources=["4:0"],
                next_steps="Repair the return branch, then rerun the failing module",
            ),
        ),
        response(edit("return 3", "return 5"), RunTestsArgs(test_files=["test_calc.py"])),
    ]
    path = str(tmp_path / "checkpoint.sqlite")
    flow = LangGraphFlow.create(
        root=root,
        agent=agent,
        max_iterations=10,
        checkpoint_path=path,
        trajectory_id="replan-proof",
        action_policy={"variant": "a6"},
    )

    class Crash(BaseException):
        pass

    original = flow._journal_node

    def wrapped(name, fn):
        call = original(name, fn)

        async def run(state, runtime):
            result = await call(state, runtime)
            if (
                name == "assess"
                and result.get("repair_state", {}).get("replans", {}).get(family, {}).get("status") == "accepted"
            ):
                raise Crash()
            return result

        return run

    flow._journal_node = wrapped
    with pytest.raises(Crash):
        await flow.run(workspace=workspace)
    assert len(runtime.calls) == 3
    restored = LangGraphFlow.create(
        root=root,
        agent=agent,
        max_iterations=10,
        checkpoint_path=path,
        trajectory_id="replan-proof",
        action_policy={"variant": "a6"},
    )
    last = await restored.run(workspace=workspace)
    assert restored.is_finished() == "repair_exhausted", restored.graph_state["repair_state"]
    assert len(runtime.calls) == 4 and agent.completion_model._calls == 5
    saved = restored.graph_state["repair_state"]
    assert (
        saved["replans"][family]["status"] == "accepted" and saved["failure_families"][family]["repair_attempts"] == 3
    )
    assert last.completion_status["status"] == "unverified"


@pytest.mark.asyncio
async def test_cleanup_exceeding_grace_leaves_pending_operation_and_resume_cannot_replay(tmp_path):
    directory = tmp_path / "repo"
    directory.mkdir()
    agent, root, workspace, runtime = await fixture(directory)
    released = asyncio.Event()
    calls = []

    async def stuck(**kwargs):
        calls.append(1)
        try:
            await asyncio.Event().wait()
        finally:
            await released.wait()

    runtime.run_tests = stuck
    for action in agent.actions:
        if isinstance(action, RunTests):
            action.timeout, action.cleanup_grace = 0.02, 0.03
    agent.tool_timeout = 2
    agent.completion_model._responses = [
        response(RegisterAcceptancePlanArgs(plan=plan()), edit(), RunTestsArgs(test_files=["test_calc.py"]))
    ]
    path = str(tmp_path / "checkpoint.sqlite")
    kwargs = dict(
        root=root,
        agent=agent,
        max_iterations=5,
        checkpoint_path=path,
        trajectory_id="uncertain-cleanup",
        action_policy={"variant": "a6"},
        tool_timeout=2,
    )
    flow = LangGraphFlow.create(**kwargs)
    try:
        with pytest.raises(WorkspaceConsistencyError):
            await flow.run(workspace=workspace)
        assert flow.graph_state["error_kind"] == "workspace_consistency"
        with sqlite3.connect(path + ".outcomes.sqlite") as connection:
            assert connection.execute("SELECT COUNT(*) FROM flow_operations WHERE status='started'").fetchone()[0] == 1
        restored = LangGraphFlow.create(**kwargs)
        with pytest.raises(RecoveryConflict):
            await restored.run(workspace=workspace)
        assert calls == [1]
    finally:
        released.set()
        await asyncio.sleep(0.03)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setting,value", [("python_executable", "/usr/bin/python3"), ("test_command", "pytest --changed-runner")]
)
async def test_completed_checkpoint_rechecks_compiler_and_runner_inputs_before_reusing_verified(
    tmp_path, setting, value
):
    directory = tmp_path / "repo"
    directory.mkdir()
    agent, root, workspace, runtime = await fixture(directory)
    agent.completion_model._responses = [
        response(
            RegisterAcceptancePlanArgs(plan=plan()),
            edit(),
            RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"),
        )
    ]
    kwargs = dict(
        root=root,
        agent=agent,
        max_iterations=5,
        checkpoint_path=str(tmp_path / "checkpoint.sqlite"),
        trajectory_id="completed-binding",
        action_policy={"variant": "a6"},
    )
    flow = LangGraphFlow.create(**kwargs)
    last = await flow.run(workspace=workspace)
    assert last.completion_status["status"] == "verified"
    calls = len(runtime.calls)
    setattr(runtime, setting, value)
    restored = LangGraphFlow.create(**kwargs)
    with pytest.raises(RecoveryConflict, match="verification inputs changed"):
        await restored.run(workspace=workspace)
    assert len(runtime.calls) == calls
