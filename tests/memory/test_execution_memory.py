"""Test evidence is version-bound and Finish is not official resolution."""

import sys

import pytest

from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.local_coding import LocalBash, LocalRunTests, LocalTestArgs
from moatless.actions.bash import BashArgs
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.actions.task_state import UpdateTaskStateArgs
from moatless.execution_memory import classify_completion, failure_records, test_records as recorded_tests
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.task_state import EvidenceRef, PlanStep
from moatless.testing.schema import TestResult as Result, TestStatus as Status
from tests.memory.test_repository_context import execute, setup


@pytest.mark.asyncio
async def test_cli_pass_edit_stale_and_rerun_refreshes_without_losing_original(tmp_path):
    (tmp_path / "code.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path, actions=[LocalRunTests(), LocalBash(), Finish()])
    tested = await execute(agent, root, LocalTestArgs(command="test -f code.py"))
    original = tested.observation.properties["test_run"].copy()
    assert original["status"] == "passed" and original["return_code"] == 0
    assert classify_completion(tested, workspace)["status"] == "verified"
    edited = await execute(agent, tested, BashArgs(command="printf 'value = 2\\n' > code.py"))
    assert recorded_tests(edited, workspace)[0]["effective_status"] == "stale"
    finished = await execute(agent, edited, FinishArgs(finish_reason="Claim done"))
    assert finished.completion_status["status"] == "unverified"
    assert finished.completion_status["official_harness"] == "not_run"
    assert finished.completion_status["resolved"] is None
    assert tested.observation.properties["test_run"] == original
    rerun = await execute(agent, edited, LocalTestArgs(command="test -f code.py"), FinishArgs(finish_reason="Checked"))
    assert recorded_tests(rerun, workspace)[0]["superseded"]
    assert rerun.completion_status["status"] == "verified"


@pytest.mark.asyncio
async def test_tests_that_change_code_cannot_verify_final_version(tmp_path):
    (tmp_path / "code.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path, actions=[LocalRunTests(), Finish()])
    node = await execute(agent, root, LocalTestArgs(command="printf 'value = 2\\n' > code.py"), FinishArgs(finish_reason="Done"))
    record = recorded_tests(node, workspace)[0]
    assert record["status"] == "passed" and record["effective_status"] == "stale"
    assert record["workspace_before"] != record["workspace_after"]
    assert node.completion_status["status"] == "unverified"


@pytest.mark.asyncio
async def test_unknown_and_missing_tests_do_not_pass_even_if_cached_result_passed(tmp_path):
    (tmp_path / "test_code.py").write_text("def test_code(): pass\n")
    agent, root, workspace = await setup(tmp_path, actions=[RunTests(), Finish()], runtime=ResultsRuntime([]))
    root.file_context.add_test_file("test_code.py", [Result(status=Status.PASSED, name="previous", file_path="test_code.py")])
    node = await execute(agent, root, RunTestsArgs(test_files=["test_code.py"]), FinishArgs(finish_reason="Done"))
    assert node.action_steps[0].observation.properties["test_run"]["status"] == "unknown"
    assert node.completion_status["status"] == "unverified"
    missing = await execute(agent, root, RunTestsArgs(test_files=["does_not_exist.py"]), FinishArgs(finish_reason="Done"))
    assert missing.action_steps[0].observation.properties["test_run"]["status"] == "not_run"
    assert missing.completion_status["status"] == "unverified"


class ResultsRuntime(RuntimeEnvironment):
    def __init__(self, results):
        self.results = results
        self.return_code = 1 if any(result.status in {Status.FAILED, Status.ERROR} for result in results) else 0

    async def run_tests(self, patch=None, test_files=None):
        self.last_test_execution = [{"command": "fixture-test", "target": path, "return_code": self.return_code}
                                    for path in test_files or []]
        return self.results


@pytest.mark.asyncio
@pytest.mark.parametrize("result_status,expected", [
    (Status.PASSED, "verified"), (Status.FAILED, "unverified"),
    (Status.ERROR, "unverified"), (Status.UNKNOWN, "unverified"), (Status.SKIPPED, "unverified"),
])
async def test_batch_result_statuses_are_not_inferred_from_legacy_summary(tmp_path, result_status, expected):
    (tmp_path / "test_code.py").write_text("def test_code(): pass\n")
    agent, root, workspace = await setup(tmp_path, actions=[RunTests(), Finish()],
        runtime=ResultsRuntime([Result(status=result_status, name="test_code", file_path="test_code.py")]))
    (tmp_path / "code.py").write_text("value = 1\n")
    root.file_context.add_file("code.py").apply_changes("value = 2\n")
    node = await execute(agent, root, RunTestsArgs(test_files=["test_code.py"]), FinishArgs(finish_reason="Claim success"))
    assert node.completion_status["status"] == expected
    assert node.action_steps[0].observation.properties["test_run"]["return_code"] == agent.workspace.runtime.return_code
    if result_status in {Status.UNKNOWN, Status.SKIPPED}:
        assert node.action_steps[0].observation.properties["test_run"]["status"] == "unknown"


@pytest.mark.asyncio
async def test_failed_different_check_is_not_erased_by_success_and_failure_attempts_retained(tmp_path):
    agent, root, workspace = await setup(tmp_path, actions=[LocalRunTests(), Finish()])
    failed = await execute(agent, root, LocalTestArgs(command="false"), LocalTestArgs(command="false"))
    assert len(failure_records(failed)) == 1 and failure_records(failed)[0]["attempts"] == 2
    node = await execute(agent, failed, LocalTestArgs(command="true"), FinishArgs(finish_reason="Done"))
    assert node.completion_status["status"] == "unverified"
    assert [r["effective_status"] for r in recorded_tests(node, workspace) if not r["superseded"]] == ["failed", "passed"]


@pytest.mark.asyncio
async def test_blocked_claim_requires_cited_failed_tool_and_retains_claim_provenance(tmp_path):
    agent, root, workspace = await setup(tmp_path, actions=[LocalRunTests(), Finish()])
    no_evidence = await execute(agent, root, UpdateTaskStateArgs(phase="blocked", blocked_by=["Missing dependency"]),
                                FinishArgs(finish_reason="Blocked"))
    assert no_evidence.completion_status["status"] == "unverified"
    assert no_evidence.completion_status["blocked_claim"] == ["Missing dependency"]
    failed = await execute(agent, root, LocalTestArgs(command="exit 2"))
    blocked = await execute(agent, failed, UpdateTaskStateArgs(
        phase="blocked", blocked_by=["Test command cannot complete; cause unknown"],
        plan=[PlanStep(id="P1", description="Diagnose command", evidence=[EvidenceRef(node_id=failed.node_id, step_index=0)])]),
        FinishArgs(finish_reason="Blocked"))
    assert blocked.completion_status["status"] == "blocked"
    assert blocked.completion_status["blocked_evidence"] == [f"{failed.node_id}:0"]
    assert "cause unknown" in blocked.completion_status["blocked_claim"][0]
    rerun = await execute(agent, blocked.parent, LocalTestArgs(command="exit 2"))
    # A code change invalidates the cited failure; it cannot prove a current block.
    (tmp_path / "changed.py").write_text("new code\n")
    rerun.task_state = blocked.task_state.model_copy(deep=True)
    assert classify_completion(rerun, workspace)["status"] == "unverified"


@pytest.mark.asyncio
async def test_environment_change_invalidates_results_and_node_roundtrip_keeps_raw_records(tmp_path):
    from moatless.node import Node

    agent, root, workspace = await setup(tmp_path, actions=[LocalRunTests()])
    node = await execute(agent, root, LocalTestArgs(command="true"))
    workspace._environment.env = {"TASK_TEST_OPTION": "changed"}
    assert recorded_tests(node, workspace)[0]["effective_status"] == "stale"
    restored = Node.reconstruct(root.model_dump(), repo=workspace.repository).get_node_by_id(node.node_id)
    assert restored.action_steps[0].observation.properties["test_run"] == node.observation.properties["test_run"]
    assert classify_completion(restored, workspace)["status"] == "unverified"


@pytest.mark.asyncio
async def test_cli_failure_is_pipeline_safe_and_real_pytest_is_recorded(tmp_path):
    import shlex

    (tmp_path / "test_code.py").write_text("def test_code(): assert False\n")
    agent, root, workspace = await setup(tmp_path, actions=[LocalRunTests()])
    command = f"{shlex.quote(sys.executable)} -m pytest -q | tail -3"
    failed = await execute(agent, root, LocalTestArgs(command=command))
    record = failed.observation.properties["test_run"]
    assert record["return_code"] == 1 and record["status"] == "failed"
    assert record["command"] == command
