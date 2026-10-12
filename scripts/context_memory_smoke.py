"""Offline debugging replay through both production entry points.

Uses scripted responses and real pytest; never invokes a model provider or
official SWE-bench tests. Only an explicit --output saves replay artifacts.
"""

import argparse
import asyncio
import json
import re
import shlex
import tempfile
from pathlib import Path
import sys

from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.local_coding import LocalTestArgs
from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.actions.task_state import UpdateTaskStateArgs
from moatless.agent.agent import ActionAgent
from moatless.cli.service import SessionService
from moatless.cli.worktree import git
from moatless.execution_memory import failure_records, test_records
from moatless.flow.langgraph_flow import LangGraphFlow
from moatless.message_history.managed import ManagedMessageHistoryGenerator
from moatless.message_history.policy import ContextPolicy
from moatless.node import Node
from moatless.repository.file import FileRepository
from moatless.task_state import EvidenceRef, Hypothesis
from moatless.workspace import Workspace
from scripts.migration_smoke import PytestSmokeRuntime, SOURCE, TEST, ScriptedCompletion, response


class DebuggingCompletion(ScriptedCompletion):
    def __init__(self, cli=False):
        test = LocalTestArgs(command=f"{shlex.quote(sys.executable)} -m pytest -q") if cli else RunTestsArgs(test_files=["test_calc.py"])
        super().__init__([
            response(UpdateTaskStateArgs(actual_behavior="Addition returns the wrong number", expected_behavior="Return the sum",
                hypotheses=[Hypothesis(id="H1", statement="Multiplication will fix it")], next_step="Inspect calc.py"), cost=0),
            response(ReadFileArgs(file_path="calc.py"), cost=0),
            response(StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a * b"), cost=0),
            response(test, cost=0),
            response(UpdateTaskStateArgs(), cost=0),
            response(StringReplaceArgs(path="calc.py", old_str="return a * b", new_str="return a + b"), cost=0),
            response(test, cost=0),
            response(FinishArgs(finish_reason="Corrected addition and passed the task-local regression"), cost=0),
        ], max_tokens=512)

    async def create_completion(self, messages, system_prompt=None):
        if self._calls == 4:
            latest = next(message for message in reversed(messages) if message.get("role") == "tool")
            match = re.fullmatch(r"node_(\d+)_step_(\d+)", latest["tool_call_id"])
            self._responses[4] = response(UpdateTaskStateArgs(phase="modification", hypotheses=[Hypothesis(
                id="H1", statement="Multiplication will fix it", status="rejected",
                evidence=[EvidenceRef(node_id=int(match[1]), step_index=int(match[2]))])],
                next_step="Replace multiplication with addition and rerun the regression"), cost=0)
        return await super().create_completion(messages, system_prompt)


def evidence(flow, node, model, workspace):
    return {"kind": "offline scripted debugging replay", "provider_calls": 0,
            "official_harness": "not_run", "model_fixture_calls": model._calls,
            "completion_status": node.completion_status,
            "tests": test_records(node, workspace), "failures": failure_records(node),
            "messages": model._messages, "nodes": flow.root.dump_as_list(),
            "patch": node.file_context.generate_git_patch()}


async def replay_batch(target: Path):
    target.mkdir()
    (target / "calc.py").write_text(SOURCE)
    (target / "test_calc.py").write_text(TEST)
    runtime = PytestSmokeRuntime(target)
    model = DebuggingCompletion()
    agent = ActionAgent(completion_model=model, actions=[ReadFile(), StringReplace(), RunTests(), Finish()],
                        system_prompt="Repair addition", memory=ManagedMessageHistoryGenerator(
                            policy=ContextPolicy(context_window=32768)))
    workspace = Workspace(repository=FileRepository(repo_path=str(target)), runtime=runtime)
    initial = await runtime.run_tests(test_files=["test_calc.py"])
    assert initial[0].status.value == "FAILED"
    flow = LangGraphFlow.create(message="Fix add", agent=agent, project_id="context-memory", trajectory_id="batch", max_iterations=12)
    node = await flow.run(workspace=workspace)
    assert node.completion_status["status"] == "verified"
    assert (target / "calc.py").read_text() == SOURCE
    assert [r["returncode"] for r in runtime.runs] == [1, 1, 0]
    saved = Node.reconstruct(flow.root.model_dump(), repo=workspace.repository).get_last_node()
    assert saved.completion_status == node.completion_status
    assert saved.task_state.hypotheses[0].status == "rejected"
    return dict(evidence(flow, node, model, workspace), pytest_runs=runtime.runs)

async def replay_cli(target: Path, directory: Path):
    target.mkdir()
    (target / "calc.py").write_text(SOURCE)
    (target / "test_calc.py").write_text(TEST)
    (target / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(target, "init", "-q")
    git(target, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "add", ".")
    git(target, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
    service = SessionService(directory)
    session = await service.create(str(target), "Fix add", {"max_iterations": 12,
        "env": {"PYTHONDONTWRITEBYTECODE": "1"}, "context_policy": {"context_window": 32768}})
    model = DebuggingCompletion(cli=True)
    configured_model = session.flow.agent.completion_model
    model.initialize([action.args_schema for action in session.flow.agent.actions], session.flow.agent.system_prompt)
    session.flow.agent.completion_model = model
    try:
        node = await service.start(session)
        assert node.completion_status["status"] == "verified"
        checks = service.tests(session)
        assert [r["recorded_status"] for r in checks] == ["failed", "passed"]
        assert checks[0]["test_status"] == "stale" and checks[1]["test_status"] == "passed"
        assert (target / "calc.py").read_text() == SOURCE
        result = dict(evidence(session.flow, node, model, session.workspace), cli_tests=checks,
                      session_info_status=session.info["completion_status"], patch=service.diff(session))
        # Restore the configured production model before testing persistence;
        # an ephemeral scripted fixture is intentionally not loadable config.
        session.flow.agent.completion_model = configured_model
        service.save(session)
        loaded = SessionService(directory).load(session.id)
        assert loaded.flow.root.get_last_node().task_state == node.task_state
        assert loaded.info["completion_status"] == node.completion_status
        result["reload_verified"] = True
        return result
    finally:
        await service.close()


async def replay():
    with tempfile.TemporaryDirectory(prefix="moatless-context-replay-") as temporary:
        target = Path(temporary)
        return {"batch": await replay_batch(target / "batch"),
                "cli": await replay_cli(target / "repo", target / "sessions")}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = await replay()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {"completion_status": value["completion_status"],
                            "model_fixture_calls": value["model_fixture_calls"]} for name, value in result.items()}))


if __name__ == "__main__":
    asyncio.run(main())
