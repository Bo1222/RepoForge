"""Offline M0/M1 replay with real tools and pytest in an isolated patched copy.

Run with: python -m scripts.migration_smoke --flow loop --output PATH
Fixtures are only replaced by an explicit --output invocation, never by tests.
"""

import argparse
import asyncio
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.actions.respond import Respond, RespondArgs
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.actions.view_diff import ViewDiff, ViewDiffArgs
from moatless.agent.agent import ActionAgent
from moatless.completion.base import BaseCompletionModel, CompletionResponse
from moatless.completion.stats import CompletionAttempt, CompletionInvocation, Usage
from moatless.flow.loop import AgenticLoop
from moatless.repository.file import FileRepository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.testing.schema import TestResult, TestStatus
from moatless.workspace import Workspace

PROMPT = "Repair the arithmetic bug using the available tools."
SOURCE = "def add(a, b):\n    return a - b\n"
TEST = "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


class ScriptedCompletion(BaseCompletionModel):
    """Fixed responses with realistic invocation accounting, without API access."""

    _responses: list[Any] = PrivateAttr(default_factory=list)
    _calls: int = PrivateAttr(default=0)
    _messages: list[Any] = PrivateAttr(default_factory=list)

    def __init__(self, responses, **kwargs):
        super().__init__(model="migration-mock", **kwargs)
        self._responses = responses

    async def create_completion(self, messages, system_prompt=None):
        self._calls += 1
        self._messages.append(copy.deepcopy(messages))
        if self._calls > len(self._responses):
            raise RuntimeError("Script exhausted (bounded replay detected a stalled loop)")
        response = self._responses[self._calls - 1]
        if isinstance(response, Exception):
            raise response
        return response.model_copy(deep=True)

    def _validate_completion(self, completion_response):
        raise NotImplementedError("ScriptedCompletion returns validated fixtures directly")


def response(*actions, cost=0.01):
    return CompletionResponse(
        structured_outputs=list(actions),
        thought="scripted reasoning",
        text_response="scripted response",
        thinking_blocks=[{"type": "thinking", "thinking": "scripted thinking"}],
        completion_invocation=CompletionInvocation(
            model="migration-mock",
            attempts=[CompletionAttempt(usage=Usage(prompt_tokens=10, completion_tokens=5, completion_cost=cost))],
        ),
    )


class PytestSmokeRuntime(RuntimeEnvironment):
    """Test-only adapter: apply each patch to a fresh baseline and execute pytest."""

    def __init__(self, baseline: Path):
        self.baseline = baseline
        self.runs: list[dict] = []

    async def run_tests(self, patch=None, test_files=None):
        with tempfile.TemporaryDirectory(prefix="moatless-pytest-") as tmp:
            target = Path(tmp)
            for path in self.baseline.glob("*.py"):
                (target / path.name).write_text(path.read_text())
            if patch:
                apply = subprocess.run(
                    ["git", "apply", "-"],
                    input=patch,
                    text=True,
                    cwd=target,
                    capture_output=True,
                    timeout=10,
                )
                if apply.returncode:
                    raise RuntimeError(f"Smoke patch failed: {apply.stderr}")
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *(test_files or ["test_calc.py"])],
                cwd=target,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.runs.append({"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
            self.last_test_execution = [{"target": path, "command": "real pytest", "return_code": proc.returncode}
                                        for path in test_files or ["test_calc.py"]]
            return [
                TestResult(
                    name="test_add",
                    file_path="test_calc.py",
                    status=TestStatus.PASSED if proc.returncode == 0 else TestStatus.FAILED,
                    failure_output=proc.stdout + proc.stderr if proc.returncode else None,
                )
            ]


def normalize(value):
    """Remove volatile timing only; preserve arguments, results, accounting and IDs."""
    if isinstance(value, dict):
        return {
            k: normalize(v)
            for k, v in value.items()
            if k not in {"timestamp", "start_time", "end_time", "test_duration", "execution_time", "started_at", "duration"}
        }
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def historical_test_projection(value):
    """D07: project out only additive workspace-test provenance for M0 checks.

    Raw replays retain all execution receipts, including newly added return codes.
    Arguments, output, usage, patches and artifact changes remain compared;
    v2 execution facts are independently tested against actual pytest processes.
    """
    result = copy.deepcopy(value)
    additive = {"test_source", "test_origins", "product_patch_hash", "tested_patch_hash",
                "result_statuses", "result_count", "product_source", "runner", "environment", "test_bundle_hash",
                "input_identity", "check_id", "introduced_failures", "public_baseline", "test_files",
                "result_counts", "test_results", "test_execution", "return_code", "timed_out", "complete",
                "patch_stale", "continuation_safe", "targets", "execution_status", "recipe_version", "base_identity",
                "verification_inputs", "product_inputs_hash", "runner_fingerprint", "input_identity_after", "product_files",
                "started_at", "duration", "commands", "partial", "uncompleted_targets", "related_targets"}
    for node in result.get("trajectory", {}).get("nodes", []):
        observations = [node.get("output", {})]
        observations.extend(step.get("observation", {}) for step in node.get("action_steps", [])
                            )
        for observation in observations:
            props = observation.get("properties") or {}
            props.pop("syntax_checked", None)
            props.pop("syntax_checks", None)
            if props.get("test_source") == "workspace":
                for key in additive:
                    props.pop(key, None)
                if not props:
                    observation.pop("properties", None)
    return result


async def replay(flow_class=AgenticLoop, coding=True, managed=False):
    events = []

    async def capture(event):
        events.append(event.model_dump(mode="json"))

    with tempfile.TemporaryDirectory(prefix="moatless-coding-") as tmp:
        baseline = Path(tmp)
        (baseline / "calc.py").write_text(SOURCE)
        (baseline / "test_calc.py").write_text(TEST)
        runtime = PytestSmokeRuntime(baseline)
        if coding:
            responses = [
                response(ReadFileArgs(file_path="calc.py")),
                response(StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b")),
                response(ViewDiffArgs()),
                response(RunTestsArgs(test_files=["test_calc.py"])),
                response(FinishArgs(finish_reason="test passed")),
            ]
            actions = [ReadFile(), StringReplace(), ViewDiff(), RunTests(), Finish()]
        else:
            responses = [response(RespondArgs(message="Offline Respond-only smoke."))]
            actions = [Respond()]
        model = ScriptedCompletion(responses)
        memory_options = {}
        if managed:
            from moatless.message_history.managed import ManagedMessageHistoryGenerator
            from moatless.message_history.policy import ContextPolicy
            memory_options["memory"] = ManagedMessageHistoryGenerator(policy=ContextPolicy(
                context_window=1048576, enable_task_state=False, load_repo_instructions=False))
        agent = ActionAgent(completion_model=model, system_prompt=PROMPT, actions=actions, **memory_options)
        flow = flow_class.create(
            message="Fix add" if coding else "Respond",
            agent=agent,
            project_id="migration",
            trajectory_id="coding" if coding else "respond",
            on_event=capture,
            max_iterations=10,
        )
        workspace = Workspace(repository=FileRepository(repo_path=tmp), runtime=runtime, artifact_handlers=[])
        if coding:
            before = await runtime.run_tests()
            assert before[0].status == TestStatus.FAILED
        node = await flow.run(workspace=workspace)
        trajectory = flow.get_trajectory_data()
        # Graph control metadata is compared separately from the legacy business trajectory.
        business = {k: v for k, v in trajectory.items() if k not in {"graph_state", "finish_reason"}}
        result = normalize(
            {
                "trajectory": business,
                "events": events,
                "usage": flow.total_usage().model_dump(),
                "patch": node.file_context.generate_git_patch(),
                "model_calls": model._calls,
                "messages": model._messages,
                "finish_reason": flow.is_finished(),
            }
        )
        if coding:
            assert runtime.runs[-1]["returncode"] == 0
            assert (baseline / "calc.py").read_text() == SOURCE  # shadow edits did not alter the baseline
        return result, runtime.runs


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", choices=["loop", "graph"], default="loop")
    parser.add_argument("--respond", action="store_true")
    parser.add_argument("--managed", action="store_true", help="Same-source Managed facts migration comparison")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    flow_class = AgenticLoop
    if args.flow == "graph":
        from moatless.flow.langgraph_flow import LangGraphFlow

        flow_class = LangGraphFlow
    result, runs = await replay(flow_class, coding=not args.respond, managed=args.managed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        args.output.with_suffix(".pytest.json").write_text(json.dumps(runs, indent=2) + "\n")
    print(
        json.dumps(
            {
                "flow": args.flow,
                "model_calls": result["model_calls"],
                "finish_reason": result["finish_reason"],
                "usage": result["usage"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
