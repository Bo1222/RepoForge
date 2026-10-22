"""Offline counterexamples for the current issue-repair workflow.

Runs real project functions, scripted model responses and, where noted, real
pytest subprocesses in temporary repositories. No provider requests or hidden
benchmark inputs are used. This is an audit artifact, not a production patch.
"""

import asyncio
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch as mock_patch
import xml.etree.ElementTree as ET
from pydantic import PrivateAttr
from litellm import ModelResponse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
for name, value in {
    "OTEL_SDK_DISABLED": "true",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
}.items():
    os.environ[name] = value

from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.find_code_snippet import FindCodeSnippet, FindCodeSnippetArgs
from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.actions.read_observation import ReadObservationArgs
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.actions.schema import Observation
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.actions.test_targets import discover_test_targets
from moatless.completion.base import CompletionRetryError
from moatless.completion.tool_call import ToolCallCompletionModel
from moatless.evaluation.generation import select_submitted_patch
from moatless.evaluation.verification import snapshot
from moatless.execution_memory import classify_completion, test_records
from moatless.flow.graph_state import GraphRuntime, initial_state
from moatless.flow.langgraph_flow import LangGraphFlow
from moatless.message_history.policy import content_hash
from moatless.message_history.policy import ContextBudgetExceeded
from moatless.node import ActionStep
from moatless.policy.action import extract_state
from moatless.policy import ActionPolicy, ActionPolicyConfig, BudgetSnapshot
from moatless.policy.failures import interpret
from moatless.repository.file import FileRepository
from moatless.repository.repository import InMemRepository
from moatless.repository_context import file_version
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.testing.schema import TestResult, TestStatus
from scripts.migration_smoke import response
from tests.memory.test_repository_context import execute, setup

logging.getLogger().setLevel(logging.CRITICAL)
RESULTS = []


def record(patch, **changes):
    result = {
        "superseded": False,
        "tested_patch_hash": content_hash(patch),
        "product_patch_hash": content_hash(patch),
        "targets": ["tests/test_code.py"],
        "test_source": "base_public",
        "test_origins": {"tests/test_code.py": "original"},
        "effective_status": "passed",
        "status": "passed",
        "introduced_failures": [],
        "product_patch": patch,
    }
    result.update(changes)
    return result


def emit(case, **details):
    RESULTS.append({"case": case, **details})
    print(json.dumps(RESULTS[-1], ensure_ascii=False), flush=True)


class ScriptedRuntime(RuntimeEnvironment):
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []
        self.last_test_execution = []
        self.last_test_results = []

    async def run_tests(self, patch=None, test_files=None):
        self.calls.append({"patch": patch, "test_files": test_files})
        self.last_test_execution = []
        self.last_test_results = []
        result = await self.behavior(len(self.calls), patch, test_files)
        code = 1 if any(r.status in {TestStatus.FAILED, TestStatus.ERROR} for r in result) else 0
        self.last_test_execution = [
            {"target": path, "command": "scripted-test", "return_code": code} for path in test_files
        ]
        self.last_test_results = result
        return result


class RealPytestRuntime(RuntimeEnvironment):
    """Apply the supplied real diff to a fresh public base and run real pytest."""

    def __init__(self, files):
        self.files = files
        self.calls = []
        self.last_test_execution = []
        self.last_test_results = []

    async def run_tests(self, patch=None, test_files=None):
        with tempfile.TemporaryDirectory(prefix="workflow-real-tests-") as tmp:
            directory = Path(tmp)
            for name, content in self.files.items():
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            if patch:
                applied = subprocess.run(
                    ["git", "apply", "-"], input=patch, text=True,
                    capture_output=True, cwd=directory, timeout=10,
                )
                if applied.returncode:
                    raise AssertionError(applied.stderr)
            report = directory / "report.xml"
            env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH=str(directory))
            command = [
                sys.executable, "-m", "pytest", "-q", "-c", "/dev/null",
                "--junitxml", str(report), *test_files,
            ]
            proc = subprocess.run(
                command, cwd=directory, env=env, capture_output=True, text=True, timeout=15,
            )
            results = []
            if report.exists():
                for item in ET.parse(report).iter("testcase"):
                    status = TestStatus.PASSED
                    failure = item.find("failure")
                    error = item.find("error")
                    if failure is not None:
                        status = TestStatus.FAILED
                    elif error is not None:
                        status = TestStatus.ERROR
                    elif item.find("skipped") is not None:
                        status = TestStatus.SKIPPED
                    results.append(TestResult(
                        status=status, name=item.attrib.get("name"),
                        file_path=test_files[0],
                        failure_output=(failure.text if failure is not None
                                        else error.text if error is not None else None),
                    ))
            self.last_test_execution = [
                {"target": test_files[0], "command": "real pytest", "return_code": proc.returncode}
            ]
            self.last_test_results = results
            self.calls.append({
                "patch": patch, "test_files": test_files,
                "return_code": proc.returncode,
                "statuses": [r.status.value for r in results],
            })
            return results


class RawScriptedModel(ToolCallCompletionModel):
    _raw_responses: list = PrivateAttr(default_factory=list)
    _requests: list = PrivateAttr(default_factory=list)

    async def _execute_completion(self, messages, invocation):
        self._requests.append(deepcopy(messages))
        return self._raw_responses[len(self._requests) - 1]


def seed(directory, files):
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def make_flow(agent, root, variant="a5", **kwargs):
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=20,
                                action_policy={"variant": variant}, **kwargs)
    state = initial_state(root, flow.project_id, flow.trajectory_id, root.node_id, False)
    flow._graph_state = state
    flow._loaded_snapshot_version = state["snapshot_version"]
    runtime = SimpleNamespace(context=GraphRuntime(agent=agent, workspace=agent.workspace))
    return flow, state, runtime


async def phase_replay(agent, root, variant="a5"):
    """Exercise actual business phases; exclude LangGraph/checkpoint transport."""
    flow, state, runtime = make_flow(agent, root, variant)
    phases = {
        "prepare": flow._prepare_iteration, "generate": flow._generate_actions,
        "policy": flow._policy_decision, "execute": flow._execute_action,
        "record": flow._record_observation, "commit": flow._commit_iteration,
    }
    trace = []
    for _ in range(150):
        if state["phase"] == "end":
            return flow, trace
        phase = state["phase"]
        state = await phases[phase](state, runtime)
        trace.append({"phase": phase, "next": state["phase"],
                      "finish_reason": state["finish_reason"]})
    raise AssertionError("phase replay exceeded its audit bound")


async def run_cases():
    empty = snapshot(content_hash(""), None, [record("")])
    missing = snapshot("current", None, [record("old", tested_patch_hash=None, product_patch_hash=None)])
    bypass = snapshot(content_hash("current"), None,
                      [record("current", introduced_failures=["known_regression"])])
    assert empty.state.value == missing.state.value == bypass.state.value == "PUBLIC_VERIFIED"
    emit("verification_admission",
         empty_patch_hash_state=empty.state.value,
         missing_binding_state=missing.state.value,
         known_regression_state=bypass.state.value)

    old = record("known_bad_candidate", superseded=True, effective_status="stale")
    later_failure = record("known_bad_candidate", status="failed", effective_status="stale")
    submitted, used = select_submitted_patch("new_candidate", [old, later_failure])
    assert used and submitted == "known_bad_candidate"
    emit("submission_ignores_later_failure", selected=submitted,
         old_pass_superseded=True, same_candidate_later_status=later_failure["status"])

    files = {
        "calc.py": "value = 1\n",
        "tests/test_code.py": "from calc import value\ndef test_code():\n    assert value == 2\n",
    }
    with tempfile.TemporaryDirectory(prefix="workflow-discovery-") as tmp:
        directory = Path(tmp)
        seed(directory, files)
        disk = discover_test_targets(FileRepository(repo_path=tmp), ["calc.py"], require_relation=True)
        memory = discover_test_targets(InMemRepository(files), ["calc.py"], require_relation=True)
        assert disk == [] and memory == ["tests/test_code.py"]
        emit("test_discovery_disk_vs_memory", real_file_repository=disk, in_memory_repository=memory)

    ranked_files = {
        "alpha.py": "value = 1\n",
        "tests/a_noise.py": "# alpha appears only in this comment\ndef test_noise():\n    assert True\n",
        "tests/alpha.py": "from alpha import value\ndef test_alpha():\n    assert value == 2\n",
    }
    with tempfile.TemporaryDirectory(prefix="workflow-ranking-") as tmp:
        directory = Path(tmp)
        seed(directory, ranked_files)
        agent, root, workspace = await setup(directory, actions=[ReadFile(), RunTests(), Finish()],
                                             runtime=ScriptedRuntime(lambda *_: None), shadow=True)
        root.file_context.add_file("alpha.py").apply_changes("value = 2\n")
        for path in ["tests/a_noise.py", "tests/alpha.py"]:
            root.file_context.add_file(path)
        root.action_steps = [ActionStep(action=FinishArgs(finish_reason="done"))]
        ranking = discover_test_targets(workspace.repository, ["alpha.py"],
                                       ["tests/a_noise.py", "tests/alpha.py"], require_relation=True)
        features, _ = extract_state(root, 0, workspace)
        assert ranking[0] == "tests/alpha.py" and features.target == "tests/a_noise.py"
        emit("ranked_targets_reordered_alphabetically",
             discovery_order=ranking, actual_policy_target=features.target)

    async def pass_then_timeout(call, patch, targets):
        if call > 1:
            await asyncio.sleep(2)
        return [TestResult(status=TestStatus.PASSED, name="test_code", file_path=targets[0])]

    with tempfile.TemporaryDirectory(prefix="workflow-timeout-") as tmp:
        directory = Path(tmp)
        seed(directory, files)
        runtime = ScriptedRuntime(pass_then_timeout)
        tool = RunTests(timeout=1)
        agent, root, workspace = await setup(directory, actions=[tool, Finish()], runtime=runtime, shadow=True)
        root.file_context.add_file("calc.py").apply_changes("value = 2\n")
        first = await execute(agent, root, RunTestsArgs(test_files=["tests/test_code.py"], test_source="base_public"))
        second = await execute(agent, first, RunTestsArgs(test_files=["tests/test_code.py"], test_source="base_public"))
        props = second.observation.properties
        current = test_records(second, workspace)[-1]
        state = snapshot(content_hash(second.file_context.generate_git_patch(ignore_tests=True)), None, [current])
        assert props["timed_out"] and current["status"] == "passed"
        assert state.state.value == "PUBLIC_VERIFIED"
        emit("timeout_reuses_cached_pass", actual_timeout=props["timed_out"],
             recorded_status=current["status"], effective_status=current["effective_status"],
             verification_state=state.state.value)

    with tempfile.TemporaryDirectory(prefix="workflow-modified-tests-") as tmp:
        directory = Path(tmp)
        seed(directory, files)
        runtime = RealPytestRuntime(files)
        agent, root, workspace = await setup(directory, actions=[RunTests(), Finish()],
                                             runtime=runtime, shadow=True)
        root.file_context.add_file("calc.py").apply_changes("value = 2\n")
        root.file_context.add_file("tests/test_code.py").apply_changes(
            "from calc import value\ndef test_code():\n    assert value == 3\n")
        node = await execute(agent, root, RunTestsArgs(test_files=["tests/test_code.py"]))
        full_patch = node.file_context.generate_git_patch()
        full_results = await runtime.run_tests(patch=full_patch, test_files=["tests/test_code.py"])
        state = classify_completion(node, workspace)
        assert node.observation.properties["auto_base_public"]
        assert node.observation.properties["test_run"]["status"] == "passed"
        assert any(r.status == TestStatus.FAILED for r in full_results)
        assert state["status"] == "verified"
        emit("modified_tests_not_executed", requested_source="workspace",
             executed_source=node.observation.properties["test_source"],
             reported_status=node.observation.properties["test_run"]["status"],
             actual_workspace_statuses=[r.status.value for r in full_results],
             classified_completion=state["status"], scope="real pytest on both actual diffs")

    with tempfile.TemporaryDirectory(prefix="workflow-unrelated-pass-") as tmp:
        directory = Path(tmp)
        public_files = {
            "calc.py": "value = 1\n",
            "tests/test_issue.py": "from calc import value\ndef test_issue():\n    assert value == 3\n",
            "tests/test_unrelated.py": "def test_unrelated():\n    assert True\n",
        }
        seed(directory, public_files)
        runtime = RealPytestRuntime(public_files)
        agent, root, workspace = await setup(directory, actions=[RunTests(), Finish()],
                                             runtime=runtime, shadow=True)
        root.file_context.add_file("calc.py").apply_changes("value = 2\n")
        tested = await execute(agent, root, RunTestsArgs(
            test_files=["tests/test_unrelated.py"], test_source="base_public"))
        finish = tested.create_child()
        finish.action_steps = [ActionStep(action=FinishArgs(finish_reason="done"))]
        features, _ = extract_state(finish, 0, workspace)
        decision = ActionPolicy(ActionPolicyConfig(variant="a5")).decide(
            features, finish.action_steps[0], BudgetSnapshot(remaining_rounds=10))
        issue_results = await runtime.run_tests(
            patch=tested.file_context.generate_git_patch(ignore_tests=True),
            test_files=["tests/test_issue.py"])
        assert features.target is None and features.public_test_passed
        assert decision.kind == "allow"
        assert issue_results[0].status == TestStatus.FAILED
        emit("unrelated_public_pass_allows_finish", related_target=features.target,
             verification_state=features.verification_state, finish_decision=decision.kind,
             public_issue_test_status=issue_results[0].status.value,
             scope="real pytest; issue test is already public")

    with tempfile.TemporaryDirectory(prefix="workflow-new-test-binding-") as tmp:
        directory = Path(tmp)
        seed(directory, files)
        runtime = RealPytestRuntime(files)
        agent, root, workspace = await setup(directory, actions=[RunTests(), Finish()],
                                             runtime=runtime, shadow=True)
        root.file_context.add_file("calc.py").apply_changes("value = 2\n")
        root.file_context.add_file("tests/test_new.py").apply_changes(
            "from calc import value\ndef test_new():\n    assert value == 2\n")
        tested = await execute(agent, root, RunTestsArgs(test_files=["tests/test_new.py"]))
        records = test_records(tested, workspace)
        state = snapshot(content_hash(tested.file_context.generate_git_patch(ignore_tests=True)), None, records)
        assert records[-1]["effective_status"] == "passed" and state.state.value == "STALE"
        emit("new_test_current_run_discarded_as_stale", execution_status=records[-1]["effective_status"],
             tests=records[-1]["targets"], verification_state=state.state.value,
             tested_equals_product_hash=records[-1]["tested_patch_hash"] == records[-1]["product_patch_hash"],
             scope="real pytest on current product plus newly added regression test")

    async def fail_then_slow_baseline(call, patch, targets):
        if call > 1:
            await asyncio.sleep(2)
        return [TestResult(status=TestStatus.FAILED, name="test_code",
                           file_path=targets[0], failure_output="AssertionError")]

    with tempfile.TemporaryDirectory(prefix="workflow-baseline-timeout-") as tmp:
        directory = Path(tmp)
        seed(directory, files)
        runtime = ScriptedRuntime(fail_then_slow_baseline)
        agent, root, workspace = await setup(directory, actions=[RunTests(timeout=1), Finish()],
                                             runtime=runtime, shadow=True)
        root.file_context.add_file("calc.py").apply_changes("value = 2\n")
        root.action_steps = [ActionStep(action=RunTestsArgs(
            test_files=["tests/test_code.py"], test_source="base_public"))]
        flow, state, context = make_flow(agent, root, tool_timeout=1.2)
        state["phase"] = "execute"
        result = await flow._execute_action(state, context)
        assert result["finish_reason"] == "tool_timeout" and len(runtime.calls) == 2
        emit("baseline_escapes_invocation_timeout", calls=len(runtime.calls),
             invocation_timeout=1, outer_timeout=1.2,
             finish_reason=result["finish_reason"],
             observation_present=root.action_steps[0].observation is not None)

    async def failed_assertion(call, patch, targets):
        return [TestResult(status=TestStatus.FAILED, name="test_code",
                           file_path=targets[0], failure_output="AssertionError: expected 3")]

    with tempfile.TemporaryDirectory(prefix="workflow-correction-") as tmp:
        directory = Path(tmp)
        seed(directory, files)
        runtime = ScriptedRuntime(failed_assertion)
        agent, root, workspace = await setup(directory, actions=[StringReplace(), RunTests(), Finish()],
                                             runtime=runtime, shadow=True)
        root.file_context.add_file("calc.py")
        root.file_context.add_file("tests/test_code.py")
        agent.completion_model._responses = [
            response(StringReplaceArgs(path="calc.py", old_str="value = 1", new_str="value = 2"),
                     FinishArgs(finish_reason="first finish")),
            response(FinishArgs(finish_reason="finish after failed verification")),
        ]
        flow, trace = await phase_replay(agent, root)
        decisions = [d["decision"] for n in flow.root.get_all_nodes() for d in (n.policy_decisions or [])]
        assert [d["kind"] for d in decisions] == ["allow", "verify", "stop"]
        emit("verification_attempt_consumes_repair_key", decisions=decisions,
             model_calls=agent.completion_model._calls,
             outcome=flow.graph_state["finish_reason"],
             scope="actual business phases; LangGraph transport excluded")

    for label, props, message in [
        ("passing_modified_tests", {"test_run": {"status": "passed",
          "test_origins": {"tests/test_code.py": "modified"}}}, "1 passed"),
        ("ordinary_search_miss", {"fail_reason": "no_search_hits"}, "No search results found"),
        ("actual_assertion_in_modified_test", {"test_run": {"status": "failed",
          "test_origins": {"tests/test_code.py": "modified"}}}, "AssertionError: expected 3"),
    ]:
        signal = interpret(props, message)
        emit("failure_interpreter_" + label, kind=signal.kind.value,
             suggested_action=signal.suggested_action)

    syntax = CodeModificationMixin().validate_python_syntax("code.py", "value = 1\n", "return 1\n")
    try:
        compile("return 1\n", "code.py", "exec")
    except SyntaxError as exc:
        compile_error = str(exc)
    else:
        raise AssertionError("audit compile fixture should be invalid")
    assert syntax is None
    emit("ast_parse_accepts_compile_error", edit_validation_rejected=False,
         real_compile_error=compile_error)

    with tempfile.TemporaryDirectory(prefix="workflow-shadow-search-") as tmp:
        directory = Path(tmp)
        seed(directory, {"calc.py": "def calculate():\n    return 1\n"})
        agent, root, workspace = await setup(directory, actions=[ReadFile(), Finish()], shadow=True)
        root.file_context.add_file("calc.py").apply_changes("def calculate():\n    return 2\n")
        search = FindCodeSnippet()
        search._workspace = workspace
        agent._action_map[FindCodeSnippetArgs] = search
        async def synchronous_grep_transport(repository, search_text, file_pattern=None):
            # Keep the real grep command and base files; bypass only asyncio's
            # child-watcher transport, which stalls in the audit sandbox.
            proc = subprocess.run(
                ["grep", "-n", "-F", search_text, file_pattern],
                cwd=repository.repo_path, capture_output=True, text=True, timeout=3,
            )
            return [(file_pattern, int(line.split(":", 1)[0]))
                    for line in proc.stdout.splitlines()]

        with mock_patch.object(FileRepository, "find_exact_matches", synchronous_grep_transport):
            node = await execute(agent, root, FindCodeSnippetArgs(
                code_snippet="return 1", file_pattern="calc.py"))
        properties = node.observation.properties
        messages = await agent.memory.generate_messages(node, workspace)
        source = f"{node.node_id}:0"
        current_digest = file_version(workspace, node.file_context, "calc.py")
        assert "return 1" in node.observation.message
        assert properties["code_view"]["files"]["calc.py"] == current_digest
        assert source not in node.context_build["stale_sources"]
        emit("shadow_search_old_content_marked_current",
             actual_patch_content=node.file_context._files["calc.py"].content,
             search_output=node.observation.message,
             labeled_with_current_hash=True, marked_stale=False,
             prompt_contains_old_code="return 1" in str(messages),
             scope="real grep; synchronous subprocess transport; actual rendering/versioning")

    model = RawScriptedModel(model="offline-audit")
    model.initialize([StringReplaceArgs, ReadFileArgs, FinishArgs], "offline audit")
    calls = [
        {"id": "call_a", "type": "function", "function": {
            "name": "StringReplace", "arguments": json.dumps({
                "path": "calc.py", "old_str": "x" * 3000, "new_str": "y"})}},
        {"id": "call_b", "type": "function", "function": {
            "name": "ReadFile", "arguments": json.dumps({"file_path": "calc.py"})}},
    ]
    model._raw_responses = [
        ModelResponse(choices=[{"message": {"role": "assistant", "tool_calls": calls},
                                "finish_reason": "tool_calls"}]),
        ModelResponse(choices=[{"message": {"role": "assistant", "tool_calls": [
            {"id": "call_c", "type": "function", "function": {
                "name": "Finish", "arguments": json.dumps({"finish_reason": "done"})}}
        ]}, "finish_reason": "tool_calls"}]),
    ]
    validations = 0

    async def reject_first(outputs, text):
        nonlocal validations
        validations += 1
        if validations == 1:
            raise CompletionRetryError("post validation rejected these proposals")
        return outputs, text

    model._post_validation_fn = reject_first
    await model._create_completion_with_retries([{"role": "user", "content": "fix"}])
    retry_messages = model._requests[1]
    assistant = next(m for m in retry_messages if m.get("tool_calls"))
    replied = {m.get("tool_call_id") for m in retry_messages if m.get("role") == "tool"}
    unpaired = [c["id"] for c in assistant["tool_calls"] if c["id"] not in replied]
    try:
        json.loads(assistant["tool_calls"][0]["function"]["arguments"])
    except json.JSONDecodeError:
        arguments_valid = False
    else:
        arguments_valid = True
    assert unpaired == ["call_b"] and not arguments_valid
    emit("retry_breaks_call_pairing_and_json", dangling_call_ids=unpaired,
         truncated_arguments_valid_json=arguments_valid, provider_calls=0)

    with tempfile.TemporaryDirectory(prefix="workflow-context-ids-") as tmp:
        directory = Path(tmp)
        seed(directory, {"calc.py": "def calculate():\n    return 1\n"})
        agent, root, workspace = await setup(directory, actions=[ReadFile(), Finish()], shadow=True)
        read = await execute(agent, root, ReadFileArgs(file_path="calc.py"))
        messages = await agent.memory.generate_messages(read, workspace)
        included = read.context_build["included_sources"]
        assert f"node:{read.node_id}" in included and "current_code:calc.py" in included
        emit("fresh_read_dedup_source_id_mismatch",
             interaction_source=f"node:{read.node_id}", code_view_source=f"{read.node_id}:0",
             current_code_still_included=True,
             code_occurrences=str(messages).count("return 1"))

        current = root
        for i in range(12):
            current = current.create_child()
            current.action_steps = [ActionStep(
                action=ReadFileArgs(file_path="calc.py"),
                observation=Observation.create(f"archive {i}\n" + "retained details " * 2000),
            )]
        agent.memory.policy.recent_interactions = 1
        catalog = None
        chosen_budget = None
        for budget in [1600, 2400, 3200, 4800, 6400]:
            agent.memory.policy.max_input_tokens = budget
            try:
                projected = await agent.memory.generate_messages(current, workspace)
            except ContextBudgetExceeded:
                continue
            catalog = next((m.get("content") for m in projected
                            if "<omitted_sources>" in (m.get("content") or "")), None)
            if catalog:
                chosen_budget = budget
                break
        assert catalog is not None
        ids = [line for line in catalog.splitlines() if line.startswith("node:")]
        rejected = []
        for source in ids:
            try:
                ReadObservationArgs(source_id=source)
            except ValueError:
                rejected.append(source)
        assert ids and rejected == ids
        emit("omitted_catalog_references_not_readable",
             input_budget=chosen_budget, catalog_source_ids=ids,
             rejected_by_read_observation=rejected)


async def main():
    started = time.monotonic()
    await run_cases()
    relevant = [
        "moatless/policy/action.py", "moatless/evaluation/verification.py",
        "moatless/evaluation/generation.py", "moatless/actions/run_tests.py",
        "moatless/actions/test_targets.py", "moatless/policy/failures.py",
        "moatless/execution_memory.py", "moatless/flow/langgraph_flow.py",
        "moatless/completion/base.py",
        "moatless/actions/search_base.py", "moatless/actions/find_code_snippet.py",
        "moatless/message_history/managed.py", "moatless/repository_context.py",
    ]
    result = {
        "python": sys.version.split()[0],
        "provider_calls": 0,
        "hidden_benchmark_inputs": False,
        "seconds": round(time.monotonic() - started, 3),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in relevant},
        "cases": RESULTS,
    }
    destination = Path(__file__).with_name("probe-results.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Saved " + str(destination), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
