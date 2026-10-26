"""Regression counterparts of the workflow audit's execution counterexamples."""
import asyncio
from copy import deepcopy
import json
import re

from litellm.types.utils import ModelResponse
from pydantic import PrivateAttr
import pytest

from repoforge.actions.finish import Finish, FinishArgs
from repoforge.actions.find_code_snippet import FindCodeSnippet, FindCodeSnippetArgs
from repoforge.actions.read_file import ReadFile, ReadFileArgs
from repoforge.actions.read_observation import ReadObservation, ReadObservationArgs
from repoforge.actions.schema import Observation
from repoforge.actions.string_replace import StringReplaceArgs
from repoforge.completion.base import CompletionRetryError
from repoforge.completion.tool_call import ToolCallCompletionModel
from repoforge.evaluation.generation import select_submitted_patch
from repoforge.evaluation.verification import snapshot, VerificationState
from repoforge.node import ActionStep
from repoforge.policy.failures import interpret
from tests.memory.test_repository_context import setup, execute


class OfflineModel(ToolCallCompletionModel):
    _responses: list = PrivateAttr(default_factory=list)
    _requests: list = PrivateAttr(default_factory=list)

    async def _execute_completion(self, messages, invocation):
        self._requests.append(deepcopy(messages))
        response = self._responses[len(self._requests) - 1]
        if isinstance(response, Exception):
            raise response
        return response


def tool_response(calls):
    return ModelResponse(choices=[{"message": {"role": "assistant", "tool_calls": calls}, "finish_reason": "tool_calls"}])


def call(identity, name, **arguments):
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


@pytest.mark.asyncio
async def test_retry_rejects_every_call_with_valid_compact_json_and_archives_original():
    model = OfflineModel(model="offline-audit")
    model.initialize([StringReplaceArgs, ReadFileArgs, FinishArgs], "offline")
    original = [call("a", "StringReplace", path="calc.py", old_str="x" * 10000, new_str="y"),
                call("b", "ReadFile", file_path="calc.py")]
    model._responses = [tool_response(original), tool_response([call("done", "Finish", finish_reason="done")])]
    validations = 0
    async def validate(outputs, text):
        nonlocal validations
        validations += 1
        if validations == 1:
            raise CompletionRetryError("Rejected by schema validation")
        return outputs, text
    model._post_validation_fn = validate
    result = await model._create_completion_with_retries([{"role": "user", "content": "fix"}])
    retry = model._requests[1]
    assistant = next(m for m in retry if m.get("tool_calls"))
    assert {c["id"] for c in assistant["tool_calls"]} == {m["tool_call_id"] for m in retry if m["role"] == "tool"}
    for c in assistant["tool_calls"]:
        assert json.loads(c["function"]["arguments"])["not_executed"]
    assert len(str(retry)) < 2500
    assert len(result.completion_invocation.attempts) == 2
    assert "x" * 10000 in str(result.completion_invocation.attempts[0].invalid_response)


@pytest.mark.asyncio
async def test_empty_length_responses_have_three_total_attempts_without_replaying_content():
    model = OfflineModel(model="offline-audit")
    model.initialize([FinishArgs], "offline")
    model._responses = [ModelResponse(choices=[{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}])] * 2 + [
        tool_response([call("done", "Finish", finish_reason="done")])]
    result = await model._create_completion_with_retries([{"role": "user", "content": "fix"}])
    assert len(model._requests) == len(result.completion_invocation.attempts) == 3
    assert model._requests[1] == model._requests[0]


@pytest.mark.asyncio
async def test_transient_and_protocol_retries_share_three_attempt_budget():
    model = OfflineModel(model="offline-audit")
    model.initialize([FinishArgs], "offline")
    invalid = tool_response([call("invalid", "Unknown", value=1)])
    valid = tool_response([call("done", "Finish", finish_reason="done")])
    model._responses = [asyncio.TimeoutError(), invalid, valid]
    result = await model._create_completion_with_retries([{"role": "user", "content": "fix"}])
    assert len(model._requests) == len(result.completion_invocation.attempts) == 3
    model._requests = []
    model._responses = [asyncio.TimeoutError(), asyncio.TimeoutError(), valid]
    from repoforge.exceptions import CompletionRuntimeError
    with pytest.raises(CompletionRuntimeError):
        await model._create_completion_with_retries([{"role": "user", "content": "fix"}])
    assert len(model._requests) == 2


@pytest.mark.asyncio
async def test_shadow_search_renders_current_code_and_does_not_recall_removed_text(tmp_path):
    (tmp_path / "calc.py").write_text("def calculate():\n    return 1\n")
    agent, root, workspace = await setup(tmp_path, actions=[ReadFile(), Finish()], shadow=True)
    root.file_context.add_file("calc.py").apply_changes("def calculate():\n    return 2\n")
    action = FindCodeSnippet()
    action._workspace = workspace
    agent._action_map[FindCodeSnippetArgs] = action
    node = await execute(agent, root, FindCodeSnippetArgs(code_snippet="return 2", file_pattern="calc.py"))
    assert "return 2" in node.observation.message and "return 1" not in node.observation.message
    old = await execute(agent, node, FindCodeSnippetArgs(code_snippet="return 1", file_pattern="calc.py"))
    assert "return 1" not in old.observation.message


@pytest.mark.asyncio
async def test_new_shadow_file_is_searchable_and_fresh_read_is_not_duplicated(tmp_path):
    agent, root, workspace = await setup(tmp_path, actions=[ReadFile(), Finish()], shadow=True)
    root.file_context.add_file("new.py").apply_changes("def calculate():\n    return 42\n")
    action = FindCodeSnippet()
    action._workspace = workspace
    agent._action_map[FindCodeSnippetArgs] = action
    found = await execute(agent, root, FindCodeSnippetArgs(code_snippet="return 42"))
    assert "return 42" in found.observation.message
    messages = await agent.memory.generate_messages(found, workspace)
    assert not any("<current_code " in (m.get("content") or "") for m in messages)


@pytest.mark.asyncio
async def test_older_selected_fresh_read_also_avoids_duplicate_current_code(tmp_path):
    (tmp_path / "one.py").write_text("value = 1\n")
    (tmp_path / "two.py").write_text("value = 2\n")
    agent, root, workspace = await setup(tmp_path, actions=[ReadFile(), Finish()], shadow=True)
    node = await execute(agent, root, ReadFileArgs(file_path="one.py"))
    node = await execute(agent, node, ReadFileArgs(file_path="two.py"))
    node = await execute(agent, node, ReadFileArgs(file_path="two.py"))
    messages = await agent.memory.generate_messages(node, workspace)
    assert "value = 1" in str(messages)
    assert "current_code:one.py" not in node.context_build["included_sources"]


@pytest.mark.asyncio
async def test_catalog_contains_final_omitted_observation_refs_accepted_by_read_observation(tmp_path):
    (tmp_path / "calc.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path, actions=[ReadFile(), Finish()], shadow=True)
    node = root
    for i in range(12):
        node = node.create_child()
        node.action_steps = [ActionStep(action=ReadFileArgs(file_path="calc.py"),
            observation=Observation.create(f"original {i}\n" + "retained details " * 2000))]
    agent.memory.policy.max_input_tokens = 2400
    agent.memory.policy.recent_interactions = 1
    messages = await agent.memory.generate_messages(node, workspace)
    catalog = next(m["content"] for m in messages if "<omitted_sources>" in (m.get("content") or ""))
    refs = re.findall(r"^\d+:\d+$", catalog, re.M)
    assert refs
    omitted_nodes = {int(s.split(":")[1]) for s in node.context_build["omitted_sources"] if s.startswith("node:")}
    for ref in refs:
        assert int(ref.split(":")[0]) in omitted_nodes
        result = await ReadObservation().execute_for_node(ReadObservationArgs(source_id=ref), node)
        assert "original" in result.message


def test_missing_recipe_binding_never_verifies_and_export_never_rolls_back():
    historical = {"test_source": "base_public", "product_patch": "old patch", "status": "passed",
                  "effective_status": "passed", "product_patch_hash": None, "tested_patch_hash": None}
    assert snapshot("current", None, [historical]).state != VerificationState.PUBLIC_VERIFIED
    assert select_submitted_patch("current failed patch", [historical]) == ("current failed patch", False)


def test_failure_interpreter_distinguishes_provenance_assertion_and_search_miss():
    assert interpret({"fail_reason": "no_search_hits"}) is None
    assert interpret({"test_run": {"status": "passed", "test_origins": {"test_x.py": "modified"}}}) is None
    signal = interpret({"test_run": {"status": "failed", "test_origins": {"test_x.py": "modified"}}}, "AssertionError: 2 != 3")
    assert signal.kind.value == "assertion_failure"

