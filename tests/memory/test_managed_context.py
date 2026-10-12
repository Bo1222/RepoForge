from copy import deepcopy

import pytest
from pydantic import ValidationError

from moatless.actions.read_observation import ReadObservation, ReadObservationArgs
from moatless.actions.respond import RespondArgs
from moatless.actions.schema import Observation
from moatless.message_history.base import BaseMemory
from moatless.message_history.managed import ManagedMessageHistoryGenerator
from moatless.message_history.policy import ContextBudgetExceeded, ContextPolicy, TokenEstimator, request_envelope
from moatless.node import ActionStep, Node
from moatless.repository.repository import InMemRepository
from moatless.workspace import Workspace


class Model:
    model = "deepseek/deepseek-chat"
    max_tokens = 512
    model_api_key = "fake-credential-for-testing"
    _system_prompt = "System guidance"
    _completion_params = {"tools": [{"type": "function", "function": {"name": "Respond", "description": "Respond"}}]}

    def _prepare_messages(self, messages, prompt):
        return [{"role": "system", "content": prompt}, *deepcopy(messages)]


def setup(policy=None):
    memory = ManagedMessageHistoryGenerator(policy=policy or ContextPolicy(context_window=8192))
    memory.bind_model(Model())
    root = Node(node_id=0, user_message="Fix the issue")
    child = root.create_child()
    child.action_steps = [ActionStep(action=RespondArgs(message="done"), observation=Observation(
        message="header\n" + "large output\n" * 3000 + "FAILED: important diagnostic",
        properties={"return_code": 1, "fail_reason": "command_failed"},
    ))]
    return memory, root, child, Workspace(repository=InMemRepository())


def test_policy_requires_capacity_and_supported_version():
    with pytest.raises(ValidationError):
        ContextPolicy()
    with pytest.raises(ValidationError):
        ContextPolicy(context_window=8192, version=2)
    assert ContextPolicy(context_window=8192).input_budget(512) == 6656
    with pytest.raises(ValueError, match="finite"):
        ContextPolicy(context_window=8192).input_budget(None)


@pytest.mark.parametrize("key,value", [("messages", []), ("model", "another/model")])
def test_managed_params_cannot_override_bound_context(key, value):
    memory = ManagedMessageHistoryGenerator(policy=ContextPolicy(context_window=8192))
    model = Model()
    model._completion_params = {key: value}
    with pytest.raises(ValueError, match="overriding"):
        memory.bind_model(model)


@pytest.mark.asyncio
async def test_budgeted_observation_retains_reference_exit_status_and_original():
    memory, root, child, workspace = setup(ContextPolicy(context_window=8192, max_observation_tokens=200))
    messages = await memory.generate_messages(child, workspace)
    result = next(m for m in messages if m["role"] == "tool")
    assert "ReadObservation source_id=2:0" in result["content"]
    assert '"return_code":1' in result["content"] and "FAILED" in result["content"]
    assert TokenEstimator(Model.model).count(result["content"]) <= 200
    assert len(child.action_steps[0].observation.message) > 30000
    tool = ReadObservation()
    output = await tool.execute_for_node(ReadObservationArgs(source_id="2:0", start_line=3002, end_line=3002), child)
    assert "FAILED: important diagnostic" in output.message
    assert child.context_build["shortened_sources"] == ["2:0"]
    assert root.user_message in str(messages)


@pytest.mark.asyncio
async def test_original_observation_read_is_ancestor_scoped_and_bounded():
    _, root, child, _ = setup()
    sibling = root.create_child()
    sibling.action_steps = [ActionStep(action=RespondArgs(message="secret"), observation=Observation(message="other branch"))]
    tool = ReadObservation()
    missing = await tool.execute_for_node(ReadObservationArgs(source_id=f"{sibling.node_id}:0"), child)
    assert missing.properties["fail_reason"] == "observation_not_found"
    output = await tool.execute_for_node(ReadObservationArgs(source_id="2:0", end_line=5000), child)
    assert output.properties["end_line"] == 200
    with pytest.raises(ValidationError):
        ReadObservationArgs(source_id="../../secret")


@pytest.mark.asyncio
async def test_system_prompt_and_tool_schema_are_inside_the_budget():
    memory, _, child, workspace = setup(ContextPolicy(context_window=4096, max_input_tokens=300))
    model = Model()
    model._system_prompt = "Large system guidance " * 2000
    memory.bind_model(model)
    with pytest.raises(ContextBudgetExceeded, match="request envelope"):
        await memory.generate_messages(child, workspace)
    model._system_prompt = "short"
    model._completion_params = {"tools": [{"description": "large schema " * 2000}]}
    with pytest.raises(ContextBudgetExceeded):
        await memory.generate_messages(child, workspace)


@pytest.mark.asyncio
async def test_final_guard_catches_retry_growth_and_redacts_credentials():
    memory, _, child, workspace = setup()
    messages = await memory.generate_messages(child, workspace)
    prepared = memory._model._prepare_messages(messages, "System guidance")
    prepared.append({"role": "user", "content": Model.model_api_key})
    memory.record_request(child, prepared)
    record = child.context_build["prepared_requests"][-1]
    assert Model.model_api_key not in str(record)
    assert record["credentials_redacted"]
    assert record["request_hash"]
    assert "fallback" in child.context_build["estimator"]
    estimator = TokenEstimator(Model.model)
    assert estimator.count(request_envelope(prepared, memory._model)) == record["estimated_input_tokens"]
    with pytest.raises(ContextBudgetExceeded):
        memory.record_request(child, prepared + [{"role": "user", "content": "too long " * 20000}])


def test_managed_policy_component_roundtrip_and_unbound_error():
    memory = ManagedMessageHistoryGenerator(policy=ContextPolicy(context_window=8192))
    restored = BaseMemory.from_dict(memory.model_dump())
    assert isinstance(restored, ManagedMessageHistoryGenerator)
    assert restored.policy.context_window == 8192


@pytest.mark.asyncio
async def test_audit_survives_node_roundtrip_and_user_requirements_are_pinned():
    memory, root, child, workspace = setup()
    child.user_message = "Preserve compatibility"
    messages = await memory.generate_messages(child, workspace)
    memory.record_request(child, memory._model._prepare_messages(messages, "System guidance"))
    restored = Node.from_dict({"nodes": root.dump_as_list()})
    saved = restored.get_node_by_id(child.node_id)
    assert saved.context_build == child.context_build
    assert root.user_message in str(messages) and child.user_message in str(messages)


@pytest.mark.asyncio
async def test_two_recent_groups_survive_with_compressed_bodies():
    memory, root, child, workspace = setup(ContextPolicy(
        context_window=4096, max_input_tokens=800, max_observation_tokens=2000,
    ))
    next_node = child.create_child()
    next_node.action_steps = [ActionStep(action=RespondArgs(message="another"), observation=Observation(
        message="second very long output " * 3000,
    ))]
    messages = await memory.generate_messages(next_node, workspace)
    assert len([m for m in messages if m["role"] == "tool"]) == 2
    assert TokenEstimator(Model.model).count(request_envelope(
        memory._model._prepare_messages(messages, Model._system_prompt), memory._model
    )) <= 800


@pytest.mark.asyncio
async def test_agent_final_wire_request_is_audited_and_schema_checked(monkeypatch):
    import json
    import litellm
    from moatless.actions.respond import Respond
    from moatless.agent.agent import ActionAgent
    from moatless.completion.tool_call import ToolCallCompletionModel

    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return litellm.ModelResponse(choices=[{
            "finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{
                "id": "call1", "type": "function", "function": {
                    "name": "SendMessage", "arguments": json.dumps({"message": "done"}),
                },
            }]},
        }], usage={"prompt_tokens": 123, "completion_tokens": 5, "total_tokens": 128})

    monkeypatch.setattr(litellm, "acompletion", complete)
    memory = ManagedMessageHistoryGenerator(policy=ContextPolicy(context_window=8192))
    agent = ActionAgent(completion_model=ToolCallCompletionModel(model="gpt-4o-mini"),
                        actions=[Respond()], system_prompt="Repair the issue", memory=memory)
    await agent.initialize(Workspace(repository=InMemRepository()))
    node = Node(node_id=0, user_message="Fix").create_child()
    await agent.generate_actions(node)
    assert len(requests) == 1
    audit = node.context_build["prepared_requests"][-1]
    assert audit["request"]["messages"] == requests[0]["messages"]
    assert audit["request"]["tools"] == requests[0]["tools"]
    assert node.context_build["actual_usage"]["prompt_tokens"] == 123
    assert "ReadObservation" in str(requests[0]["tools"])


@pytest.mark.asyncio
async def test_admission_checks_budget_before_any_provider_call(monkeypatch):
    import litellm
    from moatless.actions.respond import Respond
    from moatless.agent.agent import ActionAgent
    from moatless.completion.tool_call import ToolCallCompletionModel

    calls = []
    async def unexpected(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Provider must not be called")
    monkeypatch.setattr(litellm, "acompletion", unexpected)
    memory = ManagedMessageHistoryGenerator(policy=ContextPolicy(context_window=8192, max_input_tokens=120))
    agent = ActionAgent(completion_model=ToolCallCompletionModel(model="gpt-4o-mini"),
                        actions=[Respond()], system_prompt="large guidance " * 1000, memory=memory)
    await agent.initialize(Workspace(repository=InMemRepository()))
    node = Node(node_id=0, user_message="Fix").create_child()
    with pytest.raises(ContextBudgetExceeded):
        await agent.generate_actions(node)
    assert calls == []


@pytest.mark.asyncio
async def test_validation_retry_cannot_bypass_final_request_budget(monkeypatch):
    import litellm
    from moatless.actions.respond import Respond
    from moatless.agent.agent import ActionAgent
    from moatless.completion.tool_call import ToolCallCompletionModel

    calls = []
    async def invalid(**kwargs):
        calls.append(deepcopy(kwargs))
        return litellm.ModelResponse(choices=[{
            "finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{
                "id": "bad", "type": "function", "function": {"name": "unknown_tool" * 2000, "arguments": "{}"},
            }]},
        }])
    monkeypatch.setattr(litellm, "acompletion", invalid)
    agent = ActionAgent(completion_model=ToolCallCompletionModel(model="gpt-4o-mini"), actions=[Respond()],
                        system_prompt="Fix", memory=ManagedMessageHistoryGenerator(
                            policy=ContextPolicy(context_window=8192, max_input_tokens=1200)))
    await agent.initialize(Workspace(repository=InMemRepository()))
    node = Node(node_id=0, user_message="Fix").create_child()
    with pytest.raises(ContextBudgetExceeded):
        await agent.generate_actions(node)
    assert len(calls) == 1
    assert "context_budget_exceeded" in node.error
