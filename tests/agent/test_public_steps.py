import pytest

from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.respond import Respond, RespondArgs
from moatless.agent.agent import ActionAgent
from moatless.context_data import current_action_step
from moatless.exceptions import CompletionRejectError, RuntimeError
from moatless.node import Node
from scripts.migration_smoke import ScriptedCompletion, response


def setup_agent(*responses):
    model = ScriptedCompletion(list(responses))
    agent = ActionAgent(completion_model=model, system_prompt="fixed", actions=[Finish(), Respond()])
    node = Node.create_root("issue").create_child()
    return agent, model, node


@pytest.mark.asyncio
async def test_generation_and_execution_are_independent_and_reuse_response():
    agent, model, node = setup_agent(response(RespondArgs(message="done")))
    events = []

    async def capture(event):
        events.append((event.event_type, current_action_step.get()))

    agent._on_event = capture
    await agent.generate_actions(node)
    assert node.observation is None
    assert model._calls == 1
    assert node.thoughts.text == "scripted reasoning"
    assert node.assistant_message == "scripted response"
    assert node.thinking_blocks
    assert node.usage().completion_cost == 0.01
    await agent.generate_actions(node)
    assert model._calls == 1
    await agent.execute_action_step(node, 0)
    assert node.observation.message == "done"
    assert events == [("run", None), ("created", None), ("executed", 0)]
    assert current_action_step.get() is None
    with pytest.raises(RuntimeError):
        await agent.execute_action_step(node, 0)


@pytest.mark.asyncio
async def test_terminal_prevents_remaining_step_and_index_validation():
    agent, _, node = setup_agent(response(FinishArgs(finish_reason="done"), RespondArgs(message="unused")))
    await agent.generate_actions(node)
    with pytest.raises(IndexError):
        await agent.execute_action_step(node, -1)
    with pytest.raises(IndexError):
        await agent.execute_action_step(node, 2)
    await agent.execute_action_step(node, 0)
    with pytest.raises(RuntimeError, match="terminal"):
        await agent.execute_action_step(node, 1)
    assert node.action_steps[1].observation is None


@pytest.mark.asyncio
async def test_rejection_keeps_completion_usage():
    invocation = response().completion_invocation
    agent, _, node = setup_agent(CompletionRejectError("invalid", completion_invocation=invocation))
    await agent.generate_actions(node)
    assert node.error
    assert node.usage().completion_cost == 0.01


@pytest.mark.asyncio
async def test_execution_failure_restores_outer_context(monkeypatch):
    agent, _, node = setup_agent(response(RespondArgs(message="done")))
    await agent.generate_actions(node)

    async def fail(self, args, **kwargs):
        raise ValueError("tool failed")

    monkeypatch.setattr(Respond, "execute", fail)
    token = current_action_step.set(42)
    try:
        with pytest.raises(ValueError, match="tool failed"):
            await agent.execute_action_step(node, 0)
        assert current_action_step.get() == 42
        assert node.error
    finally:
        current_action_step.reset(token)
