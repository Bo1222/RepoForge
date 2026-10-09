"""Executable observations of the legacy Loop; keep these expectations frozen."""

import json
from pathlib import Path

import pytest

from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.respond import Respond, RespondArgs
from moatless.actions.think import Think, ThinkArgs
from moatless.agent.agent import ActionAgent
from moatless.completion.schema import ResponseSchema
from moatless.exceptions import CompletionRejectError, RejectError
from moatless.flow.loop import AgenticLoop
from moatless.node import ActionStep
from scripts.migration_smoke import ScriptedCompletion, replay, response

BASELINE = Path(__file__).resolve().parents[2] / "experiments/baselines/M0"


def make_flow(responses, **kwargs):
    model = ScriptedCompletion(responses)
    agent = ActionAgent(
        completion_model=model, system_prompt="fixed contract prompt", actions=[Respond(), Finish(), Think()]
    )
    return AgenticLoop.create(message="issue", agent=agent, **kwargs), model


@pytest.mark.asyncio
@pytest.mark.parametrize("coding", [False, True])
async def test_frozen_replay(coding):
    actual, _ = await replay(coding=coding)
    expected = json.loads((BASELINE / ("coding.json" if coding else "respond.json")).read_text())
    assert actual == expected


@pytest.mark.asyncio
async def test_root_counts_and_iteration_boundary():
    flow, model = make_flow([response(RespondArgs(message="done"))], max_iterations=1)
    assert (await flow.run()).node_id == 0
    assert flow.is_finished() == "max_iterations"
    assert model._calls == 0
    flow.max_iterations = 2
    assert (await flow.run()).node_id == 1
    assert flow.is_finished() == "max_iterations"  # iteration reason precedes terminal


@pytest.mark.asyncio
async def test_cost_checked_after_whole_response():
    flow, model = make_flow([response(RespondArgs(message="done"), cost=0.02)], max_cost=0.01)
    await flow.run()
    assert model._calls == 1
    assert flow.total_usage().completion_cost == 0.02
    assert flow.is_finished() == "max_cost"


@pytest.mark.asyncio
async def test_zero_node_id_is_unspecified():
    flow, model = make_flow([response(RespondArgs(message="done"))], max_iterations=1)
    assert (await flow.run(node_id=0)).node_id == 0
    assert model._calls == 0


@pytest.mark.asyncio
async def test_positive_node_id_bypasses_budget_and_returns_one_round():
    flow, model = make_flow([response(RespondArgs(message="done"))], max_iterations=1)
    child = flow._create_next_node(flow.root)
    assert (await flow.run(node_id=child.node_id)).node_id == child.node_id
    assert model._calls == 1


@pytest.mark.asyncio
async def test_message_creates_child_and_prebuilt_actions_skip_completion():
    flow, model = make_flow([response(RespondArgs(message="done"))])
    node = await flow.run(message="followup")
    assert node.user_message == "followup"
    assert node.parent is flow.root
    flow2, model2 = make_flow([])
    child = flow2._create_next_node(flow2.root)
    child.action_steps = [ActionStep(action=RespondArgs(message="prebuilt"))]
    assert (await flow2.run()).observation.message == "prebuilt"
    assert model2._calls == 0


@pytest.mark.asyncio
async def test_terminal_stops_remaining_actions():
    flow, _ = make_flow([response(FinishArgs(finish_reason="done"), RespondArgs(message="must not execute"))])
    node = await flow.run()
    assert node.action_steps[0].observation.terminal
    assert node.action_steps[1].observation is None
    assert node.is_terminal()
    assert not node.is_executed()  # legacy uses only the last ActionStep


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [response(), CompletionRejectError("invalid"), RejectError("rejected")])
async def test_rejection_finishes_as_terminal_with_error(failure):
    flow, model = make_flow([failure])
    node = await flow.run()
    assert node.error
    assert flow.is_finished() == "terminal"
    assert model._calls == 1


@pytest.mark.asyncio
async def test_model_exception_propagates_and_keeps_error():
    flow, _ = make_flow([ValueError("model unavailable")])
    with pytest.raises(ValueError, match="model unavailable"):
        await flow.run()
    assert "model unavailable" in flow.get_last_node().error


@pytest.mark.asyncio
async def test_duplicate_is_skipped_then_executed_on_next_loop_pass():
    flow, model = make_flow([response(RespondArgs(message="same"))])
    sibling = flow._create_next_node(flow.root)
    sibling.action_steps = [ActionStep(action=RespondArgs(message="same"))]
    current = flow._create_next_node(flow.root)
    result = await flow.run()
    assert result is current
    assert result.is_duplicate
    assert result.observation.message == "same"
    assert model._calls == 1


@pytest.mark.asyncio
async def test_non_action_responses_retry_same_node_and_overwrite_invocation():
    flow, model = make_flow([response(ResponseSchema()), response(ResponseSchema())], max_cost=0.02)
    with pytest.raises(RuntimeError, match="Script exhausted"):
        await flow.run()
    assert model._calls == 3  # two rejected-by-filter responses plus bounded abort
    assert len(flow.root.get_all_nodes()) == 2
    assert flow.total_usage().completion_cost == 0.01  # old build_action entry was overwritten


@pytest.mark.asyncio
async def test_tool_exception_propagates_with_model_usage(monkeypatch):
    flow, model = make_flow([response(RespondArgs(message="broken"))])

    async def fail(self, args, **kwargs):
        raise ValueError("tool failed")

    monkeypatch.setattr(Respond, "execute", fail)
    with pytest.raises(ValueError, match="tool failed"):
        await flow.run()
    assert flow.root.get_last_node().action_steps[0].observation is None
    assert flow.total_usage().completion_cost == 0.01
    assert model._calls == 1


@pytest.mark.asyncio
async def test_sparse_existing_ids_collide_in_legacy_expansion():
    flow, _ = make_flow([response(RespondArgs(message="done"))])
    node = flow.root.create_child()  # existing Node API assigns 2 to the first child
    node.action_steps = [ActionStep(action=ThinkArgs(thought="continue"))]
    await flow.run()
    assert [n.node_id for n in flow.root.get_all_nodes()] == [0, 2, 2]
