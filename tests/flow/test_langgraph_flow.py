import json
from pathlib import Path

import pytest
from langgraph.errors import GraphRecursionError

from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.respond import Respond, RespondArgs
from moatless.actions.schema import Observation
from moatless.actions.think import Think, ThinkArgs
from moatless.agent.agent import ActionAgent
from moatless.completion.schema import ResponseSchema
from moatless.context_data import current_action_step, current_node_id
from moatless.exceptions import CompletionError, CompletionRejectError, RejectError
from moatless.flow.langgraph_flow import LangGraphFlow
from moatless.flow.loop import AgenticLoop
from moatless.flow.schema import FlowStatus
from moatless.node import ActionStep
from scripts.migration_smoke import ScriptedCompletion, replay, response

BASELINE = Path(__file__).resolve().parents[2] / "experiments/baselines/M0"


def make_flow(responses, flow_class=LangGraphFlow, **kwargs):
    model = ScriptedCompletion(responses)
    agent = ActionAgent(completion_model=model, system_prompt="fixed", actions=[Think(), Respond(), Finish()])
    return flow_class.create(message="issue", agent=agent, project_id="p", trajectory_id="t", **kwargs), model


def align_event_ids(result):
    # D06 is an intentional metadata addition in M1, checked separately below.
    result = json.loads(json.dumps(result))
    for event in result["events"]:
        if event["scope"] in {"agent", "action"}:
            event["project_id"] = None
            event["trajectory_id"] = None
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("coding", [False, True])
async def test_m0_m1_frozen_business_replay(coding):
    result, _ = await replay(LangGraphFlow, coding=coding)
    expected = json.loads((BASELINE / ("coding.json" if coding else "respond.json")).read_text())
    assert align_event_ids(result) == expected
    assert all(
        e["project_id"] == "migration" and e["trajectory_id"] == ("coding" if coding else "respond")
        for e in result["events"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_iterations,max_cost,expected",
    [(1, None, "max_iterations"), (2, None, "max_iterations"), (10, 0.01, "max_cost"), (10, 0, "terminal")],
)
async def test_loop_budget_boundaries_and_reason_precedence(max_iterations, max_cost, expected):
    outputs = [response(RespondArgs(message="done"), cost=0.02)]
    graph, graph_model = make_flow(outputs, max_iterations=max_iterations, max_cost=max_cost)
    loop, loop_model = make_flow(outputs, AgenticLoop, max_iterations=max_iterations, max_cost=max_cost)
    graph_node, loop_node = await graph.run(), await loop.run()
    assert graph_node.node_id == loop_node.node_id
    assert graph.is_finished() == loop.is_finished() == expected
    assert graph_model._calls == loop_model._calls
    assert graph.total_usage() == loop.total_usage()


@pytest.mark.asyncio
async def test_multiple_actions_are_graph_steps_not_iterations():
    flow, model = make_flow(
        [response(ThinkArgs(thought="a"), ThinkArgs(thought="b"), FinishArgs(finish_reason="done"))], max_iterations=3
    )
    node = await flow.run()
    assert len(flow.root.get_all_nodes()) == 2
    assert len(node.action_steps) == 3
    assert all(step.observation for step in node.action_steps)
    assert model._calls == 1
    assert flow.graph_state["action_cursor"] == 3
    assert flow.is_finished() == "terminal"
    # The compiled graph itself retained the completed business/control snapshot.
    checkpoint = await flow._compiled_graph.aget_state(flow._graph_config)
    assert checkpoint.values == flow.graph_state
    assert {"generate_actions", "policy_decision", "execute_action", "record_observation"} <= set(
        flow._compiled_graph.get_graph().nodes
    )


@pytest.mark.asyncio
async def test_terminal_leaves_remaining_actions_unexecuted():
    flow, model = make_flow([response(FinishArgs(finish_reason="done"), ThinkArgs(thought="unused"))])
    node = await flow.run()
    assert node.action_steps[1].observation is None
    assert node.terminal
    assert flow.graph_state["action_cursor"] == 1
    assert flow.graph_state["finish_reason"] == "terminal"
    assert model._calls == 1
    # Re-running a terminal trajectory makes no additional model/tool call.
    await flow.run()
    assert model._calls == 1


@pytest.mark.asyncio
async def test_message_continuation_and_single_step_budget_bypass():
    flow, model = make_flow([response(ThinkArgs(thought="a")), response(RespondArgs(message="done"))], max_iterations=1)
    node = flow._create_next_node(flow.root)
    first = await flow.run(node_id=node.node_id)
    assert first is node
    assert flow.graph_state["finish_reason"] == "single_step"
    # A one-round return alone is not a completed trajectory (raise the old budget).
    flow.max_iterations = 10
    assert flow.status == FlowStatus.RUNNING
    second = await flow.run(message="continue")
    assert second.parent is first
    assert second.user_message == "continue"
    assert model._calls == 2
    assert flow.graph_state["finish_reason"] == "terminal"


@pytest.mark.asyncio
async def test_explicit_zero_node_id_and_positive_executed_expansion():
    flow, model = make_flow([response(ThinkArgs(thought="a")), response(RespondArgs(message="done"))], max_iterations=1)
    node = await flow.run(node_id=0)
    assert node.node_id == 1
    assert flow.graph_state["finish_reason"] == "single_step"
    next_node = await flow.run(node_id=node.node_id)
    assert next_node.parent is node
    assert model._calls == 2


@pytest.mark.asyncio
async def test_prebuilt_partial_response_does_not_repeat_tools_or_completion():
    flow, model = make_flow([])
    node = flow._create_next_node(flow.root)
    node.action_steps = [
        ActionStep(action=ThinkArgs(thought="completed"), observation=Observation.create(message="already done")),
        ActionStep(action=RespondArgs(message="pending")),
    ]
    events = []

    async def capture(event):
        events.append(
            (
                event.event_type,
                getattr(event, "action_name", None),
                current_action_step.get(),
                current_node_id.get(),
                event.project_id,
                event.trajectory_id,
            )
        )

    flow._on_event = capture
    token = current_node_id.set(99)
    try:
        result = await flow.run()
        assert current_node_id.get() == 99
    finally:
        current_node_id.reset(token)
    assert result.action_steps[0].observation.message == "already done"
    assert result.action_steps[1].observation.message == "pending"
    assert model._calls == 0
    executed = [e for e in events if e[0] == "executed"]
    assert executed == [("executed", "SendMessage", 1, node.node_id, "p", "t")]


@pytest.mark.asyncio
async def test_duplicate_is_an_explicit_bounded_stop():
    flow, model = make_flow([response(RespondArgs(message="same"))])
    sibling = flow._create_next_node(flow.root)
    sibling.action_steps = [ActionStep(action=RespondArgs(message="same"))]
    flow._create_next_node(flow.root)
    node = await flow.run()
    assert node.is_duplicate
    assert node.observation is None
    assert flow.is_finished() == "duplicate"
    assert model._calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,reason,kind",
    [
        (response(), "no_actions", "model_rejection"),
        (response(ResponseSchema()), "no_actions", "model_rejection"),
        (
            CompletionRejectError("invalid", completion_invocation=response().completion_invocation),
            "generation_error",
            "model_rejection",
        ),
        (ValueError("model down"), "generation_error", "model"),
    ],
)
async def test_generation_failure_exits_once_with_snapshot(output, reason, kind):
    events = []

    async def capture(event):
        events.append((event.scope, event.event_type))

    flow, model = make_flow([output], on_event=capture)
    with pytest.raises((RejectError, ValueError)):
        await flow.run()
    assert model._calls == 1
    assert flow.graph_state["finish_reason"] == reason
    assert flow.graph_state["error_kind"] == kind
    assert flow.status == FlowStatus.ERROR
    assert events[-1] == ("flow", "error")
    data = flow.get_trajectory_data()
    assert data["nodes"] == flow.graph_state["business_snapshot"]
    assert data["finish_reason"] == reason
    if isinstance(output, CompletionRejectError):
        assert flow.total_usage().completion_cost == 0.01


@pytest.mark.asyncio
async def test_tool_failure_retains_completed_steps_usage_and_events(monkeypatch):
    flow, model = make_flow([response(ThinkArgs(thought="done"), RespondArgs(message="broken"))])

    async def fail(self, args, **kwargs):
        raise ValueError("tool failed")

    monkeypatch.setattr(Respond, "execute", fail)
    with pytest.raises(ValueError, match="tool failed"):
        await flow.run()
    node = flow.root.get_last_node()
    assert node.action_steps[0].observation
    assert node.action_steps[1].observation is None
    assert flow.graph_state["action_cursor"] == 1
    assert flow.graph_state["error_kind"] == "tool"
    assert flow.total_usage().completion_cost == 0.01
    assert model._calls == 1
    assert current_action_step.get() is None


@pytest.mark.asyncio
async def test_graph_recursion_limit_is_not_agent_iteration_limit():
    flow, model = make_flow([response(RespondArgs(message="done"))], graph_recursion_limit=1)
    with pytest.raises(GraphRecursionError):
        await flow.run()
    assert flow.graph_state["finish_reason"] == "graph_recursion_limit"
    assert flow.graph_state["error_kind"] == "graph_recursion"
    assert model._calls == 0


@pytest.mark.asyncio
async def test_missing_node_raises_without_model_call():
    flow, model = make_flow([])
    with pytest.raises(ValueError, match="not found"):
        await flow.run(node_id=999)
    assert model._calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,kind",
    [
        (CompletionError("tool model down", completion_invocation=response().completion_invocation), "tool_model"),
        (
            CompletionRejectError("tool model invalid", completion_invocation=response().completion_invocation),
            "tool_rejection",
        ),
    ],
)
async def test_tool_completion_failure_keeps_invocation_without_double_charge(monkeypatch, failure, kind):
    flow, model = make_flow([response(RespondArgs(message="broken"))])

    async def fail(self, args, **kwargs):
        raise failure

    monkeypatch.setattr(Respond, "execute", fail)
    with pytest.raises((CompletionError, RejectError)):
        await flow.run()
    assert flow.graph_state["error_kind"] == kind
    assert flow.total_usage().completion_cost == 0.02  # generation + tool model
    assert model._calls == 1
    assert flow.root.get_last_node().action_steps[0].completion == failure.completion_invocation


@pytest.mark.asyncio
async def test_graph_rebinds_replacement_workspace():
    from moatless.repository.repository import InMemRepository
    from moatless.runtime.runtime import NoEnvironment
    from moatless.workspace import Workspace

    flow, _ = make_flow([response(ThinkArgs(thought="first")), response(RespondArgs(message="done"))])
    first_repo, second_repo = InMemRepository(), InMemRepository()
    first_runtime, second_runtime = NoEnvironment(), NoEnvironment()
    node = await flow.run(node_id=0, workspace=Workspace(repository=first_repo, runtime=first_runtime))
    assert node.file_context.repository is first_repo
    result = await flow.run(workspace=Workspace(repository=second_repo, runtime=second_runtime))
    assert result.file_context.repository is second_repo
    assert result.file_context._runtime is second_runtime


@pytest.mark.asyncio
async def test_sparse_existing_ids_expand_without_collision():
    flow, _ = make_flow([response(RespondArgs(message="done"))])
    node = flow.root.create_child()
    node.action_steps = [ActionStep(action=ThinkArgs(thought="continue"))]
    await flow.run()
    assert [n.node_id for n in flow.root.get_all_nodes()] == [0, 2, 3]
