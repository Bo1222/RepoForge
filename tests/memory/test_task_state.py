import pytest
from pydantic import ValidationError

from repoforge.actions.respond import RespondArgs
from repoforge.actions.schema import Observation
from repoforge.actions.task_state import UpdateTaskState, UpdateTaskStateArgs
from repoforge.flow.graph_state import initial_state, restore_nodes
from repoforge.node import ActionStep, Node
from repoforge.task_state import EvidenceRef, Hypothesis, TaskState


def setup():
    root = Node(node_id=0, user_message="Empty input crashes")
    node = root.create_child()
    node.action_steps = [ActionStep(action=RespondArgs(message="reproduce"), observation=Observation(message="Crash reproduced"))]
    return root, node


@pytest.mark.asyncio
async def test_notes_update_and_evidence_preserved_in_snapshot():
    root, node = setup()
    update = UpdateTaskState()
    outcome = await update.execute_for_node(UpdateTaskStateArgs(
        actual_behavior="Empty input crashes", expected_behavior="Return empty result", phase="investigation",
        hypotheses=[Hypothesis(id="H1", statement="Missing empty input guard", status="supported",
                               evidence=[EvidenceRef(node_id=node.node_id, step_index=0)])],
        next_step="Inspect guard and run regression test",
    ), node)
    assert not outcome.properties.get("fail_reason")
    assert node.task_state.revision == 1
    assert node.action_steps[0].observation.message == "Crash reproduced"
    restored = restore_nodes(initial_state(root, "project", "issue", node.node_id))
    assert restored.get_node_by_id(node.node_id).task_state == node.task_state


@pytest.mark.asyncio
async def test_bad_evidence_rejects_without_state_mutation():
    root, node = setup()
    sibling = root.create_child()
    sibling.action_steps = [ActionStep(action=RespondArgs(message="other"), observation=Observation(message="Other branch"))]
    tool = UpdateTaskState()
    for evidence in ([], [EvidenceRef(node_id=sibling.node_id, step_index=0)],
                     [EvidenceRef(node_id=node.node_id, step_index=10)]):
        outcome = await tool.execute_for_node(UpdateTaskStateArgs(hypotheses=[Hypothesis(
            id="H1", statement="Claim", status="supported", evidence=evidence,
        )]), node)
        assert outcome.properties["fail_reason"] == "invalid_task_evidence"
        assert node.task_state is None


def test_task_state_cannot_write_execution_facts_and_version_is_checked():
    with pytest.raises(ValidationError):
        UpdateTaskStateArgs(return_code=0, tests_passed=True)
    with pytest.raises(ValidationError):
        TaskState(version=2)


@pytest.mark.asyncio
async def test_branch_notes_are_independent_and_updates_do_not_erase_unmentioned_fields():
    root, node = setup()
    await UpdateTaskState().execute_for_node(UpdateTaskStateArgs(expected_behavior="Return empty result"), node)
    left = node.create_child()
    right = node.create_child()
    await UpdateTaskState().execute_for_node(UpdateTaskStateArgs(next_step="Write guard"), left)
    assert left.task_state.expected_behavior == "Return empty result"
    assert right.task_state.next_step is None and node.task_state.next_step is None
    assert left.task_state.revision == 2 and right.task_state.revision == 1
    legacy = Node.from_dict({"nodes": [{"node_id": 0, "user_message": "Old issue"}]})
    assert legacy.task_state is None


@pytest.mark.asyncio
async def test_task_update_is_a_real_agent_action_and_prompt_load_is_idempotent():
    from repoforge.agent.agent import ActionAgent
    from repoforge.actions.respond import Respond
    from repoforge.message_history.managed import ManagedMessageHistoryGenerator
    from repoforge.message_history.policy import ContextPolicy
    from repoforge.workspace import Workspace
    from repoforge.repository.repository import InMemRepository
    from scripts.migration_smoke import ScriptedCompletion, response

    model = ScriptedCompletion([response(UpdateTaskStateArgs(next_step="Investigate"))])
    agent = ActionAgent(completion_model=model, actions=[Respond()], system_prompt="Fix issue",
                        memory=ManagedMessageHistoryGenerator(policy=ContextPolicy(context_window=8192)))
    workspace = Workspace(repository=InMemRepository())
    await agent.initialize(workspace)
    root = Node(node_id=0, user_message="Fix issue")
    node = root.create_child()
    await agent.generate_actions(node)
    await agent.execute_action_step(node, 0)
    assert node.task_state.next_step == "Investigate"
    assert node.action_steps[0].observation.properties["task_state_revision"] == 1
    dump = agent.model_dump()
    # ScriptedCompletion is intentionally not a persistent configured model.
    dump.pop("completion_model")
    loaded = ActionAgent.from_dict(dump)
    assert loaded.system_prompt.count("# Task context and memory") == 1
    assert len([a for a in loaded.actions if isinstance(a, UpdateTaskState)]) == 1
