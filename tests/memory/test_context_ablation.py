import pytest

from repoforge.actions.respond import Respond
from repoforge.agent.agent import ActionAgent
from repoforge.message_history.managed import ManagedMessageHistoryGenerator
from repoforge.message_history.policy import ContextPolicy
from repoforge.node import Node
from repoforge.task_state import TaskState
from repoforge.workspace import Workspace
from scripts.migration_smoke import ScriptedCompletion


@pytest.mark.asyncio
async def test_task_state_ablation_keeps_budgeting_and_read_tool_without_note_prompt():
    model = ScriptedCompletion([], max_tokens=512)
    memory = ManagedMessageHistoryGenerator(policy=ContextPolicy(context_window=8192, enable_task_state=False))
    agent = ActionAgent(completion_model=model, actions=[Respond()], system_prompt="Fix issue", memory=memory)
    assert "ReadObservation" in [action.name for action in agent.actions]
    assert "UpdateTaskState" not in [action.name for action in agent.actions]
    assert "# Task context and memory" not in agent.system_prompt
    memory.bind_model(model)
    node = Node(node_id=0, user_message="Fix issue", task_state=TaskState(next_step="Ablated note"))
    messages = await memory.generate_messages(node, Workspace())
    assert "task_state" not in str(messages)
    assert node.task_state.next_step == "Ablated note"
    assert node.context_build["input_budget"] > 0
    restored = ActionAgent.from_dict({key: value for key, value in agent.model_dump().items() if key != "completion_model"})
    assert restored.memory.policy.enable_task_state is False
    assert "UpdateTaskState" not in [action.name for action in restored.actions]
