import copy
import json

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from repoforge.actions.string_replace import StringReplace, StringReplaceArgs
from repoforge.agent.agent import ActionAgent
from repoforge.flow.graph_state import initial_state, restore_nodes, snapshot_nodes
from repoforge.node import Node
from repoforge.repository.repository import InMemRepository
from repoforge.runtime.runtime import NoEnvironment
from repoforge.workspace import Workspace
from scripts.migration_smoke import ScriptedCompletion, response


@pytest.mark.asyncio
async def test_json_roundtrip_rebinds_patch_runtime_usage_and_does_not_mutate():
    repo = InMemRepository(files={"calc.py": "def add(a, b):\n    return a - b\n"})
    runtime = NoEnvironment()
    workspace = Workspace(repository=repo, runtime=runtime)
    model = ScriptedCompletion(
        [response(StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b"))]
    )
    agent = ActionAgent(system_prompt="fixed", completion_model=model, actions=[StringReplace()])
    await agent.initialize(workspace)
    root = Node.create_root("fix")
    root.file_context.repository = repo
    root.file_context._runtime = runtime
    root.file_context.add_file("calc.py", show_all_spans=True)
    node = root.create_child()
    await agent.run(node)
    state = initial_state(root, "p", "t", node.node_id)
    state["action_cursor"] = 1
    raw = json.loads(json.dumps(state))
    original = copy.deepcopy(raw)
    restored = restore_nodes(raw, workspace)
    restored_again = restore_nodes(raw, workspace)
    assert raw == original
    assert restored.get_last_node().file_context.generate_git_patch() == node.file_context.generate_git_patch()
    assert restored.get_last_node().file_context.repository is repo
    assert restored.get_last_node().file_context._runtime is runtime
    assert restored.total_usage() == root.total_usage()
    assert snapshot_nodes(restored) == snapshot_nodes(restored_again)
    assert "return a + b" in restored.get_last_node().file_context.get_file("calc.py").content
    with pytest.raises(ValueError, match="schema"):
        restore_nodes({**raw, "schema_version": 99}, workspace)
    with pytest.raises(ValueError, match="cursor"):
        restore_nodes({**raw, "action_cursor": 9}, workspace)


@pytest.mark.asyncio
async def test_async_sqlite_checkpoint_install_smoke(tmp_path):
    class State(TypedDict):
        value: int

    builder = StateGraph(State)
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    config = {"configurable": {"thread_id": "install-smoke"}}
    db = str(tmp_path / "checkpoint.sqlite")
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = builder.compile(checkpointer=saver)
        assert await graph.ainvoke({"value": 1}, config) == {"value": 2}
        assert (await graph.aget_state(config)).values == {"value": 2}
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        assert (await saver.aget_tuple(config)).checkpoint["channel_values"]["value"] == 2
