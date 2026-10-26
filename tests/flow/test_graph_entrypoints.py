from unittest.mock import AsyncMock, MagicMock

import pytest

from repoforge.actions.respond import Respond, RespondArgs
from repoforge.agent.agent import ActionAgent
from repoforge.completion.tool_call import ToolCallCompletionModel
from repoforge.evaluation.manager import EvaluationManager
from repoforge.evaluation.schema import Evaluation
from repoforge.flow import AgenticFlow, AgenticLoop, LangGraphFlow, OneOffFlow, SearchTree
from repoforge.flow.manager import FlowManager
from repoforge.flow.run_flow import handle_flow_event, persist_trajectory_data, setup_flow
from repoforge.events import FlowStartedEvent
from repoforge.node import Node
from repoforge.storage.file_storage import FileStorage
from repoforge.expander.expander import Expander
from repoforge.selector.simple import SimpleSelector
from scripts.migration_smoke import ScriptedCompletion, response


def config(flow_class=LangGraphFlow):
    agent = ActionAgent(
        system_prompt="fixed", completion_model=ToolCallCompletionModel(model="test-model"), actions=[Respond()]
    )
    options = {"selector": SimpleSelector(), "expander": Expander()} if flow_class is SearchTree else {}
    return flow_class(agent=agent, id="graph-test", **options).model_dump()


@pytest.mark.parametrize("flow_class", [LangGraphFlow, AgenticLoop, SearchTree, OneOffFlow])
def test_typed_config_discovery_roundtrip(flow_class):
    data = config(flow_class)
    result = AgenticFlow.from_config(data)
    assert type(result) is flow_class
    assert result.model_dump() == data
    assert AgenticFlow.get_component_by_classpath(data["flow_class"]) is flow_class


def test_untyped_historical_request_stays_search_tree():
    data = config(SearchTree)
    data.pop("flow_class")
    assert isinstance(AgenticFlow.from_config(data), SearchTree)


@pytest.mark.asyncio
async def test_manager_defaults_and_explicit_old_choice(tmp_path):
    storage = FileStorage(tmp_path)
    agent_manager, model_manager = MagicMock(), MagicMock()
    agent_manager.get_agent.side_effect = lambda **kw: ActionAgent(system_prompt="fixed", actions=[Respond()])
    model_manager.create_completion_model.side_effect = lambda *a: ToolCallCompletionModel(model="test-model")
    manager = FlowManager(
        runner=MagicMock(),
        storage=storage,
        eventbus=MagicMock(),
        agent_manager=agent_manager,
        model_manager=model_manager,
    )
    data = config()
    data.pop("flow_class")
    await manager.create_config(data)
    assert isinstance(await manager.get_flow_config("graph-test"), LangGraphFlow)
    await manager.update_config(data)
    assert isinstance(await manager.get_flow_config("graph-test"), LangGraphFlow)
    data["id"] = "legacy"
    data["flow_class"] = AgenticLoop.get_class_name()
    await manager.create_config(data)
    assert isinstance(await manager.get_flow_config("legacy"), AgenticLoop)
    graph = await manager.create_loop("graph", "project", "agent", "model", "issue")
    loop = await manager.create_loop(
        "loop", "project", "agent", "model", "issue", flow_class=AgenticLoop.get_class_name()
    )
    assert type(graph) is LangGraphFlow and type(loop) is AgenticLoop


@pytest.mark.asyncio
async def test_manager_to_runner_setup_loads_graph_and_exports_snapshots(tmp_path, monkeypatch):
    storage = FileStorage(tmp_path)
    manager = FlowManager(
        runner=MagicMock(), storage=storage, eventbus=MagicMock(), agent_manager=MagicMock(), model_manager=MagicMock()
    )
    await manager.create_flow(
        flow_config=AgenticFlow.from_config(config()), message="issue", project_id="p", trajectory_id="t"
    )
    import repoforge.settings as settings

    monkeypatch.setattr(settings, "get_storage", AsyncMock(return_value=storage))
    import importlib

    run_module = importlib.import_module("repoforge.flow.run_flow")
    monkeypatch.setattr(run_module, "get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(settings, "get_event_bus", AsyncMock(return_value=AsyncMock()))
    monkeypatch.setenv("REPOFORGE_DIR", str(tmp_path))
    loaded = await setup_flow("p", "t")
    assert type(loaded) is LangGraphFlow and loaded.checkpoint_path.endswith(".sqlite")
    model = ScriptedCompletion([response(RespondArgs(message="done"))])
    loaded.agent.completion_model = model
    await loaded.run()
    trajectory = await storage.read_from_trajectory("trajectory.json", "p", "t")
    assert trajectory["finish_reason"] == "terminal"
    assert trajectory["graph_state"]["snapshot_version"] == loaded.graph_state["snapshot_version"]
    restored = AgenticFlow.from_dicts(config(), trajectory)
    assert restored.graph_state["action_cursor"] == 1
    assert restored.get_trajectory_data()["nodes"] == trajectory["nodes"]


@pytest.mark.asyncio
async def test_actual_run_flow_entry_and_environment_error(tmp_path, monkeypatch):
    import importlib
    from repoforge.workspace import Workspace
    from repoforge.repository.repository import InMemRepository

    run_module = importlib.import_module("repoforge.flow.run_flow")
    storage = FileStorage(tmp_path)
    flow = LangGraphFlow.create(
        message="issue",
        agent=ActionAgent(
            system_prompt="fixed",
            actions=[Respond()],
            completion_model=ScriptedCompletion([response(RespondArgs(message="done"))]),
        ),
        project_id="p",
        trajectory_id="t",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_module, "get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(run_module, "setup_flow", AsyncMock(return_value=flow))
    monkeypatch.setattr(run_module, "setup_workspace", AsyncMock(return_value=Workspace(repository=InMemRepository())))
    events = AsyncMock()
    monkeypatch.setattr(run_module, "emit_event", events)
    await run_module.run_flow("p", "t", node_id=0)
    assert (await storage.read_from_trajectory("trajectory.json", "p", "t"))["finish_reason"] == "single_step"
    assert flow.root.get_last_node().terminal and flow.root.get_last_node().observation.message == "done"
    monkeypatch.setattr(run_module, "setup_workspace", AsyncMock(side_effect=ValueError("missing environment")))
    with pytest.raises(ValueError, match="missing environment"):
        await run_module.run_flow("p", "t", node_id=0)
    assert events.call_args.kwargs["data"]["error_kind"] == "environment"


@pytest.mark.asyncio
async def test_grading_error_is_separate_from_agent_failure(monkeypatch):
    import repoforge.evaluation.run_instance as runner
    from repoforge.runtime.runtime import NoEnvironment
    from repoforge.workspace import Workspace
    from repoforge.repository.repository import InMemRepository

    flow = LangGraphFlow.create(
        message="issue",
        agent=ActionAgent(
            system_prompt="fixed",
            actions=[Respond()],
            completion_model=ScriptedCompletion([response(RespondArgs(message="done"))]),
        ),
    )
    node = await flow.run(workspace=Workspace(repository=InMemRepository()))
    runtime = MagicMock(spec=NoEnvironment)
    runtime.swebench_evaluate = AsyncMock(side_effect=ValueError("grading unavailable"))
    events = AsyncMock()
    monkeypatch.setattr(runner, "_emit_event", events)
    with pytest.raises(ValueError, match="grading unavailable"):
        await runner.evaluate_instance("e", "i", flow, node, runtime, AsyncMock())
    assert flow.graph_state["error"] is None and node.evaluation_result is None
    assert events.call_args.kwargs["data"]["error_kind"] == "grading"


@pytest.mark.asyncio
async def test_swebench_runner_entry_loads_graph_with_stub_environment(tmp_path, monkeypatch):
    import repoforge.evaluation.run_instance as runner
    from repoforge.index.code_index import CodeIndex
    from repoforge.repository.repository import InMemRepository
    from repoforge.runtime.runtime import NoEnvironment
    from repoforge.node import ActionStep

    flow = AgenticFlow.from_config(config())
    flow._root = Node.create_root(user_message="issue")
    node = flow._create_next_node(flow.root)
    node.action_steps = [ActionStep(action=RespondArgs(message="done"))]
    runtime = NoEnvironment()
    runtime.swebench_evaluate = AsyncMock(return_value={"t": {"resolved": True, "stub_environment": True}})
    storage = FileStorage(tmp_path)
    import repoforge.settings as settings

    monkeypatch.setattr(settings, "get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(runner, "setup_flow", AsyncMock(return_value=flow))
    monkeypatch.setattr(runner, "setup_swebench_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(runner, "GitRepository", lambda **kw: InMemRepository())
    monkeypatch.setattr(CodeIndex, "from_persist_dir", MagicMock(return_value=None))
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    from pathlib import Path

    monkeypatch.setattr(runner, "Path", lambda p: tmp_path / Path(p).name)
    persist = AsyncMock()
    monkeypatch.setattr(runner, "persist_trajectory_data", persist)
    monkeypatch.setattr(runner, "_emit_event", AsyncMock())
    monkeypatch.setenv("REPO_PATH", str(tmp_path))
    monkeypatch.setenv("INDEX_STORE_DIR", str(tmp_path))
    await runner.run_swebench_instance("p", "t")
    assert type(flow) is LangGraphFlow
    assert flow.graph_state["finish_reason"] == "terminal"
    assert node.evaluation_result.details["stub_environment"] is True
    assert persist.await_count >= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retry", "resume", "execute", "reset"])
async def test_explicit_manager_changes_get_fresh_run_identity(tmp_path, monkeypatch, operation):
    import repoforge.settings as settings
    from repoforge.node import ActionStep, Node
    from repoforge.flow.schema import StartTrajectoryRequest

    storage = FileStorage(tmp_path)
    monkeypatch.setattr(settings, "get_storage", AsyncMock(return_value=storage))
    flow = AgenticFlow.from_config(config())
    flow.project_id, flow.trajectory_id = "p", "t"
    flow._root = Node.create_root(user_message="issue")
    node = flow._create_next_node(flow.root)
    node.action_steps = [ActionStep(action=RespondArgs(message="done"))]
    await flow.run()
    manager = FlowManager(
        runner=AsyncMock(), storage=storage, eventbus=MagicMock(), agent_manager=MagicMock(), model_manager=MagicMock()
    )
    await manager.save_trajectory("p", "t", flow)
    if operation == "retry":
        await manager.retry_trajectory("p", "t")
    elif operation == "resume":
        await manager.resume_trajectory("p", "t", StartTrajectoryRequest(agent_id="a", model_id="m", message="more"))
    elif operation == "execute":
        await manager.execute_node("p", "t", node.node_id)
    else:
        await manager.reset_node("p", "t", node.node_id)
    loaded = await manager.get_flow("p", "t")
    assert loaded.checkpoint_run_id != flow.checkpoint_run_id
    assert loaded.graph_state is None
    if operation == "retry":
        assert not loaded.root.children
    elif operation == "resume":
        assert loaded.root.get_last_node().user_message == "more"
    elif operation == "execute":
        assert len(loaded.root.children) == 2
        assert not loaded.root.get_last_node().is_executed()
    else:
        assert not loaded.root.get_last_node().action_steps


@pytest.mark.asyncio
async def test_evaluation_schema_and_update_keep_typed_flow(tmp_path):
    storage = FileStorage(tmp_path)
    manager = EvaluationManager(runner=MagicMock(), storage=storage, eventbus=MagicMock(), flow_manager=MagicMock())
    await manager.update_config("evaluation", config())
    data = await storage.read_from_project("flow.json", "evaluation")
    assert type(AgenticFlow.from_config(data)) is LangGraphFlow
    evaluation = Evaluation(evaluation_name="e", dataset_name="test", instances=[], flow=AgenticFlow.from_config(data))
    assert type(Evaluation.model_validate(evaluation.model_dump()).flow) is LangGraphFlow


@pytest.mark.asyncio
async def test_legacy_event_save_is_awaited_and_storage_retry_is_bounded(monkeypatch):
    import repoforge.settings as settings
    import importlib

    run_module = importlib.import_module("repoforge.flow.run_flow")
    storage = AsyncMock()
    storage.write_to_trajectory.side_effect = [OSError("temporary"), OSError("temporary"), None]
    monkeypatch.setattr(settings, "get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(run_module, "get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(settings, "get_event_bus", AsyncMock(return_value=AsyncMock()))
    flow = AgenticLoop.create(message="issue", agent=AgenticFlow.from_config(config()).agent)
    await handle_flow_event(flow, FlowStartedEvent())
    assert storage.write_to_trajectory.await_count == 3
    storage.write_to_trajectory.reset_mock(side_effect=True)
    storage.write_to_trajectory.side_effect = OSError("unavailable")
    with pytest.raises(OSError, match="unavailable"):
        await persist_trajectory_data(flow)
    assert storage.write_to_trajectory.await_count == 3
