import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import litellm

from repoforge.completion.log_handler import LogHandler
from repoforge.context_data import current_project_id, current_trajectory_id
from repoforge.environment.local import LocalBashEnvironment
from repoforge.events import BaseEvent
from repoforge.flow.flow import AgenticFlow
from repoforge.flow.langgraph_flow import LangGraphFlow
from repoforge.flow.durable import RecoveryConflict, default_checkpoint_path
from repoforge.index.code_index import CodeIndex
from repoforge.repository.git import GitRepository
from repoforge.runner.utils import (
    cleanup_job_logging,
    emit_event,
    setup_job_logging,
)
from repoforge.runtime.runtime import RuntimeEnvironment
from repoforge.settings import get_storage
from repoforge.workspace import Workspace

logger = logging.getLogger(__name__)


_flow_lock = asyncio.Lock()


async def setup_flow(project_id: str, trajectory_id: str) -> AgenticFlow:
    from repoforge.settings import get_storage

    storage = await get_storage()

    current_project_id.set(project_id)
    current_trajectory_id.set(trajectory_id)

    litellm.callbacks = [LogHandler(storage=storage)]

    logger.info(f"setup_flow(project_id: {project_id}, trajectory_id: {trajectory_id})")

    try:
        settings: dict | None = await storage.read_from_trajectory(
            path="flow.json", trajectory_id=trajectory_id, project_id=project_id
        )  # type: ignore
    except Exception:
        settings = await storage.read_from_project(path="flow.json", project_id=project_id)  # type: ignore

        logger.info("Settings not found in trajectory, using project settings")
    else:
        logger.info("Settings found in trajectory")

    trajectory_dict: dict | None = await storage.read_from_trajectory(
        path="trajectory.json", trajectory_id=trajectory_id, project_id=project_id
    )  # type: ignore
    flow = AgenticFlow.from_dicts(settings=settings, trajectory=trajectory_dict)

    async def on_event(event: BaseEvent) -> None:
        await handle_flow_event(flow, event)

    flow._on_event = on_event
    if isinstance(flow, LangGraphFlow):
        if not flow.checkpoint_path:
            flow.checkpoint_path = default_checkpoint_path(project_id, trajectory_id, flow.checkpoint_run_id)
        flow._on_snapshot = persist_trajectory_data

    return flow


async def setup_swebench_runtime() -> RuntimeEnvironment:
    storage = await get_storage()
    from repoforge.runtime.local import SweBenchLocalEnvironment

    instance_path = os.environ.get("INSTANCE_PATH")
    if not instance_path:
        raise ValueError("INSTANCE_PATH is not set")

    if not os.path.exists(instance_path):
        raise FileNotFoundError(f"Instance path {instance_path} does not exist")

    with open(instance_path) as f:
        swebench_instance = json.loads(f.read())

    try:
        return SweBenchLocalEnvironment(
            repo_path=Path("/testbed"),
            swebench_instance=swebench_instance,
            storage=storage,
        )
    except Exception as e:
        logger.error(f"Error setting up SWEbench runtime with instance {swebench_instance}")
        raise e


async def setup_workspace(project_id: str, trajectory_id: str) -> Workspace:
    logger.info(f"setup_workspace(project_id: {project_id}, trajectory_id: {trajectory_id})")

    storage = await get_storage()
    instance_path = os.environ.get("INSTANCE_PATH")
    repo_path = None
    runtime = None
    if instance_path:
        runtime = await setup_swebench_runtime()
        repo_path = "/testbed"
    elif await storage.exists(f"projects/{project_id}/settings.json"):
        project_settings = await storage.read(f"projects/{project_id}/settings.json")
        repo_path = project_settings["repository_path"]
        runtime = None

    elif os.environ.get("REPO_PATH"):
        # TODO: Use Local bash environment
        runtime = None
        repo_path = os.environ.get("REPO_PATH")

    logger.info(f"repo_path: {repo_path}")

    if repo_path:
        repository = GitRepository(repo_path=repo_path, shadow_mode=True)
        environment = LocalBashEnvironment(cwd=repo_path)
    else:
        repository = None
        environment = None

    if os.environ.get("INDEX_STORE_DIR"):
        index_store_dir = os.environ.get("INDEX_STORE_DIR")
    else:
        index_store_dir = None

    if index_store_dir:
        logger.info(f"Using index store dir: {index_store_dir}")
        code_index = CodeIndex.from_persist_dir(
            persist_dir=index_store_dir,
            file_repo=repository,
        )
    else:
        code_index = None
        logger.info("No index store dir provided, skipping code index")

    return Workspace(
        repository=repository,
        code_index=code_index,
        runtime=runtime,
        environment=environment,
        storage=storage,
    )


async def persist_trajectory_data(flow: AgenticFlow) -> None:
    storage = await get_storage()

    if isinstance(flow, LangGraphFlow):
        flow.persist_snapshot()
    trajectory_data = flow.get_trajectory_data()
    # Retrying this exact overwrite is idempotent; external model/tool calls are
    # never retried here. Bound transient storage failures to three attempts.
    for attempt in range(3):
        try:
            await storage.write_to_trajectory("trajectory.json", trajectory_data, flow.project_id, flow.trajectory_id)
            break
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            await asyncio.sleep(0.05 * (attempt + 1))
    logger.info(
        f"Trajectory data with {len(trajectory_data['nodes'])} nodes written to {flow.project_id}/{flow.trajectory_id}/trajectory.json"
    )


async def handle_flow_event(flow: AgenticFlow, event: BaseEvent) -> None:
    """Handle flow events by persisting data and publishing to event bus.

    Args:
        flow: The flow instance that generated the event
    """
    from repoforge.settings import get_event_bus

    try:
        event_bus = await get_event_bus()
        await event_bus.publish(event)
    except Exception as e:
        logger.error(f"Error publishing event: {e}")

    # Graph snapshots are exported by the awaited callback after node commit.
    # Events inside an external operation may describe effects not yet committed.
    if not isinstance(flow, LangGraphFlow):
        if (event.scope == "node" and event.event_type == "expanded") or event.scope == "flow":
            async with _flow_lock:
                await persist_trajectory_data(flow)


async def run_flow(project_id: str, trajectory_id: str, node_id: int | None = None) -> None:
    """Run an instance's agentic flow."""
    print(f"Running instance {trajectory_id} with node_id {node_id} for project {project_id}")

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(f"./logs/logs_{date_str}.log")
    original_handlers = setup_job_logging(log_path=log_path)
    storage = await get_storage()
    flow = None
    error_kind = "environment"

    try:
        flow = await setup_flow(project_id, trajectory_id)

        if node_id is None and flow.is_finished():
            logger.warning(f"Flow already finished for instance {trajectory_id}")
            return None

        workspace = await setup_workspace(project_id, trajectory_id)

        logger.info(f"Flow created for instance {trajectory_id}")

        error_kind = "flow"
        await flow.run(workspace=workspace, node_id=node_id)

        logger.info(f"Flow completed for instance {trajectory_id}")

    except Exception as e:
        logger.exception(f"Error running instance {trajectory_id}")
        if isinstance(e, RecoveryConflict):
            error_kind = "recovery_conflict"
        elif error_kind == "flow" and isinstance(flow, LangGraphFlow) and flow.graph_state:
            error_kind = flow.graph_state.get("error_kind") or error_kind
        await emit_event(
            evaluation_name=project_id,
            instance_id=trajectory_id,
            scope="flow",
            event_type="error",
            data={"error": str(e), "error_kind": error_kind},
        )
        raise e
    finally:
        try:
            if flow is not None and not (
                isinstance(flow, LangGraphFlow) and (flow._recovery_conflict or flow._unconfirmed_operation)
            ):
                await persist_trajectory_data(flow)
        finally:
            cleanup_job_logging(original_handlers)
            trajectory_key = storage.get_trajectory_path(project_id=project_id, trajectory_id=trajectory_id)
            await storage.write_raw(f"{trajectory_key}/logs/{log_path.name}", log_path.read_text())
