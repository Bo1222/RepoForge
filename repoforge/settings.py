import logging
import os

from dotenv import load_dotenv

from litellm import Type
from repoforge.eventbus.base import BaseEventBus
from repoforge.eventbus.local_bus import LocalEventBus
from repoforge.runner.asyncio_runner import AsyncioRunner
from repoforge.runner.runner import BaseRunner
from repoforge.runner.scheduler import SchedulerRunner
from repoforge.storage.base import BaseStorage
from repoforge.storage.file_storage import FileStorage

logger = logging.getLogger(__name__)

_storage = None
_event_bus = None
_runner = None

model_manager = None
agent_manager = None
flow_manager = None
evaluation_manager = None


async def initialize_managers():
    """Initialize all manager instances asynchronously by calling their get_instance() methods.

    This function simply accesses the singleton instances, ensuring they're created.
    The actual singleton management is handled by the classes themselves.
    """
    from repoforge.completion.manager import ModelConfigManager
    from repoforge.agent.manager import AgentConfigManager
    from repoforge.flow.manager import FlowManager
    from repoforge.evaluation.manager import EvaluationManager

    global model_manager, agent_manager, flow_manager, evaluation_manager

    event_bus = await get_event_bus()
    runner = await get_runner()
    storage = await get_storage()

    try:
        logger.info("Initializing manager singletons...")

        model_manager = ModelConfigManager(storage=storage)
        agent_manager = AgentConfigManager(storage=storage)
        await model_manager.initialize()
        await agent_manager.initialize()

        if event_bus is None or runner is None:
            raise ValueError("Event bus or runner is not initialized")

        flow_manager = FlowManager(
            storage=storage,
            eventbus=event_bus,
            runner=runner,
            agent_manager=agent_manager,
            model_manager=model_manager,
        )
        await flow_manager.initialize()

        evaluation_manager = EvaluationManager(
            storage=storage,
            eventbus=event_bus,
            runner=runner,
            flow_manager=flow_manager,
        )
        await evaluation_manager.initialize()

        logger.info("All manager singletons accessed successfully")
    except Exception as e:
        logger.exception(f"Error initializing manager singletons: {e}")
        raise e


async def ensure_managers_initialized():
    """Ensure the manager singletons are initialized."""
    global model_manager, agent_manager, flow_manager, evaluation_manager
    if model_manager is None or agent_manager is None or flow_manager is None or evaluation_manager is None:
        await initialize_managers()


async def get_agent_manager():
    """Get the agent manager instance, ensuring it's initialized."""

    await ensure_managers_initialized()

    if agent_manager is None:
        raise RuntimeError("Agent manager not initialized")

    return agent_manager


async def get_model_manager():
    """Get the model manager instance, ensuring it's initialized."""
    await ensure_managers_initialized()

    if model_manager is None:
        raise RuntimeError("Model manager not initialized")

    return model_manager


async def get_flow_manager():
    """Get the flow manager instance, ensuring it's initialized."""
    await ensure_managers_initialized()

    if flow_manager is None:
        raise RuntimeError("Flow manager not initialized")

    return flow_manager


async def get_event_bus() -> BaseEventBus:
    """Get the event bus instance."""
    global _event_bus

    load_dotenv()

    if _event_bus is None:
        storage = await get_storage()
        if os.environ.get("REDIS_URL"):
            logger.info(f"Use Redis Event Bus with redis url: {os.environ.get('REDIS_URL')}")
            try:
                from repoforge.eventbus.redis_bus import RedisEventBus

                _event_bus = RedisEventBus(redis_url=os.environ.get("REDIS_URL"), storage=storage)
            except Exception as e:
                logger.error(f"Failed to initialize event bus and runner. Using local event bus instead. Error: {e}")
                _event_bus = LocalEventBus(storage=storage)
        else:
            logger.info("Use Local Event Bus")
            _event_bus = LocalEventBus(storage=storage)

    await _event_bus.initialize()

    return _event_bus


async def get_runner():
    """Get the runner instance."""
    global _runner
    load_dotenv()
    runner_type = os.environ.get("REPOFORGE_RUNNER")
    if runner_type == "docker":
        raise ValueError(
            "REPOFORGE_RUNNER=docker is no longer supported: the custom Docker runner was removed. "
            "Use REPOFORGE_RUNNER=asyncio for local execution, or kubernetes. "
            "Run the official SWE-bench Harness separately for grading."
        )
    if _runner is None:
        runner_impl: Type[BaseRunner]
        if runner_type == "kubernetes":
            from repoforge.runner.kubernetes_runner import KubernetesRunner

            runner_impl = KubernetesRunner
        else:
            logger.info("Use Local Runner")
            runner_impl = AsyncioRunner

        if os.environ.get("REDIS_URL"):
            _runner = SchedulerRunner(runner_impl, storage_type="redis", redis_url=os.environ.get("REDIS_URL"))
        else:
            _runner = runner_impl()

        logger.info(f"Runner initialized: {_runner.__class__.__name__}")

    return _runner


async def get_storage() -> BaseStorage:
    """Get the storage instance."""
    global _storage

    load_dotenv()

    if _storage is None:
        if os.environ.get("REPOFORGE_STORAGE") == "s3":
            from repoforge.storage.s3_storage import S3Storage

            _storage = S3Storage()
        elif os.environ.get("REPOFORGE_STORAGE") == "azure":
            from repoforge.storage.azure_storage import AzureBlobStorage

            _storage = AzureBlobStorage()
        else:
            if not os.environ.get("REPOFORGE_DIR"):
                raise ValueError("REPOFORGE_DIR environment variable is not set")
            _storage = FileStorage(base_dir=os.environ.get("REPOFORGE_DIR"))

        logger.info(f"Storage initialized: {_storage}")

    return _storage


async def get_evaluation_manager():
    """Get the evaluation manager instance, ensuring it's initialized."""
    await ensure_managers_initialized()

    if evaluation_manager is None:
        raise RuntimeError("Evaluation manager not initialized")

    return evaluation_manager
