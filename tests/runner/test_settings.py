from unittest.mock import MagicMock

import pytest

import moatless.settings as settings
from moatless.runner.asyncio_runner import AsyncioRunner
from moatless.runner.scheduler import SchedulerRunner


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("redis", [False, True])
async def test_removed_runner_rejected_before_cache_or_scheduler(monkeypatch, cached, redis):
    monkeypatch.setattr(settings, "load_dotenv", lambda: None)
    monkeypatch.setattr(settings, "_runner", MagicMock() if cached else None)
    monkeypatch.setenv("MOATLESS_RUNNER", "docker")
    if redis:
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    else:
        monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(ValueError, match="custom Docker runner was removed"):
        await settings.get_runner()


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_type", [None, "asyncio", "kubernetes"])
@pytest.mark.parametrize("redis", [False, True])
async def test_supported_runner_selection(monkeypatch, runner_type, redis):
    from kubernetes import config
    from moatless.runner.kubernetes_runner import KubernetesRunner

    monkeypatch.setattr(config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(AsyncioRunner, "_start_cleanup_task", lambda self: None)
    monkeypatch.setattr(AsyncioRunner, "_setup_signal_handlers", lambda self: None)
    monkeypatch.setattr(settings, "load_dotenv", lambda: None)
    monkeypatch.setattr(settings, "_runner", None)
    if runner_type is None:
        monkeypatch.delenv("MOATLESS_RUNNER", raising=False)
    else:
        monkeypatch.setenv("MOATLESS_RUNNER", runner_type)
    if redis:
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    else:
        monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(SchedulerRunner, "start_scheduler", lambda self: None)
    expected = KubernetesRunner if runner_type == "kubernetes" else AsyncioRunner
    runner = await settings.get_runner()
    if redis:
        assert isinstance(runner, SchedulerRunner)
        assert isinstance(runner.runner, expected)
    else:
        assert isinstance(runner, expected)


def test_kubernetes_update_uses_retained_script(monkeypatch):
    from pathlib import Path

    from kubernetes import config
    from moatless.runner.kubernetes_runner import KubernetesRunner

    monkeypatch.setattr(config, "load_incluster_config", lambda: None)
    runner = KubernetesRunner(image="worker:test", update_on_start=True, update_branch="main")
    job = runner._create_job_object(
        "job", "project", "django__django-11099", settings.get_runner, update_on_start=True, update_branch="main"
    )
    command = job.spec.template.spec.containers[0].args[1]
    assert "/opt/moatless/scripts/update-moatless.sh --branch main" in command
    assert "/opt/moatless/docker/" not in command
    script = Path(__file__).resolve().parents[2] / "scripts/update-moatless.sh"
    assert script.is_file()
    assert script.stat().st_mode & 0o111
