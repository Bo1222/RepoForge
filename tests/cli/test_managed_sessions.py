"""Both configured CLI sessions and component-loaded agents use managed memory."""

import pytest

from moatless.cli.service import SessionService
from moatless.actions.finish import FinishArgs
from moatless.actions.local_coding import LocalTestArgs
from moatless.actions.task_state import UpdateTaskStateArgs
from moatless.message_history.managed import ManagedMessageHistoryGenerator


@pytest.mark.asyncio
async def test_flash_config_forwards_native_toggle_and_temperature(tmp_path, repo):
    import json
    from pathlib import Path
    config = json.loads((Path(__file__).resolve().parents[2] / "moatless/evaluation/profiles/flash_nonthinking_a0.json").read_text())
    config["temperature"] = .25
    session = await SessionService(tmp_path / "flash").create(str(repo), "Inspect", config)
    model = session.flow.agent.completion_model
    assert model.temperature == .25
    assert model._completion_params["extra_body"]["thinking"]["type"] == "disabled"
    assert model.model_base_url == "https://api.deepseek.com"
    assert session.flow.agent.memory.policy.enable_task_state is False
    assert not any(action.name == "UpdateTaskState" for action in session.flow.agent.actions)


@pytest.mark.asyncio
async def test_cli_context_policy_is_opt_in_and_survives_reload(tmp_path, repo, scripted):
    outputs, calls = scripted
    service = SessionService(tmp_path / "sessions")
    session = await service.create(str(repo), "Repair", {"context_policy": {"context_window": 8192}})
    assert isinstance(session.flow.agent.memory, ManagedMessageHistoryGenerator)
    assert any(action.name == "ReadObservation" for action in session.flow.agent.actions)
    outputs.extend([[UpdateTaskStateArgs(next_step="Run reproduction")],
                    [LocalTestArgs(command="python -m pytest -q")],
                    [FinishArgs(finish_reason="investigation complete")]])
    await service.start(session)
    loaded = SessionService(tmp_path / "sessions").load(session.id)
    assert isinstance(loaded.flow.agent.memory, ManagedMessageHistoryGenerator)
    assert loaded.flow.root.get_last_node().context_build["policy"]["version"] == 1
    assert loaded.flow.root.get_last_node().task_state.next_step == "Run reproduction"
    assert "Repair" in str(calls[0])


def test_confirmed_task_notes_survive_process_loss_without_replay(tmp_path, repo):
    import subprocess
    import sys

    def worker(mode):
        return subprocess.run([sys.executable, "-m", "tests.cli.recovery_driver", mode, str(tmp_path), str(repo)],
                              capture_output=True, text=True, timeout=45)

    first = worker("managed-confirmed")
    assert first.returncode == 23, first.stderr
    resumed = worker("managed-resume")
    assert resumed.returncode == 0, resumed.stderr
    assert not (tmp_path / "error").exists()
    service = SessionService(tmp_path / "sessions")
    session = service.load((tmp_path / "id").read_text())
    state = session.flow.root.get_last_node().task_state
    assert state.next_step == "Run verification" and state.revision == 1
    assert (tmp_path / "model-calls").read_text().splitlines() == ["call"]
    assert (session.directory / "worktree" / "counter").read_text().splitlines() == ["executed"]


def test_managed_unknown_side_effect_is_not_replayed(tmp_path, repo):
    import os
    import signal
    import subprocess
    import sys

    def worker(mode):
        return subprocess.run([sys.executable, "-m", "tests.cli.recovery_driver", mode, str(tmp_path), str(repo)],
                              capture_output=True, text=True, timeout=45)

    try:
        interrupted = worker("managed-unknown")
        assert interrupted.returncode == 23, interrupted.stderr
        calls = (tmp_path / "model-calls").read_text()
        resumed = worker("managed-resume")
        assert resumed.returncode == 0, resumed.stderr
        assert "Unknown side effects" in (tmp_path / "error").read_text()
        assert (tmp_path / "model-calls").read_text() == calls
    finally:
        process = tmp_path / "process-group"
        if process.exists():
            try:
                os.killpg(int(process.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
