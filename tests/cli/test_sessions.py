import asyncio
import json
from pathlib import Path
import subprocess

import pytest

from moatless.actions.bash import BashArgs
from moatless.actions.finish import FinishArgs
from moatless.actions.read_file import ReadFileArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.actions.create_file import CreateFileArgs
from moatless.actions.local_coding import LocalTestArgs
from moatless.cli.service import SessionService
from moatless.cli.worktree import git, patch_hash
from moatless.completion.base import CompletionResponse
from moatless.flow.durable import RecoveryConflict


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing", "not-git", "no-commit"])
async def test_invalid_source_reports_path_before_model_call(tmp_path, scripted, kind):
    source = tmp_path / kind
    if kind != "missing":
        source.mkdir()
    if kind == "no-commit":
        git(source, "init", "-q")
    service = SessionService(tmp_path / "sessions")
    with pytest.raises(ValueError) as error:
        await service.create(str(source), "读取代码")
    assert str(source) in str(error.value)
    assert ("HEAD" if kind == "no-commit" else "--repo") in str(error.value)
    assert scripted[1] == []
    assert not service.list_sessions()
    assert not list(service.directory.iterdir())


@pytest.mark.asyncio
async def test_coding_followup_export_apply_and_reload(repo, tmp_path, scripted):
    outputs, calls = scripted
    outputs.extend(
        [
            [ReadFileArgs(file_path="calc.py"), LocalTestArgs(command="python -m pytest -q")],
            [StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b")],
            [LocalTestArgs(command="python -m pytest -q"), FinishArgs(finish_reason="修复完成")],
        ]
    )
    service = SessionService(tmp_path / "sessions")
    session = await service.create(
        str(repo), "修复加法\n并测试", {"model": "deepseek/deepseek-chat", "instance_id": "owner__repo-1"}
    )
    await service.start(session)
    assert session.info["status"] == "terminal"
    assert [t["test_status"] for t in service.tests(session)] == ["failed", "passed"]
    assert "return a - b" in (repo / "calc.py").read_text()
    assert "return a + b" in Path(session.info["worktree"], "calc.py").read_text()
    old_usage = service.usage(session)
    service2 = SessionService(service.directory)
    loaded = service2.load(session.id)
    outputs.extend(
        [[CreateFileArgs(path="说明.txt", file_text="中文多行\n追加任务\n"), FinishArgs(finish_reason="续聊完成")]]
    )
    await service2.start(loaded, "追加中文说明文件")
    assert service2.usage(loaded) == old_usage
    assert "修复加法" in str(calls[-1]) and "追加中文说明" in str(calls[-1])
    patch = service2.diff(loaded)
    assert "说明.txt" in patch or "\\350" in patch
    target = service2.export(loaded, str(tmp_path / "export"))
    assert (target / "events.jsonl").exists()
    assert json.loads((target / "predictions.jsonl").read_text())["instance_id"] == "owner__repo-1"
    with pytest.raises(ValueError, match="preview"):
        service2.apply(loaded, "wrong")
    service2.apply(loaded, patch_hash(patch))
    assert (repo / "说明.txt").read_text() == "中文多行\n追加任务\n"
    assert "return a + b" in (repo / "calc.py").read_text()
    assert git(repo, "rev-parse", "HEAD").strip() == session.info["base_commit"]
    with pytest.raises(ValueError, match="clean"):
        service2.apply(loaded, patch_hash(patch))


@pytest.mark.asyncio
async def test_source_clean_and_recovery_workspace_conflict(repo, tmp_path, scripted):
    service = SessionService(tmp_path / "sessions")
    (repo / "new.txt").write_text("user changes")
    with pytest.raises(ValueError, match="clean"):
        await service.create(str(repo), "任务")
    assert not service.list_sessions()
    (repo / "new.txt").unlink()
    scripted[0].append([FinishArgs(finish_reason="done")])
    s = await service.create(str(repo), "task")
    await service.start(s)
    Path(s.info["worktree"], "calc.py").write_text("changed externally")
    restored = SessionService(service.directory).load(s.id)
    with pytest.raises(RecoveryConflict, match="changed"):
        await SessionService(service.directory).start(restored, "followup")


@pytest.mark.asyncio
async def test_cancel_process_and_unknown_side_effect_no_replay(repo, tmp_path, scripted):
    service = SessionService(tmp_path / "sessions")
    scripted[0].append([BashArgs(command="echo started > marker; sleep 120; echo repeated >> marker")])
    s = await service.create(str(repo), "cancel")
    running = service.start(s)
    marker = Path(s.info["worktree"], "marker")
    for _ in range(200):
        if marker.exists():
            break
        await asyncio.sleep(0.01)
    assert marker.exists()
    with pytest.raises(ValueError, match="running"):
        service.start(s, "concurrent")
    process_ids = [p.pid for p in s.workspace.environment.processes]
    await service.cancel(s)
    assert running.cancelled()
    for pid in process_ids:
        with pytest.raises(ProcessLookupError):
            import os

            os.kill(pid, 0)
    restored_service = SessionService(service.directory)
    with pytest.raises(RecoveryConflict, match="Unknown"):
        await restored_service.start(restored_service.load(s.id))
    assert marker.read_text() == "started\n"
    assert len(scripted[1]) == 1


@pytest.mark.asyncio
async def test_long_output_failures_and_path_escape(repo, tmp_path, scripted):
    outputs, _ = scripted
    outputs.append(
        [
            BashArgs(command="python -c 'print(\"z\" * 20000)'"),
            BashArgs(command="exit 7"),
            ReadFileArgs(file_path="../secret"),
            LocalTestArgs(command="exit 5"),
            FinishArgs(finish_reason="failures shown"),
        ]
    )
    service = SessionService(tmp_path / "sessions")
    s = await service.create(str(repo), "failure tools")
    await service.start(s)
    assert service.tests(s)[-1]["test_status"] == "failed"
    events = [json.loads(line) for line in (s.directory / "events.jsonl").read_text().splitlines()]
    completed = [e for e in events if e["scope"] == "tool" and e["event_type"] == "completed"]
    assert len(completed[0]["data"]["observation"]["message"]) > 20000
    assert len([e for e in events if e["scope"] == "tool" and e["event_type"] == "failed"]) == 3
    assert service.tests(await service.create(str(repo), "no tests")) == []


@pytest.mark.asyncio
async def test_model_failure_and_iteration_budget(repo, tmp_path, scripted):
    service = SessionService(tmp_path / "sessions")
    scripted[0].append(ValueError("model failed"))
    s = await service.create(str(repo), "model error")
    with pytest.raises(ValueError, match="model failed"):
        await service.start(s)
    assert s.info["status"] == "failed"
    scripted[0].append([ReadFileArgs(file_path="calc.py")])
    s = await service.create(str(repo), "budget", {"max_iterations": 2})
    await service.start(s)
    assert s.info["status"] == "max_iterations"


@pytest.mark.parametrize("mode", ["confirmed", "unknown"])
def test_cross_process_recovery(repo, tmp_path, mode):
    import os
    import signal
    import sys

    driver = Path(__file__).with_name("recovery_driver.py")
    root = tmp_path / mode
    root.mkdir()
    first = subprocess.run([sys.executable, str(driver), mode, str(root), str(repo)], capture_output=True, timeout=45)
    assert first.returncode == 23, first.stderr.decode()
    if mode == "unknown":
        os.killpg(int((root / "process-group").read_text()), signal.SIGKILL)
    second = subprocess.run(
        [sys.executable, str(driver), "resume", str(root), str(repo)], capture_output=True, timeout=45
    )
    assert second.returncode == 0, second.stderr.decode()
    assert (root / "model-calls").read_text() == "call\n"
    if mode == "confirmed":
        info = json.loads((root / "export" / "session.json").read_text())
        assert info["status"] == "terminal"
        assert Path(info["worktree"], "counter").read_text() == "executed\n"
        assert json.loads((root / "export" / "tests.json").read_text())[-1]["test_status"] == "passed"
    else:
        assert "RecoveryConflict" in (root / "error").read_text()


@pytest.mark.asyncio
async def test_zero_cost_budget_does_not_call_model(repo, tmp_path, scripted):
    service = SessionService(tmp_path / "sessions")
    s = await service.create(str(repo), "zero budget", {"max_cost": 0})
    await service.start(s)
    assert s.info["status"] == "max_cost"
    assert scripted[1] == []


@pytest.mark.asyncio
async def test_cost_usage_preserved_on_resume_and_followup(repo, tmp_path, scripted):
    from moatless.completion.stats import Usage, CompletionAttempt, CompletionInvocation

    invocation = CompletionInvocation(
        model="deepseek/deepseek-chat",
        attempts=[CompletionAttempt(usage=Usage(completion_cost=0.02, prompt_tokens=20, completion_tokens=10))],
    )
    scripted[0].append(
        CompletionResponse(structured_outputs=[ReadFileArgs(file_path="calc.py")], completion_invocation=invocation)
    )
    service = SessionService(tmp_path / "sessions")
    s = await service.create(str(repo), "cost budget", {"max_cost": 0.01})
    await service.start(s)
    assert s.info["status"] == "max_cost"
    assert service.usage(s)["completion_cost"] == 0.02
    restarted = SessionService(service.directory)
    loaded = restarted.load(s.id)
    await restarted.start(loaded)
    assert restarted.usage(loaded)["completion_cost"] == 0.02
    await restarted.start(loaded, "new requirement")
    assert len(scripted[1]) == 1
    assert restarted.usage(loaded)["completion_cost"] == 0.02


@pytest.mark.asyncio
async def test_stale_client_preserves_latest_history(repo, tmp_path, scripted):
    scripted[0].extend([[FinishArgs(finish_reason="first")], [FinishArgs(finish_reason="second")]])
    service = SessionService(tmp_path / "sessions")
    s = await service.create(str(repo), "original")
    stale_service = SessionService(service.directory)
    stale = stale_service.load(s.id)
    await service.start(s)
    await stale_service.start(stale, "followup from stale client")
    history = stale.flow.root.get_all_nodes()
    assert [step.action.finish_reason for n in history for step in n.action_steps] == ["first", "second"]
    assert "original" in str(scripted[1][-1])


@pytest.mark.asyncio
async def test_initial_drift_timeout_and_process_lock(repo, tmp_path, scripted):
    from filelock import FileLock

    service = SessionService(tmp_path / "sessions")
    s = await service.create(str(repo), "initial drift")
    Path(s.info["worktree"], "calc.py").write_text("external changes")
    with pytest.raises(RecoveryConflict, match="changed"):
        await service.start(s)
    s = await service.create(str(repo), "locked")
    with FileLock(str(s.directory / "session.lock")):
        with pytest.raises(RecoveryConflict, match="another process"):
            await service.start(s)
    scripted[0].append([BashArgs(command="sleep 120")])
    s = await service.create(str(repo), "timeout", {"tool_timeout": 0.1})
    with pytest.raises(asyncio.TimeoutError):
        await service.start(s)
    assert not s.workspace.environment.processes
    restored = SessionService(service.directory)
    with pytest.raises(RecoveryConflict, match="Unknown"):
        await restored.start(restored.load(s.id))


@pytest.mark.asyncio
async def test_clean_source_patch_conflict(repo, tmp_path, scripted):
    scripted[0].append(
        [
            StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b"),
            FinishArgs(finish_reason="done"),
        ]
    )
    service = SessionService(tmp_path / "sessions")
    s = await service.create(str(repo), "edit")
    await service.start(s)
    patch = service.diff(s)
    (repo / "calc.py").write_text("def different():\n    return 100\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "source advanced")
    with pytest.raises(ValueError, match="patch"):
        service.apply(s, patch_hash(patch))
    assert "different" in (repo / "calc.py").read_text()
    assert git(repo, "status", "--porcelain") == ""


@pytest.mark.asyncio
async def test_read_search_edit_bash_and_tests_share_disk(repo, tmp_path, scripted):
    from moatless.actions.grep_tool import GrepToolArgs

    scripted[0].append(
        [
            ReadFileArgs(file_path="calc.py"),
            BashArgs(command="sed -i 's/a - b/a * b/' calc.py"),
            ReadFileArgs(file_path="calc.py"),
            GrepToolArgs(pattern="return a", include="*.py"),
            StringReplaceArgs(path="calc.py", old_str="return a * b", new_str="return a + b"),
            LocalTestArgs(command="python -m pytest -q"),
            FinishArgs(finish_reason="same disk"),
        ]
    )
    service = SessionService(tmp_path / "sessions")
    s = await service.create(str(repo), "same disk")
    await service.start(s)
    steps = s.flow.root.get_last_node().action_steps
    assert "return a - b" in steps[0].observation.message
    assert "return a * b" in steps[2].observation.message
    assert "calc.py:" in steps[3].observation.message
    assert service.tests(s)[-1]["test_status"] == "passed"
    assert "return a - b" in (repo / "calc.py").read_text()
    with pytest.raises(ValueError, match="outside"):
        await SessionService(repo / "sessions").create(str(repo), "nested worktree")
