"""Exercise real completion parsing, Agent, durable Flow and UI with an offline SDK boundary."""

import json
from pathlib import Path

import pytest
from litellm.types.utils import ModelResponse

from moatless.actions.finish import FinishArgs
from moatless.cli import worktree
from moatless.cli.app import CodingApp, DetailScreen, TaskInput
from moatless.cli.main import parser, main
from moatless.cli.service import SessionService
from moatless.completion.base import CompletionRetryError
from moatless.completion.tool_call import ToolCallCompletionModel
from moatless.flow.durable import RecoveryConflict


def text_response(text):
    return ModelResponse(
        model="deepseek/deepseek-chat",
        choices=[{"message": {"role": "assistant", "content": text}, "index": 0, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


@pytest.fixture
def sdk(monkeypatch):
    replies, calls = [], []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key-not-real")

    async def complete(**kwargs):
        calls.append(kwargs)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr("litellm.acompletion", complete)
    return replies, calls


@pytest.mark.asyncio
async def test_chat_roundtrip_and_recovery_without_git(tmp_path, monkeypatch, sdk):
    def forbid_git(*args, **kwargs):
        raise AssertionError("ordinary chat must not invoke Git")

    monkeypatch.setattr(worktree, "git", forbid_git)
    replies, calls = sdk
    replies.extend([text_response("当前配置为 deepseek/deepseek-chat。"), text_response("列表可变，元组不可变。")])
    service = SessionService(tmp_path / "sessions")
    s = await service.create(None, "你是什么模型？")
    await service.start(s)
    assert s.info["mode"] == "chat"
    assert s.info["source"] is s.info["worktree"] is s.info["base_commit"] is None
    assert not (s.directory / "worktree").exists()
    assert [a.name for a in s.flow.agent.actions] == ["Finish"]
    assert s.info["status"] == "terminal"
    assert len(calls) == 1
    assert "deepseek/deepseek-chat" in calls[0]["messages"][0]["content"]
    assert [tool["function"]["name"] for tool in calls[0]["tools"]] == ["Finish"]
    invocation = s.flow.root.get_last_node().completions["build_action"]
    assert len(invocation.attempts) == 1
    assert service.usage(s)["prompt_tokens"] == 10
    assert service.usage(s)["completion_tokens"] == 5
    cost = service.usage(s)["completion_cost"]
    assert cost > 0
    for operation in [service.diff, service.changes, lambda session: service.apply(session, "ignored")]:
        with pytest.raises(ValueError, match="/repo"):
            operation(s)
    for operation in [
        s.workspace.environment.execute("echo hi"),
        s.workspace.environment.read_file("x"),
        s.workspace.environment.write_file("x", "data"),
    ]:
        with pytest.raises(ValueError, match="/repo"):
            await operation
    export = service.export(s, str(tmp_path / "export"))
    assert (export / "patch.diff").read_text() == ""
    assert json.loads((export / "tests.json").read_text()) == []
    assert not (export / "predictions.jsonl").exists()
    assert "offline-test-key-not-real" not in (s.directory / "state.json").read_text()
    resumed_service = SessionService(tmp_path / "sessions")
    restored = resumed_service.load(s.id)
    await resumed_service.start(restored)
    assert len(calls) == 1  # Finished model responses are not replayed or charged again.
    assert resumed_service.usage(restored)["completion_cost"] == cost
    await resumed_service.start(restored, "列表和元组有什么区别？")
    assert len(calls) == 2
    assert "当前配置为 deepseek/deepseek-chat" in json.dumps(calls[1]["messages"], ensure_ascii=False)
    assert resumed_service.usage(restored)["prompt_tokens"] == 20
    assert resumed_service.usage(restored)["completion_cost"] == pytest.approx(cost * 2)


@pytest.mark.asyncio
async def test_chat_ui_once_and_repository_switch(repo, tmp_path, sdk):
    replies, calls = sdk
    answer = "当前配置模型是 deepseek/deepseek-chat。"
    replies.extend([text_response(answer), text_response("这是代码仓库中的一次回答。")])
    service = SessionService(tmp_path / "sessions")
    app = CodingApp(service, None)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.submit("你是什么模型？")
        await app.workers.wait_for_complete()
        await pilot.pause()
        original = app.session
        messages = [str(widget.render()) for widget in app.query("#transcript Static")]
        assert sum(answer in message for message in messages) == 1
        assert len(calls) == 1
        await app.submit("/model")
        assert len(calls) == 1
        assert "deepseek/deepseek-chat" in str(app.query_one("#transcript").children[-1].render())
        await app.submit("/architecture")
        assert isinstance(app.screen, DetailScreen)
        assert "无（不执行文件或命令工具）" in str(app.architecture().renderables[-1])
        await pilot.press("escape")
        editor = app.query_one("#input", TaskInput)
        editor.load_text("/repo /不存在的仓库")
        await app.submit(editor.text)
        assert app.session is original
        assert editor.text == "/repo /不存在的仓库"
        await app.submit("/repo " + str(repo))
        assert app.session is None
        assert app.repo == str(repo)
        assert original.id in service.sessions
        assert len(calls) == 1
        await app.submit("介绍这个仓库")
        await app.workers.wait_for_complete()
        assert app.session.id != original.id
        assert app.session.info["mode"] == "coding"
        assert (Path(app.session.info["worktree"]) / "calc.py").exists()
        assert "你是什么模型" not in json.dumps(calls[1]["messages"], ensure_ascii=False)
        await app.submit("/repo --clear")
        assert app.repo is None and app.session is None
        assert len(service.list_sessions()) == 2


@pytest.mark.asyncio
async def test_plain_text_opt_in_and_invalid_tools(sdk):
    strict = ToolCallCompletionModel(model="test")
    strict.initialize([FinishArgs], "test")
    with pytest.raises(CompletionRetryError):
        await strict._validate_completion(text_response("普通答案"))
    enabled = ToolCallCompletionModel(model="test", allow_plain_text=True)
    enabled.initialize([FinishArgs], "test")
    actions, text, _ = await enabled._validate_completion(text_response("普通答案"))
    assert actions[0].finish_reason == "普通答案" and text is None
    with pytest.raises(CompletionRetryError):
        await enabled._validate_completion(text_response("  "))
    bad = text_response("不能把未知工具误当作普通答案")
    bad.choices[0].message.tool_calls = [
        {"id": "bad", "type": "function", "function": {"name": "Bash", "arguments": '{"command":"echo unsafe"}'}}
    ]
    # Let LiteLLM construct typed tool-call objects.
    bad = ModelResponse(**bad.model_dump())
    with pytest.raises(CompletionRetryError):
        await enabled._validate_completion(bad)


@pytest.mark.asyncio
async def test_chat_unknown_model_outcome_is_not_replayed(tmp_path, sdk):
    import asyncio

    service = SessionService(tmp_path / "sessions")
    replies, calls = sdk

    async def pending(**kwargs):
        calls.append(kwargs)
        await asyncio.Event().wait()

    from unittest.mock import patch

    s = await service.create(None, "等待模型回答")
    with patch("litellm.acompletion", pending):
        task = service.start(s)
        for _ in range(200):
            if calls:
                break
            await asyncio.sleep(0.01)
        assert len(calls) == 1
        await service.cancel(s)
        assert task.cancelled()
    restored_service = SessionService(tmp_path / "sessions")
    restored = restored_service.load(s.id)
    with pytest.raises(RecoveryConflict, match="Unknown"):
        await restored_service.start(restored)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_missing_key_and_redacted_api_error(tmp_path, monkeypatch, sdk):
    from moatless.exceptions import CompletionRuntimeError

    replies, calls = sdk
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    service = SessionService(tmp_path / "sessions")
    s = await service.create(None, "问答")
    with pytest.raises(CompletionRuntimeError, match="密钥"):
        await service.start(s)
    assert not calls and s.info["error_stage"] == "模型请求"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key-not-real")
    replies.append(CompletionRuntimeError("API failed: offline-test-key-not-real"))
    s = await service.create(None, "问答")
    with pytest.raises(CompletionRuntimeError):
        await service.start(s)
    assert "offline-test-key-not-real" not in s.info["error"]
    assert "offline-test-key-not-real" not in (s.directory / "state.json").read_text()
    assert "offline-test-key-not-real" not in (s.directory / "events.jsonl").read_text()


def test_cli_defaults_and_explicit_repo_preflight(tmp_path, monkeypatch):
    assert parser().parse_args(["chat"]).repo is None
    assert parser().parse_args(["run", "你是什么模型"]).repo is None
    monkeypatch.setattr(CodingApp, "run", lambda self: pytest.fail("invalid repo must fail before UI"))
    assert main(["--session-dir", str(tmp_path / "sessions"), "chat", "--repo", str(tmp_path / "missing")]) == 1


@pytest.mark.asyncio
async def test_legacy_coding_session_keeps_strict_parser(repo, tmp_path, sdk):
    service = SessionService(tmp_path / "sessions")
    session = await service.create(str(repo), "旧会话")
    info_file, state_file = session.directory / "session.json", session.directory / "state.json"
    info, state = json.loads(info_file.read_text()), json.loads(state_file.read_text())
    info.pop("mode")
    state["flow"]["agent"]["completion_model"].pop("allow_plain_text")
    info_file.write_text(json.dumps(info))
    state_file.write_text(json.dumps(state))
    restored_service = SessionService(tmp_path / "sessions")
    restored = restored_service.load(session.id)
    assert not restored.flow.agent.completion_model.allow_plain_text
    assert restored.workspace.repository.repo_path == session.info["worktree"]
    reply = ModelResponse(
        model="deepseek/deepseek-chat",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "finish",
                            "type": "function",
                            "function": {"name": "Finish", "arguments": '{"finish_reason":"旧会话完成"}'},
                        }
                    ],
                },
                "index": 0,
                "finish_reason": "tool_calls",
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    sdk[0].append(reply)
    await restored_service.start(restored)
    assert restored.info["status"] == "terminal"
    assert len(sdk[1]) == 1


@pytest.mark.asyncio
async def test_repo_switch_blocked_during_chat_request(repo, tmp_path, sdk):
    import asyncio
    from unittest.mock import patch

    service = SessionService(tmp_path / "sessions")
    app = CodingApp(service, None)
    started = asyncio.Event()

    async def pending(**kwargs):
        started.set()
        await asyncio.Event().wait()

    with patch("litellm.acompletion", pending):
        async with app.run_test(size=(120, 40)):
            await app.submit("等待回答")
            await asyncio.wait_for(started.wait(), timeout=5)
            original = app.session
            await app.submit("/repo " + str(repo))
            assert app.session is original and app.repo is None
            assert app.session.running
            await app.submit("/model")
            assert app.session.running
            await app.action_cancel()
            await app.workers.wait_for_complete()
            await app.submit("/repo " + str(repo))
            assert app.session is None and app.repo == str(repo)
            assert original.id in service.sessions
