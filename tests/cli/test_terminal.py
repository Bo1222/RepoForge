import json

import pytest
from textual.widgets import Collapsible

from moatless.cli.app import CodingApp, TaskInput, DetailScreen, SessionPicker
from moatless.cli.service import SessionService
from moatless.cli.main import parser, read_task


@pytest.mark.asyncio
async def test_chinese_multiline_commands_and_layout(tmp_path):
    service = SessionService(tmp_path / "sessions")
    app = CodingApp(service, str(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.query_one("#input", TaskInput)
        editor.load_text("中文第一行")
        await pilot.press("shift+enter")
        await pilot.pause()
        assert "\n" in editor.text
        editor.insert("中文第二行")
        assert "中文第二行" in editor.text
        # First Enter completes; second Enter executes without adding a newline.
        editor.load_text("/help")
        await pilot.pause()
        await pilot.press("enter")
        assert editor.text == "/help"
        await pilot.press("enter")
        await pilot.pause()
        assert editor.text == ""
        assert isinstance(app.screen, DetailScreen)
        await pilot.press("escape")
        assert not app.query("#sidebar")
        await app.submit("/model deepseek/deepseek-chat")
        await app.submit("/tools read,search,edit,tests")
        assert app.config["tools"] == ["read", "search", "edit", "tests"]
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"max_cost": 0.01}))
        await app.submit(f"/config {config}")
        assert app.config["max_cost"] == 0.01
        await app.submit("/sessions")
        assert isinstance(app.screen, SessionPicker)
        await pilot.press("escape")
        await app.submit("/new")
        await app.submit("/tests")
        assert app.session is None


def test_task_file_and_arguments(tmp_path):
    task = tmp_path / "task.md"
    task.write_text("修复函数\n运行测试")
    args = parser().parse_args(["run", "--repo", ".", "--model", "deepseek/deepseek-chat", "--task-file", str(task)])
    assert read_task(args) == "修复函数\n运行测试"
    args.task = "duplicate"
    with pytest.raises(ValueError):
        read_task(args)


@pytest.mark.asyncio
async def test_tools_followup_cancel_and_apply_ui(repo, tmp_path, scripted):
    from moatless.actions.read_file import ReadFileArgs
    from moatless.actions.bash import BashArgs
    from moatless.actions.string_replace import StringReplaceArgs
    from moatless.actions.finish import FinishArgs
    from moatless.actions.local_coding import LocalTestArgs
    from moatless.cli.app import ApplyConfirmation

    outputs, _ = scripted
    outputs.extend(
        [
            [
                ReadFileArgs(file_path="calc.py"),
                StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b"),
                BashArgs(command="python -c 'print(\"x\" * 20000)'"),
                LocalTestArgs(command="python -m pytest -q"),
                FinishArgs(finish_reason="完成"),
            ],
            [FinishArgs(finish_reason="追加要求完成")],
        ]
    )
    service = SessionService(tmp_path / "sessions")
    app = CodingApp(service, str(repo))
    async with app.run_test(size=(130, 45)) as pilot:
        await app.submit("修复加法\n并运行测试")
        for _ in range(100):
            await pilot.pause(0.05)
            if app.session and not app.session.running and app.session.info["status"] == "terminal":
                break
        assert app.session.info["status"] == "terminal"
        await app.workers.wait_for_complete()
        assert app.query(Collapsible)
        assert service.tests(app.session)[-1]["test_status"] == "passed"
        assert len(app.tool_widgets) == 4
        assert len(app.query(Collapsible)) == 4
        assert all(widget.has_class("-collapsed") for widget, _ in app.tool_widgets.values())
        assert all(not widget.query_one("Contents").display for widget, _ in app.tool_widgets.values())
        await pilot.press("ctrl+o")
        assert all(not widget.collapsed for widget, _ in app.tool_widgets.values())
        assert all(widget.query_one("Contents").display for widget, _ in app.tool_widgets.values())
        await pilot.press("ctrl+o")
        assert all(widget.collapsed for widget, _ in app.tool_widgets.values())
        await app.submit("追加要求")
        for _ in range(100):
            await pilot.pause(0.03)
            if not app.session.running:
                break
        for command in ("/diff", "/tests", "/usage", "/files", "/architecture"):
            await app.submit(command)
            assert isinstance(app.screen, DetailScreen)
            await pilot.press("escape")
        await app.submit("/sessions")
        assert isinstance(app.screen, SessionPicker)
        # Selecting a saved session loads history without starting the model.
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, SessionPicker)
        assert len(scripted[1]) == 2
        await app.submit("/resume " + app.session.id)
        for _ in range(100):
            await pilot.pause(0.03)
            if not app.session.running:
                break
        assert len(scripted[1]) == 2
        await app.submit("/export " + str(tmp_path / "ui-export"))
        assert (tmp_path / "ui-export" / "patch.diff").exists()
        await app.submit("/apply")
        await pilot.pause()
        assert isinstance(app.screen, ApplyConfirmation)
        await pilot.click("#confirm-apply")
        await pilot.pause()
        assert "return a + b" in (repo / "calc.py").read_text()
        await app.submit("/quit")


@pytest.mark.asyncio
async def test_busy_draft_and_ctrl_c_cancel(repo, tmp_path, scripted):
    from moatless.actions.bash import BashArgs

    scripted[0].append([BashArgs(command="sleep 120")])
    service = SessionService(tmp_path / "sessions")
    app = CodingApp(service, str(repo))
    async with app.run_test(size=(120, 40)) as pilot:
        await app.submit("运行长命令")
        for _ in range(100):
            await pilot.pause(0.03)
            if app.session.workspace.environment.processes:
                break
        assert app.session.running
        editor = app.query_one("#input", TaskInput)
        editor.load_text("下一条消息\n继续测试")
        editor.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert editor.text == "下一条消息\n继续测试"
        assert len(scripted[1]) == 1
        await pilot.press("ctrl+c")
        for _ in range(100):
            await pilot.pause(0.03)
            if not app.session.running:
                break
        assert app.session.info["status"] == "cancelled"
        assert not app.session.workspace.environment.processes


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_command_palette_layout_and_architecture(tmp_path, size):
    service = SessionService(tmp_path / "sessions")
    app = CodingApp(service, str(tmp_path / ("long-target-" * 8)))
    async with app.run_test(size=size) as pilot:
        editor = app.query_one("#input", TaskInput)
        assert not app.query("#sidebar, #buttons, #status")
        assert editor.region.bottom <= size[1] - 2
        assert app.query_one("#transcript").region.height > 0
        editor.load_text("/")
        await pilot.pause()
        assert app.menu_open
        await pilot.press("down", "tab")
        assert editor.text == "/sessions"
        assert not app.menu_open
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, SessionPicker)
        await pilot.press("escape")
        editor.load_text("/ar")
        await pilot.pause()
        await pilot.press("enter")
        assert editor.text == "/architecture"
        assert app.session is None
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, DetailScreen)
        assert "待创建" in str(app.architecture().renderables[-1])
        await pilot.press("escape")
        assert editor.has_focus
        editor.load_text("/ex")
        await pilot.pause()
        await pilot.press("tab")
        assert editor.text == "/export "
        assert not app.menu_open
        editor.load_text("/mo")
        await pilot.pause()
        await pilot.press("escape")
        assert editor.text == "/mo"
        assert not app.menu_open
        editor.load_text("中文\n" * 12)
        await pilot.pause()
        assert editor.region.height == 8
        assert app.query_one("#transcript").region.height > 0
        assert editor.region.bottom <= size[1] - 2


@pytest.mark.asyncio
async def test_live_tool_identity_failure_and_history(repo, tmp_path, scripted):
    from moatless.actions.local_coding import LocalTestArgs
    from moatless.actions.finish import FinishArgs

    scripted[0].append([LocalTestArgs(command="python -m pytest -q"), FinishArgs(finish_reason="测试失败")])
    service = SessionService(tmp_path / "sessions")
    app = CodingApp(service, str(repo))
    async with app.run_test(size=(120, 40)) as pilot:
        await app.submit("运行测试")
        for _ in range(100):
            await pilot.pause(0.05)
            if app.session and not app.session.running:
                break
        await app.workers.wait_for_complete()
        assert len(app.tool_widgets) == 1
        widget, _ = next(iter(app.tool_widgets.values()))
        assert widget.has_class("failed")
        assert "failed" in widget.title
        assert "exit 1" in widget.title
        assert "失败" in widget.title
        # A delayed replay updates the existing canonical tool row.
        events = [json.loads(line) for line in (app.session.directory / "events.jsonl").read_text().splitlines()]
        event = next(e for e in reversed(events) if e["scope"] == "tool" and e["data"]["name"] == "RunTests")
        app.receive_event(app.session.id, event)
        await pilot.pause()
        assert len(app.tool_widgets) == len(app.query(Collapsible)) == 1
        assert "failed" in next(iter(app.tool_widgets.values()))[0].title
