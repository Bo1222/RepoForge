"""Textual terminal UI. All execution is delegated to SessionService."""

import asyncio
import json
import os
import shlex
from pathlib import Path

from rich.syntax import Syntax
from rich.text import Text
from rich.table import Table
from rich.console import Group
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Label, ListItem, ListView, Static, TextArea

from repoforge.cli.service import SessionService, effective_tools, error_message, session_mode
from repoforge.cli import worktree
from repoforge.cli.worktree import patch_hash


# One definition drives completion and /help; parameters remain user supplied.
COMMANDS = [
    ("/new", "[任务]", "新建会话"),
    ("/sessions", "", "选择已保存会话"),
    ("/resume", "[ID]", "继续当前或指定会话"),
    ("/diff", "", "查看代码补丁"),
    ("/files", "", "查看文件变更"),
    ("/tests", "", "查看真实测试记录"),
    ("/usage", "", "查看 Token 与费用"),
    ("/architecture", "", "查看项目模块与 CLI 的对应关系"),
    ("/export", "PATH", "导出会话与补丁"),
    ("/apply", "", "预览并确认应用补丁"),
    ("/model", "[MODEL]", "查看当前模型或设置下一会话模型"),
    ("/repo", "[PATH | --clear]", "查看、绑定或清除代码仓库"),
    ("/tools", "read,search,edit,create,bash,tests", "设置新会话工具"),
    ("/config", "FILE", "加载新会话配置"),
    ("/cancel", "", "取消当前任务"),
    ("/help", "", "查看所有命令"),
    ("/quit", "", "取消任务并退出"),
]


class TaskInput(TextArea):
    class Submitted(Message):
        def __init__(self, text):
            self.text = text
            super().__init__()

    async def _on_key(self, event):
        if self.app.menu_open and event.key in {"up", "down", "tab", "enter", "escape"}:
            event.stop()
            event.prevent_default()
            if event.key in {"up", "down"}:
                self.app.move_menu(-1 if event.key == "up" else 1)
            elif event.key == "escape":
                self.app.dismiss_menu()
            else:
                self.app.complete_menu()
        elif event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
        elif event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
        else:
            await super()._on_key(event)


class DetailScreen(ModalScreen):
    BINDINGS = [("escape", "dismiss", "返回")]

    def __init__(self, title, content):
        super().__init__()
        self.heading, self.content = title, content

    def compose(self):
        with Vertical(classes="detail-dialog"):
            yield Label(self.heading, classes="dialog-title")
            with VerticalScroll(classes="detail-scroll"):
                yield Static(self.content, markup=False)
            yield Label("Esc 返回对话", classes="dialog-hint")

    def action_dismiss(self):
        self.dismiss()


class SessionPicker(ModalScreen):
    BINDINGS = [("escape", "dismiss", "返回")]

    def __init__(self, sessions):
        super().__init__()
        self.sessions = sessions

    def compose(self):
        with Vertical(classes="detail-dialog"):
            yield Label("会话 · 选择后查看历史，/resume 继续执行", classes="dialog-title")
            yield ListView(*(SessionItem(info) for info in self.sessions), id="sessions")
            if not self.sessions:
                yield Label("暂无保存的会话。输入任务创建会话。")
            yield Label("↑↓ 选择 · Enter 打开 · Esc 返回", classes="dialog-hint")

    def on_mount(self):
        self.query_one(ListView).focus()

    def on_list_view_selected(self, event):
        self.dismiss(event.item.identity)

    def action_dismiss(self):
        self.dismiss(None)


class ApplyConfirmation(ModalScreen):
    CSS = """
    ApplyConfirmation { align: center middle; }
    #apply-dialog { width: 94%; max-width: 130; height: 90%; border: solid #d99a39; background: #0b0d0f; padding: 1 2; }
    #apply-preview { height: 1fr; }
    """

    def __init__(self, source, patch):
        super().__init__()
        self.source, self.patch = source, patch

    def compose(self):
        with Vertical(id="apply-dialog"):
            yield Label(f"应用到 {self.source}？来源仓库必须干净；不会创建提交。", classes="dialog-title")
            with VerticalScroll(id="apply-preview"):
                yield Static(Syntax(self.patch, "diff", word_wrap=True, background_color="#0b0d0f"))
            with Horizontal():
                yield Button("确认应用", id="confirm-apply", variant="warning")
                yield Button("取消", id="cancel-apply")

    BINDINGS = [("escape", "reject", "取消")]

    def action_reject(self):
        self.dismiss(False)

    def on_button_pressed(self, event):
        self.dismiss(event.button.id == "confirm-apply")


class SessionItem(ListItem):
    def __init__(self, info):
        super().__init__(Label(f"{info['title'][:26]}\n{info['id'][:8]} · {info['status']}"))
        self.identity = info["id"]


class CodingApp(App):
    TITLE = "RepoForge Coding Agent"
    CSS = """
    Screen { background: #0b0d0f; color: #d5d9dc; }
    #body { height: 1fr; padding: 0 1; }
    #transcript { height: 1fr; padding: 0 1; scrollbar-color: #375256; }
    #command-menu { display: none; height: auto; max-height: 11; border: solid #263038; padding: 0 1; }
    #input { height: 3; min-height: 3; max-height: 8; border: solid #a77430; background: #0b0d0f; padding: 0 1; }
    #input:focus { border: solid #d99a39; }
    #usage-line, #key-hints { height: 1; padding: 0 1; color: #88949d; }
    .chat { height: auto; margin: 1 0; }
    .user-message { color: #37d9d1; }
    .error-message { color: #ee8877; }
    Collapsible { height: auto; background: #0b0d0f; border: none; padding: 0; margin: 0; }
    Collapsible > CollapsibleTitle { color: #aac2c5; padding: 0; }
    Collapsible > Contents { padding: 0 2; }
    Collapsible.failed > CollapsibleTitle, Collapsible.cancelled > CollapsibleTitle { color: #ee8877; }
    Collapsible.running > CollapsibleTitle { color: #d99a39; }
    Collapsible.completed > CollapsibleTitle { color: #91c69c; }
    ModalScreen { align: center middle; background: #000000 65%; }
    .detail-dialog { width: 94%; max-width: 130; height: 90%; border: solid #37d9d1; background: #0b0d0f; padding: 1 2; }
    .dialog-title { height: auto; color: #37d9d1; margin-bottom: 1; }
    .detail-scroll, #sessions { height: 1fr; }
    .detail-scroll Static { height: auto; }
    .dialog-hint { height: 1; color: #88949d; }
    ListView { background: #0b0d0f; }
    ListView > ListItem { padding: 0 1; }
    ListView > ListItem.--highlight { background: #15383c; }
    Button { background: #263038; }
    """
    BINDINGS = [
        Binding("ctrl+c", "cancel", "取消任务", priority=True),
        Binding("ctrl+q", "quit", "退出"),
        Binding("ctrl+n", "new", "新会话"),
        Binding("ctrl+d", "diff", "Diff"),
        Binding("ctrl+l", "focus_input", "输入"),
        Binding("ctrl+o", "toggle_tools", "工具详情"),
    ]

    class ServiceEvent(Message):
        def __init__(self, identity, data):
            self.identity, self.data = identity, data
            super().__init__()

    def __init__(self, service: SessionService, repo: str | None, config=None, session=None, autostart=False):
        super().__init__()
        self.service, self.repo = service, repo
        self.config = config or {}
        self.session = session
        self.autostart = autostart
        self.tool_widgets = {}
        self.history_lock = asyncio.Lock()
        self.menu_matches = []
        self.menu_index = 0
        self.menu_suppressed_text = None
        self.tools_expanded = False
        self.service.listeners.append(self.receive_event)

    def compose(self) -> ComposeResult:
        with Vertical(id="body"):
            yield VerticalScroll(id="transcript")
            yield Static(id="command-menu")
            yield TaskInput(id="input", soft_wrap=True)
            yield Static(id="usage-line")
            yield Static(id="key-hints")

    async def on_mount(self):
        if self.session:
            await self.render_history()
        else:
            await self.say("输入任务开始，或输入 / 查看命令。")
        self.query_one("#input").focus()
        self.update_status()
        self.set_interval(0.5, self.update_status)
        if self.autostart and self.session:
            self.launch(None)

    @property
    def menu_open(self):
        return bool(self.menu_matches) and self.query_one("#command-menu").display

    def on_text_area_changed(self, event):
        if event.text_area.id != "input":
            return
        editor = event.text_area
        editor.styles.height = min(8, max(3, editor.wrapped_document.height + 2))
        text = editor.text
        if text != self.menu_suppressed_text:
            self.menu_suppressed_text = None
        matches = (
            [entry for entry in COMMANDS if entry[0].startswith(text)]
            if (text.startswith("/") and not any(char.isspace() for char in text) and text != self.menu_suppressed_text)
            else []
        )
        if matches != self.menu_matches:
            self.menu_index = 0
        self.menu_matches = matches
        self.render_menu()

    def render_menu(self):
        menu = self.query_one("#command-menu", Static)
        menu.display = bool(self.menu_matches)
        if not self.menu_matches:
            return
        start = max(0, self.menu_index - 6)
        content = Text()
        for index, (name, params, description) in enumerate(self.menu_matches[start : start + 7], start):
            selected = index == self.menu_index
            content.append(
                f"{'›' if selected else ' '} {name:<15} {description}\n",
                style="bold #37d9d1 on #15383c" if selected else "#a8b5bb",
            )
        content.append("↑↓ 选择 · Tab / Enter 补全 · Esc 关闭", style="#88949d")
        menu.update(content)

    def move_menu(self, direction):
        self.menu_index = (self.menu_index + direction) % len(self.menu_matches)
        self.render_menu()

    def dismiss_menu(self):
        self.menu_suppressed_text = self.query_one("#input", TaskInput).text
        self.menu_matches = []
        self.render_menu()

    def complete_menu(self):
        name, params, _ = self.menu_matches[self.menu_index]
        value = name + (" " if params else "")
        self.menu_suppressed_text = value
        self.menu_matches = []
        self.render_menu()
        editor = self.query_one("#input", TaskInput)
        editor.load_text(value)
        editor.move_cursor((0, len(value)))

    def on_resize(self):
        if self.is_mounted:
            self.update_status()

    def show_detail(self, title, content):
        self.push_screen(DetailScreen(title, content))

    async def select_session(self, identity):
        if identity:
            try:
                self.session = self.service.load(identity)
                self.repo = self.session.info["source"]
                self.dismiss_menu()
                await self.render_history()
            except Exception as exc:
                await self.say(f"无法加载会话：{exc}")
        self.query_one("#input").focus()

    async def say(self, text, title=None):
        transcripts = list(self.query("#transcript"))
        if not transcripts:
            return
        classes = "chat user-message" if str(text).startswith("You: ") else "chat"
        if str(text).startswith(("Error:", "ValueError:", "RecoveryConflict:")):
            classes += " error-message"
        value = str(text).replace("You: ", "› ", 1) if classes.startswith("chat user") else str(text)
        value = value.removeprefix("Agent: ")
        widget = Static(Text(value), classes=classes)
        if title:
            widget = Collapsible(widget, title=title, collapsed=True)
        await transcripts[0].mount(widget)
        transcripts[0].scroll_end(animate=False)

    @staticmethod
    def tool_title(name, status, arguments, elapsed=None, observation=None, error=""):
        symbols = {"waiting": "○", "running": "◌", "completed": "✓", "failed": "×", "cancelled": "■"}
        labels = {"waiting": "等待", "running": "执行中", "completed": "完成", "failed": "失败", "cancelled": "已取消"}
        summary = (
            arguments.get("file_path")
            or arguments.get("path")
            or arguments.get("command")
            or arguments.get("query")
            or ""
        )
        summary = " ".join(str(summary).split())[:70]
        properties = (observation or {}).get("properties") or {}
        if properties.get("test_status"):
            summary += f" · {properties['test_status']} (exit {properties.get('return_code', '未知')})"
        elif status in {"failed", "cancelled"}:
            reason = error or properties.get("fail_reason") or (observation or {}).get("message", "")
            summary += " · " + " ".join(str(reason).split())[:100] if reason else ""
        return f"{symbols.get(status, '○')} {name}  {summary} · {labels.get(status, status)}" + (
            f" · {elapsed:.2f}s" if elapsed is not None else ""
        )

    async def upsert_tool(self, node_id, index, name, arguments, status, elapsed=None, observation=None, error=""):
        transcript = self.query_one("#transcript")
        follow = transcript.is_vertical_scroll_end
        key = (self.session.id, node_id, index)
        if key not in self.tool_widgets:
            body = Static()
            widget = Collapsible(body, title=name, collapsed=not self.tools_expanded)
            self.tool_widgets[key] = (widget, body)
            await self.query_one("#transcript").mount(widget)
        widget, body = self.tool_widgets[key]
        widget.remove_class("waiting", "running", "completed", "failed", "cancelled")
        widget.add_class(status)
        widget.title = self.tool_title(name, status, arguments, elapsed, observation, error)
        output = (observation or {}).get("message", error)
        if len(output) > 12000:
            output = output[:12000] + "\n…完整输出已保存到 events.jsonl 和 trajectory。"
        parts = [Text(json.dumps(arguments, ensure_ascii=False, indent=2))]
        if "old_str" in arguments and "new_str" in arguments:
            diff = "\n".join(
                ["- " + line for line in arguments["old_str"].splitlines()]
                + ["+ " + line for line in arguments["new_str"].splitlines()]
            )
            parts.append(Syntax(diff, "diff", word_wrap=True, background_color="#0b0d0f"))
        parts.append(Text("\n" + output))
        body.update(Group(*parts))
        if follow:
            transcript.call_after_refresh(transcript.scroll_end, animate=False)

    async def render_history(self):
        async with self.history_lock:
            await self._render_history()

    async def _render_history(self):
        await self.query_one("#transcript").remove_children()
        self.tool_widgets.clear()
        # Saved lifecycle records supply cancelled/failed states without replacing Node history.
        lifecycle = {}
        log = self.session.directory / "events.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    event = json.loads(line)
                    if event.get("scope") == "tool":
                        data = event["data"]
                        lifecycle[(data["node_id"], data["step"])] = event
                except (ValueError, KeyError):
                    continue
        for node in self.session.flow.root.get_all_nodes():
            if node.user_message:
                await self.say("You: " + node.user_message)
            if node.assistant_message:
                await self.say("Agent: " + node.assistant_message)
            for index, step in enumerate(node.action_steps):
                if step.action.name == "Finish":
                    if step.observation:
                        await self.say("Agent: " + step.action.finish_reason)
                    continue
                observation = step.observation.model_dump(mode="json") if step.observation else None
                event = lifecycle.get((node.node_id, index), {})
                status = event.get("event_type", "waiting")
                if observation:
                    properties = observation.get("properties") or {}
                    status = (
                        "failed"
                        if properties.get("fail_reason") or properties.get("test_status") == "failed"
                        else "completed"
                    )
                elapsed = (
                    (step.end_time - step.start_time).total_seconds() if step.end_time and step.start_time else None
                )
                await self.upsert_tool(
                    node.node_id,
                    index,
                    step.action.name,
                    step.action.model_dump(),
                    status,
                    elapsed,
                    observation,
                    (event.get("data") or {}).get("error", ""),
                )
        self.query_one("#transcript").scroll_end(animate=False)
        self.update_status()

    def update_status(self):
        narrow = self.size.width < 90
        if not self.session:
            footer = ("代码任务" if self.repo is not None else "普通聊天") + " · 尚未创建会话 · Token / 费用：未知"
        else:
            s = self.session
            usage = self.service.usage(s)
            rounds = max(0, len(s.flow.root.get_all_nodes()) - 1)
            cost = usage.get("completion_cost")
            cost_text = f"${cost:.6f}" if cost is not None else "未知"
            mode_label = "代码任务" if session_mode(s.info) == "coding" else "普通聊天"
            footer = f"{mode_label} · {rounds} 轮 · ↑{usage.get('prompt_tokens', '未知')} ↓{usage.get('completion_tokens', '未知')} Token · {cost_text}"
            if not narrow and session_mode(s.info) == "coding":
                footer += f" · worktree: {s.id[:8]}/worktree"
        for usage_line in self.query("#usage-line"):
            usage_line.update(Text(footer))
        hint = "Enter 发送 · Shift+Enter 换行 · / 命令 · Ctrl+C 取消 · Ctrl+Q 退出"
        if not narrow:
            hint += " · Ctrl+O 工具详情"
        for hints in self.query("#key-hints"):
            hints.update(Text(hint))

    def architecture(self):
        table = Table(title="RepoForge 实现结构 → CLI 功能", expand=True, border_style="#263038")
        table.add_column("实现模块", style="#37d9d1", ratio=2)
        table.add_column("界面 / 数据来源", ratio=3)
        for module, description in [
            ("cli/main.py", "启动参数 → 仓库、模型与配置"),
            ("cli/service.py · SessionService", "会话 → 创建、选择、恢复、导出、应用补丁"),
            ("flow/langgraph_flow.py", "执行 → 阶段、轮数、停止状态、checkpoint"),
            ("agent/session_agent.py + actions/local_coding.py", "工具 → 生命周期事件、参数、Observation、耗时"),
            ("Workspace + cli/worktree.py", "目标仓库 → 实际工作目录、文件变更、Diff"),
            ("Node + CompletionInvocation", "轨迹 → 对话历史、测试记录、Token 与费用"),
        ]:
            table.add_row(module, description)
        details = Text("\n本表是 RepoForge 自身的实现结构；下面是任务操作的目标仓库与会话。\n", style="#a8b5bb")
        if self.session:
            s = self.session
            config = s.info["config"]
            tests = self.service.tests(s)
            values = {
                "会话": s.id,
                "模型": config["model"],
                "模式": session_mode(s.info),
                "启用工具": ", ".join(effective_tools(s.info) + ["Finish（回答结束）"]),
                "状态 / 阶段": f"{s.info['status']} / {(s.flow.graph_state or {}).get('phase', 'ready')}",
                "目标来源仓库": s.info["source"] or "未绑定（普通聊天）",
                "工作目录 worktree": s.info["worktree"] or "无（不执行文件或命令工具）",
                "测试": tests[-1]["test_status"] if tests else "未运行",
                "用量（Node 汇总）": json.dumps(self.service.usage(s), ensure_ascii=False),
                "记录目录": str(s.directory),
                "记录文件": "session.json · state.json · events.jsonl · checkpoint.sqlite",
            }
        else:
            values = {
                "目标仓库": self.repo or "未绑定（普通聊天）",
                "新会话模型": self.config.get("model") or os.environ.get("REPOFORGE_MODEL", "deepseek/deepseek-chat"),
                "新会话工具": ", ".join(
                    (
                        self.config.get("tools", ["read", "search", "edit", "create", "bash", "tests"])
                        if self.repo is not None
                        else []
                    )
                    + ["Finish（回答结束）"]
                ),
                "会话 / worktree / 记录": "待创建",
                "测试": "未运行",
                "Token / 费用": "未知",
            }
        for label, value in values.items():
            details.append(f"{label}：", style="#37d9d1")
            details.append(str(value) + "\n")
        return Group(table, details)

    def receive_event(self, identity, data):
        self.post_message(self.ServiceEvent(identity, data))

    async def on_coding_app_service_event(self, event):
        async with self.history_lock:
            await self._render_service_event(event)

    async def _render_service_event(self, event):
        if not self.session or event.identity != self.session.id or not list(self.query("#transcript")):
            return
        data = event.data
        details = data.get("data") or {}
        if data["scope"] == "action" and data["event_type"] == "created":
            for index, args in enumerate(details.get("arguments", [])):
                name = args.get("action_args_class", details["name"]).removesuffix("Args")
                if name != "Finish":
                    await self.upsert_tool(data["node_id"], index, name, args, "waiting")
        elif data["scope"] == "tool" and details["name"] != "Finish":
            status = data["event_type"]
            observation = details.get("observation")
            if ((observation or {}).get("properties") or {}).get("test_status") == "failed":
                status = "failed"
            await self.upsert_tool(
                details["node_id"],
                details["step"],
                details["name"],
                details["arguments"],
                status,
                details.get("elapsed"),
                observation,
                details.get("error", ""),
            )
        elif data["event_type"] == "error":
            await self.say("Error: " + error_message(data.get("error") or details, self.session.info["config"]))
        self.update_status()

    def launch(self, message):
        session = self.session
        task = self.service.start(session, message)
        self.run_worker(self.execute(session, task), exclusive=False, group="sessions", exit_on_error=False)

    async def execute(self, s, task):
        try:
            await task
            if self.session is s:
                await self.render_history()
                if s.info["status"] not in {"terminal", "completed", "finished"}:
                    await self.say("任务已停止：" + s.info["status"])
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                if self.session is s:
                    await self.render_history()
                    await self.say("任务已取消；已保存状态。恢复时会核对未确认副作用。")
            elif isinstance(exc, Exception):
                if self.session is s:
                    stage = s.info.get("error_stage", "执行")
                    await self.say(f"{stage}: {type(exc).__name__}: {error_message(exc, s.info['config'])}")
            else:
                raise
        finally:
            self.update_status()

    async def on_task_input_submitted(self, event):
        await self.submit(event.text)

    async def submit(self, text):
        if not text.strip():
            return
        try:
            self.dismiss_menu()
            if text.startswith("/"):
                await self.command(text)
            else:
                if self.session and self.session.running:
                    raise ValueError("当前任务仍在运行。可继续编辑下一条消息，结束后再发送。")
                if not self.session:
                    self.session = await self.service.create(self.repo, text, self.config)
                    await self.render_history()
                    self.launch(None)
                else:
                    await self.say("You: " + text)
                    self.launch(text)
            self.query_one("#input", TaskInput).load_text("")
        except Exception as exc:
            await self.say(f"会话准备: {type(exc).__name__}: {error_message(exc, self.config)}")

    async def command(self, text):
        parts = shlex.split(text)
        command, args = parts[0], parts[1:]
        if command == "/help":
            help_text = Text()
            for name, params, description in COMMANDS:
                help_text.append(f"{name} {params}\n", style="#37d9d1")
                help_text.append(f"  {description}\n\n")
            self.show_detail("命令帮助", help_text)
        elif command == "/architecture":
            self.show_detail("项目结构与当前会话", self.architecture())
        elif command == "/new":
            self.session = None
            self.tool_widgets.clear()
            await self.query_one("#transcript").remove_children()
            if args:
                await self.submit(" ".join(args))
            else:
                await self.say("输入任务开始，或输入 / 查看命令。")
        elif command == "/sessions":
            self.push_screen(SessionPicker(self.service.list_sessions()), self.select_session)
        elif command == "/repo" and len(args) <= 1:
            if not args:
                await self.say("当前目标仓库：" + (self.repo or "未绑定（普通聊天）。/repo PATH 绑定仓库。"))
            else:
                if self.session and self.session.running:
                    raise ValueError("任务运行中不能切换仓库，请等待完成或先 /cancel。")
                repo = None if args[0] == "--clear" else (await asyncio.to_thread(worktree.validate, args[0]))[0]
                # Only change the current context after validation succeeds.
                self.repo = repo
                self.session = None
                self.tool_widgets.clear()
                await self.query_one("#transcript").remove_children()
                await self.say(
                    "已绑定仓库：" + repo + "。输入任务创建独立代码会话；原会话已保留。"
                    if repo
                    else "已切换到普通聊天，原会话已保留。输入问题开始。"
                )
        elif command == "/model" and not args:
            next_model = self.config.get("model") or os.environ.get("REPOFORGE_MODEL", "deepseek/deepseek-chat")
            current = self.session.info["config"]["model"] if self.session else next_model
            await self.say(f"当前配置的 API 模型：{current}\n下一会话模型：{next_model}")
        elif command == "/model" and len(args) == 1:
            self.config["model"] = args[0]
            await self.say("New session model: " + args[0])
        elif command == "/tools" and len(args) == 1:
            from repoforge.cli.service import TOOLS

            names = args[0].split(",")
            if set(names) - TOOLS.keys():
                raise ValueError("Unknown tools")
            self.config["tools"] = names
            await self.say("New session tools: " + args[0])
        elif command == "/config" and len(args) == 1:
            self.config.update(json.loads(Path(args[0]).expanduser().read_text()))
            await self.say("Loaded configuration for new sessions")
        elif command == "/quit":
            await self.action_quit()
        elif command == "/resume":
            if args:
                self.session = self.service.load(args[0])
                self.repo = self.session.info["source"]
                await self.render_history()
            if not self.session:
                raise ValueError("Specify a session ID")
            self.launch(None)
        elif command == "/cancel":
            await self.action_cancel()
        else:
            if not self.session:
                raise ValueError("Create or select a session first")
            s = self.session
            if command == "/diff":
                self.show_detail(
                    "Diff · 相对基准提交",
                    Syntax(self.service.diff(s) or "无变更", "diff", word_wrap=True, background_color="#0b0d0f"),
                )
            elif command == "/files":
                self.show_detail("文件变更 · worktree", Text(self.service.changes(s) or "无变更"))
            elif command == "/tests":
                tests = self.service.tests(s)
                self.show_detail(
                    "测试 · 实际执行记录",
                    Syntax(
                        json.dumps(tests, ensure_ascii=False, indent=2) if tests else "未运行测试（unknown）",
                        "json",
                        word_wrap=True,
                        background_color="#0b0d0f",
                    ),
                )
            elif command == "/usage":
                self.show_detail(
                    "用量 · Node 汇总",
                    Syntax(
                        json.dumps(self.service.usage(s), indent=2), "json", word_wrap=True, background_color="#0b0d0f"
                    ),
                )
            elif command == "/export" and len(args) == 1:
                target = self.service.export(s, args[0])
                await self.say(f"Exported: {target}. Official Harness: not run.")
            elif command == "/apply":
                if s.running:
                    raise ValueError("Wait for the task to stop before applying")
                patch = self.service.diff(s)
                if not patch:
                    raise ValueError("No patch to apply")

                def confirmed(answer):
                    if answer:
                        self.run_worker(self.apply_patch(s, patch), exit_on_error=False)

                self.push_screen(ApplyConfirmation(s.info["source"], patch), confirmed)
            else:
                raise ValueError("Unknown command or arguments; use /help")
        self.update_status()

    async def apply_patch(self, session, patch):
        try:
            self.service.apply(session, patch_hash(patch))
            await self.say("Patch applied to source repository; no commit created.")
        except Exception as exc:
            await self.say(f"Apply conflict: {exc}")

    def action_toggle_tools(self):
        self.tools_expanded = not self.tools_expanded
        for widget, _ in self.tool_widgets.values():
            widget.collapsed = not self.tools_expanded

    async def action_cancel(self):
        if self.session:
            await self.service.cancel(self.session)

    async def action_quit(self):
        await self.service.close()
        self.exit()

    async def action_new(self):
        await self.submit("/new")

    async def action_diff(self):
        await self.submit("/diff")

    def action_focus_input(self):
        self.query_one("#input").focus()
