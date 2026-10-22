"""Shared local service for Textual, batch CLI, and programmatic clients."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from filelock import FileLock, Timeout

from moatless.actions.finish import Finish
from moatless.actions.local_coding import (
    LocalReadFile,
    LocalStringReplace,
    LocalCreateFile,
    LocalSearch,
    LocalBash,
    LocalRunTests,
)
from moatless.agent.session_agent import SessionAgent
from moatless.completion.tool_call import ToolCallCompletionModel
from moatless.exceptions import CompletionRuntimeError
from moatless.flow.flow import AgenticFlow
from moatless.flow.langgraph_flow import LangGraphFlow
from moatless.flow.durable import OutcomeJournal, RecoveryConflict, thread_id, workspace_fingerprint
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.message_history.managed import ManagedMessageHistoryGenerator
from moatless.repository.file import FileRepository
from moatless.repository.repository import InMemRepository
from moatless.workspace import Workspace
from moatless.cli.environment import ChatEnvironment, SessionEnvironment
from moatless.cli import worktree

SYSTEM_PROMPT = """You are a coding agent working in an isolated local Git worktree.
Read current files and search before editing. All tools use the same actual disk files.
Use Bash for exploration and RunTests for test commands so their exit status is recorded.
Do not claim tests passed without a successful RunTests observation. Fix failures when possible.
Do not commit, switch branches, or modify .git. Finish with a clear user-facing summary in finish_reason.
When the user adds requirements, use the existing history and implement them in this worktree.
For questions that need no file operations, answer directly in plain text.
For coding tasks, execute the necessary tools before reporting results.
"""
CHAT_PROMPT = """You are a conversational assistant. Answer the user's questions directly in plain text.
Use the existing conversation history. No repository is bound and no file or shell tools are available.
If the user asks to inspect or edit local code, explain that /repo PATH binds a Git repository.
Do not pretend to have read files, run commands, changed code, or tested anything.
"""


def session_mode(info):
    return info.get("mode", "coding")


def effective_tools(info):
    return info["config"]["tools"] if session_mode(info) == "coding" else []


def error_message(exc, config):
    # Never include a configured API credential in a user-visible diagnostic.
    message = str(exc)
    key = api_key({"model": os.environ.get("MOATLESS_MODEL", "deepseek/deepseek-chat"), **config})
    if key:
        message = message.replace(key, "[API key redacted]")
    return message


TOOLS = {
    "read": LocalReadFile,
    "search": LocalSearch,
    "edit": LocalStringReplace,
    "create": LocalCreateFile,
    "bash": LocalBash,
    "tests": LocalRunTests,
}


def atomic_json(path: Path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def api_key(config: dict) -> str | None:
    name = config.get("api_key_env")
    if name:
        return os.environ.get(name)
    model = config["model"]
    provider = model.split("/", 1)[0].upper()
    return os.environ.get(f"{provider}_API_KEY") or (os.environ.get("OPENAI_API_KEY") if "/" not in model else None)


def safe_config(config: dict, secret: str | None = None) -> dict:
    config = deepcopy(config)

    # Credentials must be supplied through the environment, never session files.
    def scrub(value):
        if isinstance(value, dict):
            return {
                k: (None if k.lower() in {"model_api_key", "api_key", "authorization"} else scrub(v))
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str) and secret:
            return value.replace(secret, "[API key redacted]")
        return value

    return scrub(config)


@dataclass
class Session:
    directory: Path
    info: dict
    flow: LangGraphFlow
    workspace: Workspace
    task: asyncio.Task | None = None

    @property
    def id(self):
        return self.info["id"]

    @property
    def running(self):
        return self.task is not None and not self.task.done()


class SessionService:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or os.environ.get("MOATLESS_CLI_DIR", "~/.moatless/cli")).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, Session] = {}
        self.listeners = []

    def list_sessions(self):
        result = []
        for path in self.directory.glob("*/session.json"):
            result.append(json.loads(path.read_text()))
        return sorted(result, key=lambda info: info["updated"], reverse=True)

    def _directory(self, identity: str):
        if not re.fullmatch(r"[0-9a-f]{32}", identity):
            raise ValueError("Invalid session ID; use /sessions to list IDs")
        return self.directory / identity

    async def create(self, repo: str | None, task: str, config: dict | None = None, completion_model=None):
        if not task.strip():
            raise ValueError("Task cannot be empty")
        config = {
            "model": os.environ.get("MOATLESS_MODEL", "openai/deepseek-flash"),
            "max_iterations": 30,
            "max_cost": 2.0,
            "model_timeout": 180,
            "tool_timeout": 120,
            "tools": list(TOOLS),
            "env": {},
            **(config or {}),
        }
        if config["model"] == "openai/deepseek-flash":
            config.setdefault("base_url", "https://api.deepseek.com")
            config.setdefault("api_key_env", "DEEPSEEK_API_KEY")
            # Preserve explicit overrides; the ordinary new-session default is non-thinking.
            config["params"] = deepcopy(config.get("params", {}))
            config["params"].setdefault("extra_body", {}).setdefault("thinking", {"type": "disabled"})
        if config["max_iterations"] < 2 or config["max_cost"] is not None and config["max_cost"] < 0:
            raise ValueError("max_iterations must be >=2 and max_cost must be nonnegative")
        from moatless.policy import ActionPolicyConfig
        if ActionPolicyConfig(**config.get("action_policy", {})).variant != "off":
            raise ValueError("ActionPolicy v1 is batch-only; CLI requires action_policy off")
        unknown = set(config["tools"]) - TOOLS.keys()
        if unknown:
            raise ValueError(f"Unknown tools: {sorted(unknown)}")
        if "model_api_key" in config or "api_key" in config:
            raise ValueError("Use api_key_env in config instead of storing an API key")
        identity = uuid.uuid4().hex
        directory = self._directory(identity)
        directory.mkdir(mode=0o700)
        try:
            mode = "coding" if repo is not None else "chat"
            source, base = (
                (await asyncio.to_thread(worktree.create, repo, directory / "worktree"))
                if repo is not None
                else (None, None)
            )
            model = completion_model or ToolCallCompletionModel(
                model=config["model"],
                model_base_url=config.get("base_url"),
                model_api_key=api_key(config),
                max_tokens=config.get("max_tokens", 4096),
                temperature=config.get("temperature", 0.0),
                reasoning_effort=config.get("reasoning_effort"),
                timeout=config["model_timeout"],
                message_cache=False,
                params=config.get("params", {}),
                thoughts_in_action=config.get("thoughts_in_action", False),
                allow_plain_text=True,
            )
            base_prompt = config.get("system_prompt") or (SYSTEM_PROMPT if mode == "coding" else CHAT_PROMPT)
            context = (
                f"\nRuntime configuration: API model identifier is {config['model']}. "
                "When asked about your model, report this configured identifier; "
                "do not infer the provider's internal backend version. "
                f"Session mode is {mode}. Available coding tools: {', '.join(config['tools']) if mode == 'coding' else 'none'}."
            )
            agent = SessionAgent(
                completion_model=model,
                system_prompt=base_prompt + context,
                actions=([TOOLS[name]() for name in config["tools"]] if mode == "coding" else []) + [Finish()],
                memory=ManagedMessageHistoryGenerator(policy=config["context_policy"]) if config.get("context_policy") else MessageHistoryGenerator(
                    include_file_context=False, include_git_patch=False, max_tokens_per_observation=4000
                ),
                shadow_mode=False,
            )
            flow = LangGraphFlow.create(
                message=task,
                agent=agent,
                project_id="cli",
                trajectory_id=identity,
                max_iterations=config["max_iterations"],
                max_cost=config["max_cost"],
                checkpoint_path=str(directory / "checkpoint.sqlite"),
                model_timeout=config["model_timeout"],
                tool_timeout=config["tool_timeout"],
            )
            info = {
                "id": identity,
                "title": task[:80],
                "mode": mode,
                "source": source,
                "base_commit": base,
                "worktree": str(directory / "worktree") if mode == "coding" else None,
                "config": safe_config(config),
                "status": "ready",
                "updated": datetime.now(timezone.utc).isoformat(),
                "official_harness": "not_run",
            }
            session = Session(directory, info, flow, self._workspace(info))
            info["initial_fingerprint"] = workspace_fingerprint(session.workspace, flow.checkpoint_path)
            self.sessions[identity] = session
            self._bind(session)
            self.save(session)
            return session
        except BaseException:
            if (directory / "worktree").exists():
                await asyncio.to_thread(
                    worktree.git, repo, "worktree", "remove", "--force", str(directory / "worktree")
                )
            shutil.rmtree(directory)
            raise

    def _workspace(self, info):
        if session_mode(info) == "chat":
            return Workspace(repository=InMemRepository(), environment=ChatEnvironment())
        return Workspace(
            repository=FileRepository(repo_path=info["worktree"], shadow_mode=False),
            environment=SessionEnvironment(info["worktree"], info["config"].get("env")),
        )

    def _bind(self, session):
        async def event(event):
            data = event.model_dump(mode="json")
            if event.scope == "action" and event.event_type == "created":
                # The generated steps already belong to the canonical Node history.
                node = session.flow.get_node_by_id(event.node_id)
                data["data"] = {
                    "name": event.action_name,
                    "arguments": [s.action.model_dump() for s in node.action_steps],
                }
            data = safe_config(data, secret=api_key(session.info["config"]))
            with (session.directory / "events.jsonl").open("a") as log:
                log.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
                log.flush()
            for listener in self.listeners:
                listener(session.id, data)

        session.flow._on_event = event

        async def snapshot(flow):
            self.save(session)

        session.flow._on_snapshot = snapshot

    def save(self, session):
        session.info["updated"] = datetime.now(timezone.utc).isoformat()
        # The flow and trajectory share a single atomically replaced envelope.
        state = safe_config(
            {"flow": session.flow.model_dump(), "trajectory": session.flow.get_trajectory_data()},
            secret=api_key(session.info["config"]),
        )
        atomic_json(session.directory / "state.json", state)
        atomic_json(session.directory / "session.json", session.info)

    def load(self, identity: str) -> Session:
        if identity in self.sessions:
            return self.sessions[identity]
        directory = self._directory(identity)
        info = json.loads((directory / "session.json").read_text())
        state = json.loads((directory / "state.json").read_text())
        state["flow"]["agent"]["completion_model"]["model_api_key"] = api_key(info["config"])
        flow = AgenticFlow.from_dicts(state["flow"], state["trajectory"])
        if not isinstance(flow, LangGraphFlow):
            raise ValueError("CLI sessions require LangGraphFlow")
        session = Session(directory, info, flow, self._workspace(info))
        self._bind(session)
        self.sessions[identity] = session
        return session

    def verify(self, session):
        if (
            session_mode(session.info) == "coding"
            and worktree.git(session.info["worktree"], "rev-parse", "HEAD").strip() != session.info["base_commit"]
        ):
            raise RecoveryConflict("Session worktree HEAD differs from the recorded base commit")
        path = Path(f"{session.flow.checkpoint_path}.outcomes.sqlite")
        expected = session.info.get("initial_fingerprint")
        if path.exists():
            journal = OutcomeJournal(
                str(path),
                thread_id("cli", session.id, session.flow.checkpoint_run_id),
                session.flow._configuration_signature(),
            )
            try:
                saved = journal.read()
                if journal.pending():
                    raise RecoveryConflict("Unknown side effects or model charges; automatic replay is disabled")
                if saved:
                    expected = saved["workspace_ref"].get("fingerprint")
            finally:
                journal.close()
        if expected and expected != workspace_fingerprint(session.workspace, session.flow.checkpoint_path):
            raise RecoveryConflict("Session workspace or environment changed since the last confirmed snapshot")

    def _refresh(self, session):
        """A client loaded before another run finished must use the latest canonical history."""
        info = json.loads((session.directory / "session.json").read_text())
        if info["updated"] != session.info["updated"]:
            state = json.loads((session.directory / "state.json").read_text())
            state["flow"]["agent"]["completion_model"]["model_api_key"] = api_key(info["config"])
            session.flow = AgenticFlow.from_dicts(state["flow"], state["trajectory"])
            session.info = info
            session.workspace = self._workspace(info)
            self._bind(session)

    def start(self, session: Session, message: str | None = None) -> asyncio.Task:
        if session.running:
            raise ValueError("This session already has a running task; wait or cancel before sending")
        if message is not None and not message.strip():
            raise ValueError("Message cannot be empty")
        session.task = asyncio.create_task(self._run(session, message))
        return session.task

    async def _run(self, session, message):
        lock = FileLock(str(session.directory / "session.lock"), timeout=0)
        try:
            lock.acquire()
        except Timeout as exc:
            raise RecoveryConflict("This session is running in another process") from exc
        try:
            self._refresh(session)
            await asyncio.to_thread(self.verify, session)
            # Existing library flows treat max_cost=0 as unlimited; the CLI's
            # explicit zero budget must stop before any model/tool operation.
            if session.flow.max_cost is not None and self.usage(session)["completion_cost"] >= session.flow.max_cost:
                session.info["status"] = "max_cost"
                return session.flow.root.get_last_node()
            key_env = session.info["config"].get("api_key_env")
            if key_env and not api_key(session.info["config"]):
                session.info["error_stage"] = "模型请求"
                raise CompletionRuntimeError(f"缺少模型密钥环境变量 {key_env}，请设置后再创建新会话。")
            if message:
                # Only completed turns accept follow-up messages. Interrupted turns must resume first.
                state = session.flow.graph_state
                if state and state["phase"] != "end":
                    raise RecoveryConflict("Resume the interrupted turn before appending requirements")
                session.flow.begin_new_run()
                session.flow.max_iterations = (
                    len(session.flow.root.get_all_nodes()) + session.info["config"]["max_iterations"]
                )
            session.info["status"] = "running"
            session.info.pop("completion_status", None)
            self.save(session)
            node = await session.flow.run(message=message, workspace=session.workspace)
            if node.completion_status:
                session.info["completion_status"] = node.completion_status
            reason = session.flow.get_trajectory_data().get("finish_reason")
            session.info["status"] = reason or "completed"
            return node
        except asyncio.CancelledError:
            session.info["status"] = "cancelled"
            raise
        except RecoveryConflict as exc:
            session.info["status"] = "recovery_conflict"
            session.info["error"] = error_message(exc, session.info["config"])
            raise
        except Exception as exc:
            state = session.flow.graph_state or {}
            stage = state.get("error_kind") or session.info.get("error_stage", "会话准备")
            session.info["error_stage"] = {
                "model": "模型请求",
                "model_rejection": "响应解析",
                "tool": "工具执行",
                "timeout": "超时",
                "control": "执行控制",
            }.get(stage, stage)
            session.info["status"] = "failed"
            session.info["error"] = error_message(exc, session.info["config"])
            raise
        finally:
            await session.workspace.environment.close()
            self.save(session)
            lock.release()

    async def cancel(self, session):
        if session.running:
            session.task.cancel()
            try:
                await session.task
            except asyncio.CancelledError:
                pass

    async def close(self):
        await asyncio.gather(*(self.cancel(s) for s in self.sessions.values()), return_exceptions=True)

    def _idle(self, session):
        if session.running:
            raise ValueError("Wait for the running task or cancel it first")
        lock = FileLock(str(session.directory / "session.lock"), timeout=0)
        try:
            lock.acquire()
        except Timeout as exc:
            raise RecoveryConflict("Session is running in another process") from exc
        return lock

    @staticmethod
    def require_repository(session):
        if session_mode(session.info) != "coding":
            raise ValueError("当前是普通聊天，尚未绑定仓库。请先输入 /repo /实际Git仓库路径。")

    def diff(self, session):
        self.require_repository(session)
        return worktree.diff(session.info["worktree"], session.info["base_commit"])

    def usage(self, session):
        return session.flow.root.total_usage().model_dump()

    def tests(self, session):
        if session.info["config"].get("context_policy"):
            from moatless.execution_memory import test_records
            node = session.flow.root.get_last_node()
            records = test_records(node, session.workspace)
            outputs = {f"{n.node_id}:{index}": step.observation.message
                       for n in node.get_trajectory() for index, step in enumerate(n.action_steps) if step.observation}
            return [dict(record, test_status=record["effective_status"],
                         recorded_status=record["status"], output=outputs[record["source_id"]],
                         node_id=int(record["source_id"].split(":")[0]))
                    for record in records]
        return [
            dict(node_id=n.node_id, **s.observation.properties, output=s.observation.message)
            for n in session.flow.root.get_all_nodes()
            for s in n.action_steps
            if s.observation and "test_status" in (s.observation.properties or {})
        ]

    def changes(self, session):
        self.require_repository(session)
        return worktree.git(session.info["worktree"], "status", "--short")

    def apply(self, session, confirmed_hash: str):
        self.require_repository(session)
        with self._idle(session):
            self._refresh(session)
            self.verify(session)
            patch = self.diff(session)
            if worktree.patch_hash(patch) != confirmed_hash:
                raise ValueError("Patch changed after preview; review it again")
            worktree.apply(session.info["source"], patch)

    def export(self, session, destination: str):
        with self._idle(session):
            self._refresh(session)
            target = Path(destination).expanduser().resolve()
            if target.exists():
                raise ValueError("Export destination must not already exist")
            if target.is_relative_to(session.directory) or (
                session.info["worktree"] and target.is_relative_to(Path(session.info["worktree"]))
            ):
                raise ValueError("Export outside the session worktree")
            target.mkdir(parents=True)
            self.save(session)
            for name in ("session.json", "state.json", "events.jsonl"):
                source = session.directory / name
                if source.exists():
                    shutil.copy2(source, target / name)
            patch = self.diff(session) if session_mode(session.info) == "coding" else ""
            (target / "patch.diff").write_text(patch)
            secret = api_key(session.info["config"])
            atomic_json(target / "trajectory.json", safe_config(session.flow.get_trajectory_data(), secret))
            atomic_json(target / "tests.json", safe_config(self.tests(session), secret))
            atomic_json(target / "usage.json", self.usage(session))
            instance = session.info["config"].get("instance_id")
            if (
                session_mode(session.info) == "coding"
                and instance
                and re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+", instance)
            ):
                prediction = {
                    "instance_id": instance,
                    "model_name_or_path": session.info["config"]["model"],
                    "model_patch": patch,
                }
                (target / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
            return target
