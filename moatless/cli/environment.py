"""Cancelable local process groups; cancellation never falls back/replays a command."""

import asyncio
import os
from pathlib import Path
import signal

from moatless.environment.local import LocalBashEnvironment
from moatless.environment.base import BaseEnvironment, EnvironmentExecutionError
from moatless.runtime.native_tools import tool_process


class SessionEnvironment(LocalBashEnvironment):
    def __init__(self, cwd: str, env: dict[str, str] | None = None):
        super().__init__(cwd=cwd, env=env)
        self.processes = set()

    def path(self, name: str) -> Path:
        base = Path(self.cwd).resolve()
        path = (base / name).resolve()
        if not path.is_relative_to(base) or ".git" in path.relative_to(base).parts:
            raise ValueError("File path must stay inside the session worktree and outside .git")
        return path

    async def read_file(self, path: str) -> str:
        return await asyncio.to_thread(self.path(path).read_text)

    async def write_file(self, path: str, content: str):
        target = self.path(path)

        def write():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

        await asyncio.to_thread(write)

    async def command(self, command: str) -> tuple[int, str]:
        argv, environment = tool_process(["bash", "-c", command], {**os.environ, **(self.env or {})})
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self.processes.add(process)
        try:
            output, _ = await process.communicate()
            return process.returncode, output.decode(errors="replace")
        except asyncio.CancelledError:
            await self.stop(process)
            raise
        finally:
            self.processes.discard(process)

    async def stop(self, process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
        except asyncio.TimeoutError:
            pass
        # The shell can exit before a child ignoring SIGTERM; kill the group too.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.communicate()

    async def close(self):
        await asyncio.gather(*(self.stop(p) for p in list(self.processes)))

    async def execute(self, command: str, fail_on_error=False, patch=None) -> str:
        if patch:
            raise ValueError("Session tools use actual files; shadow patches are unsupported")
        code, output = await self.command(command)
        if code and fail_on_error:
            raise EnvironmentExecutionError(f"Command exited {code}", code, output)
        return output


class ChatEnvironment(BaseEnvironment):
    """Conversation sessions have no filesystem or process execution capability."""

    def recovery_state(self):
        return {"mode": "chat", "version": 1}

    async def execute(self, command, fail_on_error=False, patch=None):
        raise ValueError("普通聊天不能执行命令，请先通过 /repo PATH 绑定仓库。")

    async def read_file(self, path):
        raise ValueError("普通聊天不能读取文件，请先通过 /repo PATH 绑定仓库。")

    async def write_file(self, path, content):
        raise ValueError("普通聊天不能修改文件，请先通过 /repo PATH 绑定仓库。")

    async def close(self):
        pass
