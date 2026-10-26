"""Separate-process fixture for confirmed-snapshot and unconfirmed-operation recovery."""

import asyncio
import json
import os
from pathlib import Path
import sys

from repoforge.actions.bash import BashArgs
from repoforge.actions.finish import FinishArgs
from repoforge.actions.read_file import ReadFileArgs
from repoforge.actions.string_replace import StringReplaceArgs
from repoforge.actions.local_coding import LocalTestArgs
from repoforge.actions.task_state import UpdateTaskStateArgs
from repoforge.cli.service import SessionService
from repoforge.completion.base import CompletionResponse
from repoforge.completion.tool_call import ToolCallCompletionModel


async def main():
    mode, root, repo = sys.argv[1:]
    managed = mode.startswith("managed-")
    mode = mode.removeprefix("managed-")
    root = Path(root)
    service = SessionService(root / "sessions")

    async def completion(self, messages, **kwargs):
        with (root / "model-calls").open("a") as log:
            log.write("call\n")
        if mode == "unknown":
            outputs = [BashArgs(command="echo started >> counter; sleep 120")]
        else:
            outputs = [
                ReadFileArgs(file_path="calc.py"),
                StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b"),
                BashArgs(command="echo executed >> counter"),
                LocalTestArgs(command="python -m pytest -q"),
                FinishArgs(finish_reason="recovered"),
            ]
            if managed:
                outputs.insert(0, UpdateTaskStateArgs(next_step="Run verification"))
        return CompletionResponse(structured_outputs=outputs)

    ToolCallCompletionModel.create_completion = completion
    if mode == "resume":
        s = service.load((root / "id").read_text())
        try:
            await service.start(s)
        except Exception as exc:
            (root / "error").write_text(type(exc).__name__ + ": " + str(exc))
            return
        service.export(s, str(root / "export"))
        return
    config = {"context_policy": {"context_window": 16384}} if managed else None
    s = await service.create(repo, "recovery task", config)
    (root / "id").write_text(s.id)
    saved = s.flow._on_snapshot

    async def snapshot(flow):
        await saved(flow)
        steps = flow.root.get_last_node().action_steps
        index = 2 if managed else 1
        if mode == "confirmed" and len(steps) > index and steps[index].observation:
            os._exit(23)

    s.flow._on_snapshot = snapshot
    task = service.start(s)
    if mode == "unknown":
        while not Path(s.info["worktree"], "counter").exists():
            await asyncio.sleep(0.01)
        (root / "process-group").write_text(str(next(iter(s.workspace.environment.processes)).pid))
        os._exit(23)
    await task


asyncio.run(main())
