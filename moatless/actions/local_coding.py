"""Disk-based CLI tools sharing one worktree, with observations in the existing trajectory."""

import asyncio
import shlex

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.actions.read_file import ReadFileArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.actions.create_file import CreateFileArgs
from moatless.actions.grep_tool import GrepToolArgs
from moatless.actions.bash import BashArgs


class LocalReadFile(Action):
    args_schema = ReadFileArgs

    async def execute(self, args, file_context=None):
        try:
            text = await self.workspace.environment.read_file(args.file_path)
            lines = text.splitlines()
            start = args.start_line or 1
            end = min(args.end_line or start + 199, start + 199)
            if start < 1 or end < start:
                raise ValueError("Invalid line range")
            return Observation.create(
                message="\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines) if start <= i + 1 <= end)
            )
        except (OSError, ValueError) as exc:
            return Observation.create(message=str(exc), properties={"fail_reason": "read_error"})


class LocalStringReplace(Action):
    args_schema = StringReplaceArgs

    async def execute(self, args, file_context=None):
        try:
            env = self.workspace.environment
            text = await env.read_file(args.path)
            if text.count(args.old_str) != 1:
                raise ValueError("old_str must match exactly once; read the current disk file first")
            await env.write_file(args.path, text.replace(args.old_str, args.new_str, 1))
            return Observation.create(message=f"Updated {args.path}")
        except (OSError, ValueError) as exc:
            return Observation.create(message=str(exc), properties={"fail_reason": "edit_error"})


class LocalCreateFile(Action):
    args_schema = CreateFileArgs

    async def execute(self, args, file_context=None):
        try:
            env = self.workspace.environment
            target = env.path(args.path)

            # Exclusive creation prevents accidental overwrite, including concurrent creation.
            def write():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("x") as stream:
                    stream.write(args.file_text)

            await asyncio.to_thread(write)
            return Observation.create(message=f"Created {args.path}")
        except (OSError, ValueError) as exc:
            return Observation.create(message=str(exc), properties={"fail_reason": "create_error"})


class LocalSearch(Action):
    args_schema = GrepToolArgs

    async def execute(self, args, file_context=None):
        env = self.workspace.environment
        # No index required; grep includes newly created worktree files.
        command = (
            "grep -rnE --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=node_modules"
        )
        if args.include:
            command += " --include=" + shlex.quote(args.include)
        command += " -- " + shlex.quote(args.pattern) + " ."
        code, output = await env.command(command)
        return Observation.create(
            message="\n".join(output.splitlines()[: args.max_results]) or "No matches",
            properties={"return_code": code, **({"fail_reason": "search_error"} if code > 1 else {})},
        )


class LocalBash(Action):
    args_schema = BashArgs

    async def execute(self, args, file_context=None):
        code, output = await self.workspace.environment.command(args.command)
        return Observation.create(
            message=f"Exit code: {code}\n{output}",
            properties={"return_code": code, **({"fail_reason": "command_failed"} if code else {})},
        )


class LocalTestArgs(ActionArguments):
    """Execute a test command in the session worktree and record its exit status."""

    model_config = ConfigDict(title="RunTests")
    command: str = Field(..., description="Test command, for example python -m pytest -q")


class LocalRunTests(Action):
    args_schema = LocalTestArgs

    async def execute(self, args, file_context=None):
        # A trailing tail/tee must not turn a failed test process into success.
        code, output = await self.workspace.environment.command("set -o pipefail\n" + args.command)
        return Observation.create(
            message=f"Test exit code: {code}\n{output}",
            properties={
                "test_command": args.command,
                "return_code": code,
                "test_status": "passed" if code == 0 else "failed",
                **({"fail_reason": "tests_failed"} if code else {}),
            },
        )
