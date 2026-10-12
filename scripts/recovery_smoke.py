"""Offline, two-process recovery probe; fault injection is opt-in.

Run with --directory PATH, then repeat the same command to reopen SQLite.
No model API, Docker, index or official grading is used.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.actions.respond import Respond, RespondArgs
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.agent.agent import ActionAgent
from moatless.flow.durable import OutcomeJournal, RecoveryConflict
from moatless.flow.langgraph_flow import LangGraphFlow
from moatless.repository.file import FileRepository
from moatless.workspace import Workspace
from scripts.migration_smoke import SOURCE, ScriptedCompletion, response


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--fault", choices=["after_generation", "before_tool", "after_effect", "after_commit"])
    args = parser.parse_args()
    directory = Path(args.directory)
    repo_path = directory / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    source = repo_path / "calc.py"
    if not source.exists():
        source.write_text(SOURCE)

    def count(kind):
        with (directory / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    model = ScriptedCompletion(
        [
            response(
                ReadFileArgs(file_path="calc.py"),
                StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b"),
                RespondArgs(message="done"),
            )
        ]
    )
    original_model = ScriptedCompletion.create_completion
    original_tool = ActionAgent.execute_action_step
    original_save, original_begin = OutcomeJournal.save, OutcomeJournal.begin

    async def generate(self, *a, **kw):
        count("model")
        return await original_model(self, *a, **kw)

    async def execute(self, node, cursor):
        count(f"tool:{cursor}")
        await original_tool(self, node, cursor)
        if args.fault == "after_effect" and cursor == 1:
            os._exit(88)

    def save(self, state, operation=None):
        original_save(self, state, operation)
        if args.fault == "after_generation" and state["phase"] == "policy":
            os._exit(88)
        if args.fault == "after_commit" and state["phase"] == "record" and state["action_cursor"] == 1:
            os._exit(88)

    def begin(self, state, kind):
        result = original_begin(self, state, kind)
        if args.fault == "before_tool" and kind == "tool" and state["action_cursor"] == 1:
            os._exit(88)
        return result

    ScriptedCompletion.create_completion = generate
    ActionAgent.execute_action_step = execute
    OutcomeJournal.save, OutcomeJournal.begin = save, begin
    agent = ActionAgent(
        completion_model=model,
        system_prompt="fixed",
        actions=[ReadFile(), StringReplace(), Respond()],
        shadow_mode=args.shadow,
    )
    flow = LangGraphFlow.create(
        message="fix add",
        agent=agent,
        project_id="recovery",
        trajectory_id="coding",
        checkpoint_path=str(directory / "checkpoint.sqlite"),
        checkpoint_run_id=args.run_id,
    )
    workspace = Workspace(repository=FileRepository(repo_path=str(repo_path), shadow_mode=args.shadow))
    try:
        await flow.run(workspace=workspace)
    except RecoveryConflict as exc:
        (directory / "conflict.json").write_text(json.dumps({"error_kind": "recovery_conflict", "error": str(exc)}))
        raise
    trajectory = flow.get_trajectory_data()
    trajectory["usage"] = flow.total_usage().model_dump()
    trajectory["patch"] = flow.root.get_last_node().file_context.generate_git_patch()
    (directory / "result.json").write_text(json.dumps(trajectory, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
