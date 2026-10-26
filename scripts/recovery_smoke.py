"""Offline, two-process recovery probe; fault injection is opt-in.

Run with --directory PATH, then repeat the same command to reopen SQLite.
No model API, Docker, index or official grading is used.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from repoforge.actions.read_file import ReadFile, ReadFileArgs
from repoforge.actions.respond import Respond, RespondArgs
from repoforge.actions.string_replace import StringReplace, StringReplaceArgs
from repoforge.agent.agent import ActionAgent
from repoforge.flow.durable import OutcomeJournal, RecoveryConflict
from repoforge.flow.langgraph_flow import LangGraphFlow
from repoforge.repository.file import FileRepository
from repoforge.workspace import Workspace
from scripts.migration_smoke import SOURCE, ScriptedCompletion, response


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--swe-bundle", type=Path, help="Public pytest bundle in a provisioned container; real SWE runtime/conda, no model API")
    parser.add_argument("--fault", choices=["after_generation", "before_tool", "after_effect", "after_commit"])
    args = parser.parse_args()
    directory = Path(args.directory)
    repo_path = directory / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    source = repo_path / "calc.py"
    if not source.exists():
        source.write_text(SOURCE)
    runtime = None
    actions = [ReadFile(), StringReplace(), Respond()]
    scripted_actions = [ReadFileArgs(file_path="calc.py"),
                        StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b")]
    if args.swe_bundle:
        from repoforge.evaluation.generation import environment_identity, load_bundle
        from repoforge.evaluation.experiment import read_json, write_json
        from repoforge.runtime.local import SweBenchLocalEnvironment
        from repoforge.actions.run_tests import RunTests, RunTestsArgs
        from scripts.migration_smoke import TEST
        bundle = load_bundle(args.swe_bundle)
        if bundle["instance"]["repo"] != "pytest-dev/pytest":
            raise ValueError("Arithmetic recovery probe uses the real pytest testbed environment")
        test = repo_path / "test_calc.py"
        if not test.exists():
            test.write_text(TEST)
        identity = environment_identity(os.environ["GENERATION_IMAGE_ID"])
        ready = directory / "runtime-ready.json"
        if ready.exists() and read_json(ready) != identity:
            raise RecoveryConflict("Real runtime environment identity changed")
        runtime = SweBenchLocalEnvironment(repo_path, bundle["instance"], None, generation_only=True,
                                          recovery_identity=identity, resume_ready=ready.exists())
        await runtime._wait_for_install()
        write_json(ready, identity)
        if not args.shadow:
            original_tests = runtime.run_tests
            async def actual_file_tests(patch=None, **kwargs):
                return await original_tests(patch=None, **kwargs)
            runtime.run_tests = actual_file_tests
        actions.insert(2, RunTests())
        scripted_actions.append(RunTestsArgs(test_files=["test_calc.py"]))
    scripted_actions.append(RespondArgs(message="done"))

    def count(kind):
        with (directory / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    model = ScriptedCompletion(
        [
            response(*scripted_actions)
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
        if args.fault == "after_commit" and state["phase"] == "record" and state["action_cursor"] == (2 if args.swe_bundle else 1):
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
        actions=actions,
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
    workspace = Workspace(repository=FileRepository(repo_path=str(repo_path), shadow_mode=args.shadow),
                          runtime=runtime, environment=runtime)
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
    async def run():
        # Test-only timer for sandboxes suppressing cross-thread socketpair
        # wakeups. Real graph/SQLite operations and crash faults are unchanged.
        timer = None
        if os.environ.get("REPOFORGE_TEST_SELECTOR_TICK") == "1":
            loop = asyncio.get_running_loop()
            def tick():
                nonlocal timer
                timer = loop.call_later(0.01, tick)
            tick()
        try:
            await main()
        finally:
            if timer:
                timer.cancel()
    asyncio.run(run())
