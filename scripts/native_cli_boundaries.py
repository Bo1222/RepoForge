"""Real CLI processes and recovery on a public fixture, with fixed model responses.

This is synthetic workflow evidence, never SWE-bench scoring evidence.
"""

import argparse
import asyncio
import os
from pathlib import Path
from unittest.mock import patch

from repoforge.actions.bash import BashArgs
from repoforge.actions.create_file import CreateFileArgs
from repoforge.actions.finish import FinishArgs
from repoforge.actions.local_coding import LocalTestArgs
from repoforge.actions.read_file import ReadFileArgs
from repoforge.actions.string_replace import StringReplaceArgs
from repoforge.cli.service import SessionService
from repoforge.cli.worktree import git, patch_hash
from repoforge.completion.base import CompletionResponse
from repoforge.completion.stats import CompletionAttempt, CompletionInvocation, Usage
from repoforge.completion.tool_call import ToolCallCompletionModel
from repoforge.evaluation.experiment import read_json, write_json
from repoforge.flow.durable import RecoveryConflict, workspace_fingerprint


async def run(bundle, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir()
    git(source, "init", "-q")
    (source / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (source / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n")
    (source / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(source, "add", ".")
    git(source, "-c", "user.name=NativeCLI", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "public CLI fixture")
    config = {**bundle["cli_config"], "env": {"PATH": os.environ["PATH"],
              "PYTEST_ADDOPTS": "", "PYTHONDONTWRITEBYTECODE": "1"}}
    service = SessionService(output / "sessions")
    outputs, calls, cases = [], [], []

    async def complete(model, messages, **kwargs):
        calls.append(messages)
        write_json(output / "model-contexts.json", calls)
        if not outputs:
            raise ValueError("Fixture responses exhausted; no provider fallback")
        response = outputs.pop(0)
        return response if isinstance(response, CompletionResponse) else CompletionResponse(structured_outputs=response)

    async def create(task, **overrides):
        session = await service.create(str(source), task, {**config, **overrides})
        env = {**session.info["config"]["env"], "PYTHONPATH": session.info["worktree"]}
        session.info["config"]["env"] = env
        session.workspace.environment.env = env
        session.info["initial_fingerprint"] = workspace_fingerprint(session.workspace, session.flow.checkpoint_path)
        service.save(session)
        return session

    async def must_conflict(session, reason):
        restarted = SessionService(service.directory)
        try:
            await restarted.start(restarted.load(session.id))
        except RecoveryConflict as exc:
            assert reason in str(exc), str(exc)
        else:
            raise AssertionError("Unsafe resume was accepted")
        finally:
            await restarted.close()

    try:
        with patch.object(ToolCallCompletionModel, "create_completion", complete):
            outputs.extend([
                [ReadFileArgs(file_path="calc.py"), LocalTestArgs(command="python -m pytest -q test_calc.py")],
                [StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b")],
                [LocalTestArgs(command="python -m pytest -q test_calc.py"), FinishArgs(finish_reason="verified fixture")],
            ])
            session = await create("Read, reproduce, repair and test the public addition fixture")
            await service.start(session)
            assert session.info["status"] == "terminal", session.info["status"]
            tests = service.tests(session)
            assert [row.get("recorded_status", row["test_status"]) for row in tests] == ["failed", "passed"]
            assert tests[-1]["test_status"] == "passed" and tests[-1]["return_code"] == 0
            assert "return a - b" in (source / "calc.py").read_text()
            cases.append({"case": "read_edit_test", "passed": True, "session_id": session.id,
                          "original_and_patched_tests": tests})
            # A terminal confirmed checkpoint resumes without replaying a call.
            before = len(calls)
            loaded = service.load(session.id)
            await service.start(loaded)
            assert len(calls) == before
            cases.append({"case": "confirmed_resume", "passed": True, "session_id": session.id})
            outputs.append([CreateFileArgs(path="followup.txt", file_text="continued\n"),
                            LocalTestArgs(command="python -m pytest -q test_calc.py"),
                            FinishArgs(finish_reason="followup verified")])
            await service.start(loaded, "Add followup.txt and test again")
            assert "Read, reproduce" in str(calls[-1]) and "Add followup.txt" in str(calls[-1])
            assert service.tests(loaded)[-1]["return_code"] == 0
            exported = service.export(loaded, str(output / "export"))
            patch_text = service.diff(loaded)
            try:
                service.apply(loaded, "wrong-preview-hash")
            except ValueError:
                pass
            else:
                raise AssertionError("Unconfirmed patch applied")
            # Conflict evidence uses only the fixture source, never a user checkout.
            (source / "calc.py").write_text("external fixture change\n")
            try:
                service.apply(loaded, patch_hash(patch_text))
            except ValueError:
                pass
            else:
                raise AssertionError("Patch applied to a changed source")
            (source / "calc.py").write_text("def add(a, b):\n    return a - b\n")
            service.apply(loaded, patch_hash(patch_text))
            assert "return a + b" in (source / "calc.py").read_text()
            cases.append({"case": "followup_export_apply_conflicts", "passed": True,
                          "session_id": session.id, "export": str(exported)})
            git(source, "reset", "--hard", "HEAD")
            (source / "followup.txt").unlink()

            before = len(calls)
            zero = await create("zero budget", max_cost=0)
            await service.start(zero)
            assert zero.info["status"] == "max_cost" and len(calls) == before
            cases.append({"case": "zero_budget", "passed": True, "session_id": zero.id})
            invocation = CompletionInvocation(model=config["model"], attempts=[CompletionAttempt(
                usage=Usage(completion_cost=0.02, prompt_tokens=20, completion_tokens=10))])
            outputs.append(CompletionResponse(structured_outputs=[ReadFileArgs(file_path="calc.py")],
                                               completion_invocation=invocation))
            cost = await create("cost budget", max_cost=0.01)
            await service.start(cost)
            assert cost.info["status"] == "max_cost" and service.usage(cost)["completion_cost"] == 0.02
            before = len(calls)
            await service.start(service.load(cost.id))
            assert len(calls) == before and service.usage(service.load(cost.id))["completion_cost"] == 0.02
            cases.append({"case": "cost_budget_resume", "passed": True, "session_id": cost.id})
            outputs.append([ReadFileArgs(file_path="calc.py")])
            rounds = await create("round budget", max_iterations=2)
            await service.start(rounds)
            assert rounds.info["status"] == "max_iterations"
            cases.append({"case": "round_budget", "passed": True, "session_id": rounds.id})

            outputs.append([BashArgs(command="echo started > started; sleep 30; echo late > late")])
            canceled = await create("cancel live process")
            running = service.start(canceled)
            marker = Path(canceled.info["worktree"]) / "started"
            for _ in range(200):
                if marker.exists():
                    break
                await asyncio.sleep(0.025)
            assert marker.exists() and canceled.workspace.environment.processes
            process_ids = [process.pid for process in canceled.workspace.environment.processes]
            before = len(calls)
            await service.cancel(canceled)
            assert running.cancelled() and not canceled.workspace.environment.processes
            for pid in process_ids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise AssertionError("Canceled command still live")
            await must_conflict(canceled, "Unknown")
            assert len(calls) == before and not (marker.parent / "late").exists()
            cases.append({"case": "cancel_unknown_resume_refused", "passed": True,
                          "session_id": canceled.id, "observed_live_processes": process_ids})

            outputs.append([BashArgs(command="echo started > started; sleep 30; echo late > late")])
            timed = await create("timeout live process", tool_timeout=0.2)
            try:
                await service.start(timed)
            except asyncio.TimeoutError:
                pass
            else:
                raise AssertionError("Tool did not time out")
            marker = Path(timed.info["worktree"]) / "started"
            assert marker.exists() and not timed.workspace.environment.processes
            before = len(calls)
            await must_conflict(timed, "Unknown")
            assert len(calls) == before and not (marker.parent / "late").exists()
            cases.append({"case": "timeout_unknown_resume_refused", "passed": True, "session_id": timed.id})
            drift = await create("initial drift")
            Path(drift.info["worktree"], "calc.py").write_text("external edit\n")
            before = len(calls)
            await must_conflict(drift, "changed")
            assert len(calls) == before
            cases.append({"case": "initial_workspace_drift", "passed": True, "session_id": drift.id})
    finally:
        await service.close()
        write_json(output / "result.json", {"manifest_hash": bundle["job"]["manifest_hash"],
            "cases": cases, "passed": len(cases) == 9 and all(row["passed"] for row in cases),
            "provider_calls": 0, "synthetic_workflow": True, "official_harness": "not_run",
            "fixture_model_calls": len(calls)})
    if len(cases) != 9:
        raise ValueError("Native CLI boundary probe incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(read_json(args.bundle), args.output))


if __name__ == "__main__":
    main()
