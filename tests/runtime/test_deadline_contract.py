"""One real deadline, bounded cleanup and cancellation propagation."""

import asyncio
import time
import os
from pathlib import Path
import shlex

import pytest

from moatless.exceptions import WorkspaceConsistencyError
from moatless.runtime.deadline import invoke, _unconfirmed
from moatless.runtime.local import SweBenchLocalEnvironment
from tests.runtime.test_generation_runtime import PUBLIC, PATCH, CREATE


@pytest.mark.asyncio
async def test_execution_and_cleanup_have_independent_bounded_budgets():
    released = asyncio.Event()
    entered = asyncio.Event()

    async def operation():
        try:
            await asyncio.Event().wait()
        finally:
            entered.set()
            await released.wait()  # intentionally uncooperative cleanup

    start = time.monotonic()
    try:
        with pytest.raises(WorkspaceConsistencyError, match="grace"):
            await invoke(operation(), timeout=0.03, cleanup_grace=0.04)
        assert entered.is_set() and 0.05 <= time.monotonic() - start < 0.5
        assert _unconfirmed
    finally:
        released.set()
        await asyncio.sleep(0.02)
    assert not _unconfirmed


@pytest.mark.asyncio
async def test_user_cancellation_propagates_after_cleanup():
    cleaned = []

    async def operation():
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.01)
            cleaned.append(True)

    task = asyncio.create_task(invoke(operation(), timeout=1, cleanup_grace=0.1))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == [True]


@pytest.mark.asyncio
async def test_real_test_process_killed_reaped_and_patch_restored(tmp_path, monkeypatch):
    monkeypatch.setenv("SKIP_INSTALL", "true")
    monkeypatch.setenv("SKIP_CONDA_ACTIVATE", "true")
    (tmp_path / "data.txt").write_text("old\n")
    (tmp_path / "data.txt").chmod(0o640)
    runtime = SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True)
    pidfile = tmp_path / "pid.txt"
    # exec keeps the child PID identical to the process-group leader.
    command = "exec /usr/bin/python3 -c " + shlex.quote(
        f"import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));print('started',flush=True);time.sleep(10)"
    )
    runtime._test_script = lambda targets: command
    _, timed_out = await invoke(
        runtime.run_tests(PATCH + CREATE, ["test.py"], managed_deadline=True), timeout=0.3, cleanup_grace=1
    )
    assert timed_out and runtime.last_test_cleanup_confirmed
    assert (tmp_path / "data.txt").read_text() == "old\n"
    assert (tmp_path / "data.txt").stat().st_mode & 0o777 == 0o640
    assert not (tmp_path / "sub").exists()
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_cleanup_failure_is_recovery_error_but_cancellation_still_propagates():
    async def failed_cleanup():
        try:
            await asyncio.Event().wait()
        finally:
            raise OSError("restore failed")

    with pytest.raises(WorkspaceConsistencyError, match="cleanup"):
        await invoke(failed_cleanup(), timeout=0.01, cleanup_grace=0.1)
    task = asyncio.create_task(invoke(failed_cleanup(), timeout=1, cleanup_grace=0.1))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
