"""Public generation inputs, real patch cleanup, and readiness/recovery contracts."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from moatless.runtime.local import SweBenchLocalEnvironment
from moatless.exceptions import RuntimeError
from moatless.flow.durable import workspace_fingerprint
from moatless.repository.file import FileRepository
from moatless.workspace import Workspace


PUBLIC = {"instance_id": "pytest-dev__pytest-7490", "repo": "pytest-dev/pytest",
          "version": "6.0", "base_commit": "base", "problem_statement": "Repair issue"}
PATCH = "--- a/data.txt\n+++ b/data.txt\n@@ -1 +1 @@\n-old\n+new\n"
CREATE = "--- /dev/null\n+++ b/sub/new.txt\n@@ -0,0 +1 @@\n+new\n"


@pytest.fixture(autouse=True)
def skip_real_install(monkeypatch):
    monkeypatch.setenv("SKIP_INSTALL", "true")


@pytest.mark.asyncio
async def test_generation_rejects_gold_and_grading(tmp_path):
    with pytest.raises(ValueError, match="non-public"):
        SweBenchLocalEnvironment(tmp_path, {**PUBLIC, "test_patch": "secret"}, None, generation_only=True)
    runtime = SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True)
    await runtime._wait_for_install()
    assert runtime.test_spec is None
    with pytest.raises(RuntimeError, match="unavailable"):
        await runtime.swebench_evaluate("grade", PATCH)


@pytest.mark.asyncio
@pytest.mark.parametrize("finished", [False, True])
async def test_failed_install_is_never_ignored(tmp_path, finished):
    runtime = SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True)
    await runtime._install_task
    async def failure():
        return False
    runtime._install_task = asyncio.create_task(failure())
    if finished:
        await runtime._install_task
    with pytest.raises(RuntimeError, match="returned failure"):
        await runtime._wait_for_install()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError("test failure"), asyncio.CancelledError()])
async def test_run_tests_restores_exact_contents_even_on_exception(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("SKIP_CONDA_ACTIVATE", "true")
    (tmp_path / "data.txt").write_text("old\n")
    runtime = SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True)
    async def applied(**kwargs):
        assert (tmp_path / "data.txt").read_text() == "new\n"
        assert (tmp_path / "sub/new.txt").read_text() == "new\n"
        raise failure
    runtime._run_tests_applied = applied
    with pytest.raises(type(failure)):
        await runtime.run_tests(PATCH + CREATE, ["test.py"])
    assert (tmp_path / "data.txt").read_text() == "old\n"
    assert not (tmp_path / "sub").exists()


@pytest.mark.asyncio
async def test_resume_requires_checked_identity_and_does_not_reinstall(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True, resume_ready=True)
    identity = {"container_id": "container", "image_id": "image", "dependencies": "digest"}
    runtime = SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True,
                                      recovery_identity=identity, resume_ready=True)
    assert runtime._install_task is None
    assert runtime.recovery_state()["identity"] == identity
    workspace = Workspace(repository=FileRepository(repo_path=str(tmp_path)), runtime=runtime, environment=runtime)
    assert workspace_fingerprint(workspace, str(tmp_path / "checkpoint"))["runtime"]["recoverable"]
    opaque = SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True)
    await opaque._wait_for_install()
    workspace = Workspace(repository=FileRepository(repo_path=str(tmp_path)), runtime=opaque, environment=opaque)
    assert not workspace_fingerprint(workspace, str(tmp_path / "checkpoint"))["runtime"]["recoverable"]


@pytest.mark.asyncio
async def test_real_command_preserves_quotes_and_timeout_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("SKIP_CONDA_ACTIVATE", "true")
    runtime = SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True)
    await runtime._wait_for_install()
    code_output = await runtime._execute_command("printf '%s' 'a b; literal'", timeout=2)
    assert code_output == ("a b; literal", 0)
    output, code = await runtime._execute_command("sleep 5", timeout=0.05)
    assert code == -1 and "timed out" in output


@pytest.mark.asyncio
async def test_nonzero_exit_cannot_be_verified_by_passing_parser(tmp_path, monkeypatch):
    from moatless.testing.schema import TestResult, TestStatus
    runtime = SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True)
    runtime._execute_command = AsyncMock(return_value=("test passed", 1))
    monkeypatch.setattr("moatless.runtime.local.parse_log", lambda *a: [
        TestResult(name="test", file_path="test.py", status=TestStatus.PASSED)])
    results = await runtime.run_tests(test_files=["test.py"])
    assert any(r.status == TestStatus.ERROR for r in results)
    assert runtime.last_test_execution[0]["return_code"] == 1
