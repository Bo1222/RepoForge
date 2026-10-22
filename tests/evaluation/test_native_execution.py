"""Real namespace and process-lifecycle checks, without model calls or Docker."""

import asyncio
import shlex

import pytest

from moatless.evaluation.native_execution import NativeExecutor, NativeMount


def executor(tmp_path, **kwargs):
    repo = tmp_path / "public"
    repo.mkdir()
    inputs = tmp_path / "input"
    inputs.mkdir()
    return NativeExecutor([NativeMount(repo, "/testbed", True), NativeMount(inputs, "/input")],
                          network=False, **kwargs)


@pytest.mark.asyncio
async def test_namespace_exposes_only_public_mounts_and_explicit_environment(tmp_path, monkeypatch):
    private = tmp_path / "private-labels.txt"
    private.write_text("must not reach the generator")
    monkeypatch.setenv("HOST_ONLY_SECRET", "not inherited")
    runner = executor(tmp_path, environment={"PUBLIC_SETTING": "allowed"})
    (tmp_path / "input" / "bundle.json").write_text("public instance")
    result = await runner.run(["/bin/bash", "-c",
        f"test ! -e {shlex.quote(str(private))} && "
        "test ! -e /var/run/docker.sock && test -z \"${HOST_ONLY_SECRET+x}\" && "
        "test \"$PUBLIC_SETTING\" = allowed && cat /input/bundle.json && "
        "! touch /input/forbidden && echo edited > /testbed/result.txt"], timeout=10)
    assert result.return_code == 0, result.output
    assert "public instance" in result.output
    assert (tmp_path / "public" / "result.txt").read_text() == "edited\n"
    assert not (tmp_path / "input" / "forbidden").exists()


@pytest.mark.asyncio
async def test_real_timeout_kills_descendants_and_preserves_output(tmp_path):
    runner = executor(tmp_path)
    result = await runner.run(["/bin/bash", "-c",
        "echo started; (sleep 1; echo leaked > /testbed/late.txt) & wait"], timeout=0.2)
    assert result.timed_out and result.return_code == -1
    assert "started" in result.output
    await asyncio.sleep(1.1)
    assert not (tmp_path / "public" / "late.txt").exists()


@pytest.mark.asyncio
async def test_real_cancellation_kills_descendants(tmp_path):
    runner = executor(tmp_path)
    task = asyncio.create_task(runner.run(["/bin/bash", "-c",
        "echo started > /testbed/started.txt; (sleep 1; echo leaked > /testbed/late.txt) & wait"], timeout=10))
    for _ in range(100):
        if (tmp_path / "public" / "started.txt").exists() or task.done():
            break
        await asyncio.sleep(0.01)
    assert (tmp_path / "public" / "started.txt").exists(), "Cancellation requires a live child process"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(1.1)
    assert not (tmp_path / "public" / "late.txt").exists()


@pytest.mark.parametrize("target", ["/", "/proc", "/etc", "/usr", "/usr/private", "/home", "/x/../etc", "relative"])
def test_mount_cannot_replace_namespace_or_system_boundaries(tmp_path, target):
    with pytest.raises(ValueError):
        NativeExecutor([NativeMount(tmp_path, target, True)])


def test_overlapping_mounts_cannot_hide_readonly_boundary(tmp_path):
    with pytest.raises(ValueError, match="Overlapping"):
        NativeExecutor([NativeMount(tmp_path, "/input"), NativeMount(tmp_path, "/input/escape", True)])


def test_root_symlink_mount_is_rejected(tmp_path):
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        NativeExecutor([NativeMount(linked, "/input")])


def test_missing_bubblewrap_refuses_host_fallback(tmp_path, monkeypatch):
    runner = executor(tmp_path)
    monkeypatch.setattr("moatless.evaluation.native_execution.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="refusing unisolated"):
        runner.command(["/bin/true"])


@pytest.mark.asyncio
async def test_credentials_never_appear_in_argv_and_output_can_be_redacted(tmp_path):
    secret = "native-test-only-value"
    runner = executor(tmp_path, environment={"MODEL_KEY": secret})
    argv = ["/bin/bash", "-c", "echo \"$MODEL_KEY\""]
    assert secret not in " ".join(runner.command(argv))
    result = await runner.run(argv, timeout=10, redact=[secret])
    assert result.return_code == 0 and result.output == "[REDACTED]\n"
