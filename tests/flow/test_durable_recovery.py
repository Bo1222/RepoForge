import asyncio
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

import pytest
from filelock import FileLock

from moatless.actions.respond import Respond
from moatless.flow.durable import OutcomeJournal, RecoveryConflict, thread_id
from moatless.flow.graph_state import initial_state
from moatless.repository.repository import InMemRepository
from moatless.runtime.runtime import NoEnvironment
from moatless.workspace import Workspace
from moatless.node import EvaluationResult
from tests.flow.test_langgraph_flow import make_flow, response, RespondArgs


def worker(directory, *options):
    return subprocess.run(
        [sys.executable, "-m", "scripts.recovery_smoke", "--directory", str(directory), *options],
        capture_output=True,
        text=True,
        timeout=40,
    )


@pytest.mark.parametrize("fault", ["after_generation", "before_tool", "after_effect", "after_commit"])
@pytest.mark.parametrize("shadow", [False, True])
def test_real_process_crash_boundaries(tmp_path, fault, shadow):
    options = ["--shadow"] if shadow else []
    first = worker(tmp_path, *options, "--fault", fault)
    assert first.returncode == 88, first.stderr
    calls_before = (tmp_path / "calls.jsonl").read_text()
    resumed = worker(tmp_path, *options)
    if fault in {"before_tool", "after_effect"}:
        assert resumed.returncode != 0
        assert "Unconfirmed external operations" in resumed.stderr
        assert (tmp_path / "calls.jsonl").read_text() == calls_before
        assert json.loads((tmp_path / "conflict.json").read_text())["error_kind"] == "recovery_conflict"
    else:
        assert resumed.returncode == 0, resumed.stderr
        result = json.loads((tmp_path / "result.json").read_text())
        assert result["finish_reason"] == "terminal"
        assert result["usage"]["completion_cost"] == 0.01
        assert "return a + b" in result["patch"] and "return a - b" in result["patch"]
        assert result["graph_state"]["action_cursor"] == 3
        calls = [json.loads(line)["kind"] for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        assert calls == ["model", "tool:0", "tool:1", "tool:2"]
        content = (tmp_path / "repo/calc.py").read_text()
        assert ("return a - b" if shadow else "return a + b") in content
        # Reopening a completed thread must not repeat effects or charges either.
        again = worker(tmp_path, *options)
        assert again.returncode == 0, again.stderr
        assert (tmp_path / "calls.jsonl").read_text().splitlines() == [json.dumps({"kind": c}) for c in calls]
        with sqlite3.connect(tmp_path / "checkpoint.sqlite") as conn:
            assert conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0] > 0
        with sqlite3.connect(tmp_path / "checkpoint.sqlite.outcomes.sqlite") as conn:
            assert conn.execute("SELECT count(*) FROM flow_operations WHERE status='started'").fetchone()[0] == 0
            assert conn.execute("SELECT count(*) FROM flow_operations").fetchone()[0] == 4


def test_changed_disk_refuses_recovery(tmp_path):
    assert worker(tmp_path, "--fault", "after_generation").returncode == 88
    (tmp_path / "repo/calc.py").write_text("changed externally\n")
    resumed = worker(tmp_path)
    assert "differs from the committed snapshot" in resumed.stderr
    assert (tmp_path / "calls.jsonl").read_text().count("model") == 1


def test_changed_git_commit_refuses_recovery(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    for command in [
        ["git", "init", "-q"],
        ["git", "add", "calc.py"],
        ["git", "-c", "user.name=Recovery Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"],
    ]:
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    assert worker(tmp_path, "--fault", "after_generation").returncode == 88
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Recovery Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "changed baseline",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert "differs from the committed snapshot" in worker(tmp_path).stderr


@pytest.mark.asyncio
async def test_thread_isolation_and_config_conflict(tmp_path):
    path = str(tmp_path / "state.sqlite")
    flow, model = make_flow([response(RespondArgs(message="done"))], checkpoint_path=path)
    await flow.run()
    usage = flow.total_usage()
    await flow.run()
    assert model._calls == 1 and flow.total_usage() == usage
    other, second_model = make_flow(
        [response(RespondArgs(message="other"))], checkpoint_path=path, checkpoint_run_id="repeat"
    )
    await other.run()
    assert second_model._calls == 1
    assert thread_id("p", "t", "default") != thread_id("p", "t", "repeat")
    assert thread_id("p", "t", "default") != thread_id("p2", "t", "default")
    assert thread_id("p", "t", "default") != thread_id("p", "t2", "default")
    flow.max_iterations += 1
    with pytest.raises(RecoveryConflict, match="configuration changed"):
        await flow.run()
    assert model._calls == 1


@pytest.mark.asyncio
async def test_concurrent_same_thread_rejected(tmp_path):
    path = str(tmp_path / "state.sqlite")
    flow, model = make_flow([response(RespondArgs(message="done"))], checkpoint_path=path)
    with FileLock(f"{path}.{thread_id('p', 't', 'default')}.lock"):
        with pytest.raises(RecoveryConflict, match="already running"):
            await flow.run()
    assert model._calls == 0


@pytest.mark.asyncio
async def test_runtime_state_and_shadow_mode_are_checked(tmp_path):
    class StatefulRuntime(NoEnvironment):
        version = "v1"

        def recovery_state(self):
            return {"version": self.version}

    runtime = StatefulRuntime()
    path = str(tmp_path / "state.sqlite")
    workspace = Workspace(repository=InMemRepository(files={"a": "base"}), runtime=runtime)
    flow, model = make_flow([response(RespondArgs(message="done"))], checkpoint_path=path)
    await flow.run(workspace=workspace)
    runtime.version = "v2"
    with pytest.raises(RecoveryConflict, match="differs"):
        await flow.run(workspace=workspace)
    assert model._calls == 1
    runtime.version = "v1"
    flow.agent.shadow_mode = False
    with pytest.raises(RecoveryConflict, match="configuration changed"):
        await flow.run(workspace=workspace)


@pytest.mark.asyncio
async def test_opaque_runtime_cannot_auto_recover(tmp_path):
    from moatless.runtime.runtime import RuntimeEnvironment

    class OpaqueRuntime(RuntimeEnvironment):
        async def run_tests(self, **kwargs):
            return []

    path = str(tmp_path / "state.sqlite")
    flow, model = make_flow([response(RespondArgs(message="done"))], checkpoint_path=path)
    workspace = Workspace(repository=InMemRepository(), runtime=OpaqueRuntime())
    await flow.run(workspace=workspace)
    with pytest.raises(RecoveryConflict, match="Opaque runtime"):
        await flow.run(workspace=workspace)
    assert model._calls == 1


@pytest.mark.asyncio
async def test_timeout_keeps_operation_uncertain_and_has_no_retry(tmp_path, monkeypatch):
    path = str(tmp_path / "state.sqlite")
    flow, model = make_flow([response(RespondArgs(message="done"))], checkpoint_path=path, tool_timeout=0.01)

    async def slow(self, *args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(Respond, "execute", slow)
    with pytest.raises(asyncio.TimeoutError):
        await flow.run()
    assert flow.graph_state["error_kind"] == "timeout"
    assert flow.graph_state["finish_reason"] == "tool_timeout"
    with pytest.raises(RecoveryConflict, match="Unconfirmed"):
        await flow.run()
    with pytest.raises(RecoveryConflict, match="Reconcile unfinished"):
        flow.begin_new_run()
    assert model._calls == 1


@pytest.mark.asyncio
async def test_snapshot_callback_failure_does_not_replay_committed_model(tmp_path):
    path = str(tmp_path / "state.sqlite")
    flow, model = make_flow([response(RespondArgs(message="done"))], checkpoint_path=path)

    async def fail_after_generation(flow):
        if flow.graph_state["phase"] == "policy":
            raise OSError("export unavailable")

    flow._on_snapshot = fail_after_generation
    with pytest.raises(OSError):
        await flow.run()
    assert model._calls == 1
    # Infrastructure errors are committed as graph errors; recovery needs an
    # explicit new run, rather than generating a response again automatically.
    assert flow.graph_state["error_kind"] == "graph"
    with pytest.raises(Exception, match="Saved graph"):
        await flow.run()
    assert model._calls == 1


def test_journal_started_model_refuses_replay(tmp_path):
    flow, _ = make_flow([])
    state = initial_state(flow.root, "p", "t", 0)
    journal = OutcomeJournal(str(tmp_path / "state.sqlite"), "thread", "signature")
    try:
        operation = journal.begin(state, "model")
        assert journal.pending() == [(operation, "model")]
        with pytest.raises(RecoveryConflict, match="already started"):
            journal.begin(state, "model")
        journal.save(state, operation)
        assert not journal.pending()
        with pytest.raises(RecoveryConflict, match="already committed"):
            journal.begin(state, "model")
    finally:
        journal.close()


@pytest.mark.asyncio
async def test_post_run_grading_uses_the_same_durable_snapshot(tmp_path):
    path = str(tmp_path / "state.sqlite")
    flow, _ = make_flow([response(RespondArgs(message="done"))], checkpoint_path=path)
    await flow.run()
    timestamp = datetime.now(timezone.utc)
    flow.root.get_last_node().evaluation_result = EvaluationResult(
        resolved=True, details={"offline_test": True}, start_time=timestamp, end_time=timestamp
    )
    flow.persist_snapshot()
    exported = flow.get_trajectory_data()
    restored, model = make_flow([], checkpoint_path=path)
    await restored.run()
    assert model._calls == 0
    assert restored.root.get_last_node().evaluation_result.resolved is True
    assert restored.graph_state["snapshot_version"] >= exported["graph_state"]["snapshot_version"]
    with sqlite3.connect(f"{path}.outcomes.sqlite") as conn:
        saved = json.loads(conn.execute("SELECT state FROM flow_snapshots").fetchone()[0])
        assert saved == restored.graph_state


@pytest.mark.asyncio
async def test_invalid_snapshot_is_a_conflict_without_overwriting_it(tmp_path):
    path = str(tmp_path / "state.sqlite")
    flow, model = make_flow([response(RespondArgs(message="done"))], checkpoint_path=path)
    await flow.run()
    with sqlite3.connect(f"{path}.outcomes.sqlite") as conn:
        saved = json.loads(conn.execute("SELECT state FROM flow_snapshots").fetchone()[0])
        saved["schema_version"] = 999
        conn.execute("UPDATE flow_snapshots SET state=?", (json.dumps(saved),))
    with pytest.raises(RecoveryConflict, match="Invalid durable snapshot"):
        await flow.run()
    assert model._calls == 1
    with sqlite3.connect(f"{path}.outcomes.sqlite") as conn:
        assert json.loads(conn.execute("SELECT state FROM flow_snapshots").fetchone()[0]) == saved
