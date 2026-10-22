"""PruningAsyncSqliteSaver keeps one checkpoint per (thread, namespace) without replay."""

import sqlite3

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.base.id import uuid6

from moatless.actions.respond import RespondArgs
from moatless.flow.pruning_checkpointer import PruningAsyncSqliteSaver
from scripts.migration_smoke import response
from tests.flow.test_langgraph_flow import make_flow


def checkpoint(seq, payload=None):
    value = empty_checkpoint()
    value["id"] = str(uuid6(clock_seq=seq))
    if payload is not None:
        value["channel_values"] = {"payload": payload}
        value["channel_versions"] = {"payload": seq}
    return value


def config(thread="t1", ns="", checkpoint_id=None):
    value = {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
    if checkpoint_id:
        value["configurable"]["checkpoint_id"] = checkpoint_id
    return value


async def fetch(saver, query):
    cursor = await saver.conn.execute(query)
    try:
        return await cursor.fetchall()
    finally:
        await cursor.close()


async def seed(saver, thread, ns, value, parent=None):
    """Raw insert to build checkpoint history the pruning aput would not produce."""
    type_, blob = saver.serde.dumps_typed(value)
    await saver.conn.execute(
        "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,"
        " type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (thread, ns, value["id"], parent, type_, blob, b"{}"))
    await saver.conn.commit()


async def seed_write(saver, thread, ns, checkpoint_id, task_id, value):
    type_, blob = saver.serde.dumps_typed(value)
    await saver.conn.execute(
        "INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel,"
        " type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (thread, ns, checkpoint_id, task_id, 0, "messages", type_, blob))
    await saver.conn.commit()


@pytest.mark.asyncio
async def test_aput_keeps_only_latest_checkpoint_and_writes(tmp_path):
    async with PruningAsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
        c0, c1, c2 = checkpoint(0), checkpoint(1), checkpoint(2)
        await saver.aput(config(), c0, {"source": "loop", "step": 0}, {})
        await saver.aput_writes(config(checkpoint_id=c0["id"]), [("messages", ["m0"])], "task-0")
        await saver.aput(config(checkpoint_id=c0["id"]), c1, {"source": "loop", "step": 1}, {})
        await saver.aput_writes(config(checkpoint_id=c1["id"]), [("messages", ["m1"])], "task-1")
        await saver.aput(config(checkpoint_id=c1["id"]), c2, {"source": "loop", "step": 2}, {})
        await saver.aput_writes(config(checkpoint_id=c2["id"]), [("messages", ["m2"])], "task-2")
        assert await fetch(saver, "SELECT checkpoint_id FROM checkpoints") == [(c2["id"],)]
        assert await fetch(saver, "SELECT DISTINCT checkpoint_id FROM writes") == [(c2["id"],)]
        latest = await saver.aget_tuple(config())
        assert latest is not None and latest.checkpoint["id"] == c2["id"]
        assert [(write[1], write[2]) for write in latest.pending_writes] == [("messages", ["m2"])]


@pytest.mark.asyncio
async def test_late_writes_for_pruned_checkpoint_do_not_accumulate(tmp_path):
    async with PruningAsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
        c0, c1 = checkpoint(0), checkpoint(1)
        await saver.aput(config(), c0, {"source": "loop", "step": 0}, {})
        await saver.aput(config(checkpoint_id=c0["id"]), c1, {"source": "loop", "step": 1}, {})
        # A write batch for the already-pruned checkpoint can arrive late from the
        # worker pool; it must not accumulate.
        await saver.aput_writes(config(checkpoint_id=c0["id"]), [("messages", ["late"])], "task-late")
        assert await fetch(saver, "SELECT count(*) FROM writes") == [(0,)]


@pytest.mark.asyncio
async def test_aprune_keep_latest_per_namespace(tmp_path):
    async with PruningAsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
        await saver.setup()
        main = [checkpoint(seq) for seq in (0, 1, 2)]
        sub = [checkpoint(10 + seq) for seq in (0, 1, 2)]
        for index in range(3):
            await seed(saver, "t1", "", main[index])
            await seed(saver, "t1", "sub", sub[index])
            await seed_write(saver, "t1", "", main[index]["id"], f"m{index}", [index])
            await seed_write(saver, "t1", "sub", sub[index]["id"], f"s{index}", [index])
        other = checkpoint(20)
        await seed(saver, "t2", "", other)
        await saver.aprune(["t1"], strategy="keep_latest")
        assert await fetch(
            saver, "SELECT checkpoint_ns, checkpoint_id FROM checkpoints WHERE thread_id='t1' ORDER BY checkpoint_ns"
        ) == [("", main[2]["id"]), ("sub", sub[2]["id"])]
        assert await fetch(saver, "SELECT DISTINCT checkpoint_id FROM writes WHERE thread_id='t1'") == [
            (main[2]["id"],), (sub[2]["id"],)]
        assert await fetch(saver, "SELECT count(*) FROM checkpoints WHERE thread_id='t2'") == [(1,)]


@pytest.mark.asyncio
async def test_aprune_delete_strategy_removes_thread_rows(tmp_path):
    async with PruningAsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
        await saver.setup()
        value = checkpoint(0)
        await seed(saver, "t1", "", value)
        await seed_write(saver, "t1", "", value["id"], "task", ["v"])
        await seed(saver, "t2", "", checkpoint(1))
        await saver.aprune(["t1"], strategy="delete")
        assert await fetch(saver, "SELECT count(*) FROM checkpoints WHERE thread_id='t1'") == [(0,)]
        assert await fetch(saver, "SELECT count(*) FROM writes WHERE thread_id='t1'") == [(0,)]
        assert await fetch(saver, "SELECT count(*) FROM checkpoints WHERE thread_id='t2'") == [(1,)]
        with pytest.raises(ValueError, match="Unsupported prune strategy"):
            await saver.aprune(["t1"], strategy="unknown")


@pytest.mark.asyncio
async def test_wal_sidecar_truncated_and_latest_readable(tmp_path):
    path = tmp_path / "cp.sqlite"
    async with PruningAsyncSqliteSaver.from_conn_string(str(path)) as saver:
        c0, c1 = checkpoint(0, payload="x" * (2 << 20)), checkpoint(1)
        await saver.aput(config(), c0, {"source": "loop", "step": 0}, {})
        await saver.aput(config(checkpoint_id=c0["id"]), c1, {"source": "loop", "step": 1}, {})
        await saver.truncate_wal()
        wal = path.with_name(path.name + "-wal")
        assert not wal.exists() or wal.stat().st_size == 0
    async with PruningAsyncSqliteSaver.from_conn_string(str(path)) as reopened:
        latest = await reopened.aget_tuple(config())
        assert latest is not None and latest.checkpoint["id"] == c1["id"]


@pytest.mark.asyncio
async def test_flow_run_prunes_to_single_checkpoint_and_resume_is_charge_free(tmp_path):
    path = str(tmp_path / "state.sqlite")
    flow, model = make_flow([response(RespondArgs(message="done"))], checkpoint_path=path)
    await flow.run()
    assert model._calls == 1
    usage = flow.total_usage()
    await flow.run()
    assert model._calls == 1 and flow.total_usage() == usage
    with sqlite3.connect(path) as connection:
        checkpoint_ids = [row[0] for row in connection.execute("SELECT checkpoint_id FROM checkpoints")]
        assert len(checkpoint_ids) == 1
        stale = connection.execute(
            "SELECT count(*) FROM writes WHERE checkpoint_id != ?", (checkpoint_ids[0],)).fetchone()[0]
        assert stale == 0


@pytest.mark.asyncio
async def test_begin_new_run_keeps_one_checkpoint_per_thread(tmp_path):
    path = str(tmp_path / "state.sqlite")
    flow, model = make_flow(
        [response(RespondArgs(message="one")), response(RespondArgs(message="two"))], checkpoint_path=path)
    await flow.run()
    flow.begin_new_run()
    await flow.run()
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT thread_id, count(*) FROM checkpoints GROUP BY thread_id").fetchall()
    assert len(rows) == 2 and all(count == 1 for _, count in rows)
