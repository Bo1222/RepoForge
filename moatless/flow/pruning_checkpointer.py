"""AsyncSqliteSaver that keeps only the latest checkpoint per (thread_id, checkpoint_ns).

LangGraph persists a full GraphState snapshot per super-step (message history and
context_cache included), so graph jobs accumulate multi-GB checkpoint databases
(observed 1.3-2.5 GiB per job). moatless crash recovery is driven by the
OutcomeJournal (<checkpoint>.outcomes.sqlite), not by LangGraph checkpoint history,
and resume reads only the latest row (AsyncSqliteSaver.aget_tuple uses
``ORDER BY checkpoint_id DESC LIMIT 1`` plus that row's writes), so keeping one
checkpoint per thread/namespace preserves every recovery guarantee while bounding
disk usage to roughly one snapshot. Released jobs delete the file entirely
(moatless.evaluation.experiment_native.release_job_workspace).

Caveat: this assumes no DeltaChannel usage, no explicit ``checkpoint_id`` pinning and
no time-travel forks (moatless uses none). If DeltaChannel is introduced later,
keep-latest must preserve the ancestor chain up to the nearest delta snapshot (see
``BaseCheckpointSaver.aprune`` in langgraph for the contract).
"""

from collections.abc import Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

_LATEST = (
    "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?"
    " ORDER BY checkpoint_id DESC LIMIT 1"
)


class PruningAsyncSqliteSaver(AsyncSqliteSaver):
    """Keep only the newest checkpoint (and its writes) per thread and namespace."""

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint,
                   metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        # The just-written checkpoint id is the maximum (UUIDv6 ordering), so it is
        # the row kept. super().aput holds self.lock internally (non-reentrant):
        # prune only after it returned.
        result = await super().aput(config, checkpoint, metadata, new_versions)
        async with self.lock:
            await self._prune_thread(
                str(config["configurable"]["thread_id"]),
                str(config["configurable"].get("checkpoint_ns", "")),
            )
            await self.conn.commit()
        return result

    async def aput_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, object]],
                          task_id: str, task_path: str = "") -> None:
        await super().aput_writes(config, writes, task_id, task_path)
        # Pregel submits writes through a worker pool, so writes belonging to an
        # already-pruned checkpoint can land after the next aput. Re-prune here;
        # writes of the latest checkpoint survive because they are not deleted.
        async with self.lock:
            await self._prune_writes(
                str(config["configurable"]["thread_id"]),
                str(config["configurable"].get("checkpoint_ns", "")),
            )
            await self.conn.commit()

    async def _prune_thread(self, thread_id: str, checkpoint_ns: str) -> None:
        """Delete everything but the latest checkpoint and its writes. Lock held."""
        async with self.conn.cursor() as cur:
            await cur.execute(_LATEST, (thread_id, checkpoint_ns))
            row = await cur.fetchone()
            if row is None:
                await cur.execute(
                    "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ?",
                    (thread_id, checkpoint_ns))
                return
            keep = row[0]
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id != ?",
                (thread_id, checkpoint_ns, keep))
            await cur.execute(
                "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id != ?",
                (thread_id, checkpoint_ns, keep))

    async def _prune_writes(self, thread_id: str, checkpoint_ns: str) -> None:
        """Delete writes that no longer belong to the latest checkpoint. Lock held."""
        async with self.conn.cursor() as cur:
            await cur.execute(_LATEST, (thread_id, checkpoint_ns))
            row = await cur.fetchone()
            if row is None:
                await cur.execute(
                    "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ?",
                    (thread_id, checkpoint_ns))
                return
            await cur.execute(
                "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id != ?",
                (thread_id, checkpoint_ns, row[0]))

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        """BaseCheckpointSaver.aprune contract (unimplemented upstream in langgraph)."""
        if strategy not in {"keep_latest", "delete"}:
            raise ValueError(f"Unsupported prune strategy: {strategy}")
        await self.setup()
        async with self.lock, self.conn.cursor() as cur:
            for raw in thread_ids:
                thread_id = str(raw)
                if strategy == "delete":
                    await cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                    await cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
                    continue
                await cur.execute(
                    "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id = ?",
                    (thread_id,))
                for (checkpoint_ns,) in await cur.fetchall():
                    await cur.execute(_LATEST, (thread_id, checkpoint_ns))
                    keep = (await cur.fetchone())[0]
                    await cur.execute(
                        "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id != ?",
                        (thread_id, checkpoint_ns, keep))
                    await cur.execute(
                        "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id != ?",
                        (thread_id, checkpoint_ns, keep))
                await cur.execute(
                    "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns NOT IN"
                    " (SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id = ?)",
                    (thread_id, thread_id))
            await self.conn.commit()

    async def truncate_wal(self) -> None:
        """Fold WAL frames into the main database and truncate the sidecar file.

        Called once at the end of a clean run: running it per super-step would push
        the deleted-snapshot frames to the main file on every node. Freed pages are
        reused, so the main file stays bounded at roughly the latest snapshot size.
        """
        async with self.lock, self.conn.cursor() as cur:
            await cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await cur.fetchone()
