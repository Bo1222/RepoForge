"""One invocation deadline plus an independently bounded cleanup grace."""

import asyncio
from repoforge.exceptions import WorkspaceConsistencyError

# Retain uncertain tasks until an adapter finishes cleanup; never silently
# discard a running external operation or permit another workspace operation.
_unconfirmed = set()


def _retain(task):
    _unconfirmed.add(task)

    def done(future):
        _unconfirmed.discard(future)
        if not future.cancelled():
            future.exception()

    task.add_done_callback(done)


async def _cancel_and_reap(task, grace):
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=grace)
    if not done:
        _retain(task)
        return False
    if not task.cancelled():
        exception = task.exception()
        if exception and not isinstance(exception, asyncio.TimeoutError):
            raise WorkspaceConsistencyError("External operation failed during cancellation cleanup") from exception
    return True


async def invoke(coroutine, *, timeout, cleanup_grace):
    """Return (result, timed_out); caller reads adapter partial buffers on timeout."""
    task = asyncio.create_task(coroutine)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if done:
            try:
                return task.result(), False
            except asyncio.TimeoutError:
                return None, True
        if not await _cancel_and_reap(task, cleanup_grace):
            raise WorkspaceConsistencyError("Test cleanup exceeded its grace; external effects are unconfirmed")
        return None, True
    except asyncio.CancelledError as cancelled:
        # User cancellation remains cancellation. The journal leaves an in-flight
        # operation unconfirmed even if cleanup cannot finish within the grace.
        try:
            await _cancel_and_reap(task, cleanup_grace)
        except Exception:
            pass  # Pending journal entry remains unconfirmed; cancellation wins.
        raise cancelled
