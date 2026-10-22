"""Opt-in test timer for containers that suppress socketpair wakeups.

Usage: pytest -p tests.selector_tick ... . The real graph, subprocess and SQLite
implementations are used; this only prevents select() waiting indefinitely when
a worker queues a ready callback without delivering the socketpair wakeup.
"""
import asyncio
import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def selector_tick():
    async def tick():
        while True:
            await asyncio.sleep(0.01)
    task = asyncio.create_task(tick())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
