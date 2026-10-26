"""Storage implementations for the job scheduler.

This package provides storage interfaces and implementations for the job scheduler,
including in-memory and Redis-based storage options.
"""

from repoforge.runner.storage.storage import JobStorage
from repoforge.runner.storage.memory import InMemoryJobStorage

__all__ = ["JobStorage", "InMemoryJobStorage"]
