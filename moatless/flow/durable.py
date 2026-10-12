"""Local outcome journal: commits results before LangGraph checkpoints advance.

An unfinished external operation is a conflict, never an invitation to replay it.
The journal and its result snapshot commit in one SQLite transaction. This does
not make external effects transactional; uncertain effects require reconciliation.
"""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import platform
from pathlib import Path
from typing import Any

from moatless.exceptions import RuntimeError
from moatless.flow.graph_state import GraphState
from moatless.node import Node
from moatless.workspace import Workspace


class RecoveryConflict(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def thread_id(project_id: str | None, trajectory_id: str | None, run_id: str) -> str:
    return digest([project_id, trajectory_id, run_id])


def default_checkpoint_path(project_id: str | None, trajectory_id: str | None, run_id: str) -> str:
    base = Path(os.environ.get("MOATLESS_DIR", "./.moatless"))
    return str(base / "checkpoints" / f"{thread_id(project_id, trajectory_id, run_id)}.sqlite")


def workspace_fingerprint(workspace: Workspace | None, excluded_path: str) -> dict:
    if workspace is None:
        return {"repository": None, "runtime": None}
    try:
        repo = workspace.repository
    except ValueError:
        repo = None
    data: dict[str, Any] = {
        "repository": None,
        "runtime": None,
        "host": platform.node(),
        "python": sys.version,
        "python_prefix": sys.prefix,
    }
    if repo is not None:
        location = getattr(repo, "repo_path", None)
        contents = {}
        commit = None
        if location:
            base = Path(location)
            proc = subprocess.run(["git", "-C", str(base), "rev-parse", "HEAD"], capture_output=True, timeout=10)
            if proc.returncode == 0:
                commit = proc.stdout.decode().strip()
                tracked = subprocess.run(
                    ["git", "-C", str(base), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                paths = [base / path.decode() for path in tracked.stdout.split(b"\0") if path]
            else:
                paths = [p for p in base.rglob("*") if p.is_file()]
            excluded = Path(excluded_path).resolve()
            for path in sorted(set(paths)):
                if any(p in {".git", "__pycache__", ".pytest_cache"} for p in path.parts):
                    continue
                resolved = str(path.resolve())
                if resolved == str(excluded) or resolved.startswith((f"{excluded}-", f"{excluded}.")):
                    continue  # database, WAL, SHM, thread lock sidecars
                contents[str(path.relative_to(base))] = (
                    hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "deleted"
                )
        else:
            contents = getattr(repo, "files", {})
        data["repository"] = {
            "class": type(repo).__name__,
            "head": commit,
            "shadow_mode": repo.shadow_mode,
            "path": str(location) if location else None,
            "content_hash": digest(contents),
        }
    runtime = workspace.runtime
    if runtime is not None:
        hook = getattr(runtime, "recovery_state", None)
        data["runtime"] = {
            "class": f"{type(runtime).__module__}.{type(runtime).__name__}",
            "instance_id": getattr(runtime, "instance_id", None),
            "state": hook() if hook else None,
            "recoverable": hook is not None,
        }
    environment = workspace._environment
    if environment is not None and environment is not runtime:
        hook = getattr(environment, "recovery_state", None)
        data["environment"] = {
            "class": f"{type(environment).__module__}.{type(environment).__name__}",
            "state": hook() if hook else None,
            "recoverable": hook is not None,
        }
    return data


def context_cache(root: Node) -> dict:
    """Preserve original file bases even when an action has changed actual disk."""
    result = {}
    for node in root.get_all_nodes():
        if node.file_context:
            result[str(node.node_id)] = {
                path: {
                    "base": file._cached_base_content,
                    "content": file._cached_content,
                    "was_edited": file.was_edited,
                    "was_viewed": file.was_viewed,
                    "is_new": file._is_new,
                }
                for path, file in node.file_context._files.items()
            }
    return result


class OutcomeJournal:
    def __init__(self, path: str, identity: str, signature: str):
        self.identity = identity
        self.signature = signature
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS flow_snapshots (
                thread_id TEXT PRIMARY KEY, signature TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS flow_operations (
                thread_id TEXT NOT NULL, operation_id TEXT NOT NULL, kind TEXT NOT NULL,
                status TEXT NOT NULL, PRIMARY KEY (thread_id, operation_id));
        """)

    def read(self) -> GraphState | None:
        row = self.connection.execute(
            "SELECT signature, state FROM flow_snapshots WHERE thread_id=?", (self.identity,)
        ).fetchone()
        if row is None:
            return None
        if row[0] != self.signature:
            raise RecoveryConflict("Flow/Agent configuration changed; use a new checkpoint_run_id for a new experiment")
        return json.loads(row[1])

    def pending(self) -> list[tuple[str, str]]:
        return self.connection.execute(
            "SELECT operation_id, kind FROM flow_operations WHERE thread_id=? AND status='started'", (self.identity,)
        ).fetchall()

    def save(self, state: GraphState, operation: str | None = None):
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO flow_snapshots VALUES (?, ?, ?)",
                (self.identity, self.signature, json.dumps(state)),
            )
            if operation:
                self.connection.execute(
                    "UPDATE flow_operations SET status='committed' WHERE thread_id=? AND operation_id=?",
                    (self.identity, operation),
                )

    def begin(self, state: GraphState, kind: str) -> str:
        operation = digest([state["current_node_id"], state["action_cursor"] if kind == "tool" else None, kind])
        with self.connection:
            row = self.connection.execute(
                "SELECT status FROM flow_operations WHERE thread_id=? AND operation_id=?", (self.identity, operation)
            ).fetchone()
            if row:
                raise RecoveryConflict(f"Operation {operation} already {row[0]}; refusing replay")
            self.connection.execute(
                "INSERT INTO flow_operations VALUES (?, ?, ?, 'started')", (self.identity, operation, kind)
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO flow_snapshots VALUES (?, ?, ?)",
                (self.identity, self.signature, json.dumps(state)),
            )
        return operation

    def close(self):
        self.connection.close()
