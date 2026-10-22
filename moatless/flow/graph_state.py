"""Versioned JSON snapshots and runtime dependencies for the single-track graph."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict

from moatless.agent.agent import ActionAgent
from moatless.node import Node
from moatless.workspace import Workspace

SCHEMA_VERSION = 1
Phase = Literal["prepare", "generate", "policy", "execute", "record", "commit", "end"]


class GraphState(TypedDict):
    # All channels use LangGraph's default overwrite reducer. In particular, never
    # append nodes, steps or Usage: each snapshot is a complete business version.
    schema_version: int
    snapshot_version: int
    project_id: str | None
    trajectory_id: str | None
    current_node_id: int
    action_cursor: int
    phase: Phase
    finish_reason: str | None
    error: str | None
    error_kind: str | None
    single_step: bool
    workspace_ref: dict[str, Any]
    business_snapshot: list[dict[str, Any]]
    context_cache: NotRequired[dict[str, Any]]
    policy_state: NotRequired[dict[str, Any]]


@dataclass
class GraphRuntime:
    """Never checkpoint this object (it contains live clients and dependencies)."""

    agent: ActionAgent
    workspace: Workspace | None


def snapshot_nodes(root: Node) -> list[dict[str, Any]]:
    return root.dump_as_list(exclude_none=True, exclude_unset=True)


def validate_state(state: GraphState) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported graph schema version: {state.get('schema_version')}")
    if state["snapshot_version"] < 0 or state["action_cursor"] < 0:
        raise ValueError("Snapshot version and action cursor must be nonnegative")
    ids = [n["node_id"] for n in state["business_snapshot"]]
    if len(ids) != len(set(ids)) or state["current_node_id"] not in ids:
        raise ValueError("Invalid or duplicate snapshot node IDs")
    current = next(n for n in state["business_snapshot"] if n["node_id"] == state["current_node_id"])
    if state["action_cursor"] > len(current.get("action_steps", [])):
        raise ValueError("Action cursor is beyond the generated response")


def restore_nodes(state: GraphState, workspace: Workspace | None = None) -> Node:
    """Reconstruct without mutating checkpoint dictionaries; rebind dependencies."""
    validate_state(state)
    repo = None
    runtime = None
    if workspace:
        try:
            repo = workspace.repository
        except ValueError:
            pass
        runtime = workspace.runtime
    root = Node.from_dict({"nodes": deepcopy(state["business_snapshot"])}, repo=repo, runtime=runtime)
    for node in root.get_all_nodes():
        for path, saved in state.get("context_cache", {}).get(str(node.node_id), {}).items():
            if node.file_context and path in node.file_context._files:
                file = node.file_context._files[path]
                file._cached_base_content = saved["base"]
                file._cached_content = saved["content"]
                file.was_edited, file.was_viewed, file._is_new = (
                    saved["was_edited"],
                    saved["was_viewed"],
                    saved["is_new"],
                )
    return root


def initial_state(
    root: Node, project_id: str | None, trajectory_id: str | None, current_node_id: int, single_step: bool = False
) -> GraphState:
    return GraphState(
        schema_version=SCHEMA_VERSION,
        snapshot_version=0,
        project_id=project_id,
        trajectory_id=trajectory_id,
        current_node_id=current_node_id,
        action_cursor=0,
        phase="prepare",
        finish_reason=None,
        error=None,
        error_kind=None,
        single_step=single_step,
        workspace_ref={"project_id": project_id, "trajectory_id": trajectory_id},
        business_snapshot=snapshot_nodes(root),
    )
