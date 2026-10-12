"""LangGraph-controlled single-track flow with optional local durable recovery."""

import asyncio
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import Field, PrivateAttr

from moatless.context_data import current_node_id
from moatless.exceptions import CompletionError, RejectError, RuntimeError
from moatless.flow.events import NodeExpandedEvent
from moatless.flow.durable import (
    OutcomeJournal,
    RecoveryConflict,
    context_cache,
    default_checkpoint_path,
    digest,
    thread_id,
    workspace_fingerprint,
)
from moatless.flow.flow import AgenticFlow
from moatless.flow.graph_state import GraphRuntime, GraphState, initial_state, restore_nodes, snapshot_nodes
from moatless.flow.schema import FlowStatus
from moatless.node import Node


class LangGraphFlow(AgenticFlow):
    graph_recursion_limit: int = Field(default=10000, ge=1)
    checkpoint_path: str | None = None
    checkpoint_run_id: str = "default"
    model_timeout: float | None = Field(default=None, gt=0)
    tool_timeout: float | None = Field(default=None, gt=0)

    _graph_state: GraphState | None = PrivateAttr(default=None)
    _loaded_snapshot_version: int = PrivateAttr(default=-1)
    _pending_exception: Exception | None = PrivateAttr(default=None)
    _compiled_graph: Any = PrivateAttr(default=None)
    _graph_config: dict = PrivateAttr(default_factory=dict)
    _journal: Any = PrivateAttr(default=None)
    _checkpointer: Any = PrivateAttr(default=None)
    _recovery_conflict: bool = PrivateAttr(default=False)
    _on_snapshot: Any = PrivateAttr(default=None)
    _unconfirmed_operation: bool = PrivateAttr(default=False)
    _last_durable_version: int = PrivateAttr(default=-1)

    def _configuration_signature(self) -> str:
        return digest(
            {
                "agent": self.agent.model_dump(exclude_none=True),
                "max_iterations": self.max_iterations,
                "max_cost": self.max_cost,
                "model_timeout": self.model_timeout,
                "tool_timeout": self.tool_timeout,
            }
        )

    def persist_snapshot(self):
        """Commit post-run annotations (e.g. grading) before exporting JSON."""
        self._sync_snapshot()
        if not self.checkpoint_path or self._journal or self._recovery_conflict or self._unconfirmed_operation:
            return
        if not self._graph_state or self._loaded_snapshot_version == -1 or self._last_durable_version == -1:
            return
        identity = thread_id(self.project_id, self.trajectory_id, self.checkpoint_run_id)
        lock = FileLock(f"{self.checkpoint_path}.{identity}.lock", timeout=0)
        try:
            with lock:
                journal = OutcomeJournal(
                    f"{self.checkpoint_path}.outcomes.sqlite", identity, self._configuration_signature()
                )
                try:
                    saved = journal.read()
                    if journal.pending() or not saved or saved["snapshot_version"] != self._last_durable_version:
                        raise RecoveryConflict("Durable snapshot advanced elsewhere; refusing stale trajectory export")
                    if self._graph_state["snapshot_version"] != self._last_durable_version:
                        journal.save(self._capture_recovery_state(self._graph_state))
                        self._last_durable_version = self._graph_state["snapshot_version"]
                finally:
                    journal.close()
        except Timeout as exc:
            raise RecoveryConflict("This checkpoint thread is already running") from exc

    def restore_trajectory_metadata(self, trajectory: dict):
        if trajectory.get("graph_state"):
            self._graph_state = {
                **deepcopy(trajectory["graph_state"]),
                "business_snapshot": deepcopy(trajectory["nodes"]),
            }
            self._root = restore_nodes(self._graph_state)

    def begin_new_run(self):
        """Explicit restart/continuation gets a fresh operation/checkpoint identity."""
        if self._journal:
            raise RecoveryConflict("Cannot reset a running graph")
        if not self.checkpoint_path:
            self.checkpoint_path = default_checkpoint_path(self.project_id, self.trajectory_id, self.checkpoint_run_id)
        if self.checkpoint_path and Path(f"{self.checkpoint_path}.outcomes.sqlite").exists():
            identity = thread_id(self.project_id, self.trajectory_id, self.checkpoint_run_id)
            lock = FileLock(f"{self.checkpoint_path}.{identity}.lock", timeout=0)
            try:
                with lock:
                    journal = OutcomeJournal(
                        f"{self.checkpoint_path}.outcomes.sqlite", identity, self._configuration_signature()
                    )
                    try:
                        if journal.pending():
                            raise RecoveryConflict(
                                "Reconcile unfinished external operations before resetting this trajectory"
                            )
                    finally:
                        journal.close()
            except Timeout as exc:
                raise RecoveryConflict("Cannot reset a running graph") from exc
        self.checkpoint_run_id = str(uuid.uuid4())
        self._graph_state = None
        self._loaded_snapshot_version = self._last_durable_version = -1
        self._recovery_conflict = self._unconfirmed_operation = False

    def _capture_recovery_state(self, state: GraphState) -> GraphState:
        updated = {**state, "context_cache": context_cache(self.root)}
        updated["workspace_ref"] = {
            **state["workspace_ref"],
            "fingerprint": workspace_fingerprint(self.agent._workspace, self.checkpoint_path),
        }
        self._graph_state = updated
        return updated

    async def run(self, message=None, workspace=None, node_id=None) -> Node:
        if not self.checkpoint_path:
            return await super().run(message, workspace, node_id)
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        identity = thread_id(self.project_id, self.trajectory_id, self.checkpoint_run_id)
        lock = FileLock(f"{self.checkpoint_path}.{identity}.lock", timeout=0)
        try:
            lock.acquire()
        except Timeout as exc:
            self._recovery_conflict = True
            raise RecoveryConflict("This checkpoint thread is already running") from exc
        signature = self._configuration_signature()
        self._recovery_conflict = False
        try:
            # Independent databases prevent a synchronous journal transaction
            # from blocking the event loop while the async saver holds its lock.
            self._journal = OutcomeJournal(f"{self.checkpoint_path}.outcomes.sqlite", identity, signature)
            async with AsyncSqliteSaver.from_conn_string(self.checkpoint_path) as saver:
                self._checkpointer = saver
                try:
                    return await super().run(message, workspace, node_id)
                finally:
                    # Never certify state around an unfinished external operation.
                    if not self._recovery_conflict and not self._journal.pending() and self._graph_state:
                        self._sync_snapshot()
                        self._journal.save(self._capture_recovery_state(self._graph_state))
                        self._last_durable_version = self._graph_state["snapshot_version"]
        finally:
            self._checkpointer = None
            if self._journal:
                self._journal.close()
                self._journal = None
            lock.release()

    def _journal_node(self, name, fn):
        async def execute(state: GraphState, runtime: Runtime[GraphRuntime]):
            operation = None
            if self._journal:
                node = self._load(state, runtime)
                kind = (
                    "tool"
                    if name == "execute_action"
                    else ("model" if name == "generate_actions" and not node.action_steps else None)
                )
                if kind:
                    operation = self._journal.begin(self._capture_recovery_state(state), kind)
                    self._unconfirmed_operation = True
            result = await fn(state, runtime)
            if self._journal:
                result = self._capture_recovery_state(result)
                # Cancellation/timeout cannot confirm external effects or charges.
                self._journal.save(result, None if result["error_kind"] == "timeout" else operation)
                self._last_durable_version = result["snapshot_version"]
                self._unconfirmed_operation = bool(self._journal.pending())
            if self._on_snapshot and not self._unconfirmed_operation:
                await self._on_snapshot(self)
            return result

        return execute

    async def _emit_event(self, event):
        if event.scope == "flow" and event.event_type == "error":
            kind = "recovery_conflict" if self._recovery_conflict else (
                self._graph_state.get("error_kind") if self._graph_state else None
            )
            event.data = {**(event.data or {}), "error_kind": kind or "control"}
        await super()._emit_event(event)

    @property
    def graph_state(self) -> GraphState | None:
        """Return an independent JSON snapshot, never live Workspace/client objects."""
        self._sync_snapshot()
        return deepcopy(self._graph_state)

    def _sync_snapshot(self):
        # The base lifecycle may annotate an exception after _run has returned.
        # Export that annotation from the same versioned snapshot as the trajectory.
        if self._recovery_conflict or self._unconfirmed_operation or self._loaded_snapshot_version == -1:
            return
        if self._graph_state is not None:
            nodes = snapshot_nodes(self.root)
            if nodes != self._graph_state["business_snapshot"]:
                self._graph_state = {
                    **self._graph_state,
                    "business_snapshot": nodes,
                    "snapshot_version": self._graph_state["snapshot_version"] + 1,
                }
                self._loaded_snapshot_version = self._graph_state["snapshot_version"]

    def get_trajectory_data(self) -> dict:
        state = self.graph_state
        if state is None:
            return super().get_trajectory_data()
        return {
            "trajectory_id": self.trajectory_id,
            "project_id": self.project_id,
            "metadata": self.metadata,
            "nodes": deepcopy(state["business_snapshot"]),
            "finish_reason": state["finish_reason"],
            "graph_state": {k: deepcopy(v) for k, v in state.items() if k != "business_snapshot"},
        }

    def is_finished(self) -> str | None:
        if self._graph_state and self._graph_state["finish_reason"] not in {None, "single_step"}:
            return self._graph_state["finish_reason"]
        return self._legacy_finish()

    def _legacy_finish(self) -> str | None:
        cost = self.total_usage().completion_cost
        if self.max_cost and cost and cost >= self.max_cost:
            return "max_cost"
        if len(self.root.get_all_nodes()) >= self.max_iterations:
            return "max_iterations"
        if self.root.get_last_node().is_terminal():
            return "terminal"
        return None

    @property
    def status(self) -> FlowStatus:
        if self._graph_state and self._graph_state["error"]:
            return FlowStatus.ERROR
        return super().status

    def _load(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> Node:
        # The live tree is a working copy of a particular snapshot version. Reuse
        # it within one invocation; a different version must be reconstructed.
        if self._loaded_snapshot_version != state["snapshot_version"]:
            self._root = restore_nodes(state, runtime.context.workspace)
            self._loaded_snapshot_version = state["snapshot_version"]
        node = self.get_node_by_id(state["current_node_id"])
        if node is None:
            raise ValueError(f"Node {state['current_node_id']} is absent from snapshot")
        current_node_id.set(node.node_id)
        return node

    def _save(self, state: GraphState, **updates) -> GraphState:
        next_state = {
            **state,
            **updates,
            "snapshot_version": state["snapshot_version"] + 1,
            "business_snapshot": snapshot_nodes(self.root),
        }
        self._graph_state = next_state
        self._loaded_snapshot_version = next_state["snapshot_version"]
        return next_state

    def _fail(self, state: GraphState, node: Node, exc: Exception, kind: str, reason: str) -> GraphState:
        if not node.error:
            node.error = f"{type(exc).__name__}: {exc}"
        self._pending_exception = exc
        return self._save(
            state,
            phase="commit",
            current_node_id=node.node_id,
            finish_reason=reason,
            error=f"{type(exc).__name__}: {exc}",
            error_kind=kind,
        )

    def _create_next_node(self, parent: Node) -> Node:
        # max+1 also works for trajectories whose IDs are not contiguous.
        child = Node(
            node_id=max(n.node_id for n in self.root.get_all_nodes()) + 1,
            parent=parent,
            file_context=parent.file_context.clone() if parent.file_context else None,
        )
        parent.add_child(child)
        return child

    async def _prepare_iteration(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        node = self._load(state, runtime)
        if not state["single_step"]:
            reason = self._legacy_finish()
            if reason:
                return self._save(state, phase="end", finish_reason=reason)
        try:
            if node.is_expandable() and node.is_executed():
                child = self._create_next_node(node)
                await self._emit_event(NodeExpandedEvent(node_id=node.node_id, child_node_id=child.node_id))
                node = child
                current_node_id.set(node.node_id)
            if node.is_executed():
                raise RuntimeError(f"Node {node.node_id} has already been executed")
            if node.is_terminal():
                return self._save(state, phase="commit", current_node_id=node.node_id, finish_reason="terminal")
            cursor = next((i for i, step in enumerate(node.action_steps) if not step.is_executed()), 0)
            return self._save(state, current_node_id=node.node_id, action_cursor=cursor, phase="generate")
        except Exception as exc:
            return self._fail(state, node, exc, "control", "control_error")

    async def _generate_actions(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        node = self._load(state, runtime)
        try:
            await asyncio.wait_for(runtime.context.agent.generate_actions(node), timeout=self.model_timeout)
        except asyncio.TimeoutError as exc:
            return self._fail(state, node, exc, "timeout", "model_timeout")
        except RejectError as exc:
            reason = "no_actions" if str(exc) == "No action found" else "generation_error"
            return self._fail(state, node, exc, "model_rejection", reason)
        except Exception as exc:
            return self._fail(state, node, exc, "model", "generation_error")
        if node.error:
            return self._fail(state, node, RejectError(node.error), "model_rejection", "generation_error")
        if not node.action_steps:
            return self._fail(
                state, node, RejectError("No executable actions generated"), "model_rejection", "no_actions"
            )
        if node.is_duplicate:
            return self._save(state, phase="commit", finish_reason="duplicate")
        return self._save(state, phase="policy")

    async def _policy_decision(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        self._load(state, runtime)
        # A0: pass through the current ordered step without calls or interventions.
        return self._save(state, phase="execute")

    async def _execute_action(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        node = self._load(state, runtime)
        try:
            await asyncio.wait_for(
                runtime.context.agent.execute_action_step(node, state["action_cursor"]), timeout=self.tool_timeout
            )
        except asyncio.TimeoutError as exc:
            return self._fail(state, node, exc, "timeout", "tool_timeout")
        except Exception as exc:
            kind = "tool_model" if isinstance(exc, CompletionError) else "tool"
            return self._fail(state, node, exc, kind, "tool_error")
        if node.error:
            return self._fail(state, node, RejectError(node.error), "tool_rejection", "tool_error")
        if node.action_steps[state["action_cursor"]].observation is None:
            return self._fail(state, node, RuntimeError("Tool returned no observation"), "tool", "tool_error")
        return self._save(state, phase="record")

    async def _record_observation(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        node = self._load(state, runtime)
        cursor = state["action_cursor"] + 1
        # Skip already completed entries in a prebuilt response, without replay.
        while cursor < len(node.action_steps) and node.action_steps[cursor].is_executed():
            cursor += 1
        phase = "commit" if node.is_terminal() or cursor >= len(node.action_steps) else "policy"
        return self._save(state, action_cursor=cursor, phase=phase)

    async def _commit_iteration(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        self._load(state, runtime)
        reason = state["finish_reason"]
        if reason is None:
            reason = "single_step" if state["single_step"] else self._legacy_finish()
        return self._save(state, phase="end" if reason else "prepare", finish_reason=reason)

    def _build_graph(self):
        builder = StateGraph(GraphState, context_schema=GraphRuntime)
        nodes = {
            "prepare_iteration": self._prepare_iteration,
            "generate_actions": self._generate_actions,
            "policy_decision": self._policy_decision,
            "execute_action": self._execute_action,
            "record_observation": self._record_observation,
            "commit_iteration": self._commit_iteration,
        }
        for name, fn in nodes.items():
            builder.add_node(name, self._journal_node(name, fn))
        routes = {
            "prepare": "prepare_iteration",
            "generate": "generate_actions",
            "policy": "policy_decision",
            "execute": "execute_action",
            "record": "record_observation",
            "commit": "commit_iteration",
            "end": END,
        }
        builder.add_conditional_edges(START, lambda state: state["phase"], routes)
        for name in nodes:
            builder.add_conditional_edges(name, lambda state: state["phase"], routes)
        return builder.compile(checkpointer=self._checkpointer or InMemorySaver())

    async def _run(self, message: str | None = None, node_id: int | None = None) -> tuple[Node, str | None]:
        recovered = None
        if self._journal:
            try:
                recovered = self._journal.read()
                if recovered:
                    self._graph_state = recovered
                    self._last_durable_version = recovered["snapshot_version"]
                pending = self._journal.pending()
                if pending:
                    raise RecoveryConflict(
                        f"Unconfirmed external operations: {pending}; reconcile effects and charges before resuming"
                    )
                if recovered:
                    expected = recovered["workspace_ref"].get("fingerprint")
                    actual = workspace_fingerprint(self.agent._workspace, self.checkpoint_path)
                    if any(
                        actual.get(kind) and not actual[kind].get("recoverable", False)
                        for kind in ("runtime", "environment")
                    ):
                        raise RecoveryConflict(
                            "Opaque runtime/environment needs a recovery_state adapter before automatic recovery"
                        )
                    if expected != actual:
                        raise RecoveryConflict(
                            "Repository commit/files, shadow mode or runtime identity/state differs from the committed snapshot"
                        )
                    try:
                        self._root = restore_nodes(recovered, self.agent._workspace)
                    except (ValueError, KeyError, TypeError) as exc:
                        raise RecoveryConflict(f"Invalid durable snapshot: {exc}") from exc
                    self._graph_state = recovered
                    if recovered["error"]:
                        raise RuntimeError(
                            f"Saved {recovered['error_kind']}: {recovered['error']}; start a new run after explicit reconciliation"
                        )
                    if recovered["phase"] != "end" and (
                        message or (node_id is not None and node_id != recovered["current_node_id"])
                    ):
                        raise RecoveryConflict("Resume the pending trajectory before starting a different message/node")
            except RecoveryConflict:
                self._recovery_conflict = True
                raise
        self._graph_state = None
        self._pending_exception = None
        self._loaded_snapshot_version = -1
        if self.agent._workspace:
            # The base initializer only fills missing bindings; explicitly passed
            # replacement workspaces must also replace existing live references.
            try:
                repository = self.agent.workspace.repository
            except ValueError:
                repository = None
            for existing in self.root.get_all_nodes():
                if existing.file_context:
                    existing.file_context.repository = repository
                    existing.file_context._runtime = self.agent.workspace.runtime
        node = self.get_node_by_id(node_id) if node_id is not None else self.root.get_last_node()
        if node is None:
            raise ValueError(f"Node with ID {node_id} not found")
        if message:
            node = self._create_next_node(node)
            node.user_message = message
        state = initial_state(self.root, self.project_id, self.trajectory_id, node.node_id, node_id is not None)
        if recovered:
            if recovered["phase"] != "end" or (
                not message and node_id is None and recovered["finish_reason"] != "single_step"
            ):
                state = recovered
            else:
                state["snapshot_version"] = recovered["snapshot_version"] + 1
        if self.agent._workspace:
            try:
                repo = self.agent.workspace.repository
                state["workspace_ref"].update(repo_path=getattr(repo, "repo_dir", None), shadow_mode=repo.shadow_mode)
            except ValueError:
                pass
        self._graph_state = state
        self._loaded_snapshot_version = state["snapshot_version"]
        self._compiled_graph = self._build_graph()
        self._graph_config = {
            "configurable": {
                "thread_id": thread_id(self.project_id, self.trajectory_id, self.checkpoint_run_id),
                "checkpoint_ns": "",
            },
            "recursion_limit": self.graph_recursion_limit,
        }
        # Link Agent events to the same trajectory. M0 leaves these IDs unset;
        # this is an explicit event-metadata difference, not a business change.
        self.agent._on_event = self._emit_event
        context = GraphRuntime(agent=self.agent, workspace=self.agent._workspace)
        if self._journal and not recovered:
            state = self._capture_recovery_state(state)
            self._journal.save(state)
            self._last_durable_version = state["snapshot_version"]
        token = current_node_id.set(node.node_id)
        try:
            result = await self._compiled_graph.ainvoke(state, self._graph_config, context=context)
            self._graph_state = result
            if self._pending_exception:
                raise self._pending_exception
            return self.get_node_by_id(result["current_node_id"]), result["finish_reason"]
        except GraphRecursionError as exc:
            latest = self._graph_state or state
            node = self.get_node_by_id(latest["current_node_id"])
            self._fail(latest, node, exc, "graph_recursion", "graph_recursion_limit")
            raise
        except Exception as exc:
            if not self._pending_exception:
                latest = self._graph_state or state
                node = self.get_node_by_id(latest["current_node_id"])
                kind = "recovery_conflict" if isinstance(exc, RecoveryConflict) else "graph"
                if isinstance(exc, RecoveryConflict):
                    self._recovery_conflict = True
                self._fail(latest, node, exc, kind, kind if kind == "recovery_conflict" else "graph_error")
            raise
        finally:
            current_node_id.reset(token)
