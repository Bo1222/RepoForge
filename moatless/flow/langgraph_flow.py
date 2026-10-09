"""LangGraph-controlled single-track flow (persistent recovery belongs to T3)."""

from copy import deepcopy
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import Field, PrivateAttr

from moatless.context_data import current_node_id
from moatless.exceptions import CompletionError, RejectError, RuntimeError
from moatless.flow.events import NodeExpandedEvent
from moatless.flow.flow import AgenticFlow
from moatless.flow.graph_state import GraphRuntime, GraphState, initial_state, restore_nodes, snapshot_nodes
from moatless.flow.schema import FlowStatus
from moatless.node import Node


class LangGraphFlow(AgenticFlow):
    graph_recursion_limit: int = Field(default=10000, ge=1)

    _graph_state: GraphState | None = PrivateAttr(default=None)
    _loaded_snapshot_version: int = PrivateAttr(default=-1)
    _pending_exception: Exception | None = PrivateAttr(default=None)
    _compiled_graph: Any = PrivateAttr(default=None)
    _graph_config: dict = PrivateAttr(default_factory=dict)

    @property
    def graph_state(self) -> GraphState | None:
        """Return an independent JSON snapshot, never live Workspace/client objects."""
        self._sync_snapshot()
        return deepcopy(self._graph_state)

    def _sync_snapshot(self):
        # The base lifecycle may annotate an exception after _run has returned.
        # Export that annotation from the same versioned snapshot as the trajectory.
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
            await runtime.context.agent.generate_actions(node)
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
            await runtime.context.agent.execute_action_step(node, state["action_cursor"])
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
            builder.add_node(name, fn)
        routes = {
            "prepare": "prepare_iteration",
            "generate": "generate_actions",
            "policy": "policy_decision",
            "execute": "execute_action",
            "record": "record_observation",
            "commit": "commit_iteration",
            "end": END,
        }
        builder.add_edge(START, "prepare_iteration")
        for name in nodes:
            builder.add_conditional_edges(name, lambda state: state["phase"], routes)
        return builder.compile(checkpointer=InMemorySaver())

    async def _run(self, message: str | None = None, node_id: int | None = None) -> tuple[Node, str | None]:
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
            "configurable": {"thread_id": str((self.project_id, self.trajectory_id))},
            "recursion_limit": self.graph_recursion_limit,
        }
        # Link Agent events to the same trajectory. M0 leaves these IDs unset;
        # this is an explicit event-metadata difference, not a business change.
        self.agent._on_event = self._emit_event
        context = GraphRuntime(agent=self.agent, workspace=self.agent._workspace)
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
                self._fail(latest, node, exc, "graph", "graph_error")
            raise
        finally:
            current_node_id.reset(token)
