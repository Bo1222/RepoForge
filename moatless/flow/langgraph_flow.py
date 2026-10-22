"""LangGraph-controlled single-track flow with optional local durable recovery."""

import asyncio
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import Field, PrivateAttr

from moatless.context_data import current_node_id
from moatless.exceptions import CompletionError, RejectError, RuntimeError, WorkspaceConsistencyError
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
from moatless.flow.pruning_checkpointer import PruningAsyncSqliteSaver
from moatless.flow.schema import FlowStatus
from moatless.node import ActionStep, Node
from moatless.actions.schema import Observation
from moatless.policy import ActionPolicy, ActionPolicyConfig, BudgetSnapshot, PolicyDecision


class LangGraphFlow(AgenticFlow):
    graph_recursion_limit: int = Field(default=10000, ge=1)
    checkpoint_path: str | None = None
    checkpoint_run_id: str = "default"
    model_timeout: float | None = Field(default=None, gt=0)
    tool_timeout: float | None = Field(default=None, gt=0)
    action_policy: ActionPolicyConfig = Field(default_factory=ActionPolicyConfig)
    _budget_provider: Any = PrivateAttr(default=None)

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
        signature = {
                "agent": self.agent.model_dump(exclude_none=True),
                "max_iterations": self.max_iterations,
                "max_cost": self.max_cost,
                "model_timeout": self.model_timeout,
                "tool_timeout": self.tool_timeout,
            }
        if self.action_policy.variant != "off":
            signature["action_policy"] = self.action_policy.model_dump(mode="json")
            if self.action_policy.variant == "a6":
                signature["repair_protocol"] = "a6-six-groups-v2"
            elif self.action_policy.variant == "a7":
                # A6 checkpoints may never resume as the single-controller A7
                # protocol, even if the action-policy blob were similar.
                signature["repair_protocol"] = "a7-single-controller-v1"
        return digest(signature)

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
            async with PruningAsyncSqliteSaver.from_conn_string(self.checkpoint_path) as saver:
                self._checkpointer = saver
                try:
                    return await super().run(message, workspace, node_id)
                finally:
                    # Never certify state around an unfinished external operation.
                    if not self._recovery_conflict and not self._journal.pending() and self._graph_state:
                        self._sync_snapshot()
                        self._journal.save(self._capture_recovery_state(self._graph_state))
                        self._last_durable_version = self._graph_state["snapshot_version"]
                        await saver.truncate_wal()
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
                self._journal.save(result, None if result["error_kind"] in {"timeout", "workspace_consistency"} else operation)
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
            kind = getattr(exc, "error_kind", "model")
            return self._fail(state, node, exc, kind, "generation_error")
        if node.error:
            return self._fail(state, node, RejectError(node.error), "model_rejection", "generation_error")
        if not node.action_steps:
            return self._fail(
                state, node, RejectError("No executable actions generated"), "model_rejection", "no_actions"
            )
        if node.is_duplicate:
            if self.action_policy.variant == "a6":
                node.is_duplicate = False
                self._skip_pending(node, 0, "repeated_proposal_requires_new_evidence")
                node.policy_feedback = "Repeated proposal without new evidence. Read a related source or change the diagnostic query before another repair."
                return self._save(state, phase="assess", action_cursor=0)
            return self._save(state, phase="commit", finish_reason="duplicate")
        return self._save(state, phase="policy")

    async def _policy_decision(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        node = self._load(state, runtime)
        if self.action_policy.variant == "a6":
            return await self._repair_policy(state, runtime)
        if self.action_policy.variant == "a7":
            return await self._a7_policy_decision(state, runtime)
        if self.action_policy.variant == "off":
            return self._save(state, phase="execute")
        from moatless.message_history.managed import ManagedMessageHistoryGenerator
        from moatless.policy.action import extract_state
        if not isinstance(self.agent.memory, ManagedMessageHistoryGenerator):
            raise ValueError("ActionPolicy requires Managed execution facts")
        cursor = state["action_cursor"]
        step = node.action_steps[cursor]
        if step.origin == "policy":
            return self._save(state, phase="execute")
        saved = deepcopy(state.get("policy_state", {"version": 1, "corrections": [], "replans": {}}))
        features, history = extract_state(node, cursor, runtime.context.workspace, saved)
        window = history[-self.action_policy.repeat_window:]
        features.repeat_count = sum(row["signature"] == features.signature and row["no_gain"] for row in window)
        features.replan_attempts = saved.get("replans", {}).get(features.signature, 0)
        totals = self._budget_provider() if self._budget_provider else {
            "reference_cost": self.total_usage().completion_cost, "cost_complete": True}
        memory = self.agent.memory.policy
        budget = BudgetSnapshot(remaining_rounds=max(0, self.max_iterations - len(self.root.get_all_nodes())),
            spent_cost=totals.get("reference_cost"), max_cost=self.max_cost,
            cost_complete=totals.get("cost_complete", False), estimated_next_call=(
                memory.max_input_tokens * self.action_policy.input_price_per_million
                + (self.agent.completion_model.max_tokens or 4096) * self.action_policy.output_price_per_million) / 1_000_000)
        decision = ActionPolicy(self.action_policy).decide(features, step, budget)
        node.policy_decisions = node.policy_decisions or []
        identity = f"{node.node_id}:{cursor}:{len(node.policy_decisions)}"
        step.origin = "model"
        step.proposal_id = step.proposal_id or f"{node.node_id}:{cursor}"
        step.policy_decision_id = identity
        node.policy_decisions.append({"decision_id": identity, "config": self.action_policy.model_dump(mode="json"),
            "state": features.model_dump(mode="json"), "budget": budget.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json")})
        if decision.kind == "allow":
            return self._save(state, phase="execute", policy_state=saved)
        if decision.correction_key:
            saved["corrections"] = [*saved.get("corrections", []), decision.correction_key]
        # A rejected model proposal is retained and labelled, never executed or
        # reported as a successful tool. Appending avoids shifting source IDs.
        for index, pending in enumerate(node.action_steps[cursor:], cursor):
            if pending.observation is None:
                pending.origin = pending.origin or "model"
                pending.proposal_id = pending.proposal_id or f"{node.node_id}:{index}"
                pending.policy_decision_id = identity
                pending.observation = Observation.create(message=f"Not executed: action policy {decision.reason}.",
                    properties={"execution_status": "skipped", "policy_decision_id": identity})
        failure_feedback = ""
        if features.failure_kind:
            failure_feedback = (
                f" Failure kind: {features.failure_kind}; suggested: {features.failure_suggested_action}. "
                f"Excerpt: {features.failure_excerpt[:300]}"
            )
        node.policy_feedback = (f"Policy {decision.reason}.{failure_feedback} Evidence: {', '.join(features.evidence[-3:])}. "
            "Use a different evidence-based action; identify an existing public test if verification is missing. "
            "Do not claim task resolution from unavailable or failed checks.")
        if decision.kind == "verify":
            from moatless.actions.run_tests import RunTestsArgs
            if RunTestsArgs not in self.agent.action_map:
                raise ValueError("Verification policy requires the RunTests action")
            node.action_steps.append(ActionStep(action=RunTestsArgs(test_files=[decision.target], test_source="base_public"),
                                               origin="policy", policy_decision_id=identity))
            return self._save(state, phase="execute", action_cursor=len(node.action_steps) - 1, policy_state=saved)
        if decision.kind == "replan":
            saved.setdefault("replans", {})[features.signature] = features.replan_attempts + 1
            saved["progress_reset_after"] = features.evidence[-1] if features.evidence else None
            return self._save(state, phase="commit", action_cursor=len(node.action_steps), policy_state=saved)
        node.terminal = True
        node.completion_status = {"status": "unverified", "reason": decision.reason,
            "verification_scope": "task-local recorded checks", "official_harness": "not_run", "resolved": None}
        return self._save(state, phase="commit", action_cursor=len(node.action_steps),
                          finish_reason="policy_stop", policy_state=saved)

    async def _execute_action(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        node = self._load(state, runtime)
        if self.action_policy.variant in {"a6", "a7"}:
            from moatless.actions.run_tests import RunTests
            current = node.action_steps[state["action_cursor"]]
            executed_tests = sum(1 for previous in node.get_trajectory() for step in previous.action_steps
                if step.action.name == "RunTests" and step.observation is not None
                and (step.observation.properties or {}).get("execution_status") != "skipped")
            if self.action_policy.variant == "a6" and current.action.name == "RunTests" and executed_tests >= self.action_policy.max_verification_attempts:
                return self._repair_stop(state, node, "verification_budget_exhausted", deepcopy(state["repair_state"]))
            if self.action_policy.variant == "a7" and current.action.name == "RunTests" and executed_tests >= self.action_policy.max_verification_attempts:
                saved = deepcopy(state.get("repair_state") or self._a7_repair_state())
                self._skip_pending(node, state["action_cursor"], "verification_budget_exhausted")
                saved["stop_reason"] = "verification_budget_exhausted"
                return self._save(state, phase="policy", action_cursor=len(node.action_steps),
                                  policy_state=state.get("policy_state"), repair_state=saved)
            for action in self.agent.actions:
                if isinstance(action, RunTests) and self.tool_timeout:
                    action.timeout = min(action.timeout, self.tool_timeout)
        try:
            await asyncio.wait_for(
                runtime.context.agent.execute_action_step(node, state["action_cursor"]), timeout=(None if node.action_steps[state["action_cursor"]].action.name == "RunTests" else self.tool_timeout)
            )
        except asyncio.TimeoutError as exc:
            return self._fail(state, node, exc, "timeout", "tool_timeout")
        except Exception as exc:
            if isinstance(exc, WorkspaceConsistencyError):
                return self._fail(state, node, exc, "workspace_consistency", "recovery_required")
            kind = "tool_model" if isinstance(exc, CompletionError) else "tool"
            return self._fail(state, node, exc, kind, "tool_error")
        if node.error:
            return self._fail(state, node, RejectError(node.error), "tool_rejection", "tool_error")
        if node.action_steps[state["action_cursor"]].observation is None:
            return self._fail(state, node, RuntimeError("Tool returned no observation"), "tool", "tool_error")
        props = node.action_steps[state["action_cursor"]].observation.properties or {}
        if self.action_policy.variant in {"a6", "a7"} and props.get("timed_out") and props.get("continuation_safe") is not True:
            return self._fail(state, node, WorkspaceConsistencyError("Timeout cleanup is unconfirmed"),
                              "workspace_consistency", "recovery_required")
        return self._save(state, phase="record")

    async def _record_observation(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        node = self._load(state, runtime)
        if self.action_policy.variant in {"a6", "a7"}:
            completed = node.action_steps[state["action_cursor"]]
            if completed.action.name == "RunTests":
                self._skip_pending(node, state["action_cursor"] + 1, "test_observation_barrier")
            return self._save(state, phase="assess")
        cursor = state["action_cursor"] + 1
        # Skip already completed entries in a prebuilt response, without replay.
        while cursor < len(node.action_steps) and node.action_steps[cursor].is_executed():
            cursor += 1
        phase = "commit" if node.is_terminal() or cursor >= len(node.action_steps) else "policy"
        return self._save(state, action_cursor=cursor, phase=phase)

    @staticmethod
    def _skip_pending(node, cursor, reason):
        for step in node.action_steps[cursor:]:
            if step.observation is None:
                step.observation = Observation.create("Not executed: " + reason,
                    properties={"execution_status": "skipped", "skip_reason": reason})

    def _repair_stop(self, state, node, reason, saved):
        self._skip_pending(node, state["action_cursor"], reason)
        node.terminal = True
        node.completion_status = {"version": 2, "status": "unverified", "reason": reason,
            "assessment": saved.get("assessment"), "official_harness": "not_run", "resolved": None}
        return self._save(state, phase="commit", finish_reason=reason, repair_state=saved,
                          action_cursor=len(node.action_steps))

    async def _repair_policy(self, state, runtime):
        from moatless.policy.repair import registered_plan
        saved = deepcopy(state.get("repair_state", {"version": 1, "attempts": {}, "failure_families": {}}))
        node = self._load(state, runtime)
        step = node.action_steps[state["action_cursor"]]
        if step.action.name == "Finish":
            self._skip_pending(node, state["action_cursor"], "completion_requires_assessment")
            return self._save(state, phase="assess", repair_state=saved)
        if step.action.name == "RunTests" and len(step.action.test_files) != 1:
            self._skip_pending(node, state["action_cursor"], "single_test_module_required")
            node.policy_feedback = "Run one explicit module per invocation; register separate mandatory checks for additional modules."
            return self._save(state, phase="commit", action_cursor=len(node.action_steps), repair_state=saved)
        if (step.action.name == "StringReplace" and step.action.old_str == step.action.new_str):
            self._skip_pending(node, state["action_cursor"], "no_changes")
            node.policy_feedback = "No-op replacement rejected before execution. Read evidence or propose an actual repair."
            return self._save(state, phase="commit", action_cursor=len(node.action_steps), repair_state=saved)
        writes = {"StringReplace", "CreateFile", "InsertLine", "InsertLines", "AppendString", "TextEditor", "str_replace_editor", "Bash", "RunPythonScript"}
        pending_replans = {key: value for key, value in saved.get("replans", {}).items() if value["status"] == "required"}
        if pending_replans:
            if step.action.name == "ReplanRepair":
                return self._save(state, phase="execute", repair_state=saved)
            # Reads/searches are the evidence source for ReplanRepair.  Allow a
            # bounded number, then stop; otherwise a model can ignore the gate
            # and search until max_iterations (observed in a6 pytest/astropy).
            if step.action.name not in writes:
                for gate in pending_replans.values():
                    gate["read_attempts"] = gate.get("read_attempts", 0) + 1
                if max(gate.get("read_attempts", 0) for gate in pending_replans.values()) > 2:
                    return self._repair_stop(state, node, "replan_evidence_exhausted", saved)
                return self._save(state, phase="execute", repair_state=saved)
            for gate in pending_replans.values():
                gate["rejections"] += 1
            if max(gate["rejections"] for gate in pending_replans.values()) >= 2:
                return self._repair_stop(state, node, "replan_exhausted", saved)
            self._skip_pending(node, state["action_cursor"], "evidence_based_replan_required")
            node.policy_feedback = "Read new diagnostic evidence, then submit ReplanRepair for these families: " + str(list(pending_replans))
            return self._save(state, phase="commit", action_cursor=len(node.action_steps), repair_state=saved)
        path = getattr(step.action, "path", None)
        if step.action.name in writes and saved.get("edit_loops", {}).get(path, {}).get("blocked"):
            self._skip_pending(node, state["action_cursor"], "edit_loop")
            node.policy_feedback = f"Repeated edit failures at {path}. Read the current file/span before another edit or replan the diagnosis."
            return self._save(state, phase="commit", action_cursor=len(node.action_steps), repair_state=saved)
        if step.action.name in writes and registered_plan(node) is None:
            self._skip_pending(node, state["action_cursor"], "acceptance_plan_required")
            node.policy_feedback = "Register public acceptance criteria and concrete checks before editing product code."
            return self._save(state, phase="commit", action_cursor=len(node.action_steps), repair_state=saved)
        if self._budget_provider and not self._budget_provider().get("cost_complete", False):
            return self._repair_stop(state, node, "unknown_cost", saved)
        return self._save(state, phase="execute", repair_state=saved)

    async def _a7_policy_decision(self, state, runtime):
        """One controller: A6 pre-gates feed O2's ActionPolicy decision."""
        from moatless.message_history.managed import ManagedMessageHistoryGenerator
        from moatless.policy.action import extract_state

        node = self._load(state, runtime)
        if not isinstance(self.agent.memory, ManagedMessageHistoryGenerator):
            raise ValueError("ActionPolicy requires Managed execution facts")
        repair = deepcopy(state.get("repair_state") or self._a7_repair_state())
        policy = deepcopy(state.get("policy_state") or self._a7_policy_state())
        # A7 stops only through the policy.  Assess/execute record a fact; the
        # policy turns it into the single terminal completion status.
        forced = repair.get("stop_reason")
        if forced:
            decision = PolicyDecision(kind="stop", reason=forced)
            self._record_a7_decision(node, state["action_cursor"], repair, policy, None, None, decision)
            self._skip_pending(node, state["action_cursor"], forced)
            node.terminal = True
            node.completion_status = self._a7_completion_status("unverified", forced, repair)
            return self._save(state, phase="commit", action_cursor=len(node.action_steps),
                              finish_reason="policy_stop", policy_state=policy, repair_state=repair)
        cursor = state["action_cursor"]
        if cursor >= len(node.action_steps):
            decision = PolicyDecision(kind="stop", reason="no_pending_action")
            self._record_a7_decision(node, cursor, repair, policy, None, None, decision)
            node.terminal = True
            node.completion_status = self._a7_completion_status("unverified", decision.reason, repair)
            return self._save(state, phase="commit", action_cursor=len(node.action_steps),
                              finish_reason="policy_stop", policy_state=policy, repair_state=repair)
        # An A6 failure-family gate starts a new O2 progress epoch.
        reset = repair.pop("progress_reset_source", None)
        if reset:
            policy["progress_reset_after"] = reset
        step = node.action_steps[cursor]
        if step.origin == "policy":
            # Verification/check steps are already policy-selected.
            return self._save(state, phase="execute", policy_state=policy, repair_state=repair)
        gate = self._a7_pre_gate(node, cursor, step, repair, runtime.context.workspace)
        if gate["stop"]:
            decision = PolicyDecision(kind="stop", reason=gate["stop"])
            self._record_a7_decision(node, cursor, repair, policy, None, None, decision)
            self._skip_pending(node, cursor, gate["stop"])
            node.terminal = True
            node.completion_status = self._a7_completion_status("unverified", gate["stop"], repair)
            return self._save(state, phase="commit", action_cursor=len(node.action_steps),
                              finish_reason="policy_stop", policy_state=policy, repair_state=repair)
        if gate["skip"]:
            decision = PolicyDecision(kind="replan", reason=gate["skip"],
                                      required_action=gate.get("required_action"))
            self._record_a7_decision(node, cursor, repair, policy, None, None, decision)
            self._skip_pending(node, cursor, gate["skip"])
            if gate.get("feedback"):
                node.policy_feedback = gate["feedback"]
            if gate.get("phase") == "assess":
                return self._save(state, phase="assess", policy_state=policy, repair_state=repair)
            return self._save(state, phase="commit", action_cursor=len(node.action_steps),
                              policy_state=policy, repair_state=repair)
        saved_assessment = repair.get("assessment")
        features, history = extract_state(
            node, cursor, runtime.context.workspace, policy, variant="a7",
            repair_state=repair, assessment=saved_assessment)
        window = history[-self.action_policy.repeat_window:]
        features.repeat_count = sum(
            row["signature"] == features.signature and row["no_gain"] for row in window)
        features.replan_attempts = policy.get("replans", {}).get(features.signature, 0)
        totals = self._budget_provider() if self._budget_provider else {
            "reference_cost": self.total_usage().completion_cost, "cost_complete": True}
        memory = self.agent.memory.policy
        budget = BudgetSnapshot(
            remaining_rounds=max(0, self.max_iterations - len(self.root.get_all_nodes())),
            spent_cost=totals.get("reference_cost"), max_cost=self.max_cost,
            cost_complete=totals.get("cost_complete", False),
            estimated_next_call=(
                memory.max_input_tokens * self.action_policy.input_price_per_million
                + (self.agent.completion_model.max_tokens or 4096)
                * self.action_policy.output_price_per_million) / 1_000_000)
        decision = ActionPolicy(self.action_policy).decide(features, step, budget)
        identity = self._record_a7_decision(node, cursor, repair, policy, features, budget, decision)
        if decision.kind == "allow":
            return self._save(state, phase="execute", policy_state=policy, repair_state=repair)
        if decision.correction_key:
            policy["corrections"] = [*policy.get("corrections", []), decision.correction_key]
        # O2 contract: a non-allow decision retains every rejected proposal
        # as a labelled skipped step, never executes it, and never renumbers
        # source IDs by deleting it.
        for index, pending in enumerate(node.action_steps[cursor:], cursor):
            if pending.observation is None:
                pending.origin = pending.origin or "model"
                pending.proposal_id = pending.proposal_id or f"{node.node_id}:{index}"
                pending.policy_decision_id = identity
                pending.observation = Observation.create(
                    message=f"Not executed: action policy {decision.reason}.",
                    properties={"execution_status": "skipped", "policy_decision_id": identity,
                                "skip_reason": decision.reason})
        node.policy_feedback = self._a7_feedback(features, decision)
        if decision.kind == "verify":
            from moatless.actions.run_tests import RunTestsArgs
            if RunTestsArgs not in self.agent.action_map:
                raise ValueError("Verification policy requires the RunTests action")
            request = {
                "test_files": decision.test_files or ([decision.target] if decision.target else []),
                "test_source": decision.test_source or "base_public",
                "product_source": decision.product_source or "current",
            }
            if decision.check_id:
                request["check_id"] = decision.check_id
            if not request["test_files"]:
                node.policy_feedback = "Policy verification requires an explicit test module."
                return self._save(state, phase="commit", action_cursor=len(node.action_steps),
                                  policy_state=policy, repair_state=repair)
            node.action_steps.append(ActionStep(action=RunTestsArgs(**request), origin="policy",
                                                policy_decision_id=identity))
            return self._save(state, phase="execute", action_cursor=len(node.action_steps) - 1,
                              policy_state=policy, repair_state=repair)
        if decision.kind == "replan":
            policy.setdefault("replans", {})[features.signature] = features.replan_attempts + 1
            policy["progress_reset_after"] = features.evidence[-1] if features.evidence else None
            from moatless.policy.replanning import require_replan
            from moatless.policy.repair import registered_plan
            if decision.required_action == "ReplanRepair" and registered_plan(node) is not None:
                require_replan(repair, node, decision.family_id or "no_progress",
                               trigger=features.evidence[-1] if features.evidence else None)
            node.policy_feedback = self._a7_feedback(features, decision)
            if decision.required_action == "ReplanRepair" and registered_plan(node) is None:
                node.policy_feedback += (
                    " Register public acceptance criteria and concrete checks before a repair gate can be required.")
            self._skip_pending(node, cursor, decision.reason)
            return self._save(state, phase="commit", action_cursor=len(node.action_steps),
                              policy_state=policy, repair_state=repair)
        # stop
        node.policy_feedback = self._a7_feedback(features, decision)
        self._skip_pending(node, cursor, decision.reason)
        node.terminal = True
        node.completion_status = self._a7_completion_status("unverified", decision.reason, repair)
        return self._save(state, phase="commit", action_cursor=len(node.action_steps),
                          finish_reason="policy_stop", policy_state=policy, repair_state=repair)

    @staticmethod
    def _a7_repair_state():
        return {"version": 1, "attempts": {}, "failure_families": {},
                "discovery_attempts": 0, "edit_loops": {}, "replans": {}}

    @staticmethod
    def _a7_policy_state():
        return {"version": 1, "corrections": [], "replans": {}}

    @staticmethod
    def _a7_completion_status(status, reason, repair):
        return {"version": 2, "status": status, "reason": reason,
                "assessment": repair.get("assessment"),
                "verification_scope": "public acceptance checks",
                "official_harness": "not_run", "resolved": None}

    def _record_a7_decision(self, node, cursor, repair, policy, features, budget, decision):
        """Persist the single controller decision for audit/export."""
        node.policy_decisions = node.policy_decisions or []
        identity = f"{node.node_id}:{cursor}:{len(node.policy_decisions)}"
        if 0 <= cursor < len(node.action_steps):
            step = node.action_steps[cursor]
            step.origin = step.origin or "model"
            step.proposal_id = step.proposal_id or f"{node.node_id}:{cursor}"
            step.policy_decision_id = identity
        node.policy_decisions.append({
            "decision_id": identity,
            "config": self.action_policy.model_dump(mode="json"),
            "state": features.model_dump(mode="json") if features is not None else None,
            "budget": budget.model_dump(mode="json") if budget is not None else None,
            "decision": decision.model_dump(mode="json"),
        })
        return identity

    @staticmethod
    def _a7_feedback(features, decision):
        if features is None:
            return f"Policy {decision.reason}."
        parts = [f"Policy {decision.reason}."]
        if features.failure_kind:
            parts.append(
                f" Failure kind: {features.failure_kind}; suggested: {features.failure_suggested_action}.")
        if features.last_test_output:
            command = features.last_test_command
            parts.append(
                f" Last test (command={command!r}, return_code={features.last_test_return_code}): "
                f"{features.last_test_output[-700:]}")
        if features.failure_excerpt:
            parts.append(f" Excerpt: {features.failure_excerpt[:400]}")
        if features.acceptance_outstanding:
            parts.append(f" Outstanding acceptance: {features.acceptance_outstanding[:8]}.")
        if features.replan_required:
            parts.append(
                " Required action: read new evidence and submit ReplanRepair. Family IDs: "
                + str(features.replan_required))
        parts.append(" Use a different evidence-based action; do not claim resolution from unavailable or failed checks.")
        return "".join(parts)

    def _a7_pre_gate(self, node, cursor, step, repair, workspace):
        """A6 constraints only; never chooses or executes an action."""
        from moatless.policy.repair import registered_plan
        result = {"skip": None, "stop": None, "feedback": "", "required_action": None, "phase": "commit"}
        name = step.action.name
        if name in {"Finish", "VerifiedFinish"}:
            # O2 semantics: a terminal proposal without any product patch is
            # a stop, not an assessment request.  A finish with a patch must
            # go through the acceptance assessment before it can terminate.
            if not node.file_context.generate_git_patch(ignore_tests=True).strip():
                result["stop"] = "no_product_patch"
                return result
            result.update(skip="completion_requires_assessment", phase="assess")
            return result
        if name == "RunTests" and len(step.action.test_files) != 1:
            result.update(skip="single_test_module_required",
                          feedback="Run one explicit module per invocation; register separate mandatory checks for additional modules.")
            return result
        if name == "StringReplace" and step.action.old_str == step.action.new_str:
            result.update(skip="no_changes",
                          feedback="No-op replacement rejected before execution. Read evidence or propose an actual repair.")
            return result
        writes = {"StringReplace", "CreateFile", "InsertLine", "InsertLines", "AppendString",
                  "TextEditor", "str_replace_editor", "Bash", "RunPythonScript"}
        # Evidence/verification actions do not consume the bounded replan
        # reading budget.
        checks = {"RunTests", "CheckSyntax", "RegisterAcceptancePlan", "ExplainTestConflict", "Reject"}
        pending = {key: value for key, value in repair.get("replans", {}).items()
                   if value.get("status") == "required"}
        if pending:
            if name == "ReplanRepair":
                return result
            if name not in writes:
                # Mandatory checks are evidence, not replan reading, and do
                # not consume the bounded evidence budget.
                if name not in checks:
                    for gate in pending.values():
                        gate["read_attempts"] = gate.get("read_attempts", 0) + 1
                    if max(gate.get("read_attempts", 0) for gate in pending.values()) > 2:
                        result["stop"] = "replan_evidence_exhausted"
                        return result
                return result
            for gate in pending.values():
                gate["rejections"] = gate.get("rejections", 0) + 1
            if max(gate["rejections"] for gate in pending.values()) >= 2:
                result["stop"] = "replan_exhausted"
                return result
            result.update(
                skip="evidence_based_replan_required", required_action="ReplanRepair",
                feedback="Read new diagnostic evidence, then submit ReplanRepair for these families: "
                         + str(list(pending)))
            return result
        path = getattr(step.action, "path", None)
        if name in writes and repair.get("edit_loops", {}).get(path, {}).get("blocked"):
            result.update(skip="edit_loop",
                          feedback=f"Repeated edit failures at {path}. Read the current file/span before another edit or replan the diagnosis.")
            return result
        if name in writes and registered_plan(node) is None:
            result.update(skip="acceptance_plan_required",
                          feedback="Register public acceptance criteria and concrete checks before editing product code.")
            return result
        return result

    def _a7_stop(self, state, node, saved, reason, start=None):
        """Record a stop fact; the policy performs the single terminal write."""
        self._skip_pending(node, state["action_cursor"] + 1 if start is None else start, reason)
        saved["stop_reason"] = reason
        return self._save(state, phase="policy", action_cursor=len(node.action_steps), repair_state=saved)

    def _a7_check_key(self, request, assessment, node, workspace):
        from subprocess import SubprocessError
        from moatless.message_history.policy import content_hash
        from moatless.verification_inputs import capture_recipe
        try:
            recipe = capture_recipe(
                workspace, node.file_context, request["test_files"],
                request.get("test_source", "workspace"), request.get("product_source", "current"))
            identity = content_hash(recipe)
        except (OSError, ValueError, SubprocessError):
            identity = None
        return content_hash({"request": request, "product": assessment.patch_hash,
                             "input_identity": identity})

    async def _a7_assess_completion(self, state, runtime):
        """A6 evidence oracle. It updates facts and schedules verification only."""
        import json
        from moatless.execution_memory import test_records
        from moatless.policy.repair import assess
        from moatless.policy.replanning import accept_replan, update_repair_control

        node = self._load(state, runtime)
        saved = deepcopy(state.get("repair_state") or self._a7_repair_state())
        records = test_records(node, runtime.context.workspace)
        assessment = assess(node, runtime.context.workspace, records=records)
        saved["assessment"] = assessment.model_dump(mode="json")
        last_props = next(((step.observation.properties or {}) for step in reversed(node.action_steps)
                           if step.observation and (step.observation.properties or {}).get("execution_status") != "skipped"), {})
        if (last_props.get("fail_reason") == "environment_error"
                or any(failure["kind"] == "environment_error" for failure in assessment.failures)):
            return self._a7_stop(state, node, saved, "environment_blocked")
        index = min(state["action_cursor"], len(node.action_steps) - 1)
        if index >= 0:
            observed = node.action_steps[index]
            path = getattr(observed.action, "path", None)
            loops = saved.setdefault("edit_loops", {})
            if observed.action.name in {"StringReplace", "CreateFile", "InsertLine", "InsertLines",
                                        "AppendString", "str_replace_editor"} and last_props.get("fail_reason"):
                loop = loops.setdefault(path, {"failures": 0, "blocked": False})
                loop["failures"] += 1
                loop["blocked"] = loop["failures"] >= 2
            elif observed.action.name in {"ReadFile", "ViewCode", "ReadFiles"} and not last_props.get("fail_reason"):
                from moatless.repository_context import action_paths
                for read_path in action_paths(observed.action):
                    loops.pop(read_path, None)
        feedback = assessment.model_dump(mode="json")
        if last_props.get("fail_reason"):
            feedback["last_rejection"] = {"kind": last_props["fail_reason"], "source_id": f"{node.node_id}:{index}"}
        feedback["failures"] = feedback["failures"][:8]
        if last_props.get("test_run"):
            observed_observation = node.action_steps[index].observation if 0 <= index < len(node.action_steps) else None
            test_run = last_props["test_run"]
            feedback["last_test_output"] = ((observed_observation.message or "") if observed_observation else "")[-1200:]
            feedback["last_test_command"] = test_run.get("command") or last_props.get("commands")
            return_code = test_run.get("return_code")
            feedback["last_test_return_code"] = (
                return_code if return_code is not None else last_props.get("return_code"))
        # Persist A6's progress/repeat facts for audit only; the single O2
        # controller owns every gate and threshold.
        from moatless.policy.repair import registered_plan
        from moatless.policy.replanning import update_no_gain
        update_no_gain(saved, node, index, last_props,
                       stagnation_limit=self.action_policy.stagnation_limit,
                       repeat_limit=self.action_policy.repeat_limit,
                       plan=registered_plan(node), open_gate=False)
        node.policy_feedback = "Acceptance assessment: " + json.dumps(feedback, ensure_ascii=False)
        if assessment.state in {"ready", "partial_ready"}:
            self._skip_pending(node, state["action_cursor"] + 1, "acceptance_satisfied")
            return self._save(state, phase="finalize", repair_state=saved)
        cursor = state["action_cursor"] + 1
        while cursor < len(node.action_steps) and node.action_steps[cursor].is_executed():
            cursor += 1
        exhausted = update_repair_control(
            node, records, saved, self.action_policy.max_repair_attempts,
            assessment.failure_obligations)
        if exhausted:
            return self._a7_stop(state, node, saved, exhausted)
        proposal = last_props.get("replan_proposal")
        if proposal:
            accepted, message = accept_replan(node, proposal, saved)
            last_props["replan_validation"] = {"accepted": accepted, "message": message}
            node.policy_feedback += " " + message
            if accepted:
                saved["repair_trial_open"] = True
                saved["repair_trial_family"] = proposal.get("family_id")
            if not accepted:
                gates = [saved.get("replans", {}).get(proposal["family_id"])]
                if not gates[0]:
                    gates = [gate for gate in saved.get("replans", {}).values()
                             if gate.get("status") == "required"]
                for gate in gates:
                    if not gate:
                        continue
                    gate["rejections"] += 1
                    if gate["rejections"] >= 2:
                        return self._a7_stop(state, node, saved, "replan_exhausted")
        required = [key for key, gate in saved.get("replans", {}).items()
                    if gate.get("status") == "required"]
        if required:
            # A6's failure-family gate starts a fresh progress epoch for the
            # single O2 controller; it must not immediately hit its own
            # no-progress stop before the model can cite new evidence.
            saved["progress_reset_source"] = f"{node.node_id}:{index}"
            node.policy_feedback += (
                " Required action: read new evidence and submit ReplanRepair. Family IDs: " + str(required))
        # Continue the current response's remaining proposals through policy.
        if cursor < len(node.action_steps):
            return self._save(state, phase="policy", action_cursor=cursor, repair_state=saved)
        executed_tests = sum(
            1 for previous in node.get_trajectory() for step in previous.action_steps
            if step.action.name == "RunTests" and step.observation is not None
            and (step.observation.properties or {}).get("execution_status") != "skipped")
        if executed_tests >= self.action_policy.max_verification_attempts:
            return self._a7_stop(state, node, saved, "verification_budget_exhausted")
        if assessment.syntax_pending and not assessment.failures:
            from moatless.actions.check_syntax import CheckSyntaxArgs
            if CheckSyntaxArgs not in self.agent.action_map:
                raise ValueError("a7 requires CheckSyntax to validate the target interpreter")
            from moatless.message_history.policy import content_hash
            key = content_hash({"syntax": assessment.syntax_pending, "patch": assessment.patch_hash})
            attempts = saved.setdefault("syntax_attempts", {})
            if attempts.get(key, 0) >= 2:
                return self._a7_stop(state, node, saved, "verification_unavailable")
            attempts[key] = attempts.get(key, 0) + 1
            node.action_steps.append(ActionStep(action=CheckSyntaxArgs(paths=assessment.syntax_pending[:1]),
                                                origin="policy"))
            return self._save(state, phase="execute", action_cursor=len(node.action_steps) - 1,
                              repair_state=saved)
        if assessment.next_checks and node.file_context.generate_git_patch(ignore_tests=True).strip():
            # Mandatory checks run even while a ReplanRepair gate blocks writes.
            # Prefer not-yet-attempted checks so one failing check cannot mask
            # the rest of the ordered queue.
            from moatless.actions.run_tests import RunTestsArgs
            invalid_targets = sorted({
                target for record in records
                if not record.get("superseded") and record.get("status") == "not_run"
                for target in record.get("targets", [])
                if target in {item for request in assessment.next_checks
                              for item in request.get("test_files", [])}})
            if invalid_targets:
                node.policy_feedback += (
                    " Check targets are not executable test modules: " + str(invalid_targets)
                    + ". Revise the plan with an existing module or create the test file first.")
            repro = [item for item in assessment.outstanding if item.endswith(":public_reproduction")]
            if repro:
                from moatless.policy.repair import registered_plan
                plan = registered_plan(node)
                by_id = {check.id: check for check in (plan.checks if plan else [])}
                detail = [f"{item.split(':', 1)[0]} -> {by_id[item.split(':', 1)[0]].targets}"
                          for item in repro if item.split(":", 1)[0] in by_id]
                node.policy_feedback += (
                    " The base comparison showed no fixed tests for: " + str(repro)
                    + ((" Target(s): " + str(detail)) if detail else "")
                    + ". Add a failing reproduction to the check module or choose a check that fails on base.")
            attempts = saved.setdefault("attempts", {})
            ranked = []
            for order, request in enumerate(assessment.next_checks):
                key = self._a7_check_key(request, assessment, node, runtime.context.workspace)
                ranked.append((attempts.get(key, 0), order, key, request))
            for count, _, key, request in sorted(ranked, key=lambda row: (row[0], row[1])):
                if count >= 2:
                    continue
                attempts[key] = count + 1
                saved["check_exhausted_rounds"] = 0
                node.action_steps.append(ActionStep(action=RunTestsArgs(**request), origin="policy"))
                return self._save(state, phase="execute", action_cursor=len(node.action_steps) - 1,
                                  repair_state=saved)
            # Every known check has consumed its automatic attempts.  Give
            # the model a turn to revise the plan/test instead of stopping
            # immediately; O2 no-progress/budget remains the outer bound.
            saved["check_exhausted_rounds"] = saved.get("check_exhausted_rounds", 0) + 1
            node.policy_feedback += (
                " Automatic check attempts exhausted for: "
                + str([request.get("check_id") or request.get("test_files") for request in assessment.next_checks])
                + ". Revise the plan/test or provide a failing reproduction.")
            if saved["check_exhausted_rounds"] > 2:
                return self._a7_stop(state, node, saved, "verification_unavailable")
            return self._save(state, phase="commit", action_cursor=cursor, repair_state=saved)
        if assessment.state == "unavailable":
            saved["discovery_attempts"] = saved.get("discovery_attempts", 0) + 1
            if saved["discovery_attempts"] > 2:
                return self._a7_stop(state, node, saved, "verification_unavailable")
            node.policy_feedback += (
                " Discover a public behavioral reproduction or extend the plan with a mandatory check.")
        return self._save(state, phase="commit", action_cursor=cursor, repair_state=saved)

    async def _assess_completion(self, state, runtime):
        if self.action_policy.variant == "a7":
            return await self._a7_assess_completion(state, runtime)
        from moatless.policy.repair import assess
        from moatless.message_history.policy import content_hash
        from moatless.actions.run_tests import RunTestsArgs
        node = self._load(state, runtime)
        saved = deepcopy(state.get("repair_state", {"version": 1, "attempts": {}, "failure_families": {}}))
        assessment = assess(node, runtime.context.workspace)
        last_props = next(((step.observation.properties or {}) for step in reversed(node.action_steps)
                           if step.observation and (step.observation.properties or {}).get("execution_status") != "skipped"), {})
        if (last_props.get("fail_reason") == "environment_error"
                or any(failure["kind"] == "environment_error" for failure in assessment.failures)):
            saved["assessment"] = assessment.model_dump(mode="json")
            return self._repair_stop(state, node, "environment_blocked", saved)
        saved["assessment"] = assessment.model_dump(mode="json")
        index = min(state["action_cursor"], len(node.action_steps) - 1)
        if index >= 0:
            observed = node.action_steps[index]
            path = getattr(observed.action, "path", None)
            loops = saved.setdefault("edit_loops", {})
            if observed.action.name in {"StringReplace", "CreateFile", "InsertLine", "InsertLines", "AppendString", "str_replace_editor"} and last_props.get("fail_reason"):
                loop = loops.setdefault(path, {"failures": 0, "blocked": False})
                loop["failures"] += 1
                loop["blocked"] = loop["failures"] >= 2
            elif observed.action.name in {"ReadFile", "ViewCode", "ReadFiles"} and not last_props.get("fail_reason"):
                from moatless.repository_context import action_paths
                for read_path in action_paths(observed.action):
                    loops.pop(read_path, None)
        feedback = assessment.model_dump(mode="json")
        if last_props.get("fail_reason"):
            feedback["last_rejection"] = {"kind": last_props["fail_reason"], "source_id": f"{node.node_id}:{index}"}

        feedback["failures"] = feedback["failures"][:8]
        if last_props.get("test_run"):
            observed_observation = node.action_steps[index].observation if 0 <= index < len(node.action_steps) else None
            test_run = last_props["test_run"]
            feedback["last_test_output"] = ((observed_observation.message or "") if observed_observation else "")[-1200:]
            feedback["last_test_command"] = test_run.get("command") or last_props.get("commands")
            feedback["last_test_return_code"] = test_run.get("return_code") or last_props.get("return_code")
        node.policy_feedback = "Acceptance assessment: " + __import__("json").dumps(feedback, ensure_ascii=False)
        if assessment.state not in {"ready", "partial_ready"}:
            from moatless.policy.repair import registered_plan
            from moatless.policy.replanning import update_no_gain
            message, _ = update_no_gain(
                saved, node, index, last_props,
                stagnation_limit=self.action_policy.stagnation_limit,
                repeat_limit=self.action_policy.repeat_limit,
                plan=registered_plan(node))
            if message:
                node.policy_feedback += " " + message
        if assessment.state in {"ready", "partial_ready"}:
            self._skip_pending(node, state["action_cursor"] + 1, "acceptance_satisfied")
            return self._save(state, phase="finalize", repair_state=saved)
        cursor = state["action_cursor"] + 1
        while cursor < len(node.action_steps) and node.action_steps[cursor].is_executed():
            cursor += 1
        from moatless.execution_memory import test_records
        from moatless.policy.replanning import update_repair_control, accept_replan
        exhausted = update_repair_control(node, test_records(node, runtime.context.workspace), saved,
                                          self.action_policy.max_repair_attempts, assessment.failure_obligations)
        if exhausted:
            return self._repair_stop(state, node, exhausted, saved)
        proposal = last_props.get("replan_proposal")
        if proposal:
            accepted, message = accept_replan(node, proposal, saved)
            last_props["replan_validation"] = {"accepted": accepted, "message": message}
            node.policy_feedback += " " + message
            if not accepted:
                gates = [saved.get("replans", {}).get(proposal["family_id"])]
                if not gates[0]:
                    gates = [gate for gate in saved.get("replans", {}).values() if gate["status"] == "required"]
                for gate in gates:
                    if not gate:
                        continue
                    gate["rejections"] += 1
                    if gate["rejections"] >= 2:
                        return self._repair_stop(state, node, "replan_exhausted", saved)
        required = [key for key, gate in saved.get("replans", {}).items() if gate["status"] == "required"]
        if required:
            node.policy_feedback += " Required action: read new evidence and submit ReplanRepair. Family IDs: " + str(required)
        # Continue a fresh model repair/read proposal; dispatch checks when a
        # response is exhausted, an edit has occurred, or Finish was proposed.
        if cursor < len(node.action_steps):
            return self._save(state, phase="policy", action_cursor=cursor, repair_state=saved)
        if assessment.syntax_pending and not assessment.failures:
            from moatless.actions.check_syntax import CheckSyntaxArgs
            if CheckSyntaxArgs not in self.agent.action_map:
                raise ValueError("a6 requires CheckSyntax to validate the target interpreter")
            key = content_hash({"syntax": assessment.syntax_pending, "patch": assessment.patch_hash})
            attempts = saved.setdefault("syntax_attempts", {})
            if attempts.get(key, 0) >= 2:
                return self._repair_stop(state, node, "verification_unavailable", saved)
            attempts[key] = attempts.get(key, 0) + 1
            node.action_steps.append(ActionStep(action=CheckSyntaxArgs(paths=assessment.syntax_pending[:1]), origin="policy"))
            return self._save(state, phase="execute", action_cursor=len(node.action_steps)-1, repair_state=saved)
        if assessment.next_check and node.file_context.generate_git_patch(ignore_tests=True).strip():
            # A required replan blocks writes, not mandatory checks.  Running
            # the registered check here produces the evidence the replan gate
            # needs; otherwise a6 could reach verification_unavailable without
            # ever executing the registered check.
            total = sum(saved.get("attempts", {}).values())
            if total >= self.action_policy.max_verification_attempts:
                return self._repair_stop(state, node, "verification_budget_exhausted", saved)
            request = assessment.next_check
            key = content_hash({"request": request, "product": assessment.patch_hash,
                "workspace": node.repository_context.workspace_version})
            attempts = saved.setdefault("attempts", {})
            if attempts.get(key, 0) >= 2:
                return self._repair_stop(state, node, "verification_unavailable", saved)
            attempts[key] = attempts.get(key, 0) + 1
            node.action_steps.append(ActionStep(action=RunTestsArgs(**request), origin="policy"))
            return self._save(state, phase="execute", action_cursor=len(node.action_steps)-1, repair_state=saved)
        if assessment.state == "unavailable":
            saved["discovery_attempts"] = saved.get("discovery_attempts", 0) + 1
            if saved["discovery_attempts"] > 2:
                return self._repair_stop(state, node, "verification_unavailable", saved)
            node.policy_feedback += " Discover a public behavioral reproduction or extend the plan with a mandatory check."
        return self._save(state, phase="commit", action_cursor=cursor, repair_state=saved)

    async def _finalize_completion(self, state, runtime):
        if self.action_policy.variant == "a7":
            return await self._a7_finalize_completion(state, runtime)
        from moatless.policy.repair import assess, registered_plan, VerifiedCheckpoint
        from moatless.execution_memory import environment_version
        node = self._load(state, runtime)
        saved = deepcopy(state["repair_state"])
        assessment = assess(node, runtime.context.workspace)
        if assessment.state not in {"ready", "partial_ready"}:
            saved["assessment"] = assessment.model_dump(mode="json")
            return self._save(state, phase="assess", repair_state=saved)
        plan = registered_plan(node)
        checkpoint = VerifiedCheckpoint(product_patch=node.file_context.generate_git_patch(ignore_tests=True),
            patch_hash=assessment.patch_hash, plan_revision=plan.revision,
            receipt_sources=assessment.receipts, environment=environment_version(runtime.context.workspace))
        partial = assessment.state == "partial_ready"
        checkpoint_key = "partial_checkpoint" if partial else "verified_checkpoint"
        saved[checkpoint_key] = checkpoint.model_dump(mode="json")
        reason = "acceptance_partially_satisfied" if partial else "acceptance_satisfied"
        node.terminal = True
        node.completion_status = {"version": 2, "status": "partially_verified" if partial else "verified", "reason": reason,
            "verification_scope": "public acceptance checks", "checkpoint": saved[checkpoint_key],
            "explained_failures": assessment.explained_failures,
            "official_harness": "not_run", "resolved": None}
        return self._save(state, phase="commit", finish_reason=reason, repair_state=saved,
                          action_cursor=len(node.action_steps))

    async def _a7_finalize_completion(self, state, runtime):
        """Single completion gate: A6 acceptance plus O2 verification facts."""
        from moatless.execution_memory import environment_version, test_records
        from moatless.policy.action import extract_state
        from moatless.policy.repair import VerifiedCheckpoint, assess, registered_plan

        node = self._load(state, runtime)
        saved = deepcopy(state.get("repair_state") or self._a7_repair_state())
        policy = deepcopy(state.get("policy_state") or self._a7_policy_state())
        records = test_records(node, runtime.context.workspace)
        assessment = assess(node, runtime.context.workspace, records=records)
        saved["assessment"] = assessment.model_dump(mode="json")
        if assessment.state not in {"ready", "partial_ready"}:
            saved["stop_reason"] = "completion_not_ready"
            return self._save(state, phase="policy", action_cursor=len(node.action_steps),
                              policy_state=policy, repair_state=saved)
        cursor = max(0, len(node.action_steps) - 1)
        features, _ = extract_state(
            node, cursor, runtime.context.workspace, policy, variant="a7",
            repair_state=saved, assessment=assessment, records=records)
        partial = assessment.state == "partial_ready"
        explained_cases = {claim.get("failing_case") for claim in assessment.explained_failures}
        unexplained_regressions = [
            failure for failure in features.introduced_failures if failure not in explained_cases]
        gate_ok = features.has_product_patch and not unexplained_regressions
        if not partial:
            # A fully ready acceptance must also be visible to O2: a public
            # receipt bound to the current product patch and no failed check.
            gate_ok = gate_ok and features.public_test_passed and not features.current_test_failed
        if not gate_ok:
            saved["stop_reason"] = "completion_gate_mismatch"
            return self._save(state, phase="policy", action_cursor=len(node.action_steps),
                              policy_state=policy, repair_state=saved)
        plan = registered_plan(node)
        checkpoint = VerifiedCheckpoint(
            product_patch=node.file_context.generate_git_patch(ignore_tests=True),
            patch_hash=assessment.patch_hash, plan_revision=plan.revision,
            receipt_sources=assessment.receipts,
            environment=environment_version(runtime.context.workspace))
        checkpoint_key = "partial_checkpoint" if partial else "verified_checkpoint"
        saved[checkpoint_key] = checkpoint.model_dump(mode="json")
        reason = "acceptance_partially_satisfied" if partial else "acceptance_satisfied"
        node.terminal = True
        node.completion_status = {
            "version": 2,
            "status": "partially_verified" if partial else "verified",
            "reason": reason,
            "verification_scope": "public acceptance checks",
            "checkpoint": saved[checkpoint_key],
            "explained_failures": assessment.explained_failures,
            "official_harness": "not_run",
            "resolved": None,
        }
        return self._save(state, phase="commit", finish_reason=reason, repair_state=saved,
                          action_cursor=len(node.action_steps))

    async def _commit_iteration(self, state: GraphState, runtime: Runtime[GraphRuntime]) -> GraphState:
        self._load(state, runtime)
        reason = state["finish_reason"]
        if reason is None:
            reason = "single_step" if state["single_step"] else self._legacy_finish()
        if self.action_policy.variant == "a6" and reason and reason != "acceptance_satisfied":
            node = self._load(state, runtime)
            if not node.completion_status:
                node.completion_status = {"version": 2, "status": "unverified", "reason": reason,
                    "assessment": state.get("repair_state", {}).get("assessment"),
                    "official_harness": "not_run", "resolved": None}
        if self.action_policy.variant == "a7" and reason and reason not in {
                "acceptance_satisfied", "acceptance_partially_satisfied"}:
            node = self._load(state, runtime)
            status = (node.completion_status or {}).get("status")
            # Keep the specific stop reason recorded by the policy; only add
            # an unverified status where none exists or a false verified one.
            if not node.completion_status or status in {"verified", "partially_verified"}:
                node.completion_status = {"version": 2, "status": "unverified", "reason": reason,
                    "assessment": state.get("repair_state", {}).get("assessment"),
                    "official_harness": "not_run", "resolved": None}
        return self._save(state, phase="end" if reason else "prepare", finish_reason=reason)

    def _build_graph(self):
        builder = StateGraph(GraphState, context_schema=GraphRuntime)
        nodes = {
            "prepare_iteration": self._prepare_iteration,
            "generate_actions": self._generate_actions,
            "policy_decision": self._policy_decision,
            "execute_action": self._execute_action,
            "record_observation": self._record_observation,
            "assess": self._assess_completion,
            "finalize": self._finalize_completion,
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
            "assess": "assess",
            "finalize": "finalize",
            "commit": "commit_iteration",
            "end": END,
        }
        async def route(state):
            return state["phase"]

        builder.add_conditional_edges(START, route, routes)
        for name in nodes:
            builder.add_conditional_edges(name, route, routes)
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
                    if self.action_policy.variant in {"a6", "a7"}:
                        from moatless.policy.repair import assess
                        repair = recovered.get("repair_state", {})
                        partial = bool(repair.get("partial_checkpoint"))
                        checkpoint = repair.get("partial_checkpoint" if partial else "verified_checkpoint")
                        if checkpoint:
                            summary = assess(self.root.get_last_node(), self.agent._workspace)
                            if (summary.state != ("partial_ready" if partial else "ready")
                                    or checkpoint["patch_hash"] != summary.patch_hash):
                                raise RecoveryConflict("Saved verification inputs changed; revalidate in a new run before claiming completion")
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
        if self.action_policy.variant == "a6":
            if recovered and "repair_state" not in recovered:
                raise RecoveryConflict("Legacy trajectory cannot resume as a6; start a new run")
            state["repair_state"] = {"version": 1, "attempts": {}, "failure_families": {}, "discovery_attempts": 0}
        if self.action_policy.variant == "a7":
            if recovered and ("repair_state" not in recovered or "policy_state" not in recovered):
                raise RecoveryConflict("Legacy or a6 trajectory cannot resume as a7; start a new run")
            state["repair_state"] = self._a7_repair_state()
            state["policy_state"] = self._a7_policy_state()
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
