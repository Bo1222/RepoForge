"""Feedback policy over the next ordered action, not a candidate search policy."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from moatless.actions.test_targets import discover_test_targets, is_related_test, is_verifiable_test_module
from moatless.evaluation.verification import VerificationState, snapshot as verification_snapshot
from moatless.execution_memory import test_records
from moatless.policy.failures import interpret as interpret_failure
from moatless.message_history.policy import content_hash
from moatless.repository_context import action_paths
from moatless.utils.file import is_test


class ActionPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    variant: Literal["off", "a1", "a2", "a3", "a4", "a5"] = "off"
    repeat_window: int = Field(default=8, ge=2)
    repeat_limit: int = Field(default=2, ge=1)
    stagnation_limit: int = Field(default=4, ge=1)
    reserve_rounds: int = Field(default=3, ge=1)
    input_price_per_million: float = Field(default=.30, ge=0)
    output_price_per_million: float = Field(default=1.20, ge=0)


class BudgetSnapshot(BaseModel):
    remaining_rounds: int
    spent_cost: float | None = 0
    max_cost: float | None = None
    cost_complete: bool = True
    estimated_next_call: float = 0
    estimate_basis: str = "configured input/output ceiling at frozen uncached reference rates; estimate, not a bill"

    @property
    def low_cost(self):
        return self.max_cost is not None and self.spent_cost is not None and (
            self.max_cost - self.spent_cost <= self.estimated_next_call)


class PolicyState(BaseModel):
    node_id: int
    step_index: int
    patch_hash: str
    product_patch_hash: str
    has_product_patch: bool
    signature: str
    repeat_count: int = 0
    no_progress: int = 0
    edits_since_test: int = 0
    changed_product_files: int = 0
    public_test_passed: bool = False
    current_test_failed: bool = False
    target: str | None = None
    correction_key: str
    corrected: bool = False
    replan_attempts: int = 0
    evidence: list[str] = Field(default_factory=list)
    verification_state: str = VerificationState.UNVERIFIED.value
    verification_confidence: float = 0.0
    introduced_failures: list[str] = Field(default_factory=list)
    failure_kind: str | None = None
    failure_signature: str | None = None
    last_action_signature: str | None = None
    failure_excerpt: str = ""
    failure_suggested_action: str | None = None


class PolicyDecision(BaseModel):
    version: Literal[1] = 1
    kind: Literal["allow", "replan", "verify", "stop"]
    reason: str
    target: str | None = None
    correction_key: str | None = None


def action_signature(action, execution):
    """Version the relevant paths; global searches use the workspace version."""
    paths = action_paths(action)
    scope = execution.get("files_before") if paths else execution.get("workspace_before")
    return content_hash({"action": action.model_dump(exclude={"thoughts"}),
                         "scope": scope, "environment": execution.get("environment")})


def extract_state(node, cursor, workspace, saved=None):
    """Build the O2 controller state from ordered execution facts."""
    saved = saved or {}
    if not node.action_steps:
        raise ValueError("Cannot extract policy state without a proposed action")
    cursor = max(0, min(cursor, len(node.action_steps) - 1))
    proposed = node.action_steps[cursor].action
    context = node.file_context
    patch = context.generate_git_patch() if context else ""
    product = context.generate_git_patch(ignore_tests=True) if context else ""
    from moatless.execution_memory import environment_version
    from moatless.repository_context import file_version, workspace_version
    paths = action_paths(proposed)
    versions = {}
    for path in paths:
        try:
            versions[path] = file_version(workspace, context, path)
        except (OSError, ValueError):
            versions[path] = "unavailable"
    execution = {"workspace_before": workspace_version(workspace, context),
                 "files_before": versions,
                 "environment": environment_version(workspace)}
    signature = action_signature(proposed, execution)
    history, seen, edits, stagnant, evidence = [], set(), 0, 0, []
    last_observation = None
    candidate_targets, latest_targets = set(), []
    for previous in node.get_trajectory():
        for i, step in enumerate(previous.action_steps):
            obs = step.observation
            if obs is None or (obs.properties or {}).get("execution_status") == "skipped":
                continue
            last_observation = obs
            props = obs.properties or {}
            ex = props.get("execution_context", {})
            sig = action_signature(step.action, ex)
            failure = props.get("fail_reason")
            progress = False
            actual_change = ex.get("workspace_before") != ex.get("workspace_after")
            product_edit = actual_change and step.action.name in {
                "StringReplace", "CreateFile", "InsertLine", "AppendString", "TextEditor"} and any(
                    not is_test(p) for p in action_paths(step.action)) and not failure
            if product_edit:
                edits += 1
                marker = ("patch", props.get("product_patch_hash", ex.get("workspace_after")))
                progress = marker not in seen
                seen.add(marker)
            view = props.get("code_view")
            if view and not failure:
                marker = ("read", sig, content_hash(obs.message or ""))
                progress |= marker not in seen and bool(obs.message)
                seen.add(marker)
            record = props.get("test_run")
            if record:
                edits = 0
                latest_targets = record.get("targets", [])
                candidate_targets.update(latest_targets)
                marker = ("test", content_hash({k: record.get(k) for k in (
                    "targets", "test_source", "status", "counts", "product_patch_hash")}))
                progress |= marker not in seen
                seen.add(marker)
            search = props.get("search_hits")
            if isinstance(search, dict):
                candidate_targets.update(f["file_path"] for f in search.get("files", []) if is_test(f["file_path"]))
            stagnant = 0 if progress else stagnant + 1
            if f"{previous.node_id}:{i}" == (saved or {}).get("progress_reset_after"):
                stagnant = 0
            history.append({"signature": sig, "no_gain": not progress,
                            "feedback": props.get("failure_signature") or content_hash(obs.message or "")})
            evidence.append(f"{previous.node_id}:{i}")
    if context:
        candidate_targets.update(f.file_path for f in context.files if is_test(f.file_path))
    # Prefer an explicitly proposed target, then the last check, then visited tests.
    explicit = [p.split("::")[0] for s in node.action_steps[cursor:]
                for p in getattr(s.action, "test_files", [])]
    changed_product_files = [f.file_path for f in context.files
                             if f.patch and not is_test(f.file_path)] if context else []
    ranked = discover_test_targets(
        workspace.repository, changed_product_files, candidate_targets, require_relation=True)
    targets = list(dict.fromkeys([*explicit, *ranked]))
    explicit_set = set(explicit)
    target = None
    from moatless.repository_context import relative_path
    for raw_path in targets:
        try:
            path = relative_path(workspace, raw_path)
            # A model-explicit test is trusted.  Other candidates (last run,
            # context hits, discovered tests) must relate to a changed product
            # file; otherwise an unrelated test becomes a false verification.
            if (raw_path not in explicit_set and path not in explicit_set
                    and changed_product_files
                    and not is_related_test(workspace.repository, path, changed_product_files)):
                continue
            file = context._files.get(path) if context else None
            base = file.get_base_content() if file else workspace.repository.get_file_content(path)
            if (is_test(path) and base and not workspace.repository.is_directory(path)
                    and is_verifiable_test_module(workspace.repository, path)):
                target = path
                break
        except (OSError, ValueError):
            continue
    records = [r for r in test_records(node, workspace) if not r["superseded"]]
    verification = verification_snapshot(
        content_hash(product) if product.strip() else None, target, records)
    failure = interpret_failure(
        (last_observation.properties or {}) if last_observation else {},
        (last_observation.message or "") if last_observation else "",
    )
    public_pass = verification.state == VerificationState.PUBLIC_VERIFIED
    failures = verification.state == VerificationState.TESTED_FAIL
    key = content_hash({"product": content_hash(product), "target": target,
                        "environment": execution["environment"]})
    return PolicyState(node_id=node.node_id, step_index=cursor, patch_hash=content_hash(patch),
        product_patch_hash=content_hash(product), has_product_patch=bool(product.strip()), signature=signature,
        repeat_count=0, no_progress=stagnant, edits_since_test=edits,
        changed_product_files=sum(bool(f.patch) and not is_test(f.file_path) for f in context.files) if context else 0,
        public_test_passed=public_pass, current_test_failed=failures, target=target, correction_key=key,
        corrected=key in (saved or {}).get("corrections", []), evidence=evidence[-8:],
        verification_state=verification.state.value,
        verification_confidence=verification.confidence,
        introduced_failures=verification.introduced_failures,
        failure_kind=failure.kind.value if failure else None,
        failure_signature=failure.signature if failure else None,
        last_action_signature=history[-1]["signature"] if history else None,
        failure_excerpt=failure.excerpt if failure else "",
        failure_suggested_action=failure.suggested_action if failure else None), history


class ActionPolicy:
    def __init__(self, config):
        self.config = config

    def decide(self, state, proposed_step, budget):
        cfg = self.config
        if cfg.variant == "off":
            return PolicyDecision(kind="allow", reason="disabled")
        terminal = proposed_step.action.name in {"Finish", "VerifiedFinish"}
        if not budget.cost_complete:
            return PolicyDecision(kind="stop", reason="unknown_cost")
        if terminal and not state.has_product_patch:
            return PolicyDecision(kind="stop", reason="no_product_patch")
        if cfg.variant in {"a3", "a5"} and state.failure_signature and (
                state.signature == state.last_action_signature):
            # T3.2: never repeat an action that just produced the same failure
            # signature.  One correction is allowed, then the policy stops.
            reason = f"repeated_{state.failure_kind or 'failure'}"
            if state.corrected:
                return PolicyDecision(kind="stop", reason=reason)
            return PolicyDecision(kind="replan", reason=reason,
                                  correction_key=state.correction_key)
        hard = cfg.variant in {"a3", "a4", "a5"}
        if cfg.variant in {"a2", "a3", "a5"} and (
                state.repeat_count >= cfg.repeat_limit or state.no_progress >= cfg.stagnation_limit):
            # T2.3: a no-progress/repeat stop must not bypass verification.
            # Give the agent one bounded repair/verify cycle first.
            if not state.replan_attempts:
                pending = self._verification_decision(state)
                if pending is not None:
                    return pending
            reason = ("p2p_regression" if state.introduced_failures
                      else state.failure_kind or "no_progress")
            return PolicyDecision(kind="stop" if state.replan_attempts else "replan", reason=reason)
        needs_check = terminal and state.has_product_patch and (
            not state.public_test_passed or state.current_test_failed)
        if cfg.variant in {"a2", "a3", "a4", "a5"} and state.has_product_patch and proposed_step.action.name != "RunTests":
            threshold = 1 if state.current_test_failed else 2
            if hard:
                threshold = 1 if state.changed_product_files > 1 or state.current_test_failed else 3
            needs_check |= state.edits_since_test >= threshold
        budget_reserve = hard and (
            budget.remaining_rounds <= cfg.reserve_rounds or budget.low_cost)
        needs_check |= budget_reserve and state.has_product_patch and not state.public_test_passed
        if needs_check:
            if state.current_test_failed and not state.corrected:
                # TESTED_FAIL is evidence for repair, not for repeating the
                # same verification.  The correction key bounds the loop.
                return self._repair_decision(state)
            if state.corrected:
                if terminal or budget_reserve:
                    return PolicyDecision(kind="stop", reason="verification_unavailable_or_failed")
                return PolicyDecision(kind="allow", reason="verification_already_attempted")
            if not state.target:
                return PolicyDecision(kind="replan", reason="need_public_test_target", correction_key=state.correction_key)
            return PolicyDecision(kind="verify", reason="budget_reserve" if budget_reserve else "verification_due",
                                  target=state.target, correction_key=state.correction_key)
        if budget_reserve:
            return PolicyDecision(kind="stop", reason="budget_reserve")
        return PolicyDecision(kind="allow", reason="eligible")

    def _verification_decision(self, state):
        """Return the verification/repair step required before a stop.

        This is a deterministic policy decision; it adds no prompt text.
        """
        if not state.has_product_patch:
            return None
        if state.verification_state == VerificationState.PUBLIC_VERIFIED.value:
            return None
        if state.verification_state == VerificationState.TESTED_FAIL.value:
            return self._repair_decision(state)
        if state.target:
            return PolicyDecision(kind="verify", reason="verification_due",
                                  target=state.target, correction_key=state.correction_key)
        return PolicyDecision(kind="replan", reason="need_public_test_target",
                              correction_key=state.correction_key)

    @staticmethod
    def _repair_decision(state):
        reason = ("p2p_regression" if state.introduced_failures
                  else state.failure_kind or "verification_failed")
        return PolicyDecision(kind="replan", reason=reason,
                              correction_key=state.correction_key)
