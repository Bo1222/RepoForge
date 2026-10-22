"""a6 acceptance ledger. Models propose obligations; execution facts discharge them.

This module only uses public issue text, workspace code and recorded executions.
It does not assign a probability of correctness or claim benchmark resolution.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from moatless.execution_memory import environment_version, test_records
from moatless.message_history.policy import content_hash
from moatless.repository_overlay import CurrentRepository
from moatless.actions.test_targets import discover_test_targets, is_related_test
from moatless.utils.file import is_test


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AcceptanceCriterion(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    behavior: str = Field(min_length=8, max_length=2000)
    public_quote: str = Field(min_length=3, max_length=2000)
    kind: Literal["issue_behavior", "regression"] = "issue_behavior"


class CheckSpec(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    criteria: list[str] = Field(min_length=1)
    targets: list[str] = Field(min_length=1, max_length=20)
    test_source: Literal["workspace", "base_public"] = "base_public"
    proof: Literal["fail_to_pass", "regression"] = "fail_to_pass"


class AcceptancePlan(StrictModel):
    version: Literal[1] = 1
    revision: int = Field(default=1, ge=1)
    criteria: list[AcceptanceCriterion] = Field(min_length=1, max_length=30)
    checks: list[CheckSpec] = Field(default_factory=list, max_length=60)

    @model_validator(mode="after")
    def valid_links(self):
        ids = [item.id for item in self.criteria]
        check_ids = [check.id for check in self.checks]
        if len(set(ids)) != len(ids) or len(set(check_ids)) != len(check_ids):
            raise ValueError("IDs must be unique")
        if not any(item.kind == "issue_behavior" for item in self.criteria):
            raise ValueError("At least one public issue behavior is required")
        for check in self.checks:
            if not set(check.criteria) <= set(ids):
                raise ValueError("Check refers to unknown criterion")
            if any(item.kind == "issue_behavior" and item.id in check.criteria for item in self.criteria) and check.proof != "fail_to_pass":
                raise ValueError("Issue behavior requires a public failing reproduction, not unrelated green tests")
        return self


class CompletionAssessment(StrictModel):
    version: Literal[1] = 1
    state: Literal["partial", "repair", "unavailable", "ready", "partial_ready"] = "partial"
    outstanding: list[str] = Field(default_factory=list)
    failures: list[dict] = Field(default_factory=list)
    receipts: list[str] = Field(default_factory=list)
    next_check: dict | None = None
    # Ordered queue: one exhausted check must not permanently mask other
    # mandatory checks.  next_check stays the head for older readers.
    next_checks: list[dict] = Field(default_factory=list)
    patch_hash: str | None = None
    failure_obligations: list[dict] = Field(default_factory=list)
    comparisons: list[dict] = Field(default_factory=list)
    syntax_pending: list[str] = Field(default_factory=list)
    explained_failures: list[dict] = Field(default_factory=list)


class VerifiedCheckpoint(StrictModel):
    version: Literal[1] = 1
    product_patch: str
    patch_hash: str
    plan_revision: int
    receipt_sources: list[str]
    environment: str


def registered_plan(node) -> AcceptancePlan | None:
    for previous in reversed(node.get_trajectory()):
        for step in reversed(previous.action_steps):
            props = step.observation.properties if step.observation else {}
            if props and props.get("accepted_plan"):
                return AcceptancePlan.model_validate(props["accepted_plan"])
    return None


def validate_revision(node, plan, workspace=None):
    if any(len(check.targets) != 1 for check in plan.checks):
        raise ValueError("Each check must name one explicit module; register separate checks for additional modules")
    public_text = "\n".join(n.user_message for n in node.get_trajectory() if n.user_message)
    for item in plan.criteria:
        if item.public_quote not in public_text:
            raise ValueError(f"{item.id}: quote must occur verbatim in public task text")
    previous = registered_plan(node)
    if previous:
        if plan.revision != previous.revision + 1:
            raise ValueError("Revision must advance by one")
        old = {item.id: item for item in previous.criteria}
        new = {item.id: item for item in plan.criteria}
        if any(key not in new or new[key] != value for key, value in old.items()):
            raise ValueError("Existing acceptance obligations cannot be deleted or weakened; add a new criterion")
        checks = {check.id: check for check in plan.checks}
        from moatless.actions.test_targets import is_existing_test_module
        for check in previous.checks:
            if check.id in checks and checks[check.id] == check:
                continue
            # A check whose target is not an executable module was never
            # acceptance evidence.  Allow repairing/replacing it rather than
            # trapping the model behind an unsatisfiable obligation.
            if workspace is not None and not is_existing_test_module(
                    workspace, node.file_context, check.targets[0], check.test_source):
                continue
            raise ValueError("Existing mandatory checks cannot be removed or changed; append a check")
    elif plan.revision != 1:
        raise ValueError("Initial plan revision must be 1")


def _request(check, product_source="current"):
    return {"test_files": check.targets, "test_source": check.test_source,
            "product_source": product_source, "check_id": check.id}


def _names(record, status):
    return {f"{result.get('file_path')}::{result.get('name')}" for result in record.get("results", [])
            if str(result.get("status", "")).lower() == status and result.get("name")}


def _complete(record):
    from moatless.verification_inputs import receipt_complete
    return receipt_complete(record)


def _enqueue(result, request):
    """Append a deterministic, duplicate-free check request."""
    if request and all(content_hash(item) != content_hash(request) for item in result.next_checks):
        result.next_checks.append(request)


def assess(node, workspace, records=None) -> CompletionAssessment:
    context = node.file_context
    product = context.generate_git_patch(ignore_tests=True) if context else ""
    digest = content_hash(product)
    result = CompletionAssessment(patch_hash=digest)
    if not product.strip():
        result.outstanding.append("product_patch")
    plan = registered_plan(node)
    if plan is None:
        result.outstanding.append("acceptance_plan")
        return result
    overlay = CurrentRepository(workspace.repository, context)
    changed = [f.file_path for f in context.files if f.patch and not is_test(f.file_path)]
    environment = environment_version(workspace)
    syntax = {}
    for previous in node.get_trajectory():
        for step in previous.action_steps:
            for check in (step.observation.properties or {}).get("syntax_checks", []) if step.observation else []:
                syntax[(check.get("path"), check.get("content_hash"), check.get("environment"))] = check
    for path in changed:
        if not path.endswith(".py"):
            continue
        check = syntax.get((path, content_hash(overlay.get_file_content(path)), environment))
        if check and check.get("status") == "valid" and check.get("interpreter"):
            continue
        result.outstanding.append("target_syntax:" + path)
        if check and check.get("status") == "invalid":
            result.failures.append({"kind": "syntax_error", "file": path, "excerpt": check.get("message", "")})
        else:
            result.syntax_pending.append(path)
    records = test_records(node, workspace) if records is None else records
    env = environment_version(workspace)
    current = [r for r in records if not r["superseded"] and r.get("product_source") == "current"
               and r.get("effective_status") != "stale" and r.get("environment_after") == env]
    from moatless.policy.obligations import unresolved_failures
    obligations = unresolved_failures(records, overlay, changed, plan.checks)
    result.failure_obligations = obligations
    for failure in obligations:
        if failure["current_failure"]:
            from moatless.policy.failures import interpret
            record = next(row for row in records if row["source_id"] == failure["source_id"])
            signal = interpret({"test_run": record}, failure["excerpt"])
            kind = signal.kind.value if signal else "test_failure"
            if failure["case"] in (record.get("comparison", {}).get("introduced_failures") or []):
                kind = "p2p_regression"
            result.failures.append({**failure, "kind": kind,
                                    "tests": failure["targets"], "failing_tests": [failure["case"]]})
        else:
            result.outstanding.append("unresolved_failure:" + failure["id"])
            if not result.failures:
                _enqueue(result, {"test_files": failure["targets"], "test_source": failure["test_source"]})
    satisfied = set()
    valid_public = False
    for check in plan.checks:
        from moatless.verification_inputs import capture_recipe
        bundle = capture_recipe(workspace, context, check.targets, check.test_source)["test_bundle_hash"]
        matching = [r for r in current if set(r["targets"]) == set(check.targets)
                    and r.get("test_source") == check.test_source and r.get("test_bundle_hash") == bundle]
        passes = [r for r in matching if _complete(r) and r.get("effective_status") == "passed"
                  and r.get("return_code") == 0 and r.get("counts", {}).get("passed", 0) > 0]
        if not passes:
            result.outstanding.append(check.id)
            if not result.failures:
                _enqueue(result, _request(check))
            continue
        receipt = passes[-1]
        result.receipts.append(receipt["source_id"])
        comparison = receipt.get("comparison") or {"status": "unknown", "introduced_failures": None}
        result.comparisons.append({"check_id": check.id, **comparison})
        if comparison["status"] != "known":
            result.outstanding.append(check.id + ":baseline_comparison")
            if not result.failures:
                _enqueue(result, _request(check, "base"))
            continue
        result.receipts.append(comparison["baseline_source"])
        if comparison["introduced_failures"]:
            result.failures.append({"kind": "p2p_regression", "tests": comparison["introduced_failures"],
                                    "source_id": receipt["source_id"], "excerpt": "Introduced failures confirmed by public baseline"})
            continue
        if check.proof == "fail_to_pass" and not comparison.get("fixed_tests"):
            result.outstanding.append(check.id + ":public_reproduction")
            # Repeating an already green baseline cannot manufacture a repro.
            continue
        satisfied.update(check.criteria)
        valid_public |= check.test_source == "base_public" and any(is_related_test(overlay, path, changed) for path in check.targets)
    result.outstanding.extend(item.id for item in plan.criteria if item.id not in satisfied)
    from moatless.verification_inputs import verification_input, declared_inputs
    declared = declared_inputs(workspace, [target for check in plan.checks for target in check.targets])
    modified_auxiliary = [f.file_path for f in context.files if f.patch and verification_input(f.file_path, declared)]
    if modified_auxiliary:
        workspace_passes = [r for r in current if r.get("test_source") == "workspace" and _complete(r)
                            and r.get("effective_status") == "passed" and r.get("return_code") == 0
                            and r.get("comparison", {}).get("status") == "known"
                            and not r["comparison"].get("introduced_failures")
                            and any(is_related_test(overlay, p, changed) for p in r["targets"])]
        if workspace_passes:
            result.receipts.extend([workspace_passes[-1]["source_id"], workspace_passes[-1]["comparison"]["baseline_source"]])
        else:
            result.outstanding.append("workspace_validation_inputs")
            if not result.failures:
                targets = discover_test_targets(overlay, changed, require_relation=True)
                if targets:
                    incomplete = [r for r in current if r.get("test_source") == "workspace" and _complete(r)
                                  and r.get("effective_status") == "passed" and set(r["targets"]) == set(targets[:1])]
                    _enqueue(result, {"test_files": targets[:1], "test_source": "workspace",
                                      "product_source": "base" if incomplete else "current"})
    # Original public regression is independent of outstanding task obligations.
    # Discharging it must never erase a pending failed-case rerun or baseline.
    if not valid_public:
        public = [r for r in current if r.get("test_source") == "base_public" and _complete(r)
                  and r.get("effective_status") == "passed" and r.get("return_code") == 0
                  and any(is_related_test(overlay, p, changed) for p in r["targets"])
                  and r.get("comparison", {}).get("status") == "known"
                  and not r["comparison"].get("introduced_failures")]
        if public:
            result.receipts.extend([public[-1]["source_id"], public[-1]["comparison"]["baseline_source"]])
        else:
            result.outstanding.append("original_public_regression")
            if not result.failures:
                incomplete = [r for r in current if r.get("test_source") == "base_public" and _complete(r)
                    and r.get("effective_status") == "passed" and r.get("comparison", {}).get("status") != "known"
                    and any(is_related_test(overlay, p, changed) for p in r["targets"])]
                if incomplete:
                    _enqueue(result, {"test_files": incomplete[-1]["targets"], "test_source": "base_public", "product_source": "base"})
                else:
                    targets = discover_test_targets(workspace.repository, changed, require_relation=True)
                    if targets:
                        _enqueue(result, {"test_files": targets[:1], "test_source": "base_public"})
    from moatless.policy.conflicts import explained_conflicts
    explained = explained_conflicts(node, workspace, records, satisfied)
    if explained:
        explained_keys = {(claim["candidate_source"], claim["failing_case"]) for claim in explained}
        result.explained_failures = explained
        result.failures = [failure for failure in result.failures
                           if (failure.get("source_id"), failure.get("case")) not in explained_keys]
        # This is a partial delivery path only. The original public failure
        # and its baseline are retained verbatim, with the public conflict.
        if not result.failures and "original_public_regression" in result.outstanding:
            explained_sources = {claim["candidate_source"] for claim in explained}
            if any(r["source_id"] in explained_sources and _complete(r) for r in current):
                result.outstanding.remove("original_public_regression")
                for record in current:
                    if record["source_id"] in explained_sources:
                        result.receipts.extend([record["source_id"], record["comparison"]["baseline_source"]])
    if result.failures:
        result.state, result.next_checks = "repair", []
    elif not result.outstanding:
        result.state = "partial_ready" if explained else "ready"
    elif not result.next_checks:
        result.state = "unavailable"
    result.next_check = result.next_checks[0] if result.next_checks else None
    return result
