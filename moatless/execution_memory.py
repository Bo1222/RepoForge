"""System-owned execution facts; all records live on existing observations."""

import os
import re
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from moatless.message_history.policy import content_hash
from moatless.repository_context import action_paths, file_version, invalidate_read_cache, refresh_context, relative_path, workspace_version


class TestRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: Literal[1, 2, 3] = 3
    source_id: str
    command: str | None = None
    targets: list[str] = Field(default_factory=list)
    status: Literal["passed", "failed", "error", "unknown", "not_run"]
    return_code: int | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    workspace_before: str
    workspace_after: str
    environment_before: str
    environment_after: str
    test_source: Literal["workspace", "base_public"] = "workspace"
    test_origins: dict[str, str] = Field(default_factory=dict)
    product_patch_hash: str | None = None
    tested_patch_hash: str | None = None
    introduced_failures: list[str] | None = None
    recipe_version: int | None = None
    input_identity: str | None = None
    input_identity_after: str | None = None
    base_identity: dict | None = None
    verification_inputs: dict | None = None
    product_inputs_hash: str | None = None
    runner_fingerprint: str | None = None
    public_baseline: dict | None = None
    test_bundle_hash: str | None = None
    runner: str | None = None
    product_source: str = "current"
    check_id: str | None = None
    complete: bool | None = None
    timed_out: bool = False
    patch_stale: bool = False
    product_files: list[str] = Field(default_factory=list)
    related_targets: dict[str, bool] = Field(default_factory=dict)
    commands: list[str | None] = Field(default_factory=list)
    started_at: str | None = None
    duration: float | None = None
    partial: bool = False
    uncompleted_targets: list[str] = Field(default_factory=list)
    results: list[dict] = Field(default_factory=list)
    # Historical field retained for reading old trajectories. Export never
    # selects a patch from receipts; checkpoints support explicit recovery.
    product_patch: str | None = None


def environment_version(workspace) -> str:
    runtime, environment = workspace.runtime, getattr(workspace, "_environment", None)
    def identity(component):
        if component is None:
            return None
        from moatless.runtime.runtime import RuntimeEnvironment
        return {"class": f"{type(component).__module__}.{type(component).__name__}",
                "instance_id": getattr(component, "instance_id", None),
                "compiler": component.compilation_configuration() if isinstance(component, RuntimeEnvironment) else None,
                "env": getattr(component, "env", None), "shell": getattr(component, "shell", None)}
    # Only the digest is retained; no environment values/credentials are logged.
    return content_hash({"python": sys.version, "prefix": sys.prefix,
                         "process_env": dict(os.environ),
                         "runtime": identity(runtime), "environment": identity(environment)})


READ_ACTIONS = {"ReadFile", "ReadFiles", "ViewCode", "GrepTool", "SemanticSearch", "FindClass", "FindFunction", "FindCodeSnippet", "GlobTool"}


def before_action(node, step, workspace, policy) -> dict:
    paths = action_paths(step.action)
    try:
        invalidate_read_cache(workspace, node.file_context, paths)
    except (OSError, ValueError):
        pass  # Action retains responsibility for reporting invalid paths.
    refresh_context(node, workspace, policy.load_repo_instructions, paths)
    return {"workspace": node.repository_context.workspace_version,
            "environment": environment_version(workspace),
            "context_files": list(node.file_context._files) if node.file_context else [],
            "files": {path: data["content_hash"] for path, data in node.repository_context.files.items()}}


def observed_paths(args, observation) -> list[str]:
    paths = set(action_paths(args))
    search = (observation.properties or {}).get("search_hits")
    if isinstance(search, dict):
        paths.update(file["file_path"] for file in search.get("files", []) if file.get("file_path"))
    # Local grep and ReadFiles have path-bearing results; search/span tools can
    # additionally register their selected paths in FileContext.
    for line in (observation.message or "").splitlines():
        match = re.match(r"(?:\./)?([^:\n]+):\d+:", line) or re.match(r"```([^\s`]+)$", line)
        if match:
            paths.add(match.group(1))
    return sorted(paths)


def after_action(node, step, observation, workspace, policy, before) -> None:
    index = next(i for i, candidate in enumerate(node.action_steps) if candidate is step)
    source = f"{node.node_id}:{index}"
    paths = observed_paths(step.action, observation) if step.action.name in READ_ACTIONS else action_paths(step.action)
    refresh_context(node, workspace, policy.load_repo_instructions, paths)
    after = node.repository_context.workspace_version
    if observation.properties is None:
        observation.properties = {}
    properties = observation.properties
    if step.action.name in {"StringReplace", "InsertLine", "InsertLines", "AppendString"}:
        properties["auto_loaded"] = [path for path in action_paths(step.action)
            if node.file_context and path in node.file_context._files and path not in before.get("context_files", [])]
    execution = {"source_id": source, "workspace_before": before["workspace"], "workspace_after": after,
                 "files_before": {p: before["files"].get(p) for p in action_paths(step.action)},
                 "environment": environment_version(workspace)}
    properties["execution_context"] = execution
    if node.file_context:
        properties.setdefault("product_patch_hash", content_hash(node.file_context.generate_git_patch(ignore_tests=True)))
        properties["patch_hash"] = content_hash(node.file_context.generate_git_patch())
    if step.action.name in READ_ACTIONS and properties.get("fail_reason") in {None, "no_search_hits"}:
        versions = {}
        for path in paths:
            try:
                path = relative_path(workspace, path)
                versions[path] = file_version(workspace, node.file_context, path)
            except (OSError, ValueError):
                continue
        properties["code_view"] = {
            "files": versions, "workspace_version": after,
            "consistent": before["workspace"] == after,
            # Searches may depend on files not present in the returned hits.
            "workspace_scoped": step.action.name not in {"ReadFile", "ViewCode"},
            "start_line": getattr(step.action, "start_line", None) or 1,
            "end_line": getattr(step.action, "end_line", None),
            "ranges_by_file": {path: info["display_ranges"] for path, info in properties.get("files", {}).items()
                               if isinstance(info, dict) and info.get("display_ranges")}
                              if isinstance(properties.get("files"), dict) else {},
        }
    if step.action.name == "RunTests":
        command = getattr(step.action, "command", None)
        targets = list(getattr(step.action, "test_files", []) or [])
        counts, status = {}, "unknown"
        if command is not None:
            status = properties.get("test_status", "unknown")
            if status not in {"passed", "failed", "error", "unknown", "not_run"}:
                status = "unknown"
            code = properties.get("return_code")
            properties.setdefault("complete", code is not None and not properties.get("timed_out"))
            if status == "passed" and code != 0:
                status = "unknown"
        elif properties.get("fail_reason") == "no_test_files":
            status = "not_run"
        else:
            # Only facts returned by this invocation can determine its outcome.
            counts = properties.get("result_counts", {})
            statuses = properties.get("result_statuses", [])
            code = properties.get("return_code")
            if properties.get("timed_out") or properties.get("fail_reason") == "tool_timeout":
                status = "error"
            elif "failed" in statuses:
                status = "failed"
            elif "error" in statuses or code not in {None, 0, 1}:
                status = "error"
            elif (properties.get("complete") is True and code == 0 and statuses
                  and "passed" in statuses and all(value in {"passed", "skipped"} for value in statuses)):
                status = "passed"
        record = TestRunRecord(source_id=source, command=command, targets=targets, status=status,
                               return_code=properties.get("return_code"), counts=counts,
                               workspace_before=before["workspace"], workspace_after=after,
                               environment_before=before["environment"], environment_after=environment_version(workspace),
                               test_source=properties.get("test_source", "workspace"),
                               test_origins=properties.get("test_origins", {}),
                               product_patch_hash=properties.get("product_patch_hash"),
                               tested_patch_hash=properties.get("tested_patch_hash"),
                               introduced_failures=properties.get("introduced_failures"),
                               **{key: properties[key] for key in ("input_identity", "test_bundle_hash", "runner",
                                   "product_source", "check_id", "complete", "timed_out", "patch_stale", "recipe_version",
                                   "input_identity_after", "base_identity", "verification_inputs", "product_inputs_hash",
                                   "runner_fingerprint", "public_baseline", "product_files", "related_targets", "commands", "started_at",
                                   "duration", "partial", "uncompleted_targets") if key in properties},
                               results=properties.get("test_results", []),
                               product_patch=properties.get("product_patch"))
        properties["test_run"] = record.model_dump()
    from moatless.tool_result import describe_result
    properties["tool_result"] = describe_result(step.action.name, properties, source)
    if properties.get("fail_reason") or (properties.get("test_run", {}).get("status") in {"failed", "error"}):
        diagnostic = re.sub(r"0x[\da-fA-F]+", "0xADDRESS", (observation.message or "")[-1500:])
        properties["failure_signature"] = content_hash({"action": step.action.name,
                "reason": properties.get("fail_reason"), "diagnostic": diagnostic})
    # Classification runs after the returned observation is attached to Node.


def test_records(node, workspace) -> list[dict]:
    current = workspace_version(workspace, node.file_context)
    environment = environment_version(workspace)
    records, latest = [], {}
    for previous in node.get_trajectory():
        for step in previous.action_steps:
            raw = (step.observation.properties or {}).get("test_run") if step.observation else None
            if raw is None:
                continue
            record = TestRunRecord.model_validate(raw).model_dump()
            key = content_hash({"command": record["command"], "targets": sorted(record["targets"]),
                                "test_source": record["test_source"], "product_source": record["product_source"],
                                "bundle": record["test_bundle_hash"],
                                "environment": record["environment_before"]})
            if key in latest and record.get("complete") is True:
                records[latest[key]]["superseded"] = True
            record["superseded"] = False
            if record.get("recipe_version") == 3:
                from moatless.verification_inputs import capture_recipe
                try:
                    expected = capture_recipe(workspace, node.file_context, record["targets"],
                                              record["test_source"], record["product_source"])
                    same_inputs = record["input_identity"] == content_hash(expected) == record.get("input_identity_after")
                except (OSError, ValueError):
                    same_inputs = False
                record["effective_status"] = record["status"] if same_inputs and not record["patch_stale"] else "stale"
                if record.get("complete") is True:
                    latest[key] = len(records)
                records.append(record)
                continue
            same_workspace = record["workspace_before"] == record["workspace_after"] == current
            if record["test_source"] == "base_public":
                same_workspace = (record["workspace_before"] == record["workspace_after"]
                    and node.file_context is not None
                    and record["product_patch_hash"] == content_hash(node.file_context.generate_git_patch(ignore_tests=True)))
            record["effective_status"] = record["status"] if (
                same_workspace and not record.get("patch_stale")
                and record["environment_before"] == record["environment_after"] == environment
            ) else "stale"
            if record.get("complete") is True:
                latest[key] = len(records)
            records.append(record)
    # Comparisons are derived immutable relations, not mutations of old receipts.
    from moatless.verification_inputs import compare_runs
    for candidate in records:
        if candidate.get("product_source") != "current":
            continue
        baselines = [row for row in records if row.get("product_source") == "base"
                     and set(row["targets"]) == set(candidate["targets"])
                     and row["test_source"] == candidate["test_source"]]
        comparisons = [compare_runs(candidate, baseline) for baseline in baselines]
        comparison = next((row for row in reversed(comparisons) if row["status"] == "known"),
                          comparisons[-1] if comparisons else compare_runs(candidate, None))
        candidate["comparison"] = comparison
        candidate["public_baseline"] = {"status": comparison["status"],
                                        "source_id": comparison.get("baseline_source"),
                                        "reason": comparison.get("reason")}
        candidate["introduced_failures"] = comparison["introduced_failures"]
    return records


def failure_records(node) -> list[dict]:
    failures = {}
    for previous in node.get_trajectory():
        for i, step in enumerate(previous.action_steps):
            props = (step.observation.properties or {}) if step.observation else {}
            signature = props.get("failure_signature")
            if not signature:
                continue
            version = props["execution_context"]["workspace_after"]
            group = failures.setdefault((signature, version), {"signature": signature, "workspace_version": version,
                "reason": props.get("fail_reason"), "sources": [], "attempts": 0})
            group["sources"].append(f"{previous.node_id}:{i}")
            group["attempts"] += 1
    return list(failures.values())


def classify_completion(node, workspace) -> dict:
    from moatless.evaluation.verification import VerificationState, snapshot as verification_snapshot
    latest = [record for record in test_records(node, workspace) if not record["superseded"]]
    product_patch = ""
    if node.file_context is not None:
        product_patch = (content_hash(node.file_context.generate_git_patch(ignore_tests=True))
                         if node.file_context.generate_git_patch(ignore_tests=True).strip() else None)
    verification = verification_snapshot(product_patch, None, latest)
    status = "verified" if verification.state == VerificationState.PUBLIC_VERIFIED else "unverified"
    if latest and all(record.get("command") for record in latest):
        status = "verified" if any(record["effective_status"] == "passed" for record in latest) and not any(
            record["effective_status"] in {"failed", "error"} for record in latest) else "unverified"
    state = node.task_state
    blocked_claim = bool(state and (state.phase == "blocked" or state.blocked_by))
    current = workspace_version(workspace, node.file_context)
    superseded_tests = {record["source_id"] for record in test_records(node, workspace) if record["superseded"]}
    failures = {source for failure in failure_records(node) if failure["workspace_version"] == current
                for source in failure["sources"] if source not in superseded_tests}
    cited = {f"{ref.node_id}:{ref.step_index}" for item in [*(state.plan if state else []), *(state.hypotheses if state else [])]
             for ref in item.evidence}
    evidence = sorted(cited.intersection(failures))
    if blocked_claim and evidence:
        status = "blocked"
    return {"version": 1, "status": status, "verification_scope": "task-local recorded checks",
            "workspace_version": current, "environment_version": environment_version(workspace),
            "test_sources": [r["source_id"] for r in latest],
            "blocked_claim": state.blocked_by if blocked_claim else [], "blocked_evidence": evidence if blocked_claim else [],
            "official_harness": "not_run", "resolved": None}
