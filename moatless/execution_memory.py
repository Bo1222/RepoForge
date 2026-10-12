"""System-owned execution facts; all records live on existing observations."""

import os
import re
import sys
from typing import Literal

from pydantic import BaseModel, Field

from moatless.message_history.policy import content_hash
from moatless.repository_context import action_paths, file_version, invalidate_read_cache, refresh_context, relative_path, workspace_version


class TestRunRecord(BaseModel):
    version: Literal[1] = 1
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


def environment_version(workspace) -> str:
    runtime, environment = workspace.runtime, workspace._environment
    def identity(component):
        if component is None:
            return None
        return {"class": f"{type(component).__module__}.{type(component).__name__}",
                "instance_id": getattr(component, "instance_id", None),
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
    execution = {"source_id": source, "workspace_before": before["workspace"], "workspace_after": after}
    properties["execution_context"] = execution
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
            if status == "passed" and code != 0:
                status = "unknown"
        elif properties.get("fail_reason") == "no_test_files":
            status = "not_run"
        elif node.file_context:
            requested = {path.split("::")[0] for path in targets}
            files = [file for file in node.file_context.test_files if file.file_path in requested]
            results = [result for file in files for result in file.test_results]
            for result in results:
                key = "timed_out" if result.timed_out else result.status.value.lower()
                counts[key] = counts.get(key, 0) + 1
            if counts.get("error") or counts.get("timed_out"):
                status = "error"
            elif counts.get("failed"):
                status = "failed"
            elif (results and counts.get("passed") == len(results)
                  and all(file.test_results for file in files) and {file.file_path for file in files} == requested):
                status = "passed"
        record = TestRunRecord(source_id=source, command=command, targets=targets, status=status,
                               return_code=properties.get("return_code"), counts=counts,
                               workspace_before=before["workspace"], workspace_after=after,
                               environment_before=before["environment"], environment_after=environment_version(workspace))
        properties["test_run"] = record.model_dump()
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
            key = content_hash({"command": record["command"], "targets": sorted(record["targets"])})
            if key in latest:
                records[latest[key]]["superseded"] = True
            record["superseded"] = False
            record["effective_status"] = record["status"] if (
                record["workspace_before"] == record["workspace_after"] == current
                and record["environment_before"] == record["environment_after"] == environment
            ) else "stale"
            latest[key] = len(records)
            records.append(record)
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
    latest = [record for record in test_records(node, workspace) if not record["superseded"]]
    status = "verified" if latest and all(record["effective_status"] == "passed" for record in latest) else "unverified"
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
