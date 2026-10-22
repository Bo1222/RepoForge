"""Known failures remain obligations until fresh execution discharges them."""

import re

from moatless.actions.test_targets import is_related_test
from moatless.message_history.policy import content_hash
from moatless.verification_inputs import test_identity


def complete_current_pass(record):
    return (
        record.get("effective_status") == "passed"
        and record.get("complete") is True
        and record.get("return_code") == 0
        and not record.get("timed_out")
        and record.get("recipe_version") == 3
        and record.get("counts", {}).get("passed", 0) > 0
    )


def failure_family(record, result):
    """Cause identity, independent of requested target batches and patch hashes."""
    frames = [
        f"{frame.get('file_path')}::{frame.get('method') or ''}"
        for frame in result.get("stacktrace", [])
        if frame.get("file_path") in record.get("product_files", [])
    ]
    text = result.get("failure_output") or ""
    error = re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*(?:Error|Exception)\b", text)
    cause = {"frames": sorted(set(frames)), "error": error[-1] if error else "test_failure"}
    if not frames:
        cause["product_files"] = sorted(record.get("product_files", []))
    return content_hash(cause)


def unresolved_failures(records, repository, changed, checks):
    """Replay facts, never infer resolution from a new candidate or new target."""
    obligations = {}
    for record in records:
        if record.get("product_source", "current") != "current" or record.get("timed_out"):
            continue
        targets = record.get("targets", [])
        related = any(
            record.get("related_targets", {}).get(path) is True
            or is_related_test(repository, path, record.get("product_files") or changed)
            for path in targets
        )
        related |= any(set(targets) == set(check.targets) for check in checks)
        if not related:
            continue
        if record.get("status") in {"failed", "error"}:
            failing = [
                item for item in record.get("results", []) if str(item.get("status", "")).lower() in {"failed", "error"}
            ]
            if not failing:
                failing = [
                    {"file_path": path, "name": None, "failure_output": "Unparsed test failure"} for path in targets
                ]
            for item in failing:
                case = test_identity(item)
                key = content_hash(
                    {
                        "source": record.get("test_source"),
                        "case": case,
                        "module": item.get("file_path") or targets,
                        "bundle": record.get("test_bundle_hash"),
                    }
                )
                obligations[key] = {
                    "id": key,
                    "source_id": record["source_id"],
                    "test_source": record.get("test_source"),
                    "test_bundle_hash": record.get("test_bundle_hash"),
                    "targets": targets,
                    "case": case,
                    "module": item.get("file_path"),
                    "module_hash": (record.get("verification_inputs") or {}).get(item.get("file_path")),
                    "family": failure_family(record, item),
                    "patch_hash": record.get("product_patch_hash"),
                    "excerpt": str(item.get("failure_output") or "")[-1600:],
                    "current_failure": record.get("effective_status") in {"failed", "error"},
                }
        if complete_current_pass(record):
            passed = {
                test_identity(item)
                for item in record.get("results", [])
                if str(item.get("status", "")).lower() == "passed"
            }
            for key, failure in list(obligations.items()):
                same_bundle = failure["test_bundle_hash"] == record.get("test_bundle_hash")
                comparison = record.get("comparison") or {}
                # A newly added repro need not make the old obligation
                # impossible to discharge. If its original module is unchanged,
                # a fresh same-case fail->pass in the new, comparable bundle is
                # additional execution evidence. Editing the failed assertion
                # itself still cannot erase that failure.
                comparable_reproof = (
                    bool(failure["module_hash"])
                    and failure["module_hash"] == (record.get("verification_inputs") or {}).get(failure["module"])
                    and comparison.get("status") == "known"
                    and failure["case"] in (comparison.get("fixed_tests") or [])
                )
                if (
                    failure["test_source"] == record.get("test_source")
                    and (same_bundle or comparable_reproof)
                    and set(failure["targets"]) <= set(targets)
                    and (failure["case"] in passed or failure["case"] is None)
                ):
                    del obligations[key]
    return list(obligations.values())
