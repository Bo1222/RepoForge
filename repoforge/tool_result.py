"""Additive, versioned execution envelope; legacy Observation remains readable."""


def describe_result(action: str, props: dict, source: str) -> dict:
    reason = props.get("fail_reason")
    status = props.get("execution_status", "ok")
    if reason:
        status = {"syntax_error": "syntax_error", "no_changes": "no_changes",
                  "file_not_found": "not_found", "no_search_hits": "not_found",
                  "no_test_files": "invalid_args", "tool_timeout": "timeout",
                  "protocol_error": "protocol_error", "environment_error": "environment_error"}.get(reason, "invalid_args")
    test = props.get("test_run", {})
    return {"version": 2, "status": status,
            "outcome": test.get("status", "unknown" if reason else "ok"),
            "failure_kind": reason, "retryable": status in {"timeout", "protocol_error"},
            "evidence": {"source_id": source, "input_identity": props.get("input_identity"),
                         "files": props.get("test_files", []), "patch_hash": props.get("product_patch_hash")},
            "complete": props.get("complete", status == "ok"),
            "workspace_consistent": not props.get("patch_stale", False),
            "continuation_safe": props.get("continuation_safe", True)}
