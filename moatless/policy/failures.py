"""Map test/tool outcomes to concrete next interventions.

The interpreter is rule-based on purpose: it runs on every failed observation
and must be deterministic and cheap.  The policy consumes the resulting
``FailureSignal`` to choose a different action instead of repeating the same
failed edit/test.
"""

from enum import Enum

from pydantic import BaseModel, Field

from moatless.message_history.policy import content_hash


class FailureKind(str, Enum):
    COLLECTION_ERROR = "collection_error"
    ASSERTION_FAILURE = "assertion_failure"
    PRODUCT_EXCEPTION = "product_exception"
    TEST_TIMEOUT = "test_timeout"
    NO_TEST_TARGET = "no_test_target"
    TEST_FILES_MODIFIED = "test_files_modified"
    P2P_REGRESSION = "p2p_regression"
    F2P_STILL_FAILING = "f2p_still_failing"
    ENVIRONMENT_ERROR = "environment_error"
    PROTOCOL_ERROR = "protocol_error"
    NO_PROGRESS = "no_progress"


class FailureSignal(BaseModel):
    kind: FailureKind
    tests: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    excerpt: str = ""
    signature: str = ""
    retryable: bool = False
    suggested_action: str = "replan"


def _text(props: dict, message: str) -> str:
    parts = [message or ""]
    for key in ("fail_reason", "test_status", "test_command"):
        value = props.get(key)
        if value:
            parts.append(str(value))
    for status in props.get("result_statuses") or []:
        parts.append(str(status))
    return "\n".join(parts)


def interpret(props: dict | None, message: str = "") -> FailureSignal | None:
    props = props or {}
    test = props.get("test_run") or {}
    text = _text({**props, **test}, message)
    lower = text.lower()
    tests = list(test.get("targets") or props.get("test_files") or [])
    files = [path for path in (test.get("test_origins") or {})]
    introduced = list(test.get("introduced_failures") or props.get("introduced_failures") or [])
    status = test.get("status") or props.get("test_status")
    return_code = test.get("return_code", props.get("return_code"))
    fail_reason = props.get("fail_reason")
    excerpt = (message or "")[-800:]

    kind = None
    if props.get("timed_out") or test.get("timed_out") or fail_reason == "tool_timeout":
        kind = FailureKind.TEST_TIMEOUT
    elif fail_reason in {"protocol_error", "invalid_tool_call", "tool_call_error"} or "tool call" in lower and "invalid" in lower:
        kind = FailureKind.PROTOCOL_ERROR
    elif introduced:
        kind = FailureKind.P2P_REGRESSION
    elif fail_reason in {"no_test_files", "no_test_target"}:
        kind = FailureKind.NO_TEST_TARGET
    elif any(token in lower for token in (
            "importerror", "syntaxerror", "indentationerror",
            "error during collection", "collection error", "cannot import name")):
        kind = FailureKind.COLLECTION_ERROR
    elif any(token in lower for token in (
            "module not found", "no module named", "command not found", "permission denied",
            "connection refused", "environmenterror")):
        kind = FailureKind.ENVIRONMENT_ERROR
    elif status in {"failed"} or "assertionerror" in lower or "assert " in lower:
        kind = FailureKind.ASSERTION_FAILURE
    elif status in {"error"} or return_code == 2:
        kind = FailureKind.PRODUCT_EXCEPTION
    elif props.get("requires_base_public") and status not in {"failed", "error"}:
        kind = FailureKind.TEST_FILES_MODIFIED
    elif fail_reason in {"no_progress", "stagnation"}:
        kind = FailureKind.NO_PROGRESS
    if kind is None:
        return None

    suggested = {
        FailureKind.COLLECTION_ERROR: "repair_syntax_or_import",
        FailureKind.ASSERTION_FAILURE: "repair_target_test",
        FailureKind.PRODUCT_EXCEPTION: "repair_product_exception",
        FailureKind.TEST_TIMEOUT: "narrow_test_target",
        FailureKind.NO_TEST_TARGET: "discover_test_target",
        FailureKind.TEST_FILES_MODIFIED: "verify_base_public",
        FailureKind.P2P_REGRESSION: "repair_or_revert_regression",
        FailureKind.F2P_STILL_FAILING: "repair_failing_test",
        FailureKind.ENVIRONMENT_ERROR: "stop_environment_blocked",
        FailureKind.PROTOCOL_ERROR: "repair_tool_call",
        FailureKind.NO_PROGRESS: "switch_action_or_replan",
    }[kind]
    return FailureSignal(
        kind=kind, tests=sorted(set(tests)), files=sorted(set(files)), excerpt=excerpt,
        signature=content_hash({"kind": kind.value, "tests": sorted(set(tests)),
                                "excerpt": content_hash(excerpt)}),
        retryable=kind in {FailureKind.TEST_TIMEOUT, FailureKind.PROTOCOL_ERROR,
                           FailureKind.COLLECTION_ERROR, FailureKind.ASSERTION_FAILURE},
        suggested_action=suggested,
    )
