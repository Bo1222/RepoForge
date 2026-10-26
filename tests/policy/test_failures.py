import pytest

from repoforge.policy.failures import FailureKind, interpret


def props(**changes):
    data = {"test_run": {"status": "failed", "targets": ["tests/test_code.py"], "return_code": 1}}
    data["test_run"].update(changes)
    return data


@pytest.mark.parametrize("data,message,expected", [
    ({"timed_out": True}, "", FailureKind.TEST_TIMEOUT),
    (props(test_origins={"tests/test_code.py": "modified"}), "", FailureKind.ASSERTION_FAILURE),
    (props(introduced_failures=["tests/test_code.py::test_reg"]), "", FailureKind.P2P_REGRESSION),
    (props(status="error", return_code=2), "ImportError: cannot import name x", FailureKind.COLLECTION_ERROR),
    (props(), "AssertionError: expected 1", FailureKind.ASSERTION_FAILURE),
    (props(status="error"), "ModuleNotFoundError: No module named x", FailureKind.ENVIRONMENT_ERROR),
    ({"fail_reason": "no_test_files", "test_run": {"status": "not_run"}}, "", FailureKind.NO_TEST_TARGET),
    ({"fail_reason": "protocol_error"}, "invalid tool call", FailureKind.PROTOCOL_ERROR),
])
def test_failure_taxonomy(data, message, expected):
    signal = interpret(data, message)
    assert signal is not None
    assert signal.kind == expected
    assert signal.signature


def test_non_failure_returns_none():
    assert interpret({"test_run": {"status": "passed"}}, "all good") is None
