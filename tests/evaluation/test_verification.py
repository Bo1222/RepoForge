import pytest

from repoforge.evaluation.verification import VerificationState, snapshot


def record(**changes):
    data = {
        "superseded": False,
        "recipe_version": 3,
        "input_identity": "recipe", "input_identity_after": "recipe",
        "environment_before": "env", "environment_after": "env",
        "complete": True, "return_code": 0, "counts": {"passed": 1},
        "tested_patch_hash": "patch",
        "product_patch_hash": "patch",
        "targets": ["tests/test_code.py"],
        "test_source": "workspace",
        "test_origins": {"tests/test_code.py": "original"},
        "effective_status": "passed",
        "introduced_failures": [],
    }
    data.update(changes)
    return data


def state(*records, target=None, patch="patch"):
    return snapshot(patch, target, list(records)).state


def test_no_patch_is_not_verified():
    assert state(patch="") == VerificationState.NO_PATCH


def test_original_workspace_or_base_public_pass_is_public_verified():
    assert state(record()) == VerificationState.PUBLIC_VERIFIED
    assert state(record(test_source="base_public")) == VerificationState.PUBLIC_VERIFIED


def test_modified_tests_are_only_tested_pass():
    assert state(record(test_origins={"tests/test_code.py": "modified"})) == VerificationState.TESTED_PASS
    assert state(record(test_origins={"tests/test_code.py": "new"})) == VerificationState.TESTED_PASS


def test_failed_and_stale_are_not_verified():
    assert state(record(effective_status="failed")) == VerificationState.TESTED_FAIL
    assert state(record(effective_status="error")) == VerificationState.TESTED_FAIL
    assert state(record(product_patch_hash="old")) == VerificationState.STALE


def test_introduced_failures_prevent_public_verified():
    assert state(record(introduced_failures=["tests/test_code.py::test_regression"])) == VerificationState.TESTED_FAIL


def test_target_without_run_is_target_found():
    assert state(target="tests/test_code.py") == VerificationState.TARGET_FOUND


def test_base_public_pass_survives_workspace_test_edits():
    assert state(record(test_source="base_public",
                        test_origins={"tests/test_code.py": "modified"})) == VerificationState.PUBLIC_VERIFIED
