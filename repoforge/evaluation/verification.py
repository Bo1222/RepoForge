"""Patch-bound verification state for the coding agent.

A test is evidence only when it was run on the current product patch.  The
snapshot distinguishes public verification from lagging or modified tests; it
never claims hidden-test correctness.
"""

from enum import Enum

from pydantic import BaseModel, Field


class VerificationState(str, Enum):
    NO_PATCH = "NO_PATCH"
    UNVERIFIED = "UNVERIFIED"
    TARGET_FOUND = "TARGET_FOUND"
    TESTED_FAIL = "TESTED_FAIL"
    TESTED_PASS = "TESTED_PASS"
    PUBLIC_VERIFIED = "PUBLIC_VERIFIED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


class VerificationSnapshot(BaseModel):
    state: VerificationState
    patch_hash: str | None = None
    tests: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    introduced_failures: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def snapshot(product_patch_hash: str | None, target: str | None, records: list[dict]) -> VerificationSnapshot:
    from repoforge.verification_inputs import receipt_complete

    if not (product_patch_hash or "").strip():
        return VerificationSnapshot(state=VerificationState.NO_PATCH, patch_hash=product_patch_hash,
                                    reasons=["no product patch"])
    current = [record for record in records
               if not record.get("superseded")
               and record.get("product_patch_hash") == product_patch_hash
               and record.get("product_source", "current") == "current"
               and receipt_complete(record)]
    if records and not current:
        return VerificationSnapshot(state=VerificationState.STALE, patch_hash=product_patch_hash,
                                    reasons=["no test bound to the current patch"])
    tests = sorted({test for record in current for test in (record.get("targets") or [])})
    introduced = sorted({test for record in current for test in (record.get("introduced_failures") or [])})
    failed = [record for record in current if record.get("effective_status") in {"failed", "error"} and not record.get("timed_out")]
    if failed:
        return VerificationSnapshot(state=VerificationState.TESTED_FAIL, patch_hash=product_patch_hash,
                                    tests=tests, introduced_failures=introduced,
                                    reasons=["test failed on the current patch"])
    passed = [record for record in current if record.get("effective_status") == "passed" and record.get("complete") is True
              and record.get("return_code") == 0 and record.get("counts", {}).get("passed", 0) > 0]
    if passed and not introduced:
        # base_public runs the unchanged public tests.  A modified test file in
        # the workspace does not downgrade that evidence; the runtime patch for
        # base_public excludes test changes.
        if any(record.get("test_source") == "base_public" for record in passed):
            return VerificationSnapshot(state=VerificationState.PUBLIC_VERIFIED,
                                        patch_hash=product_patch_hash, tests=tests,
                                        confidence=0.0,
                                        reasons=["public tests passed on the current patch"])
        modified = any(origin in {"modified", "new"}
                       for record in passed for origin in (record.get("test_origins") or {}).values())
        if modified:
            return VerificationSnapshot(state=VerificationState.TESTED_PASS, patch_hash=product_patch_hash,
                                        tests=tests, confidence=0.0,
                                        reasons=["tests were modified or added"])
        if introduced:
            return VerificationSnapshot(state=VerificationState.TESTED_PASS, patch_hash=product_patch_hash,
                                        tests=tests, confidence=0.0,
                                        introduced_failures=introduced,
                                        reasons=["patch introduced test failures"])
        return VerificationSnapshot(state=VerificationState.PUBLIC_VERIFIED, patch_hash=product_patch_hash,
                                    tests=tests, confidence=0.0,
                                    reasons=["public tests passed on the current patch"])
    if introduced:
        return VerificationSnapshot(state=VerificationState.TESTED_FAIL, patch_hash=product_patch_hash,
                                    tests=tests, introduced_failures=introduced, reasons=["introduced failures"])
    if current and any(r.get("timed_out") or r.get("effective_status") == "error" for r in current):
        return VerificationSnapshot(state=VerificationState.UNAVAILABLE, patch_hash=product_patch_hash,
                                    tests=tests, reasons=["verification incomplete or unavailable"])
    if target:
        return VerificationSnapshot(state=VerificationState.TARGET_FOUND, patch_hash=product_patch_hash,
                                    tests=[target], confidence=0.0,
                                    reasons=["target selected but not run on the current patch"])
    return VerificationSnapshot(state=VerificationState.UNVERIFIED, patch_hash=product_patch_hash,
                                reasons=["no verification target"])
