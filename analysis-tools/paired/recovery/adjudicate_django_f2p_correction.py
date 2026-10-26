"""Grade django-10097 on the URL tests actually changed by its test patch.

The official dataset row lists 438 FAIL_TO_PASS tests for this instance, but
431 of them pass on the base commit, and the six URL-validator tests that the
test patch's invalid_urls.txt additions actually exercise are absent.  The
seven observed failures are TemplateDoesNotExist errors in unrelated
auth_tests (missing installed templates), i.e. environment noise.

This tool derives the corrected F2P/P2P from the same-environment baseline:
- F2P = canonical ``validators.tests.TestSimpleValidators`` URL tests that
  fail on the base commit (the six new invalid URLs);
- P2P = the remaining canonical URL tests that pass on the baseline.

It verifies the baseline and candidate reports agree, then writes corrected
``grading.json`` / ``original-baseline.json`` while preserving every raw file.
No answer, patch, question or frozen source is changed.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    args.record_dir = args.record_dir or output / "recovery-records"
    jobs = {job["job_id"]: job for job in read(output / "jobs.json")}
    if args.job not in jobs:
        raise SystemExit("Unknown job")
    job = jobs[args.job]
    if job["instance_id"] != "django__django-10097":
        raise SystemExit("This correction only applies to django__django-10097")
    directory = output / "runs" / args.job
    grading_path, baseline_path = directory / "grading.json", directory / "original-baseline.json"
    if not grading_path.is_file() or not baseline_path.is_file():
        raise SystemExit("grading.json/original-baseline.json missing")
    raw_grading, raw_baseline = read(grading_path), read(baseline_path)
    if (raw_grading.get("adjudication") or {}).get("kind") == "django-test-patch-f2p":
        print(json.dumps({"job_id": args.job, "status": "already_adjudicated"}))
        return

    case_keys = [key for key in raw_grading.get("evidence", {})
                 if "grading-native/candidates/" in key and key.endswith("/case.json")]
    if len(case_keys) != 1:
        raise SystemExit(f"Expected one candidate case.json, found {case_keys}")
    case_path = output / case_keys[0]
    case = read(case_path)
    baseline_statuses = raw_baseline.get("test_statuses") or {}
    candidate_statuses = case.get("test_statuses") or {}
    # Canonical unittest-style names are the ones the compatible report uses.
    canonical = sorted(key for key in baseline_statuses
                       if "test_URLValidator_" in key and key.endswith(")"))
    f2p = [key for key in canonical if baseline_statuses[key] != "PASSED"]
    p2p = [key for key in canonical if baseline_statuses[key] == "PASSED"]
    if len(f2p) != 6:
        raise SystemExit(f"Expected the six new URL failures on the baseline, got {f2p}")
    missing = [key for key in f2p + p2p if key not in candidate_statuses]
    if missing:
        raise SystemExit(f"Candidate report lacks URL tests: {missing[:5]}")

    f2p_success = [key for key in f2p if candidate_statuses[key] == "PASSED"]
    f2p_failure = [key for key in f2p if candidate_statuses[key] != "PASSED"]
    p2p_success = [key for key in p2p if candidate_statuses[key] == "PASSED"]
    p2p_failure = [key for key in p2p if candidate_statuses[key] != "PASSED"]
    resolved = not f2p_failure and not p2p_failure
    official_f2p = list(raw_baseline.get("FAIL_TO_PASS") or [])
    official_observed = list(raw_baseline.get("observed_ftp_failures") or [])

    timestamp = now().strftime("%Y%m%dT%H%M%SZ")
    record = {
        "kind": "django-test-patch-f2p",
        "instance_id": job["instance_id"],
        "job_id": args.job,
        "variant": job["variant"],
        "timestamp": now().isoformat(),
        "official_fail_to_pass_count": len(official_f2p),
        "official_observed_failures": official_observed,
        "official_f2p_passed_on_base": len(official_f2p) - len(official_observed),
        "corrected_fail_to_pass": f2p,
        "corrected_pass_to_pass": p2p,
        "candidate_f2p_success": f2p_success,
        "candidate_f2p_failure": f2p_failure,
        "candidate_p2p_failure": p2p_failure,
        "candidate_local_resolved": resolved,
        "baseline_case": str(case_path.parent.relative_to(output)),
        "candidate_case": case_keys[0],
        "reason": ("Official F2P lists 438 tests, 431 pass on the base commit, and the six "
                   "URL-validator tests changed by the test patch's invalid_urls.txt are absent. "
                   "The corrected F2P are exactly those six baseline failures; P2P are the other "
                   "URL-validator tests passing on the baseline. The seven official observed "
                   "failures are TemplateDoesNotExist environment errors and are excluded. "
                   "No answer or patch is changed."),
        "apply": args.apply,
    }

    if args.apply:
        raw_dir = directory / f"adjudication-{timestamp}"
        raw_dir.mkdir(exist_ok=True)
        shutil.copy2(grading_path, raw_dir / "grading.raw.json")
        shutil.copy2(baseline_path, raw_dir / "original-baseline.raw.json")
        shutil.copy2(case_path, raw_dir / "candidate-case.raw.json")
        record["raw_sha256"] = {
            "grading.json": sha(raw_dir / "grading.raw.json"),
            "original-baseline.json": sha(raw_dir / "original-baseline.raw.json"),
            "candidate-case.json": sha(raw_dir / "candidate-case.raw.json"),
        }
        operation = directory / "operation-error.json"
        if operation.is_file():
            operation_target = raw_dir / "operation-error.raw.json"
            shutil.move(operation, operation_target)
            record["archived_operation_error"] = {
                "from": str(operation.relative_to(output)),
                "to": str(operation_target.relative_to(output)),
                "sha256": sha(operation_target),
            }

        baseline_tests_status = {
            "FAIL_TO_PASS": {"success": [], "failure": f2p},
            "PASS_TO_PASS": {"success": p2p, "failure": []},
            "FAIL_TO_FAIL": {"success": [], "failure": []},
            "PASS_TO_FAIL": {"success": [], "failure": []},
        }
        corrected_baseline = {
            "answer_hash": raw_grading["answer_hash"],
            "status": "completed",
            "patch_applicable": True,
            "local_resolved": None,
            "tests_status": baseline_tests_status,
            "test_statuses": {key: baseline_statuses[key] for key in f2p + p2p},
            "original_failure_reproduced": True,
            "observed_ftp_failures": f2p,
            "FAIL_TO_PASS": f2p,
            "PASS_TO_PASS": p2p,
            "evidence": raw_baseline.get("evidence", {}),
            "adjudication": record,
        }
        corrected_tests_status = {
            "FAIL_TO_PASS": {"success": f2p_success, "failure": f2p_failure},
            "PASS_TO_PASS": {"success": p2p_success, "failure": p2p_failure},
            "FAIL_TO_FAIL": {"success": [], "failure": []},
            "PASS_TO_FAIL": {"success": [], "failure": []},
        }
        corrected_grading = {
            "answer_hash": raw_grading["answer_hash"],
            "local_grading": "completed",
            "local_resolved": resolved,
            "grader_source_sha256": raw_grading.get("grader_source_sha256"),
            "official_resolved": None,
            "official_harness": "not_run",
            "grading_backend": "native",
            "evaluation_protocol": raw_grading.get("evaluation_protocol"),
            "patch_applicable": raw_grading.get("patch_applicable"),
            "original_baseline": corrected_baseline,
            "tests_status": corrected_tests_status,
            "evidence": {**raw_grading.get("evidence", {}),
                         **raw_baseline.get("evidence", {})},
            "adjudication": record,
        }
        write(baseline_path, corrected_baseline)
        write(grading_path, corrected_grading)
        args.record_dir.mkdir(parents=True, exist_ok=True)
        record_path = args.record_dir / f"{timestamp}-{args.job}.json"
        write(record_path, record)
        record["record_path"] = str(record_path)

    print(json.dumps(record, indent=2, ensure_ascii=False))
    if args.apply:
        from repoforge.evaluation import paired
        manifest = read(output / "manifest.json")
        row = paired.job_result(output, manifest, job)
        print(json.dumps({"status": row["status"], "resolved": row["resolved"],
                          "original_failure_reproduced": row.get("original_failure_reproduced")},
                         indent=2))


if __name__ == "__main__":
    main()
