"""Adjudicate a native-grading baseline parser edge case without touching frozen source.

The swebench log parser can merge two unittest output lines into one
``test_statuses`` key when a test name is immediately followed by the next test
name on the same line.  ``NativeGrader.baseline`` then computes
``original_failure_reproduced=False`` even though the compatible report already
lists every FAIL_TO_PASS test as failed.  That makes ``paired.job_result`` mark
the job ``grading_error`` and stops the phase.

This tool only rewrites run evidence after checking both reports agree:
- baseline report: every expected FAIL_TO_PASS test failed;
- candidate case: completed with a boolean local_resolved;
- candidate report: the same FAIL_TO_PASS failures (normally unresolved).

Raw files and the original operation error are preserved.  Frozen source is
not modified, so ``validate_execution_source`` still passes.
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


def f2p_failures(tests_status):
    return list((tests_status or {}).get("FAIL_TO_PASS", {}).get("failure", []))


def f2p_expected(baseline):
    return list(baseline.get("FAIL_TO_PASS", []))


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
    directory = output / "runs" / args.job
    grading_path = directory / "grading.json"
    baseline_path = directory / "original-baseline.json"
    if not grading_path.is_file() or not baseline_path.is_file():
        raise SystemExit("grading.json/original-baseline.json missing")
    grading = read(grading_path)
    baseline = read(baseline_path)
    expected = f2p_expected(baseline)
    baseline_failures = f2p_failures(baseline.get("tests_status"))
    if baseline.get("original_failure_reproduced") is not False:
        raise SystemExit("Baseline already has a non-false reproduction flag; nothing to adjudicate")
    if not expected or set(baseline_failures) != set(expected):
        raise SystemExit(f"Baseline report does not reproduce all FAIL_TO_PASS: {baseline_failures} vs {expected}")
    case_keys = [key for key in grading.get("evidence", {})
                 if "grading-native/candidates/" in key and key.endswith("/case.json")]
    if len(case_keys) != 1:
        raise SystemExit(f"Expected one candidate case.json, found {case_keys}")
    case_path = output / case_keys[0]
    case = read(case_path)
    if (case.get("status") != "completed" or case.get("patch_applicable") is not True
            or not isinstance(case.get("local_resolved"), bool)):
        raise SystemExit(f"Candidate case is not a completed applicable run with a boolean result: "
                         f"{case.get('status')} applicable={case.get('patch_applicable')}")
    candidate_resolved = case["local_resolved"]
    tests_status = case.get("tests_status") or {}
    candidate_success = list((tests_status.get("FAIL_TO_PASS") or {}).get("success", []))
    candidate_failures = f2p_failures(tests_status)
    p2p_failures = list((tests_status.get("PASS_TO_PASS") or {}).get("failure", []))
    if candidate_resolved:
        if candidate_failures or p2p_failures or set(candidate_success) != set(expected):
            raise SystemExit("Resolved candidate contradicts its report: "
                             f"F2P success={candidate_success}, F2P failure={candidate_failures}, "
                             f"P2P failure={p2p_failures}")
    elif not candidate_failures and not p2p_failures:
        raise SystemExit("Unresolved candidate has no failing test in its compatible report")
    timestamp = now().strftime("%Y%m%dT%H%M%SZ")
    record = {
        "job_id": args.job, "instance_id": job["instance_id"], "variant": job["variant"],
        "phase": job["phase"], "timestamp": now().isoformat(),
        "answer_hash": grading.get("answer_hash"),
        "expected_fail_to_pass": expected,
        "baseline_report_failures": baseline_failures,
        "candidate_report_failures": candidate_failures,
        "candidate_f2p_success": candidate_success,
        "candidate_p2p_failures": p2p_failures,
        "candidate_case": str(case_path.relative_to(output)),
        "candidate_local_resolved": candidate_resolved,
        "decision": {"original_failure_reproduced": True,
                     "local_resolved": candidate_resolved},
        "reason": ("test_statuses keys merged two unittest output lines; the baseline compatible "
                   "report lists every FAIL_TO_PASS test as failed, so the original failure was "
                   "reproduced. The candidate outcome is taken from its completed compatible report "
                   f"(local_resolved={candidate_resolved}); no answer resampling."),
        "raw_sha256": {"grading.json": sha(grading_path), "original-baseline.json": sha(baseline_path),
                       "baseline_case.json": sha(case_path)},
        "apply": args.apply,
    }
    raw_dir = directory / f"adjudication-{timestamp}"
    if args.apply:
        raw_dir.mkdir(exist_ok=True)
        shutil.copy2(grading_path, raw_dir / "grading.raw.json")
        shutil.copy2(baseline_path, raw_dir / "original-baseline.raw.json")
        operation = directory / "operation-error.json"
        if operation.exists():
            shutil.move(operation, raw_dir / "operation-error.raw.json")
        correction = {"reason": record["reason"], "timestamp": record["timestamp"],
                      "raw_directory": str(raw_dir.relative_to(output)),
                      "baseline_report_failures": baseline_failures}
        baseline["original_failure_reproduced"] = True
        baseline["observed_ftp_failures"] = baseline_failures
        baseline["adjudication"] = correction
        grading["original_baseline"] = baseline
        grading["local_resolved"] = candidate_resolved
        grading["adjudication"] = correction
        write(baseline_path, baseline)
        write(grading_path, grading)
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
                          "cost": row["cost"], "original_failure_reproduced":
                          row.get("original_failure_reproduced")}, indent=2))


if __name__ == "__main__":
    main()
