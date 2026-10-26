"""Adjudicate a baseline collection error that is the expected F2P failure.

Some SWE-bench test patches import a symbol that the gold patch adds.  On the
base commit the whole test module then fails at collection time, so the frozen
NativeGrader sees exit code 2, records ``status=environment_error`` and cannot
derive per-test statuses from the log.  The official SWE-bench compatible
report already maps that collection ERROR to every expected FAIL_TO_PASS test,
so the original failure *is* reproduced; only the native baseline classifier
disagrees.

This tool rewrites only the cached baseline case's status/test_statuses after
checking the compatible report lists exactly the expected F2P tests as failed.
It also archives stale grading outputs and the env_blocked preparation record,
so the phase re-runs independent grading while reusing the frozen answer.
Frozen source, questions, model configuration and answers are not changed.
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


def find_baseline(output, instance):
    jobs = read(output / "jobs.json")
    related = [job for job in jobs if job["instance_id"] == instance]
    for job in related:
        path = output / "runs" / job["job_id"] / "original-baseline.json"
        if not path.is_file():
            continue
        baseline = read(path)
        cases = [key for key in baseline.get("evidence", {})
                 if "grading-native/baselines/" in key and key.endswith("/case.json")]
        reports = [key for key in baseline.get("evidence", {})
                   if "grading-native/baselines/" in key and key.endswith("/compatible-report.json")]
        if len(cases) == 1 and len(reports) == 1:
            return job, related, baseline, output / cases[0], output / reports[0]
    raise SystemExit("No baseline evidence with exactly one case.json and compatible-report.json")


def already_adjudicated(output, instance):
    paths = []
    for path in (output / "grading-native" / "baselines").glob("*/case.json"):
        data = read(path)
        if not (data.get("adjudication") and data.get("status") == "completed"):
            continue
        report = path.parent / "compatible-report.json"
        if report.is_file() and instance in read(report):
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    args.record_dir = args.record_dir or output / "recovery-records"
    instance = args.instance
    try:
        job, related, baseline, case_path, report_path = find_baseline(output, instance)
    except SystemExit:
        done = already_adjudicated(output, instance)
        if done:
            print(json.dumps({"instance_id": instance, "status": "already_adjudicated",
                              "cases": [str(path.relative_to(output)) for path in done]}))
            return
        raise
    case = read(case_path)
    report = read(report_path)
    expected = list(baseline.get("FAIL_TO_PASS") or [])
    failures = list((report.get(instance, {}).get("tests_status") or {})
                    .get("FAIL_TO_PASS", {}).get("failure", []))
    check = (case.get("checks") or [{}])[-1]

    if case.get("status") != "environment_error":
        raise SystemExit(f"Baseline status is not environment_error: {case.get('status')}")
    if check.get("return_code") != 2 or check.get("timed_out"):
        raise SystemExit(f"Baseline is not a non-timeout collection error: {check}")
    if not expected or set(failures) != set(expected):
        raise SystemExit(f"Compatible report does not fail every F2P: {failures} vs {expected}")
    if any(name in (case.get("test_statuses") or {}) for name in expected):
        raise SystemExit("Baseline already has per-test F2P statuses; nothing to adjudicate")

    timestamp = now().strftime("%Y%m%dT%H%M%SZ")
    record = {
        "instance_id": instance,
        "job_id": job["job_id"],
        "timestamp": now().isoformat(),
        "baseline_case": str(case_path.relative_to(output)),
        "compatible_report": str(report_path.relative_to(output)),
        "expected_fail_to_pass": expected,
        "compatible_report_failures": failures,
        "original_status": case.get("status"),
        "original_test_statuses": case.get("test_statuses"),
        "last_check": check,
        "decision": {"status": "completed", "original_failure_reproduced": True,
                     "observed_ftp_failures": expected},
        "reason": ("The test patch imports a symbol added by the gold patch, so the base commit "
                   "fails at collection time. The compatible report lists every expected "
                   "FAIL_TO_PASS test as failed; the native classifier only rejected the exit "
                   "code 2. Per-test ERROR markers are transcribed from that report. No answer "
                   "resampling and no source change."),
        "apply": args.apply,
    }

    if args.apply:
        raw_dir = case_path.parent / f"adjudication-{timestamp}"
        raw_dir.mkdir(exist_ok=True)
        shutil.copy2(case_path, raw_dir / "case.raw.json")
        shutil.copy2(report_path, raw_dir / "compatible-report.raw.json")
        record["raw_case_sha256"] = sha(raw_dir / "case.raw.json")
        record["raw_compatible_report_sha256"] = sha(raw_dir / "compatible-report.raw.json")

        # Stale grading outputs must be recomputed from the adjudicated baseline.
        for related_job in related:
            directory = output / "runs" / related_job["job_id"]
            if not directory.is_dir():
                continue
            stale = [directory / name for name in ("grading.json", "original-baseline.json")
                     if (directory / name).is_file()]
            if not stale:
                continue
            job_raw = directory / f"adjudication-{timestamp}"
            job_raw.mkdir(exist_ok=True)
            for path in stale:
                target = job_raw / f"{path.stem}.raw.json"
                shutil.move(path, target)
                record.setdefault("archived_grading", []).append({
                    "from": str(path.relative_to(output)),
                    "to": str(target.relative_to(output)),
                    "sha256": sha(target),
                })

        # Let provision_one use the existing receipt instead of the env_blocked
        # marker that was written when baseline grading failed.
        preparation = output / "preparation" / f"{instance}.json"
        if preparation.is_file():
            archive = output / "preparation-attempts" / f"grading-collection-adjudication-{timestamp}"
            archive.mkdir(parents=True, exist_ok=True)
            target = archive / preparation.name
            shutil.move(preparation, target)
            record["archived_preparation"] = {
                "from": str(preparation.relative_to(output)),
                "to": str(target.relative_to(output)),
                "sha256": sha(target),
            }

        case["test_statuses"] = {
            **(case.get("test_statuses") or {}),
            **{name: "ERROR" for name in expected},
        }
        case["status"] = "completed"
        case["adjudication"] = {
            "reason": record["reason"],
            "timestamp": record["timestamp"],
            "original_status": "environment_error",
            "compatible_report": record["compatible_report"],
            "expected_fail_to_pass": expected,
            "raw_case": str((raw_dir / "case.raw.json").relative_to(output)),
        }
        write(case_path, case)
        record["adjudicated_case_sha256"] = sha(case_path)
        write(raw_dir / "adjudication.json", record)

        args.record_dir.mkdir(parents=True, exist_ok=True)
        record_path = args.record_dir / f"{timestamp}-{instance}.json"
        write(record_path, record)
        record["record_path"] = str(record_path)

    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
