#!/usr/bin/env python3
"""Re-grade one holdout job with a longer independent-test timeout.

The frozen grader caps the independent candidate test command at 600 seconds.
A candidate patch can make the official suite exceed that cap even though the
suite would finish and produce a real result.  This tool preserves the original
timed-out case/grading, moves the candidate case aside, extends only the
executor timeout, and re-runs the frozen grader.  It never changes the tests,
the parser or the frozen source.
"""

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from moatless.evaluation import t7
from moatless.evaluation.experiment import load_manifest, read_json, write_json
from moatless.evaluation.experiment_native import release_job_workspace
from moatless.evaluation.native_execution import NativeExecutor
from moatless.evaluation.native_grading import NativeGrader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(output, path):
    return str(Path(path).resolve().relative_to(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    output = args.candidate.resolve()
    manifest = load_manifest(output)
    job = next((item for item in read_json(output / "jobs.json")
                if item["job_id"] == args.job_id), None)
    if job is None:
        raise SystemExit(f"unknown job: {args.job_id}")
    directory = output / "runs" / job["job_id"]
    grade = read_json(directory / "grading.json")
    if grade.get("local_grading") != "environment_error":
        raise SystemExit("job is not an environment_error grading")
    case_paths = [path for path in (grade.get("evidence") or {})
                  if "grading-native/candidates/" in path]
    if not case_paths:
        raise SystemExit("candidate case evidence is missing")
    old_case = (output / case_paths[0]).parent
    old_case_json = read_json(old_case / "case.json") or {}
    checks = old_case_json.get("checks") or []
    if not checks or checks[-1].get("timed_out") is not True:
        raise SystemExit("original candidate case did not time out; this tool is not applicable")
    preserved = directory / "grading-attempts"
    preserved.mkdir(exist_ok=True)
    old_grading = preserved / f"{uuid.uuid4().hex}.json"
    shutil.copy2(directory / "grading.json", old_grading)
    archived_case = output / "grading-native" / "candidates-timeout" / (
        old_case.name + f"-{int(time.time())}")
    archived_case.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_case), str(archived_case))
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job["job_id"], "instance_id": job["instance_id"],
        "variant": job["variant"], "phase": job["phase"],
        "original_grading": relative(output, old_grading),
        "original_grading_sha256": sha(old_grading),
        "original_case": relative(output, archived_case),
        "original_case_sha256": {p.name: sha(p) for p in archived_case.iterdir() if p.is_file()},
        "extended_timeout_seconds": args.timeout,
        "reason": args.reason,
        "user_authorization": args.authorization,
        "note": "Re-grade of a timed-out candidate case with a longer independent-test timeout; original case and grading are preserved. No tests, parser or frozen source changed.",
    }
    original_run = NativeExecutor.run

    async def run_with_extended_timeout(self, argv, *, cwd="/testbed", timeout=120, redact=()):
        if timeout == 600:
            timeout = max(timeout, args.timeout)
        return await original_run(self, argv, cwd=cwd, timeout=timeout, redact=redact)

    NativeExecutor.run = run_with_extended_timeout
    try:
        result = NativeGrader(output, manifest).grade(job)
    finally:
        NativeExecutor.run = original_run
    record["result"] = {key: result.get(key) for key in
                        ("local_grading", "local_resolved", "patch_applicable", "tests_status")}
    record["new_grading_sha256"] = sha(directory / "grading.json")
    write_json(directory / "grading-timeout-rerun.json", record)
    print(json.dumps({"job_id": job["job_id"], "instance_id": job["instance_id"],
                      "variant": job["variant"], "local_grading": result.get("local_grading"),
                      "local_resolved": result.get("local_resolved"),
                      "extended_timeout": args.timeout,
                      "original_case": str(archived_case)}, ensure_ascii=False))
    if result.get("local_grading") in {"completed", "patch_error"}:
        row = t7.accept_job(output, manifest, job, require_locked=True)
        if args.release:
            release_job_workspace(output, manifest, job)
        print(json.dumps({"accepted": True, "local_resolved": row.get("local_resolved")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
