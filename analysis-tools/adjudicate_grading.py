#!/usr/bin/env python3
"""Adjudicate a candidate-induced test-collection failure as unresolved.

The frozen native grader labels an unparseable/collection-error candidate run
as ``environment_error``.  When the patch applies, the baseline completed and
reproduced the original failure, and the compatible report says
``resolved=false``, the actual result is a candidate failure, not an
environment failure.  This tool preserves the original grading, records the
evidence and the explicit user authorization, and writes the single job's
``local_grading=completed`` / ``local_resolved=false``.  It never changes
frozen source, the lock, the manifest, the task set or any other job.
"""

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from repoforge.evaluation import t7
from repoforge.evaluation.experiment import load_manifest, read_json
from repoforge.evaluation.experiment_native import release_job_workspace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(output, path):
    return str(Path(path).resolve().relative_to(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
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
    if grade.get("patch_applicable") is not True:
        raise SystemExit("patch was not applicable; not a candidate collection failure")
    baseline = grade.get("original_baseline") or {}
    if baseline.get("status") != "completed" or baseline.get("original_failure_reproduced") is not True:
        raise SystemExit("baseline did not complete and reproduce the original failure")
    candidate_paths = [path for path in (grade.get("evidence") or {})
                       if "grading-native/candidates/" in path]
    if not candidate_paths:
        raise SystemExit("candidate grading evidence is missing")
    candidate_dir = (output / candidate_paths[0]).parent
    report = read_json(candidate_dir / "compatible-report.json") or {}
    row = report.get(job["instance_id"]) or {}
    case = read_json(candidate_dir / "case.json") or {}
    checks = case.get("checks") or []
    timed_out = any(check.get("timed_out") for check in checks)
    apply_ok = len(checks) >= 2 and all(check.get("return_code") == 0 for check in checks[:2])
    if row.get("resolved") is not False:
        raise SystemExit("compatible report does not show an unresolved patch")
    if row.get("patch_successfully_applied") is not True and not (timed_out and apply_ok):
        raise SystemExit("compatible report does not show an applied but unresolved patch")
    test_output = candidate_dir / "test-output.txt"
    if not test_output.exists():
        raise SystemExit("candidate test output is missing")
    adjudication_path = directory / "grading-adjudication.json"
    if adjudication_path.exists():
        raise SystemExit(f"adjudication already exists: {adjudication_path}")

    attempts = directory / "grading-attempts"
    attempts.mkdir(exist_ok=True)
    preserved = attempts / f"{uuid.uuid4().hex}.json"
    shutil.copy2(directory / "grading.json", preserved)
    evidence = {relative(output, candidate_dir / name): sha(candidate_dir / name)
                for name in ("test-output.txt", "parsed-tests.json",
                             "compatible-report.json", "case.json")
                if (candidate_dir / name).exists()}
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job["job_id"], "instance_id": job["instance_id"],
        "variant": job["variant"], "phase": job["phase"],
        "answer_hash": grade["answer_hash"],
        "patch_sha256": sha(directory / "patch.diff"),
        "original_grading": relative(output, preserved),
        "original_grading_sha256": sha(preserved),
        "original_local_grading": grade.get("local_grading"),
        "original_local_resolved": grade.get("local_resolved"),
        "conclusion": {"local_grading": "completed", "local_resolved": False},
        "reason": args.reason,
        "evidence": evidence,
        "user_authorization": args.authorization,
        "note": "Manual adjudication outside the frozen grader; the original grading.json is preserved under grading-attempts/. The candidate patch applies, the baseline completed and reproduced the original failure, and the compatible report is resolved=false. official_harness remains not_run.",
        "official_harness": "not_run", "official_resolved": None,
    }
    adjudication_path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    updated = dict(grade)
    updated["local_grading"] = "completed"
    updated["local_resolved"] = False
    updated["adjudication"] = {"manual": True, "path": relative(output, adjudication_path),
                               "original_local_grading": "environment_error"}
    updated_evidence = dict(updated.get("evidence") or {})
    updated_evidence[relative(output, adjudication_path)] = sha(adjudication_path)
    updated_evidence[relative(output, preserved)] = sha(preserved)
    updated["evidence"] = updated_evidence
    (directory / "grading.json").write_text(json.dumps(updated, ensure_ascii=False, indent=1))
    row = t7.accept_job(output, manifest, job, require_locked=True)
    if args.release:
        release_job_workspace(output, manifest, job)
    print(json.dumps({"job_id": job["job_id"], "instance_id": job["instance_id"],
                      "variant": job["variant"], "local_resolved": row["local_resolved"],
                      "adjudication": str(adjudication_path), "released": args.release},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
