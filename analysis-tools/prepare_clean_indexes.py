#!/usr/bin/env python3
"""Prepare shared holdout indexes only for scan-clean instances.

The user-directed rule is that non-clean instances (parse_suspect,
build_wheel_gap, environment_error, provision_failed) are not indexed and are
not executed.  This script is external to the frozen Agent source; it writes
holdout-index-coverage.json with clean_complete=true only when every clean
instance has a verified shared index, and records all non-clean instances in
the ``failed`` list so the controller skips them.
"""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from moatless.evaluation import t7
from moatless.evaluation.experiment import (
    append_json, load_manifest, now, read_json, write_json,
)
from moatless.evaluation.experiment_native import (
    run_job, shared_index, verify_index_receipt,
)

COVERAGE = "holdout-index-coverage.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scan", type=Path, default=None)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    load_dotenv(REPO_ROOT / ".env")
    manifest = load_manifest(candidate)
    lock = t7.validate_lock(candidate, manifest)
    scan = read_json(args.scan or candidate / "holdout-scan.json")
    clean = [row["instance_id"] for row in scan["rows"]
             if row["classification"] == "clean"]
    non_clean = [row["instance_id"] for row in scan["rows"]
                 if row["classification"] != "clean"]
    if len(clean) + len(non_clean) != 100:
        raise SystemExit("holdout-scan.json does not cover 100 instances")
    receipts = read_json(candidate / "provision-native.json")
    jobs = read_json(candidate / "jobs.json")
    order = {identity: index for index, identity in enumerate(manifest["split"]["holdout"])}
    lock_arms = [lock["winner"], lock["control"]]
    rows, failures = [], []
    for identity in sorted(clean, key=lambda value: order[value]):
        prepared = receipts.get(identity)
        if not prepared:
            failures.append(identity)
            print({"instance_id": identity, "failed": "missing_receipt"}, flush=True)
            continue
        job = next((item for item in jobs if item["phase"] == "holdout"
                    and item["instance_id"] == identity
                    and item["variant"] == lock["winner"]), None)
        if job is None:
            failures.append(identity)
            print({"instance_id": identity, "failed": "missing_winner_job"}, flush=True)
            continue
        try:
            store, profile, index = shared_index(candidate, manifest, prepared)
            if index is None:
                run_job(candidate, manifest, job, prepare_index_only=True)
                store, profile, index = shared_index(candidate, manifest, prepared)
            if index is None:
                raise ValueError("Index preparation published no shared index")
            receipt = verify_index_receipt(
                index, manifest, prepared["public"],
                setup_patch_sha256=prepared["environment"].get("setup_patch_sha256"))
            rows.append({"instance_id": identity, "status": "ready",
                         "index": str(index), "digest": receipt["digest"]})
            print({"instance_id": identity, "status": "ready"}, flush=True)
        except Exception as exc:
            failures.append(identity)
            print({"instance_id": identity, "failed": type(exc).__name__,
                   "message": str(exc)[:300]}, flush=True)
        write_json(candidate / COVERAGE, {
            "manifest_hash": manifest["manifest_hash"],
            "lock_hash": lock["lock_hash"],
            "rule": "only scan-clean instances are indexed and executed; all other classifications are skipped",
            "clean": clean,
            "non_clean": non_clean,
            "selected": len(rows) + len(failures),
            "ready": len(rows),
            "failed": sorted(set(non_clean) | set(failures)),
            "failed_clean": sorted(failures),
            "clean_complete": False,
            "rows": rows,
            "timestamp": now(),
        })
    coverage = {
        "manifest_hash": manifest["manifest_hash"],
        "lock_hash": lock["lock_hash"],
        "rule": "only scan-clean instances are indexed and executed; all other classifications are skipped",
        "clean": clean,
        "non_clean": non_clean,
        "selected": len(clean),
        "ready": len(rows),
        "failed": sorted(set(non_clean) | set(failures)),
        "failed_clean": sorted(failures),
        "clean_complete": len(rows) == len(clean),
        "rows": rows,
        "timestamp": now(),
    }
    write_json(candidate / COVERAGE, coverage)
    print(json.dumps({"clean": len(clean), "ready": len(rows),
                      "failed_clean": failures, "clean_complete": coverage["clean_complete"]},
                     ensure_ascii=False, indent=2))
    if coverage["clean_complete"]:
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
