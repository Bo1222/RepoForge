#!/usr/bin/env python3
"""Run a Wave2 dev30 sanity arm (A3) against the existing T7 dev environments.

This is an engineering sanity check, not the final new-split holdout.  It
creates a new candidate with the Wave2 source, re-signs the existing dev
provision receipts, runs the 30 dev A3 jobs, grades them independently, and
writes a report.  T7 lock/report/history are untouched.
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from repoforge.evaluation import t7
from repoforge.evaluation.experiment import (
    aggregate, content_hash, freeze_source, load_manifest, now, read_json, write_json,
)
from repoforge.evaluation.experiment_native import (
    grade_job, release_job_workspace, run_job, validate_agent_tests,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variant", default="a3")
    parser.add_argument("--only", help="Comma-separated instance IDs")
    parser.add_argument("--provision-override", type=Path,
                        help="merge a re-provisioned environment receipt by instance_id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO_ROOT
    reference = args.reference.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    load_dotenv(root / ".env")
    old = load_manifest(reference)
    output.mkdir(parents=True)
    manifest = json.loads(json.dumps(old))
    manifest.pop("manifest_hash", None)
    variant = args.variant
    base_profile = manifest["profiles"].get("a3") or manifest["profiles"].get("a0")
    if base_profile is None:
        raise SystemExit("reference manifest has no a3/a0 profile")
    profile = json.loads(json.dumps(base_profile))
    profile["action_policy"] = {"version": 1, "variant": variant}
    if variant == "a5":
        profile["submit_product_only"] = True
    manifest["profiles"][variant] = profile
    base_agent = manifest["agent_configs"].get("a3") or manifest["agent_configs"].get("a0")
    if base_agent is None:
        raise SystemExit("reference manifest has no a3/a0 agent config")
    manifest["agent_configs"][variant] = json.loads(json.dumps(base_agent))
    manifest["created"] = now()
    manifest["source"] = freeze_source(root, output)
    manifest["baseline_commit"] = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest["diff"] = __import__("subprocess").check_output(
        ["git", "diff", "HEAD", "--", "repoforge", "scripts"], cwd=root, text=True)
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(output / "manifest.json", manifest)

    only = {item.strip() for item in (args.only or "").split(",") if item.strip()}
    smoke = list(manifest.get("smoke_ids") or [])
    ordered_dev = [*smoke, *[identity for identity in manifest["split"]["dev"]
                              if identity not in set(smoke)]]
    if only:
        ordered_dev = [identity for identity in ordered_dev if identity in only]

    def emit(identity):
        job = {"phase": "dev", "variant": variant, "repeat": 0, "instance_id": identity,
               "manifest_hash": manifest["manifest_hash"], "entrypoint": "batch"}
        return {**job, "job_id": content_hash(job)[:24]}

    jobs = [emit(identity) for identity in ordered_dev]
    if args.limit:
        jobs = jobs[:args.limit]
    write_json(output / "jobs.json", jobs)

    receipt_file = reference / "dataset-receipt.json"
    if receipt_file.exists():
        shutil.copy2(receipt_file, output / "dataset-receipt.json")

    receipts = read_json(reference / "provision-native.json")
    if args.provision_override:
        receipts.update(read_json(args.provision_override))
    dev_ids = {job["instance_id"] for job in jobs}
    dev_receipts = {}
    for identity in dev_ids:
        prepared = receipts[identity]
        prepared = json.loads(json.dumps(prepared))
        prepared["source_sha256"] = manifest["source"]["sha256"]
        dev_receipts[identity] = prepared
    write_json(output / "provision-native.json", dev_receipts)

    cache = reference / "host-dataset-cache"
    if cache.exists():
        (output / "host-dataset-cache").symlink_to(cache, target_is_directory=True)

    if args.dry_run:
        print(json.dumps({
            "output": str(output), "variant": variant,
            "jobs": [job["instance_id"] for job in jobs],
            "profiles": sorted(manifest["profiles"]),
            "agent_configs": sorted(manifest["agent_configs"]),
            "manifest_hash": manifest["manifest_hash"],
        }, indent=1))
        return

    results = []
    for index, job in enumerate(jobs, 1):
        directory = output / "runs" / job["job_id"]
        try:
            if not (directory / "answer.json").exists():
                run_job(output, manifest, job)
            if (directory / "answer.json").exists() and not (directory / "grading.json").exists():
                validate_agent_tests(output, manifest, job)
                grade_job(output, manifest, job)
            if (directory / "grading.json").exists():
                try:
                    release_job_workspace(output, manifest, job)
                except Exception:
                    pass
                row = t7.accept_job(output, manifest, job, require_locked=False)
                rest = read_json(directory / "grading.json") or {}
                results.append({"instance_id": job["instance_id"], "resolved": row["local_resolved"],
                                "generation_status": row["generation_status"],
                                "finish_reason": row["finish_reason"],
                                "cost": row["reference_cost"],
                                "local_grading": rest.get("local_grading")})
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            else:
                raise RuntimeError("no grading.json after run")
        except Exception as exc:
            results.append({"instance_id": job["instance_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        write_json(output / "sanity-report.json", {
            "manifest_hash": manifest["manifest_hash"], "reference": str(reference),
            "variant": variant, "created": now(), "rows": results, "completed": len(results),
            "official_harness": "not_run",
        })

    try:
        report = aggregate(output, "dev")
    except ValueError as exc:
        # A subset run (for example O2 a5) is not the frozen T7 job matrix;
        # keep the per-instance results and records the aggregate limitation.
        report = {"manifest_hash": manifest["manifest_hash"], "phase": "dev",
                  "variant": variant, "rows": results, "completed": len(results),
                  "aggregate_error": str(exc), "official_harness": "not_run",
                  "provider_calls": 0}
    write_json(output / "sanity-aggregate.json", report)
    print(json.dumps({"output": str(output), "variant": variant, "completed": len(results),
                      "resolved": sum(row.get("resolved") is True for row in results),
                      "errors": sum("error" in row for row in results),
                      "manifest_hash": manifest["manifest_hash"]}, indent=1))


if __name__ == "__main__":
    main()
