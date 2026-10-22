#!/usr/bin/env python3
"""Run A7 (acceptance-repair-v3, single controller) on the historical 5 smoke instances.

Reuses the T7 native receipts and environments; does not provision a new
30-instance split.  This is a small sanity run, not a benchmark result.
A7 mixes O2's ActionPolicy (only controller) with A6 evidence/obligations.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from moatless.evaluation import t7
from moatless.evaluation.experiment import (
    A7_PROFILES, agent_config, content_hash, freeze_source, now, read_json, write_json,
)
from moatless.evaluation.experiment_native import (
    grade_job, release_job_workspace, run_job, validate_agent_tests,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provision-override", type=Path,
                        help="merge a re-provisioned environment receipt by instance_id")
    args = parser.parse_args()
    root = REPO_ROOT
    reference = args.reference.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    load_dotenv(root / ".env")
    old = read_json(reference / "manifest.json")
    output.mkdir(parents=True)
    manifest = json.loads(json.dumps(old))
    manifest.pop("manifest_hash", None)
    for key in ("t6", "t7", "exploration"):
        manifest.pop(key, None)
    manifest["created"] = now()
    manifest["source"] = freeze_source(root, output)
    manifest["baseline_commit"] = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest["diff"] = __import__("subprocess").check_output(
        ["git", "diff", "HEAD", "--", "moatless", "scripts", "pyproject.toml", "uv.lock"], cwd=root, text=True)
    manifest["profiles"] = A7_PROFILES
    manifest["agent_configs"] = {"a7": agent_config(root, "a7", manifest["model_config"])}
    manifest["repair"] = {"protocol": "acceptance-repair-v3", "variant": "a7"}
    manifest["holdout_state"] = "locked"
    manifest["official_harness"] = "not_run"
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(output / "manifest.json", manifest)

    only = {item.strip() for item in args.only.split(",") if item.strip()}
    smoke = list(manifest.get("smoke_ids") or [])
    ordered = [*smoke, *[identity for identity in manifest["split"]["dev"] if identity not in set(smoke)]]
    ordered = [identity for identity in ordered if identity in only]

    def emit(identity):
        job = {"phase": "dev", "variant": "a7", "repeat": 0, "instance_id": identity,
               "manifest_hash": manifest["manifest_hash"], "entrypoint": "batch"}
        return {**job, "job_id": content_hash(job)[:24]}

    jobs = [emit(identity) for identity in ordered]
    if args.limit:
        jobs = jobs[: args.limit]
    write_json(output / "jobs.json", jobs)

    receipt_file = reference / "dataset-receipt.json"
    if receipt_file.exists():
        shutil.copy2(receipt_file, output / "dataset-receipt.json")
    receipts = read_json(reference / "provision-native.json")
    if args.provision_override:
        override = read_json(args.provision_override)
        receipts.update(override)
    dev_receipts = {}
    for job in jobs:
        prepared = json.loads(json.dumps(receipts[job["instance_id"]]))
        prepared["source_sha256"] = manifest["source"]["sha256"]
        dev_receipts[job["instance_id"]] = prepared
    write_json(output / "provision-native.json", dev_receipts)
    cache = reference / "host-dataset-cache"
    if cache.exists():
        (output / "host-dataset-cache").symlink_to(cache, target_is_directory=True)

    if args.dry_run:
        print(json.dumps({
            "output": str(output), "variant": "a7",
            "jobs": [job["instance_id"] for job in jobs],
            "profiles": sorted(manifest["profiles"]),
            "agent_configs": sorted(manifest["agent_configs"]),
            "actions": [entry.get("action_class", "").rsplit(".", 1)[-1]
                        for entry in manifest["agent_configs"]["a7"]["actions"]],
            "repair": manifest["repair"], "manifest_hash": manifest["manifest_hash"],
        }, indent=1))
        return

    results = []
    for job in jobs:
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
                generation = read_json(directory / "generation.json") or {}
                usage = read_json(directory / "usage.json") if (directory / "usage.json").exists() else {}
                metrics = generation.get("policy_metrics") or {}
                completion = generation.get("completion_status") or {}
                results.append({"instance_id": job["instance_id"], "resolved": row["local_resolved"],
                                "generation_status": row["generation_status"],
                                "finish_reason": row["finish_reason"],
                                "cost": row["reference_cost"],
                                "local_grading": rest.get("local_grading"),
                                "completion_status": completion.get("status"),
                                "completion_reason": completion.get("reason"),
                                "steps": metrics.get("actual_tool_steps"),
                                "policy_decisions": metrics.get("decisions"),
                                "interventions": metrics.get("interventions"),
                                "prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens")})
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            else:
                raise RuntimeError("no grading.json after run")
        except Exception as exc:
            results.append({"instance_id": job["instance_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        write_json(output / "sanity-report.json", {
            "manifest_hash": manifest["manifest_hash"], "reference": str(reference),
            "variant": "a7", "created": now(), "rows": results, "completed": len(results),
            "official_harness": "not_run",
        })
    summary = {"manifest_hash": manifest["manifest_hash"], "variant": "a7",
               "rows": results, "completed": len(results),
               "resolved": sum(row.get("resolved") is True for row in results),
               "errors": sum("error" in row for row in results),
               "cost": round(sum(row.get("cost") or 0 for row in results
                                 if isinstance(row.get("cost"), (int, float))), 6),
               "steps": sum(row.get("steps") or 0 for row in results),
               "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in results),
               "completion_tokens": sum(row.get("completion_tokens") or 0 for row in results),
               "completion_status": {
                   status: sum(row.get("completion_status") == status for row in results)
                   for status in ("verified", "partially_verified", "unverified")},
               "official_harness": "not_run"}
    write_json(output / "sanity-aggregate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
