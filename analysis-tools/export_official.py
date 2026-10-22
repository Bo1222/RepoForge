#!/usr/bin/env python3
"""Export frozen T7 predictions and ID/hash manifests for the official Harness.

Read-only with respect to the experiment candidate.  All outputs go under
analysis-tools/artifacts/<candidate>/ and are never mounted into the agent or
used to change the frozen manifest, lock, answers or selection.

The official SWE-bench Harness expects one prediction JSONL per model/arm with
fields: instance_id, model_name_or_path, model_patch.  We emit one file per
phase/arm so that the same instance_id can appear in different arms without
colliding in a single predictions file.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = REPO_ROOT / "analysis-tools/artifacts"
DEV_ARMS = ("a0", "a1", "a2", "a3", "a4")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()


def ordered_dev(manifest):
    smoke = list(manifest.get("smoke_ids") or [])
    return [*smoke, *[identity for identity in manifest["split"]["dev"]
                      if identity not in set(smoke)]]


def index_map(ids):
    return {identity: index for index, identity in enumerate(ids)}


def candidate_jobs(manifest, jobs, phase, variant, order):
    selected = [job for job in jobs
                if job["phase"] == phase and job["variant"] == variant]
    indexes = index_map(order)
    selected.sort(key=lambda job: (indexes[job["instance_id"]], job["repeat"]))
    return selected


def expected_count(phase, variant):
    return 30 if phase == "dev" else 100


def job_identity(job):
    return {"job_id": job["job_id"], "instance_id": job["instance_id"],
            "variant": job["variant"], "repeat": job["repeat"],
            "phase": job["phase"]}


def verify_dataset_ids(candidate, manifest, ids):
    """Optionally verify every ID exists in the frozen dataset test split."""
    cache = candidate / "host-dataset-cache"
    if not cache.exists():
        return {"checked": False, "reason": "no host-dataset-cache"}
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"checked": False, "reason": f"datasets import failed: {exc}"}
    dataset = load_dataset(manifest["dataset"], split="test", cache_dir=str(cache))
    present = {row["instance_id"] for row in dataset}
    missing = sorted(set(ids) - present)
    receipt = read_json(candidate / "dataset-receipt.json", {}) or {}
    return {"checked": True, "dataset_instances": len(dataset),
            "fingerprint": dataset._fingerprint,
            "receipt_fingerprint": receipt.get("fingerprint"),
            "all_ids_present": not missing, "missing": missing}


def row_for(candidate, job, *, require_patch):
    directory = candidate / "runs" / job["job_id"]
    patch_path = directory / "patch.diff"
    generation_path = directory / "generation.json"
    answer_path = directory / "answer.json"
    grading_path = directory / "grading.json"
    usage_path = directory / "usage.json"
    patch_present = patch_path.exists()
    patch_text = patch_path.read_text() if patch_present else None
    answer = read_json(answer_path, {}) or {}
    grading = read_json(grading_path, {}) or {}
    generation = read_json(generation_path, {}) or {}
    usage = read_json(usage_path, {}) or {}
    if require_patch and not patch_present:
        raise SystemExit(f"Missing frozen patch for completed job {job['job_id']} "
                         f"({job['instance_id']} / {job['variant']})")
    return {
        **job_identity(job),
        "run_dir": str(directory),
        "patch_path": str(patch_path) if patch_present else None,
        "patch_present": patch_present,
        "patch_bytes": len(patch_text.encode()) if patch_text is not None else None,
        "patch_sha256": sha_file(patch_path) if patch_present else None,
        "answer_path": str(answer_path) if answer_path.exists() else None,
        "answer_hash": answer.get("answer_hash"),
        "grading_path": str(grading_path) if grading_path.exists() else None,
        "grading_sha256": sha_file(grading_path) if grading_path.exists() else None,
        "local_resolved": grading.get("local_resolved"),
        "local_grading": grading.get("local_grading"),
        "generation_sha256": sha_file(generation_path) if generation_path.exists() else None,
        "finish_reason": generation.get("finish_reason"),
        "generation_status": generation.get("generation_status"),
        "usage_sha256": sha_file(usage_path) if usage_path.exists() else None,
        "cost_complete": usage.get("cost_complete"),
        "reference_cost": usage.get("reference_cost"),
    }


def write_predictions(artifacts, phase, variant, rows):
    path = artifacts / "predictions" / f"{phase}-{variant}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            if not row["patch_present"]:
                continue
            patch = Path(row["patch_path"]).read_text()
            stream.write(json.dumps({
                "instance_id": row["instance_id"],
                "model_name_or_path": f"moatless-t7-{phase}-{variant}",
                "model_patch": patch,
            }, ensure_ascii=False) + "\n")
    return {"path": str(path), "sha256": sha_file(path),
            "instances": sum(row["patch_present"] for row in rows)}


def build(candidate, artifacts, phase, verify_dataset=True):
    candidate = Path(candidate).resolve()
    artifacts = Path(artifacts).resolve()
    manifest = read_json(candidate / "manifest.json")
    jobs = read_json(candidate / "jobs.json")
    lock = read_json(candidate / "t7-lock.json")
    if not manifest or not jobs:
        raise SystemExit("Candidate is missing manifest.json or jobs.json")
    split = manifest["split"]
    if len(split["dev"]) != 30 or len(split["holdout"]) != 100:
        raise SystemExit("Expected frozen dev30/holdout100")
    if set(split["dev"]) & set(split["holdout"]):
        raise SystemExit("dev and holdout overlap")
    dev_order, holdout_order = ordered_dev(manifest), list(split["holdout"])
    id_manifest = {
        "candidate": str(candidate),
        "candidate_name": candidate.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "manifest_hash": manifest["manifest_hash"],
        "dataset": manifest["dataset"],
        "source_digest": manifest["source"]["digest"],
        "analysis_plan_sha256": manifest["t7"]["analysis_plan_sha256"],
        "lock_hash": (lock or {}).get("lock_hash"),
        "split": {"dev": dev_order, "holdout": holdout_order,
                  "dev_sha256": digest(dev_order),
                  "holdout_sha256": digest(holdout_order)},
    }
    rows = []
    predictions = {}

    dev_jobs = {}
    for arm in DEV_ARMS:
        selected = candidate_jobs(manifest, jobs, "dev", arm, dev_order)
        if len(selected) != 30:
            raise SystemExit(f"dev {arm}: expected 30 jobs, found {len(selected)}")
        if {job["instance_id"] for job in selected} != set(split["dev"]):
            raise SystemExit(f"dev {arm}: instance IDs differ from frozen split")
        arm_rows = [row_for(candidate, job,
                            require_patch=phase in {"dev", "all"})
                    for job in selected]
        dev_jobs[arm] = {"jobs": [job_identity(job) for job in selected],
                         "jobs_sha256": digest([job_identity(job) for job in selected]),
                         "rows": arm_rows}
        rows.extend(arm_rows)
        if phase in {"dev", "all"}:
            predictions[f"dev-{arm}"] = write_predictions(artifacts, "dev", arm, arm_rows)
    id_manifest["dev"] = {
        "arms": list(DEV_ARMS),
        "instance_ids": list(split["dev"]),
        "instance_ids_sha256": digest(list(split["dev"])),
        "jobs": dev_jobs,
        "job_count": sum(len(value["jobs"]) for value in dev_jobs.values()),
    }

    frozen = {}
    for arm in DEV_ARMS:
        selected = candidate_jobs(manifest, jobs, "holdout", arm, holdout_order)
        if len(selected) != 100:
            raise SystemExit(f"holdout {arm}: expected 100 frozen jobs, found {len(selected)}")
        frozen[arm] = {"jobs": [job_identity(job) for job in selected],
                       "jobs_sha256": digest([job_identity(job) for job in selected])}
    id_manifest["holdout_frozen"] = {
        "arms": list(DEV_ARMS), "instance_ids": list(split["holdout"]),
        "instance_ids_sha256": digest(list(split["holdout"])),
        "jobs": frozen, "job_count": sum(len(value["jobs"]) for value in frozen.values()),
    }

    if not lock:
        id_manifest["holdout_execution"] = {"error": "t7-lock.json missing"}
    else:
        arms = [lock["winner"], lock["control"]]
        if arms[0] == arms[1]:
            arms = [arms[0]]
        execution = {}
        for arm in arms:
            selected = candidate_jobs(manifest, jobs, "holdout", arm, holdout_order)
            if len(selected) != 100:
                raise SystemExit(f"holdout execution {arm}: expected 100 jobs")
            if {job["instance_id"] for job in selected} != set(split["holdout"]):
                raise SystemExit(f"holdout {arm}: instance IDs differ from frozen split")
            arm_rows = [row_for(candidate, job, require_patch=False) for job in selected]
            execution[arm] = {"jobs": [job_identity(job) for job in selected],
                              "jobs_sha256": digest([job_identity(job) for job in selected]),
                              "rows": arm_rows}
            rows.extend(arm_rows)
            if phase in {"holdout", "all"}:
                predictions[f"holdout-{arm}"] = write_predictions(
                    artifacts, "holdout", arm, arm_rows)
        id_manifest["holdout_execution"] = {
            "winner": lock["winner"], "control": lock["control"],
            "arms": arms, "instance_ids": list(split["holdout"]),
            "instance_ids_sha256": digest(list(split["holdout"])),
            "jobs": execution,
            "job_count": sum(len(value["jobs"]) for value in execution.values()),
        }

    if verify_dataset:
        all_ids = [*split["dev"], *split["holdout"]]
        id_manifest["dataset_id_check"] = verify_dataset_ids(candidate, manifest, all_ids)

    write_json(artifacts / "id-manifest.json", id_manifest)
    hash_manifest = {
        "candidate": str(candidate),
        "candidate_name": candidate.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "manifest_hash": manifest["manifest_hash"],
        "lock_hash": (lock or {}).get("lock_hash"),
        "dataset": manifest["dataset"],
        "dataset_receipt": read_json(candidate / "dataset-receipt.json", {}),
        "source_digest": manifest["source"]["digest"],
        "analysis_plan_sha256": manifest["t7"]["analysis_plan_sha256"],
        "id_manifest_sha256": sha_file(artifacts / "id-manifest.json"),
        "predictions": predictions,
        "rows": rows,
        "missing_patches": [row["job_id"] for row in rows if not row["patch_present"]],
        "provider_bill": "not_available",
        "official_harness": "not_run_until_the_exported_predictions_are_evaluated",
    }
    write_json(artifacts / "hash-manifest.json", hash_manifest)
    return id_manifest, hash_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path,
                        default=None,
                        help="Output directory (default analysis-tools/artifacts/<candidate>)")
    parser.add_argument("--phase", choices=["ids", "dev", "holdout", "all"], default="ids",
                        help="ids writes only the ID manifests; dev/holdout/all also write predictions")
    parser.add_argument("--no-dataset-check", action="store_true",
                        help="Skip loading the dataset for ID verification")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    artifacts = (args.artifacts or DEFAULT_ARTIFACTS / candidate.name).resolve()
    phase = "all" if args.phase == "all" else args.phase
    # IDs are always written.  For ids we still require dev patches to exist
    # only when phase != ids; use require_patch via the phase.
    id_manifest, hash_manifest = build(candidate, artifacts, phase,
                                       verify_dataset=not args.no_dataset_check)
    print(json.dumps({
        "artifacts": str(artifacts),
        "dev_jobs": id_manifest["dev"]["job_count"],
        "holdout_frozen_jobs": id_manifest["holdout_frozen"]["job_count"],
        "holdout_execution_jobs": id_manifest.get("holdout_execution", {}).get("job_count"),
        "dev_ids_sha256": id_manifest["split"]["dev_sha256"],
        "holdout_ids_sha256": id_manifest["split"]["holdout_sha256"],
        "predictions": hash_manifest["predictions"],
        "missing_patches": len(hash_manifest["missing_patches"]),
        "dataset_id_check": id_manifest.get("dataset_id_check"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
