"""Freeze E1 metadata, split, profiles and job inventory without model calls.

This preparation tool does not run agents or grade answers. It reads only the
repository's instance-ID list, never reference patches or official test cases.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile

BASELINE = "aac2ce706514a129b19fe329dd29ce773ec1391d"
DATASET = "princeton-nlp/SWE-bench_Verified"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def repository(instance_id):
    return instance_id.rsplit("-", 1)[0].replace("__", "/", 1)


def allocate(groups, size):
    total = sum(map(len, groups.values()))
    if size < 0 or size > total:
        raise ValueError("Sample size exceeds remaining dataset")
    counts = {repo: size * len(rows) // total for repo, rows in groups.items()} if total else {}
    remainder = size - sum(counts.values())
    order = sorted(groups, key=lambda repo: (-(size * len(groups[repo]) % total), repo)) if total else []
    for repo in order[:remainder]:
        counts[repo] += 1
    return counts


def split_instances(ids, dev_size=30, holdout_size=100, seed=42):
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate instance IDs")
    groups = defaultdict(list)
    for identity in sorted(ids):
        groups[repository(identity)].append(identity)
    randomizer = random.Random(seed)
    for repo in sorted(groups):
        randomizer.shuffle(groups[repo])
    counts = allocate(groups, dev_size)
    dev = sorted(identity for repo in sorted(groups) for identity in groups[repo][:counts[repo]])
    remaining = {repo: rows[counts[repo]:] for repo, rows in groups.items()}
    counts = allocate(remaining, holdout_size)
    holdout = sorted(identity for repo in sorted(remaining) for identity in remaining[repo][:counts[repo]])
    return {"seed": seed, "method": "repository-stratified largest remainder, without replacement",
            "dev": dev, "holdout": holdout,
            "counts": {phase: dict(sorted(Counter(map(repository, rows)).items()))
                       for phase, rows in (("dev", dev), ("holdout", holdout))}}


def profiles(context_window):
    policy = {"version": 1, "context_window": context_window, "max_input_tokens": 32768,
              "max_observation_tokens": 4000, "recent_interactions": 2,
              "enable_task_state": False, "load_repo_instructions": False}
    return {
        "frozen_baseline": {"code": BASELINE, "context_policy": None},
        "integrity_only": {"code": "candidate_snapshot", "context_policy": None},
        "managed_facts": {"code": "candidate_snapshot", "context_policy": dict(policy)},
        "managed_task": {"code": "candidate_snapshot", "context_policy": {**policy, "enable_task_state": True}},
        "managed_rules": {"code": "candidate_snapshot", "context_policy": {**policy, "enable_task_state": True, "load_repo_instructions": True}},
    }


def source_hashes(root):
    # Freeze relevant Python source contents, including new untracked modules.
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("moatless", "scripts") for path in sorted((root / folder).rglob("*.py"))}


def prepare(root, model, context_window, max_iterations=80, max_cost=2.0):
    path = root / "moatless/evaluation/datasets/verified_dataset.json"
    metadata = json.loads(path.read_text())
    split = split_instances(metadata["instance_ids"])
    configuration = {"model": model, "max_tokens": 4096, "temperature": 0,
                     "max_iterations": max_iterations, "max_cost": max_cost,
                     "model_timeout": 180, "tool_timeout": 120}
    if context_window - configuration["max_tokens"] - max(1024, (context_window + 19) // 20) <= 0:
        raise ValueError("No model input capacity after output and safety reserves")
    hashes = source_hashes(root)
    manifest = {"version": 1, "dataset": DATASET, "dataset_split": "test",
                "instance_inventory_source": str(path.relative_to(root)),
                "instance_inventory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "baseline_commit": BASELINE, "candidate_source_hashes": hashes,
                "candidate_source_digest": digest(hashes), "split": split,
                "configuration": configuration, "profiles": profiles(context_window),
                "entrypoints": ["batch", "cli"], "repetitions": 3,
                "generation_inputs": ["instance_id", "repo", "base_commit", "problem_statement", "version"],
                "excluded_generation_inputs": ["patch", "hints_text", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "reference_spans"],
                "grading": {"answer_frozen_first": True, "independent_repository": True,
                            "agent_tests_separate": True, "official_harness": "not_run"},
                "selection": {"phase": "dev", "primary": "official resolved", "tie_breaker": "project-reported cost",
                              "selected_variant": None, "holdout_state": "locked_pending_dev_selection"},
                "status": "prepared_only", "provider_calls": 0}
    manifest["manifest_hash"] = digest(manifest)
    jobs = []
    for entrypoint in manifest["entrypoints"]:
        for variant, profile in manifest["profiles"].items():
            for repeat in range(3):
                for identity in split["dev"]:
                    job = {"phase": "dev", "entrypoint": entrypoint, "variant": variant,
                           "repeat": repeat, "instance_id": identity, "profile": profile,
                           "configuration": configuration, "manifest_hash": manifest["manifest_hash"]}
                    job["job_id"] = digest(job)[:24]
                    job["generation_status"] = "not_run"
                    job["official_harness"] = "not_run"
                    jobs.append(job)
    return manifest, jobs


def preflight(root):
    credential_names = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    try:
        from dotenv import dotenv_values
        configured = dotenv_values(root / ".env")
    except ImportError:
        configured = {}
    executable = shutil.which("docker")
    docker = {"executable_present": bool(executable), "return_code": None}
    if executable:
        try:
            result = subprocess.run([executable, "info", "--format", "{{json .ServerVersion}}"],
                                    capture_output=True, text=True, timeout=15)
            docker.update(return_code=result.returncode, output=(result.stdout + result.stderr).strip())
        except subprocess.TimeoutExpired:
            docker["output"] = "Docker info timed out"
    result = {"version": 1, "provider_calls": 0, "official_harness": "not_run", "docker": docker,
              "credentials_present": [name for name in credential_names if os.environ.get(name) or configured.get(name)],
              "modules": {name: bool(importlib.util.find_spec(name)) for name in ("swebench", "datasets", "pyarrow", "docker")},
              "existing_batch_environment": {"instance_path": bool(os.environ.get("INSTANCE_PATH") and Path(os.environ["INSTANCE_PATH"]).is_file()),
                    "repo_path": bool(os.environ.get("REPO_PATH") and Path(os.environ["REPO_PATH"]).is_dir()),
                    "index_path": bool(os.environ.get("INDEX_STORE_DIR") and Path(os.environ["INDEX_STORE_DIR"]).is_dir())},
              "blockers": []}
    if docker["return_code"] != 0:
        result["blockers"].append("Official Harness requires an accessible Docker daemon")
    if not all(result["existing_batch_environment"].values()):
        result["blockers"].append("Current batch entrypoint requires per-instance repository, runtime and index provisioning")
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def freeze_sources(root, output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    jobs_path = output / "dev-jobs.jsonl"
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines()]
    if any(job["generation_status"] != "not_run" for job in jobs):
        raise ValueError("Cannot rewrite the identity of an executed job")
    if source_hashes(root) != manifest["candidate_source_hashes"]:
        raise ValueError("Candidate code changed after preparation; prepare a new manifest before freezing")
    archive = subprocess.run(["git", "archive", manifest["baseline_commit"]], cwd=root,
                             capture_output=True, check=True).stdout
    baseline_path = output / "baseline-source.tar"
    baseline_path.write_bytes(archive)
    with tempfile.TemporaryDirectory(prefix="moatless-e1-freeze-") as temporary:
        target = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as baseline:
            baseline.extractall(target)
        # Only approved code paths are overlaid; .env, logs and private grading
        # artifacts are never added from the working tree.
        for path, expected in manifest["candidate_source_hashes"].items():
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"moatless", "scripts"}:
                raise ValueError("Invalid candidate code source path")
            body = (root / relative).read_bytes()
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError("Candidate source changed during freezing")
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            (target / relative).write_bytes(body)
        candidate_path = output / "candidate-source.tar"
        with tarfile.open(candidate_path, "w") as candidate:
            def deterministic(info):
                info.mtime, info.uid, info.gid = 0, 0, 0
                info.uname = info.gname = ""
                return info
            for path in sorted(target.rglob("*")):
                candidate.add(path, arcname=str(path.relative_to(target)), recursive=False, filter=deterministic)
        with tarfile.open(candidate_path) as candidate:
            for path, expected in manifest["candidate_source_hashes"].items():
                assert hashlib.sha256(candidate.extractfile(path).read()).hexdigest() == expected
    snapshots = {name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for name, path in (("baseline", baseline_path), ("candidate", candidate_path))}
    environment = {"python": sys.version, "executable": sys.executable,
                   "dependencies": dict(sorted((dist.metadata["Name"], dist.version)
                                              for dist in importlib.metadata.distributions() if dist.metadata["Name"]))}
    environment_path = output / "environment.json"
    write(environment_path, environment)
    manifest["source_snapshots"] = snapshots
    manifest["environment_sha256"] = hashlib.sha256(environment_path.read_bytes()).hexdigest()
    manifest["manifest_hash"] = digest({key: value for key, value in manifest.items() if key != "manifest_hash"})
    write(manifest_path, manifest)
    for job in jobs:
        job["manifest_hash"] = manifest["manifest_hash"]
        job["job_id"] = digest({key: value for key, value in job.items()
                                if key not in {"job_id", "generation_status", "official_harness"}})[:24]
    jobs_path.write_text("".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--model", required=True)
    preparation.add_argument("--context-window", required=True, type=int)
    preparation.add_argument("--output", required=True, type=Path)
    check = commands.add_parser("preflight")
    check.add_argument("--output", required=True, type=Path)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        manifest, jobs = prepare(root, args.model, args.context_window)
        write(args.output / "manifest.json", manifest)
        (args.output / "dev-jobs.jsonl").write_text("".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs))
        print(json.dumps({"dev": len(manifest["split"]["dev"]), "holdout": len(manifest["split"]["holdout"]),
                          "dev_jobs": len(jobs), "manifest_hash": manifest["manifest_hash"], "status": "prepared_only"}))
    elif args.command == "preflight":
        result = preflight(root)
        write(args.output, result)
        print(json.dumps({"official_harness": result["official_harness"], "blockers": result["blockers"]}))
    else:
        manifest = freeze_sources(root, args.output)
        print(json.dumps({"manifest_hash": manifest["manifest_hash"], "source_snapshots": manifest["source_snapshots"]}))


if __name__ == "__main__":
    main()
