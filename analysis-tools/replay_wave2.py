#!/usr/bin/env python3
"""Offline Wave2 replay over existing T7 traces (no model calls).

The tool reports the failure modes the Wave2 plan targets: edit preflight
misses, invalid verification targets, retry/context-budget jobs, search no-hit
and gold-file coverage, and context omission/stale/dedup counts.  It never
writes to an experiment candidate.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from repoforge.actions.test_targets import is_verifiable_test_module


class PathRepository:
    def __init__(self, root: Path):
        self.root = root

    def get_file_content(self, path: str):
        try:
            return (self.root / path).read_text()
        except (OSError, UnicodeDecodeError):
            return None


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def read_jsonl(path):
    rows = []
    try:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return rows


def gold_files(patch: str) -> set[str]:
    files = set()
    for line in patch.splitlines():
        match = re.match(r"diff --git a/(\S+) b/(\S+)", line)
        if match:
            files.add(match.group(1))
    return files


def context_counts(trajectory):
    counts = Counter()
    for node in trajectory.get("nodes") or []:
        build = node.get("context_build")
        if not isinstance(build, dict):
            continue
        counts["builds"] += 1
        for key in ("omitted_sources", "shortened_sources", "stale_sources", "deduplicated_sources"):
            counts[key] += len(build.get(key) or [])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=["dev", "holdout"], default="holdout")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    facts_path = candidate / "analysis/t7" / ("holdout/facts.jsonl" if args.phase == "holdout" else "facts.jsonl")
    facts = read_jsonl(facts_path)
    if not facts:
        raise SystemExit(f"missing facts: {facts_path}")
    receipts = read_json(candidate / "provision-native.json", {}) or {}
    jobs = {job["job_id"]: job for job in read_json(candidate / "jobs.json", []) or []}
    dataset = {}
    try:
        from datasets import load_dataset
        import os
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test",
                          cache_dir=str(candidate / "host-dataset-cache"))
        dataset = {row["instance_id"]: row for row in ds}
    except Exception:
        dataset = {}

    summary = defaultdict(Counter)
    targets = defaultdict(Counter)
    search = defaultdict(Counter)
    context = defaultdict(Counter)
    retry_jobs = []
    for row in facts:
        arm = row["variant"]
        summary[arm]["jobs"] += 1
        job_id = row["job_id"]
        directory = candidate / "runs" / job_id
        trajectory = read_json(directory / "trajectory.json", {}) or {}
        instance = row["instance_id"]
        public = (dataset.get(instance) or {})
        gold = gold_files(public.get("patch") or "")
        receipt = receipts.get(instance) or {}
        repository = PathRepository(Path((receipt.get("environment") or {}).get("repo", "")))
        seen_gold = set()
        for node in trajectory.get("nodes") or []:
            for step in node.get("action_steps") or []:
                observation = step.get("observation") or {}
                properties = observation.get("properties") or {}
                reason = properties.get("fail_reason")
                if reason:
                    summary[arm][f"fail:{reason}"] += 1
                hits = (properties.get("search_hits") or {}).get("files") or []
                call_files = {hit.get("file_path") for hit in hits if hit.get("file_path")}
                if call_files and gold:
                    search[arm]["calls"] += 1
                    search[arm]["no_gold"] += not bool(call_files & gold)
                    seen_gold |= call_files & gold
                if reason == "no_search_hits":
                    search[arm]["no_hit"] += 1
        if gold:
            search[arm]["jobs_with_gold"] += 1
            search[arm]["jobs_seen_gold"] += bool(seen_gold)
        decisions = read_json(directory / "policy-decisions.json", []) or []
        for decision in decisions if isinstance(decisions, list) else []:
            body = decision.get("decision") or {}
            if body.get("kind") != "verify":
                continue
            target = body.get("target")
            targets[arm]["verify"] += 1
            if not target:
                targets[arm]["no_target"] += 1
                continue
            if repository.root.is_dir():
                targets[arm]["target_file_exists"] += (repository.root / target).is_file()
                targets[arm]["target_is_test_module"] += bool(
                    is_verifiable_test_module(repository, target))
            ftp = public.get("FAIL_TO_PASS") or []
            if isinstance(ftp, str):
                try:
                    ftp = json.loads(ftp)
                except json.JSONDecodeError:
                    ftp = []
            target_stem = Path(target).stem
            targets[arm]["matches_f2p"] += any(
                target in name or target_stem in name for name in ftp)
        for key, value in context_counts(trajectory).items():
            context[arm][key] += value
        if row.get("error_type") == "ContextBudgetExceeded" or row.get("error_kind") == "context_budget_exceeded":
            retry_jobs.append(f"{instance}/{arm}")

    result = {"candidate": str(candidate), "phase": args.phase, "jobs": len(facts),
              "failures": {arm: dict(counter) for arm, counter in summary.items()},
              "verify_targets": {arm: dict(counter) for arm, counter in targets.items()},
              "search": {arm: dict(counter) for arm, counter in search.items()},
              "context": {arm: dict(counter) for arm, counter in context.items()},
              "retry_jobs": retry_jobs, "provider_calls": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
