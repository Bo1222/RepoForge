"""Read-only final A0/O2 analysis from frozen dev/holdout reports.

Reference patches are read only after freeze to evaluate filename coverage.
No model, provisioning, grading or retry is invoked by this command.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

import pyarrow as pa
import pyarrow.ipc as ipc


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_statistics(rows):
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row["instance_id"], {})
        assert row["variant"] not in pair
        assert row["status"] == "graded" and isinstance(row["resolved"], bool)
        pair[row["variant"]] = row["resolved"]
    assert all(set(pair) == {"a0", "a5"} for pair in pairs.values())
    both = sum(pair["a0"] and pair["a5"] for pair in pairs.values())
    a0_only = sum(pair["a0"] and not pair["a5"] for pair in pairs.values())
    a5_only = sum(pair["a5"] and not pair["a0"] for pair in pairs.values())
    discordant = a0_only + a5_only
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(a0_only, a5_only) + 1)) / 2 ** discordant)
    deltas = [int(pair["a5"]) - int(pair["a0"]) for pair in pairs.values()]
    rng = random.Random(20261023)
    draws = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000))
    return {"pairs": len(pairs), "both_resolved": both, "a0_only": a0_only, "a5_only": a5_only,
            "a0_resolved": both + a0_only, "a5_resolved": both + a5_only,
            "neither": len(pairs) - both - discordant, "delta": sum(deltas) / len(deltas),
            "exact_mcnemar_two_sided_p": p,
            "bootstrap_95_ci": [draws[249], draws[9749]],
            "bootstrap_samples": 10000, "bootstrap_seed": 20261023,
            "bootstrap_scope": f"resample the {len(pairs)} pooled task pairs; not a held-out or stratified estimate"}


def add(order, item):
    if isinstance(item, dict):
        item = item.get("file_path") or item.get("path")
    if isinstance(item, str) and item and item not in order:
        order.append(item)


def analyze(candidate: Path, reference_dataset: Path | None = None):
    reports = {phase: read(candidate / f"report-{phase}.json") for phase in ("dev", "holdout")}
    source_digests = {}
    for phase, report in reports.items():
        lock = read(candidate / f"{phase}-lock.json")
        source_digests[phase] = sha(candidate / f"report-{phase}.json")
        assert source_digests[phase] == lock["report_sha256"]
    assert reports["dev"]["manifest_hash"] == reports["holdout"]["manifest_hash"]
    assert not ({row["instance_id"] for row in reports["dev"]["rows"]}
                & {row["instance_id"] for row in reports["holdout"]["rows"]})
    rows = [row for report in reports.values() for row in report["rows"]]
    assert len(rows) == 160
    groups = {}
    for arm in ("a0", "a5"):
        selected = [row for row in rows if row["variant"] == arm]
        solved = sum(row["resolved"] is True for row in selected)
        cost = sum(row["cost"] for row in selected)
        assert len(selected) == 80
        groups[arm] = {"selected": 80, "graded": sum(row["status"] == "graded" for row in selected),
                       "resolved": solved, "resolved_rate": solved / 80,
                       "reference_generation_cost": cost, "cost_per_resolved": cost / solved}
    paired = pair_statistics(rows)
    sensitivity = pair_statistics([row for row in rows if row["instance_id"] != "django__django-10097"])
    sensitivity["excluded"] = "django__django-10097"

    dataset = reference_dataset or next((candidate / "host-dataset-cache").rglob("swe-bench_verified-test.arrow"))
    table = ipc.open_stream(pa.memory_map(str(dataset))).read_all().select(["instance_id", "patch"])
    gold = {row["instance_id"]: set(re.findall(r"^\+\+\+ b/(.+)$", row["patch"], re.M))
            for row in table.to_pylist()}


    components = {}
    for arm in ("a0", "a5"):
        counts = Counter()
        violations, cases = [], []
        for row in rows:
            if row["variant"] != arm:
                continue
            trajectory_path = candidate / "runs" / row["job_id"] / "trajectory.json"
            trajectory = read(trajectory_path)
            search, observed = [], []
            for node in trajectory.get("nodes", []):
                build = node.get("context_build")
                if isinstance(build, dict):
                    estimate, budget = build.get("estimated_input_tokens"), build.get("input_budget")
                    counts["context_builds"] += 1
                    if not isinstance(estimate, int) or not isinstance(budget, int) or estimate > budget:
                        violations.append({"job_id": row["job_id"], "node_id": node.get("node_id")})
                for step in node.get("action_steps", []):
                    properties = (step.get("observation") or {}).get("properties") or {}
                    if properties.get("execution_status") == "skipped":
                        continue
                    hits = properties.get("search_hits")
                    if isinstance(hits, dict):
                        for item in [*(hits.get("files") or []), *(hits.get("test_files") or [])]:
                            add(search, item)
                            add(observed, item)
                    action = step.get("action") or {}
                    if str(action.get("action_args_class", "")).endswith(".ViewCodeArgs"):
                        for path in properties.get("files") or {}:
                            add(observed, path)
            reference = gold[row["instance_id"]]
            assert reference
            case = {"instance_id": row["instance_id"], "phase": row["phase"],
                    "trajectory_sha256": sha(trajectory_path)}
            for kind, order in (("search", search), ("search_and_observed_view", observed)):
                for k in (5, 10):
                    key = f"{kind}_any_first_{k}"
                    case[key] = bool(set(order[:k]) & reference)
                    counts[key] += case[key]
            cases.append(case)
        assert len(cases) == 80 and not violations
        components[arm] = {"instances": 80, "context_budget_compliance": 1.0,
                           "budget_violations": violations, "counts": dict(counts),
                           "rates": {k: v / 80 for k, v in counts.items() if k != "context_builds"}, "cases": cases}

    result = {"scope": "dev30 + holdout50: descriptive combined evaluation on 80 distinct tasks",
              "manifest_hash": reports["dev"]["manifest_hash"], "source_report_sha256": source_digests,
              "groups": groups, "paired": paired, "sensitivity_without_django_10097": sensitivity,
              "component_performance": components, "original_splits_preserved": True,
              "reference_use": "post-run filenames only", "model_calls": 0, "benchmark_reruns": 0,
              "official_harness": "not_run"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="Completed candidate with frozen reports and trajectories")
    parser.add_argument("--output", type=Path, required=True, help="New analysis JSON; frozen reports are never modified")
    parser.add_argument("--reference-dataset", type=Path, help="Local SWE-bench Arrow file; defaults to the candidate cache")
    args = parser.parse_args()
    result = analyze(args.candidate.resolve(), args.reference_dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pairs": result["paired"]["pairs"], "groups": result["groups"], "official_harness": result["official_harness"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
