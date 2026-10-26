#!/usr/bin/env python3
"""Aggregate holdout trajectory facts, policy decisions and context builds."""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

C = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
FACTS = C / "analysis/t7/holdout/facts.jsonl"
OUT_JSON = C / "analysis/t7/holdout/trajectory-summary.json"
OUT_MD = C / "analysis/t7/holdout/trajectory-summary.md"


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


def quantiles(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    values = sorted(values)
    return {"n": len(values), "mean": sum(values) / len(values),
            "p50": statistics.median(values),
            "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
            "max": values[-1]}


def context_summary(trajectory):
    builds = []
    categories = Counter()
    for node in trajectory.get("nodes") or []:
        build = node.get("context_build")
        if not isinstance(build, dict):
            continue
        builds.append(build)
        for source in build.get("included_sources") or []:
            categories[str(source).split(":", 1)[0]] += 1
    estimates = [b.get("estimated_input_tokens") for b in builds if b.get("estimated_input_tokens") is not None]
    omitted = [len(b.get("omitted_sources") or []) for b in builds]
    shortened = [len(b.get("shortened_sources") or []) for b in builds]
    stale = [len(b.get("stale_sources") or []) for b in builds]
    dedup = [len(b.get("deduplicated_sources") or []) for b in builds]
    return {
        "builds": len(builds),
        "estimated_input_tokens": quantiles(estimates),
        "included_categories": dict(categories),
        "omitted_total": sum(omitted),
        "omitted_jobs": sum(1 for value in omitted if value),
        "shortened_total": sum(shortened),
        "shortened_jobs": sum(1 for value in shortened if value),
        "stale_total": sum(stale),
        "stale_jobs": sum(1 for value in stale if value),
        "deduplicated_total": sum(dedup),
        "deduplicated_jobs": sum(1 for value in dedup if value),
    }


def main():
    facts = read_jsonl(FACTS)
    if not facts:
        raise SystemExit(f"missing facts: {FACTS}")
    by_arm = defaultdict(list)
    policy_stops = defaultdict(lambda: {"count": 0, "resolved": 0, "unresolved": 0, "jobs": []})
    decision_kinds = Counter()
    decision_reasons = Counter()
    recovery_jobs = []
    context_exceeded_jobs = []
    for row in facts:
        arm = row["variant"]
        by_arm[arm].append(row)
        for reason in row.get("policy_stop_reasons") or []:
            entry = policy_stops[reason]
            entry["count"] += 1
            if row.get("local_resolved") is True:
                entry["resolved"] += 1
            elif row.get("local_resolved") is False:
                entry["unresolved"] += 1
            entry["jobs"].append(f"{row['instance_id']}/{arm}")
        decisions = read_json(C / "runs" / row["job_id"] / "policy-decisions.json", [])
        if isinstance(decisions, list):
            for decision in decisions:
                body = (decision or {}).get("decision") or {}
                decision_kinds[body.get("kind") or "unknown"] += 1
                decision_reasons[body.get("reason") or "unknown"] += 1
        attempts = read_jsonl(C / "runs" / row["job_id"] / "attempts.jsonl")
        if len(attempts) > 1 or any(item.get("resume") for item in attempts):
            recovery_jobs.append({"job_id": row["job_id"], "instance_id": row["instance_id"],
                                  "variant": arm, "attempts": len(attempts),
                                  "resumes": sum(bool(item.get("resume")) for item in attempts)})
        events = (C / "runs" / row["job_id"] / "events.jsonl")
        try:
            text = events.read_text(errors="replace")
        except OSError:
            text = ""
        if "ContextBudgetExceeded" in text or "context_budget_exceeded" in text:
            context_exceeded_jobs.append(f"{row['instance_id']}/{arm}")

    summary = {"jobs": len(facts), "arms": {}}
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        executed = [r for r in rows if r.get("generation_status") != "not_run"]
        resolved = [r for r in executed if r.get("local_resolved") is True]
        costs = [r.get("reference_cost") for r in executed if r.get("reference_cost") is not None]
        prompt = [r.get("prompt_tokens") for r in executed if r.get("prompt_tokens") is not None]
        completion = [r.get("completion_tokens") for r in executed if r.get("completion_tokens") is not None]
        cache = [r.get("cache_read_tokens") for r in executed if r.get("cache_read_tokens") is not None]
        seconds = [r.get("generation_seconds") for r in executed if r.get("generation_seconds") is not None]
        steps = [((r.get("policy_metrics") or {}).get("actual_tool_steps")) for r in executed]
        tests = [r.get("tests") or {} for r in executed]
        no_gain = [((r.get("progress") or {}).get("no_gain_rate")) for r in executed]
        context_rows = []
        for r in executed:
            trajectory = read_json(C / "runs" / r["job_id"] / "trajectory.json", {})
            context_rows.append(context_summary(trajectory or {}))
        builds = [c["builds"] for c in context_rows]
        estimate_means = [c["estimated_input_tokens"]["mean"] for c in context_rows
                          if c["estimated_input_tokens"]["mean"] is not None]
        omitted = [c["omitted_total"] for c in context_rows]
        shortened = [c["shortened_total"] for c in context_rows]
        stale = [c["stale_total"] for c in context_rows]
        dedup = [c["deduplicated_total"] for c in context_rows]
        categories = Counter()
        for c in context_rows:
            categories.update(c["included_categories"])
        summary["arms"][arm] = {
            "selected": len(rows), "executed": len(executed), "resolved": len(resolved),
            "resolved_over_executed": len(resolved) / len(executed) if executed else None,
            "finish_reasons": dict(Counter(r.get("finish_reason") for r in executed)),
            "policy_decisions": sum(r.get("policy_decisions") or 0 for r in executed),
            "policy_interventions": sum((r.get("policy_metrics") or {}).get("interventions", 0) for r in executed),
            "skipped_proposals": sum((r.get("policy_metrics") or {}).get("skipped_proposals", 0) for r in executed),
            "policy_stops": sum(1 for r in executed if (r.get("finish_reason") == "policy_stop")),
            "budget_stops": sum(1 for r in executed if (r.get("finish_reason") == "max_iterations")),
            "tool_steps": quantiles(steps),
            "no_gain_rate": quantiles(no_gain),
            "test_processes": quantiles([(r.get("policy_metrics") or {}).get("test_processes") for r in executed]),
            "effective_verification": sum(bool(t.get("effective_verification")) for t in tests) / len(tests) if tests else None,
            "test_records": quantiles([t.get("records") for t in tests]),
            "prompt_tokens": quantiles(prompt), "completion_tokens": quantiles(completion),
            "cache_read_tokens": quantiles(cache), "generation_seconds": quantiles(seconds),
            "reference_cost": quantiles(costs),
            "cost_per_resolved": (sum(costs) / len(resolved)) if costs and resolved else None,
            "context": {
                "builds": quantiles(builds),
                "estimated_input_tokens_mean": quantiles(estimate_means),
                "omitted_total": quantiles(omitted), "shortened_total": quantiles(shortened),
                "stale_total": quantiles(stale), "deduplicated_total": quantiles(dedup),
                "jobs_with_omitted": sum(1 for v in omitted if v),
                "jobs_with_shortened": sum(1 for v in shortened if v),
                "jobs_with_stale": sum(1 for v in stale if v),
                "jobs_with_dedup": sum(1 for v in dedup if v),
                "included_categories": dict(categories),
            },
        }
    by_arm_instances = {}
    for arm, rows in by_arm.items():
        by_arm_instances[arm] = {r["instance_id"]: r for r in rows if r.get("generation_status") != "not_run"}
    common = sorted(set(by_arm_instances.get("a3", {})) & set(by_arm_instances.get("a0", {})))
    transitions = {"both": [], "neither": [], "a3_only": [], "a0_only": []}
    for iid in common:
        a3 = by_arm_instances["a3"][iid].get("local_resolved") is True
        a0 = by_arm_instances["a0"][iid].get("local_resolved") is True
        key = "both" if a3 and a0 else "neither" if not a3 and not a0 else "a3_only" if a3 else "a0_only"
        transitions[key].append(iid)
    summary["paired"] = {"n": len(common), **{k: len(v) for k, v in transitions.items()},
                         "transitions": transitions}
    summary["policy_stop_reasons"] = dict(policy_stops)
    summary["decision_kinds"] = dict(decision_kinds)
    summary["decision_reasons"] = dict(decision_reasons.most_common(30))
    summary["recovery_jobs"] = recovery_jobs
    summary["context_budget_exceeded_jobs"] = context_exceeded_jobs
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({"json": str(OUT_JSON), "arms": {k: {"resolved": v["resolved"],
          "executed": v["executed"], "policy_stops": v["policy_stops"],
          "budget_stops": v["budget_stops"], "builds": v["context"]["builds"]}
          for k, v in summary["arms"].items()},
          "policy_stop_reasons": summary["policy_stop_reasons"],
          "paired": summary["paired"], "recovery_jobs": len(recovery_jobs),
          "context_budget_exceeded_jobs": len(context_exceeded_jobs)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
