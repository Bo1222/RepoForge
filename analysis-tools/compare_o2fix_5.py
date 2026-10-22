#!/usr/bin/env python3
"""Combine the fixed-code wrong-2 and rest-3 runs into a 5-instance table.

Compares the fixed O2 against the previous O2 run, the T7 A0 baseline and the
T7 A3 arm.  Offline only; no model calls.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def run_row(run, job):
    directory = run / "runs" / job["job_id"]
    grade = read(directory / "grading.json", {}) or {}
    usage = read(directory / "usage.json", {}) or {}
    generation = read(directory / "generation.json", {}) or {}
    decisions = read(directory / "policy-decisions.json", []) or []
    trajectory = read(directory / "trajectory.json", {}) or {}
    last = decisions[-1] if decisions else {}
    state = last.get("state") or {}
    tool_failures = Counter()
    tests = []
    for node in trajectory.get("nodes") or []:
        for step in node.get("action_steps") or []:
            observation = step.get("observation") or {}
            properties = observation.get("properties") or {}
            reason = properties.get("fail_reason")
            if reason:
                tool_failures[reason] += 1
            if properties.get("test_run"):
                tests.append(properties["test_run"])
    failures = {}
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = (grade.get("tests_status") or {}).get(key) or {}
        if isinstance(value, dict) and value.get("failure"):
            failures[key] = value["failure"]
    return {
        "resolved": grade.get("local_resolved"),
        "grading": grade.get("local_grading"),
        "finish_reason": generation.get("finish_reason"),
        "cost": usage.get("reference_cost"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "steps": (generation.get("policy_metrics") or {}).get("actual_tool_steps"),
        "decisions": len(decisions),
        "verification_state": state.get("verification_state"),
        "target": state.get("target"),
        "introduced_failures": state.get("introduced_failures") or [],
        "grading_failures": failures,
        "tool_failures": dict(tool_failures),
        "test_runs": [
            {"source": item.get("test_source"), "targets": item.get("targets"),
             "status": item.get("status")}
            for item in tests
        ],
    }


def total(rows, key):
    values = [row[key] for row in rows if row.get(key)]
    return {
        "resolved": sum(value.get("resolved") is True for value in values),
        "attempted": len(values),
        "cost": round(sum(value.get("cost") or 0 for value in values), 6),
        "prompt_tokens": sum(value.get("prompt_tokens") or 0 for value in values),
        "completion_tokens": sum(value.get("completion_tokens") or 0 for value in values),
        "steps": sum(value.get("steps") or 0 for value in values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrong2", type=Path, required=True)
    parser.add_argument("--rest3", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    wrong2, rest3 = args.wrong2.resolve(), args.rest3.resolve()
    previous, reference = args.previous.resolve(), args.reference.resolve()
    fixed = []
    for run in (wrong2, rest3):
        for job in read(run / "jobs.json", []) or []:
            fixed.append((run, job))
    previous_jobs = {job["instance_id"]: job for job in read(previous / "jobs.json", []) or []}
    historical = {(row["instance_id"], row["variant"]): row
                  for row in (read(reference / "report-dev.json", {}) or {}).get("rows") or []
                  if row.get("variant") in {"a0", "a3"}}
    rows = []
    for run, job in fixed:
        identity = job["instance_id"]
        row = {"instance_id": identity, "run": str(run), "fixed": run_row(run, job)}
        previous_job = previous_jobs.get(identity)
        if previous_job:
            row["previous"] = run_row(previous, previous_job)
        for variant in ("a0", "a3"):
            old = historical.get((identity, variant))
            if old:
                row[variant] = {
                    "resolved": old.get("resolved"),
                    "cost": (old.get("usage") or {}).get("reference_cost"),
                    "prompt_tokens": (old.get("flow_usage") or {}).get("prompt_tokens"),
                    "completion_tokens": (old.get("flow_usage") or {}).get("completion_tokens"),
                    "steps": (old.get("policy_metrics") or {}).get("actual_tool_steps"),
                    "finish_reason": old.get("finish_reason"),
                }
        rows.append(row)
    result = {"wrong2": str(wrong2), "rest3": str(rest3), "previous": str(previous),
              "reference": str(reference), "rows": rows,
              "totals": {key: total(rows, key) for key in ("fixed", "previous", "a0", "a3")}}
    print(json.dumps(result["totals"], indent=1))
    print()
    header = f"{'instance':34} {'fixed':>18} {'previous':>18} {'A0':>16} {'A3':>16}"
    print(header)
    for row in rows:
        def cell(value):
            if not value:
                return "-"
            return f"{'R' if value.get('resolved') is True else ('E' if value.get('grading') == 'environment_error' else '-')} ${value.get('cost') or 0:.3f} {value.get('steps') or 0}s"
        print(f"{row['instance_id']:34} {cell(row['fixed']):>18} {cell(row.get('previous')):>18} {cell(row.get('a0')):>16} {cell(row.get('a3')):>16}")
    print()
    for row in rows:
        fixed_row = row["fixed"]
        if fixed_row.get("grading_failures") or fixed_row.get("tool_failures") or fixed_row.get("grading") == "environment_error":
            print(row["instance_id"], "grading=", fixed_row.get("grading_failures"),
                  "tools=", fixed_row.get("tool_failures"),
                  "final=", fixed_row.get("verification_state"), fixed_row.get("target"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
