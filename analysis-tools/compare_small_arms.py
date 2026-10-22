#!/usr/bin/env python3
"""Compare a small O2 run against the historical T7 A0/A3 rows.

This is an offline reader: it makes no model calls and does not modify any
candidate.  It is intended for the 5-instance smoke comparison only.
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


def text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def tool_failures(trajectory_path):
    data = read(trajectory_path, {}) or {}
    counts = Counter()
    for node in data.get("nodes") or []:
        for step in node.get("action_steps") or []:
            observation = step.get("observation") or {}
            reason = (observation.get("properties") or {}).get("fail_reason")
            if reason:
                counts[reason] += 1
    return dict(counts)


def grading_failures(grade):
    status = grade.get("tests_status") or {}
    result = {}
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = status.get(key) or {}
        failures = value.get("failure") if isinstance(value, dict) else None
        if failures:
            result[key] = list(failures)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o2", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    o2, reference = args.o2.resolve(), args.reference.resolve()
    jobs = read(o2 / "jobs.json", []) or []
    ids = [job["instance_id"] for job in jobs]
    report = read(reference / "report-dev.json", {}) or {}
    historical = {(row["instance_id"], row["variant"]): row
                  for row in report.get("rows") or []
                  if row.get("instance_id") in ids and row.get("variant") in {"a0", "a3"}}

    rows = []
    for job in jobs:
        identity = job["instance_id"]
        directory = o2 / "runs" / job["job_id"]
        grade = read(directory / "grading.json", {}) or {}
        usage = read(directory / "usage.json", {}) or {}
        generation = read(directory / "generation.json", {}) or {}
        decisions = read(directory / "policy-decisions.json", []) or []
        last = decisions[-1] if decisions else {}
        state = last.get("state") or {}
        row = {
            "instance_id": identity,
            "o2": {
                "resolved": grade.get("local_resolved"),
                "grading": grade.get("local_grading"),
                "finish_reason": generation.get("finish_reason"),
                "generation_status": generation.get("generation_status"),
                "cost": usage.get("reference_cost"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "steps": (generation.get("policy_metrics") or {}).get("actual_tool_steps"),
                "decisions": len(decisions),
                "verification_state": state.get("verification_state"),
                "introduced_failures": state.get("introduced_failures") or [],
                "tool_failures": tool_failures(directory / "trajectory.json"),
                "grading_failures": grading_failures(grade),
                "patch_applicable": grade.get("patch_applicable"),
                "submit_product_only": generation.get("submit_product_only"),
                "submitted_verified_patch": generation.get("submitted_verified_patch"),
            },
        }
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

    def total(arm):
        values = [row[arm] for row in rows if arm in row]
        return {
            "resolved": sum(value.get("resolved") is True for value in values),
            "attempted": len(values),
            "cost": round(sum(value.get("cost") or 0 for value in values), 6),
            "prompt_tokens": sum(value.get("prompt_tokens") or 0 for value in values),
            "completion_tokens": sum(value.get("completion_tokens") or 0 for value in values),
            "steps": sum(value.get("steps") or 0 for value in values),
        }

    result = {"o2": str(o2), "reference": str(reference), "rows": rows,
              "totals": {arm: total(arm) for arm in ("o2", "a0", "a3")}}
    print(json.dumps(result["totals"], indent=1))
    print()
    header = f"{'instance':34} {'O2':>18} {'A0':>16} {'A3':>16}"
    print(header)
    for row in rows:
        def cell(value):
            if not value:
                return "-"
            return f"{'R' if value.get('resolved') is True else '-'} ${value.get('cost') or 0:.3f} {value.get('steps') or 0}s"
        print(f"{row['instance_id']:34} {cell(row['o2']):>18} {cell(row.get('a0')):>16} {cell(row.get('a3')):>16}")
    print()
    for row in rows:
        o2row = row["o2"]
        if o2row.get("tool_failures") or o2row.get("grading_failures"):
            print(row["instance_id"], "tools=", o2row.get("tool_failures"),
                  "grading=", o2row.get("grading_failures"),
                  "verif=", o2row.get("verification_state"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
