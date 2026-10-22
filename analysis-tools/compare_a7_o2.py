#!/usr/bin/env python3
"""Pair A7 sanity rows with the historical O2 (a5) fixed-combination rows.

Reads the A7 sanity aggregate and the frozen O2 baseline JSON.  Does not run
models, tests or grading; it only formats existing public evidence.
"""

import argparse
import json
from pathlib import Path


def load_rows(path):
    data = json.loads(Path(path).read_text())
    return {row["instance_id"]: row for row in data.get("rows", [])}


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a7", type=Path, required=True,
                        help="A7 sanity-aggregate.json (or sanity-report.json)")
    parser.add_argument("--o2", type=Path,
                        default=Path(__file__).parent / "research/a7-implementation/o2-fixed-baseline.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    a7 = load_rows(args.a7)
    o2 = load_rows(args.o2)
    rows = []
    for instance_id in sorted(set(a7) | set(o2)):
        new = a7.get(instance_id) or {}
        old = o2.get(instance_id) or {}
        new_cost, old_cost = number(new.get("cost")), number(old.get("cost"))
        new_steps, old_steps = number(new.get("steps")), number(old.get("steps"))
        rows.append({
            "instance_id": instance_id,
            "a7_resolved": new.get("resolved") is True,
            "o2_resolved": old.get("resolved") is True,
            "a7_cost": new_cost,
            "o2_cost": old_cost,
            "a7_steps": new_steps,
            "o2_steps": old_steps,
            "a7_completion": new.get("completion_status"),
            "a7_finish_reason": new.get("finish_reason"),
            "a7_error": new.get("error"),
        })
    total = {
        "a7_resolved": sum(row["a7_resolved"] for row in rows),
        "o2_resolved": sum(row["o2_resolved"] for row in rows),
        "a7_cost": round(sum(row["a7_cost"] or 0 for row in rows), 6),
        "o2_cost": round(sum(row["o2_cost"] or 0 for row in rows), 6),
        "a7_steps": sum(row["a7_steps"] or 0 for row in rows),
        "o2_steps": sum(row["o2_steps"] or 0 for row in rows),
    }
    print("| instance | A7 resolved | O2 resolved | A7 cost | O2 cost | A7 steps | O2 steps | A7 completion |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(f"| {row['instance_id']} | {row['a7_resolved']} | {row['o2_resolved']} | "
              f"{row['a7_cost']} | {row['o2_cost']} | {row['a7_steps']} | {row['o2_steps']} | "
              f"{row['a7_completion']} |")
    print()
    print(json.dumps(total, indent=1))
    if args.output:
        args.output.write_text(json.dumps({"rows": rows, "total": total}, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
