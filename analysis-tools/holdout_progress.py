#!/usr/bin/env python3
"""Compact T7 holdout progress screen (stdlib only).

Reads run directories and the controller ledgers; never writes to the
candidate.  Designed for ``watch -n 15 -d``.
"""

import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

C = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()


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


def mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


def age(value):
    return max(time.time() - value, 0) if value else None


def fmt_age(seconds):
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def main():
    manifest = read_json(C / "manifest.json", {}) or {}
    jobs = {job["job_id"]: job for job in read_json(C / "jobs.json", []) or []}
    lock = read_json(C / "t7-lock.json", {}) or {}
    coverage = read_json(C / "holdout-index-coverage.json", {}) or {}
    progress = read_json(C / "progress.json", {}) or {}
    budget = read_json(C / "budget.json", {}) or {}
    winner, control = lock.get("winner", "a3"), lock.get("control", "a0")
    clean = coverage.get("clean") or []
    non_clean = coverage.get("non_clean") or []
    expected = len(clean) * 2
    holdout_ids = {job["job_id"] for job in jobs.values()
                   if job.get("phase") == "holdout" and job.get("variant") in {winner, control}}

    verified = {}
    for row in read_jsonl(C / "verification-holdout.jsonl"):
        if row.get("job_id") in holdout_ids:
            verified[row["job_id"]] = row

    completed = []
    for job_id, row in verified.items():
        job = jobs[job_id]
        directory = C / "runs" / job_id
        start = mtime(directory / "generation-worker-started.json")
        end = mtime(directory / "grading.json") or mtime(directory / "answer.json")
        usage = read_json(directory / "usage.json", {}) or {}
        grade = read_json(directory / "grading.json", {}) or {}
        completed.append({
            "job_id": job_id, "job": job, "row": row,
            "start": start, "end": end,
            "minutes": (end - start) / 60 if start and end else None,
            "cost": usage.get("reference_cost"),
            "resolved": grade.get("local_resolved"),
        })
    completed.sort(key=lambda item: item["end"] or 0)
    by_variant = Counter(item["job"]["variant"] for item in completed)
    resolved = Counter(item["job"]["variant"] for item in completed
                       if item["resolved"] is True)
    total_by_variant = Counter({winner: len(clean), control: len(clean)})

    active_id = progress.get("active_job")
    active = jobs.get(active_id) if active_id else None
    stage = "-"
    active_elapsed = None
    sdk_lines = 0
    traj = 0
    last_event = None
    if active_id:
        directory = C / "runs" / active_id
        start = mtime(directory / "generation-worker-started.json")
        active_elapsed = age(start)
        if start:
            if not (directory / "answer.json").exists():
                stage = "generate"
            elif not (directory / "own-tests.json").exists():
                stage = "own-tests"
            elif not (directory / "grading.json").exists():
                stage = "grade"
            elif not (directory / "disposable-release.jsonl").exists():
                stage = "release"
            else:
                stage = "accept"
        else:
            stage = "starting"
        try:
            sdk_lines = len((directory / "sdk.jsonl").read_text().splitlines())
        except OSError:
            pass
        try:
            traj = (directory / "trajectory.json").stat().st_size
        except OSError:
            pass
        events = read_jsonl(directory / "events.jsonl")
        if events:
            last = events[-1]
            last_event = (last.get("event") or last.get("kind") or "-")

    now = time.time()
    recent = [item["minutes"] for item in completed[-10:] if item["minutes"]]
    rate = sum(recent) / len(recent) if recent else None
    remaining = max(expected - len(completed), 0)
    eta = rate * remaining if rate else None
    finish = datetime.fromtimestamp(now + eta * 60) if eta else None

    skipped = {row.get("job_id") for row in read_jsonl(C / "holdout-skipped.jsonl")}
    skipped = len(skipped & holdout_ids)
    expected_skips = len(non_clean) * 2
    spent = budget.get("spent_known_reference_cost") or 0
    guard = budget.get("cost_stop") or 0
    holdout_spent = (budget.get("by_phase") or {}).get("holdout", 0)
    holdout_costs = [item["cost"] for item in completed if item["cost"] is not None]
    avg_cost = sum(holdout_costs) / len(holdout_costs) if holdout_costs else None
    projected = spent + (avg_cost * remaining if avg_cost else 0)
    override = read_json(C / "budget-override-holdout-2.json", {}) or {}
    approved = override.get("approved_cost_stop")

    pct = 100.0 * len(completed) / expected if expected else 0
    filled = int(pct / 100 * 30)
    print(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    print(f"clean holdout {len(completed)}/{expected}  ({pct:5.1f}%)  "
          f"a3 {by_variant.get(winner, 0)}/{total_by_variant.get(winner, 0)}  "
          f"a0 {by_variant.get(control, 0)}/{total_by_variant.get(control, 0)}")
    print(f"[{'#' * filled}{'-' * (30 - filled)}]  "
          f"resolved {winner}={resolved.get(winner, 0)} {control}={resolved.get(control, 0)}  "
          f"excluded non-clean {len(non_clean)} ({skipped}/{expected_skips} arms skipped)")

    if active:
        print(f"\ncurrent: {active['instance_id']} / {active['variant']}  "
              f"stage={stage}  elapsed={fmt_age(active_elapsed)}")
        print(f"  sdk records={sdk_lines}  trajectory={traj / 1024:.1f}KB  "
              f"last event={last_event or '-'}")
    else:
        print("\ncurrent: between jobs" if progress.get("state") == "running" else
              f"\ncurrent: none  state={progress.get('state')}")

    print("\nlast completed:")
    for item in completed[-6:]:
        job = item["job"]
        minutes = f"{item['minutes']:.1f}m" if item["minutes"] else "-"
        cost = f"${item['cost']:.4f}" if item["cost"] is not None else "-"
        print(f"  {job['instance_id']:<40} {job['variant']}  {minutes:>6}  "
              f"resolved={item['resolved']}  {cost}")

    if rate:
        print(f"\nrate: {rate:.1f} min/job (last {len(recent)})  remaining={remaining}  "
              f"ETA={eta / 60:.1f}h  finish~{finish:%m-%d %H:%M}")
    else:
        print(f"\nremaining={remaining}  ETA=waiting for the first timed job")

    print(f"budget: ${spent:.4f}/${guard:.1f}  remaining=${guard - spent:.4f}  "
          f"holdout=${holdout_spent:.4f}"
          + (f"  avg=${avg_cost:.4f}/job  projected=${projected:.2f}" if avg_cost else ""))
    if approved and spent < guard and projected and projected > guard:
        print(f"  approved override -> ${approved:.1f} on restart after the ${guard:.1f} guard")

    try:
        free = shutil.disk_usage(C).free
        print(f"disk free: {free / 1024 ** 3:.1f}GB  controller: "
              + ("RUNNING" if progress.get("state") == "running" else str(progress.get("state"))))
    except OSError:
        print("disk free: unknown")
    if progress.get("failure_reason"):
        print(f"failure_reason: {progress['failure_reason'][:180]}")


if __name__ == "__main__":
    main()
