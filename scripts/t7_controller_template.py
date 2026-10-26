"""One serial T7 controller: dev gate chain and ablation, then locked holdout.

Run only inside a prepared T7 candidate.  Every command is source-bound, its
exit code is recorded, and no failure is retried automatically.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from filelock import FileLock

from repoforge.evaluation import t7
from repoforge.evaluation.experiment import (
    append_json, load_manifest, now, read_json, sha, validate_execution_source,
    validate_gate, verify_answer, write_json,
)

R = Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv/bin/python")
manifest = load_manifest(R)
jobs = read_json(R / "jobs.json")
counts = {phase: 0 for phase in ("smoke", "cli_smoke", "dev", "holdout")}
progress = {
    "state": "running", "stage": None, "phase": "start", "completed": counts,
    "active_job": None, "process": None, "latest_log": None, "next_action": None,
    "failure_reason": None, "manifest_hash": manifest["manifest_hash"],
    "controller": {"pid": os.getpid(),
                   "start_ticks": Path("/proc/self/stat").read_text().split()[21]},
    "official_harness": "not_run", "official_resolved": None,
    "holdout_state": "locked",
}


class BudgetStop(RuntimeError):
    pass


def save(**changes):
    progress.update(changes)
    progress["timestamp"] = now()
    write_json(R / "progress.json", progress)


def check_pause():
    if (R / "pause-request.json").exists():
        raise RuntimeError("User pause: no next command may start")


def budget():
    """Count every completed generation ledger at its full known reference cost."""
    phases = {}
    complete, total, counted = True, 0.0, 0
    by_job = {job["job_id"]: job for job in jobs}
    for path in sorted((R / "runs").glob("*/usage.json")):
        usage = read_json(path)
        cost = usage.get("reference_cost")
        if cost is None:
            cost = usage.get("known_reference_cost")
        if usage.get("cost_complete") is not True or cost is None:
            complete = False
        else:
            total += cost
            counted += 1
        phase = by_job.get(path.parent.name, {}).get("phase", "unknown")
        phases[phase] = phases.get(phase, 0.0) + (cost or 0)
    return {"manifest_hash": manifest["manifest_hash"], "timestamp": now(),
            "spent_known_reference_cost": total, "cost_complete": complete,
            "counted_jobs": counted, "by_phase": phases, "provider_bill": "not_available"}


def guard(limit):
    state = budget()
    state["cost_stop"] = limit
    state["remaining"] = limit - state["spent_known_reference_cost"]
    write_json(R / "budget.json", state)
    save(budget=state)
    if not state["cost_complete"]:
        raise BudgetStop("Unknown SDK cost; the budget guard cannot continue")
    if state["spent_known_reference_cost"] >= limit:
        raise BudgetStop(f"Cost stop reached: {state['spent_known_reference_cost']:.4f} >= {limit}")


def heartbeat(job):
    state = {}
    if job is None:
        ledgers = list(Path(manifest["native"]["environment_root"]).glob(
            "instances/*/native-commands.jsonl"))
        if ledgers:
            latest = max(ledgers, key=lambda path: path.stat().st_mtime_ns)
            lines = latest.read_text().splitlines()
            if lines:
                try:
                    state["public_preparation_last_command"] = json.loads(lines[-1])
                except json.JSONDecodeError:
                    state["public_preparation_last_command"] = "partial line"
        save(live_ledger=state)
        return
    directory = R / "runs" / job["job_id"]
    for label, path in (("sdk", directory / "sdk.jsonl"), ("events", directory / "events.jsonl"),
                        ("index_sdk", R / "index-builds" / job["instance_id"] / "sdk.jsonl")):
        if path.exists():
            lines = path.read_text().splitlines()
            state[label + "_records"] = len(lines)
            if lines:
                try:
                    record = json.loads(lines[-1])
                    state[label + "_last"] = {key: record[key] for key in (
                        "event", "kind", "status", "timestamp", "operation", "request_id", "node_id")
                        if key in record}
                except json.JSONDecodeError:
                    state[label + "_last"] = "partial line"
    save(live_ledger=state)


def command(stage, arguments, job=None, module="scripts.t4_t5_experiment"):
    check_pause()
    validate_execution_source(manifest, ROOT)
    argv = [PYTHON, "-m", module, *arguments, "--output", str(R)]
    command_id = f"{stage}-{time.time_ns()}"
    log = R / "logs" / (command_id + ".log")
    log.parent.mkdir(exist_ok=True)
    started, clock = now(), time.monotonic()
    save(active_job=job["job_id"] if job else None, next_action=stage,
         latest_log=str(log.relative_to(R)))
    environment = dict(os.environ)
    environment.update(HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with log.open("w") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        identity = {"pid": process.pid,
                    "start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split()[21]}
        row = {"tool": "exec_command/subprocess", "command_id": command_id,
               "stage": stage, "argv": argv, "cwd": str(ROOT),
               "manifest_hash": manifest["manifest_hash"],
               "source_sha256": manifest["source"]["sha256"],
               "job_id": job["job_id"] if job else None,
               "phase": job["phase"] if job else None,
               "log": str(log.relative_to(R)), "started": started, "process": identity}
        append_json(R / "commands.jsonl", {**row, "event": "started"})
        save(process=identity)
        while process.poll() is None:
            heartbeat(job)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        return_code = process.returncode
    row.update(event="completed", exit_code=return_code, ended=now(),
               wall_seconds=time.monotonic() - clock, log_sha256=sha(log))
    append_json(R / "commands.jsonl", row)
    save(process=None)
    print(json.dumps(row), flush=True)
    if return_code:
        raise RuntimeError(f"{stage} failed exit {return_code}; reconcile original effects "
                           f"before any retry; log={log}")


def public_fingerprint_preflight():
    from types import SimpleNamespace
    from repoforge.flow.durable import workspace_fingerprint
    smoke_ids = {job["instance_id"] for job in jobs if job["phase"] == "smoke"}
    rows = []
    for identity, prepared in read_json(R / "provision-native.json").items():
        if identity not in smoke_ids:
            continue
        workspace = SimpleNamespace(repository=SimpleNamespace(
            repo_path=prepared["environment"]["repo"], shadow_mode=True),
            runtime=None, _environment=None)
        started = time.monotonic()
        first = workspace_fingerprint(workspace, str(R / "excluded-preflight.sqlite"))
        second = workspace_fingerprint(workspace, str(R / "excluded-preflight.sqlite"))
        if first != second:
            raise ValueError("Public workspace changed during fingerprint preflight: " + identity)
        rows.append({"instance_id": identity,
                     "content_hash": first["repository"]["content_hash"],
                     "seconds": time.monotonic() - started})
    write_json(R / "fingerprint-preflight.json", {
        "manifest_hash": manifest["manifest_hash"], "rows": rows,
        "passed": len(rows) == 5, "provider_calls": 0})
    if len(rows) != 5:
        raise ValueError("All five public smoke workspaces must pass before any generation")


def run_dev_job(job):
    check_pause()
    directory = R / "runs" / job["job_id"]
    required = ("answer.json", "grading.json", "own-tests.json")
    if all((directory / name).exists() for name in required):
        row = t7.accept_job(R, manifest, job)
        row["status"] = "verified_existing"
    else:
        if (directory / "generation-worker-started.json").exists() and not resume:
            raise RuntimeError("Prior generation effects need reconciliation: " + job["job_id"])
        arguments = ["run-phase", "--phase", job["phase"], "--job-id", job["job_id"]]
        if resume:
            arguments.append("--resume")
        command("run-" + job["phase"], arguments, job)
        row = t7.accept_job(R, manifest, job)
        row["status"] = "verified"
    append_json(R / "verification-jobs.jsonl", row)
    counts[job["phase"]] += 1
    save(completed=counts, active_job=None, live_ledger={}, next_action="next job")
    print(json.dumps(row), flush=True)


def run_holdout_command(job):
    check_pause()
    directory = R / "runs" / job["job_id"]
    if (directory / "answer.json").exists() and (directory / "grading.json").exists():
        row = t7.accept_job(R, manifest, job, require_locked=True)
        row["status"] = "verified_existing"
    else:
        arguments = ["run-holdout", "--job-id", job["job_id"]]
        if resume:
            arguments.append("--resume")
        command("run-holdout", arguments, job, module="scripts.t7_experiment")
        row = t7.accept_job(R, manifest, job, require_locked=True)
        row["status"] = "verified"
    append_json(R / "verification-jobs.jsonl", row)
    counts["holdout"] += 1
    save(completed=counts, active_job=None, live_ledger={}, next_action="next job")
    print(json.dumps(row), flush=True)


def dev_stage(cost_stop):
    save(stage="dev", phase="gate", next_action="preflight")
    if not (R / "preflight.json").exists() or not read_json(R / "preflight.json").get("passed"):
        command("preflight", ["preflight"])
    command("provision-smoke", ["provision", "--phase", "smoke"])
    public_fingerprint_preflight()
    first = next(job for job in jobs if job["phase"] == "smoke"
                 and job["instance_id"] == "pytest-dev__pytest-7490")
    command("prepare-index", ["prepare-index", "--job-id", first["job_id"]], first)
    reference = read_json(R / "index-references" / (first["instance_id"] + ".json"))
    if reference["current_job_embedding_build"] != "reused":
        raise ValueError("First pytest index was unexpectedly rebuilt")
    for stage in ("contracts", "migration", "isolation", "recovery", "cli-boundaries"):
        command(stage, [] if stage == "migration" else [stage],
                module="scripts.native_migration_audit" if stage == "migration"
                else "scripts.t4_t5_experiment")
    for phase in ("smoke", "cli_smoke"):
        save(phase=phase)
        for job in jobs:
            if job["phase"] == phase:
                run_dev_job(job)
                guard(cost_stop)
        command("report-" + phase, ["report", "--phase", phase])
    if (R / "native-gate.json").exists():
        validate_gate(R, manifest)
    else:
        command("gate", ["gate"])
        validate_gate(R, manifest)
    write_json(R / "independent-gate-check.json", {
        "timestamp": now(), "manifest_hash": manifest["manifest_hash"],
        "source_verified": True, "gate_validated": True})
    save(phase="dev_preparation", next_action="provision dev")
    command("provision-dev", ["provision", "--phase", "dev"])
    ordered_dev = [*t7.SMOKE_IDS, *[identity for identity in manifest["split"]["dev"]
                                    if identity not in set(t7.SMOKE_IDS)]]
    prepared_jobs = {}
    for identity in ordered_dev:
        job = next(item for item in jobs if item["phase"] == "dev"
                   and item["instance_id"] == identity)
        prepared_jobs[identity] = job
        command("prepare-dev-index", ["prepare-index", "--phase", "dev",
                                      "--job-id", job["job_id"]], job)
    rows = []
    for identity in ordered_dev:
        path = R / "index-references" / (identity + ".json")
        if not path.exists():
            raise ValueError("Missing dev index reference: " + identity)
        reference = read_json(path)
        if reference.get("current_job_embedding_build") != "reused":
            raise ValueError("Dev index was rebuilt instead of reused: " + identity)
        rows.append({"instance_id": identity, "status": "reused",
                     "index": reference["index"], "digest": reference["digest"]})
    write_json(R / "index-coverage-at-dev-start.json", {
        "manifest_hash": manifest["manifest_hash"], "available_indexes": len(rows),
        "rebuilt_indexes": 0, "rows": rows})
    if len(rows) != 30:
        raise ValueError("All 30 exact-version dev indexes must be reused")
    save(phase="dev", next_action="150 dev jobs")
    for job in jobs:
        if job["phase"] != "dev":
            continue
        run_dev_job(job)
        guard(cost_stop)
    command("report-dev", ["report-dev"], module="scripts.t7_experiment")
    save(state="analysis_pending", stage="dev", phase="analysis", active_job=None,
         process=None, next_action="t7_analysis label/analyze --phase dev; review report.md, "
                                   "paired and recommendation; then lock --winner aN --confirm-user")


def holdout_stage(cost_stop):
    lock = t7.validate_lock(R, manifest)
    save(stage="holdout", phase="holdout_preparation", lock_hash=lock["lock_hash"],
         next_action="provision holdout")
    scan_path = R / t7.HOLDOUT_SCAN
    adjudication_path = R / t7.HOLDOUT_ADJUDICATION
    if not (R / "holdout-provision-summary.json").exists():
        command("provision-holdout", ["provision-holdout"], module="scripts.t7_experiment")
    if not scan_path.exists():
        command("scan-holdout", ["scan-holdout"], module="scripts.t7_experiment")
    request = R / "holdout-adjudication-request.json"
    current_scan = sha(scan_path) if scan_path.exists() else None
    adjudicated = False
    if adjudication_path.exists() and request.exists():
        adjudicated = (read_json(request).get("scan_sha256") == current_scan
                       and read_json(adjudication_path).get("scan_sha256") == current_scan
                       and read_json(adjudication_path).get("lock_hash") == lock["lock_hash"])
    if not adjudicated:
        save(state="holdout_adjudication_pending", phase="holdout_adjudication",
             active_job=None, process=None,
             next_action="Review holdout-scan.json and pending registrations; no task is replaced. "
                         "Run: python -m scripts.t7_experiment adjudicate --output <candidate> "
                         "--confirm-user")
        print(json.dumps({"state": "holdout_adjudication_pending",
                          "scan_sha256": current_scan,
                          "request": str(request)}), flush=True)
        return
    command("prepare-holdout-indexes", ["prepare-holdout-indexes"],
            module="scripts.t7_experiment")
    coverage = read_json(R / "holdout-index-coverage.json")
    blocked = set(coverage.get("failed", []))
    save(phase="holdout", next_action="200 locked holdout jobs")
    receipts = read_json(R / "provision-native.json")
    order = []
    for identity in manifest["split"]["holdout"]:
        arms = [job for job in jobs if job["phase"] == "holdout"
                and job["instance_id"] == identity
                and job["variant"] in {lock["winner"], lock["control"]}]
        arms.sort(key=lambda job: 0 if job["variant"] == lock["winner"] else 1)
        order.extend(arms)
    for job in order:
        if job["instance_id"] not in receipts or job["instance_id"] in blocked:
            append_json(R / "holdout-skipped.jsonl", {
                "timestamp": now(), "job_id": job["job_id"], "variant": job["variant"],
                "instance_id": job["instance_id"], "reason": "registered environment blocked"})
            continue
        run_holdout_command(job)
        guard(cost_stop)
    command("report-holdout", ["report-holdout"], module="scripts.t7_experiment")
    save(state="complete", phase="holdout", active_job=None, process=None,
         next_action="t7_analysis label/analyze --phase holdout; documentation and final report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dev", "holdout"])
    parser.add_argument("--cost-stop", type=float)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    global resume
    resume = arguments.resume
    if arguments.check:
        validate_execution_source(manifest, ROOT)
        state = t7.status(R)
        if arguments.stage == "holdout":
            state["lock_validated"] = bool(t7.validate_lock(R, manifest)["lock_hash"])
        state["provider_calls"] = 0
        print(json.dumps(state, ensure_ascii=False))
        return
    if arguments.stage is None:
        raise ValueError("--stage dev|holdout is required unless --check is used")
    if arguments.cost_stop is None:
        arguments.cost_stop = 10.5 if arguments.stage == "dev" else 19.5
    with FileLock(str(R / "controller.lock"), timeout=0):
        try:
            save(stage=arguments.stage, state="running")
            guard(arguments.cost_stop)
            if arguments.stage == "dev":
                dev_stage(arguments.cost_stop)
            else:
                holdout_stage(arguments.cost_stop)
        except BudgetStop as exc:
            save(state="budget_stopped", failure_reason=str(exc),
                 next_action="Review budget.json; no automatic overspend or retry")
            raise
        except BaseException as exc:
            save(state="stopped_on_error",
                 failure_reason=f"{type(exc).__name__}: {exc}",
                 next_action="Inspect the original command, worker and SDK ledger; "
                             "no automatic replay")
            raise


if __name__ == "__main__":
    main()
