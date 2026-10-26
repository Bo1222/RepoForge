"""T7 policy-ablation runner: offline checks, preparation, lock and holdout."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
from filelock import FileLock

from repoforge.evaluation.experiment import (
    aggregate, freeze_source, load_manifest, now, read_json, sha,
    validate_execution_source, write_json,
)
from repoforge.evaluation import t7

SUITES = ["tests/actions/test_t6_tool_contracts.py", "tests/policy", "tests/memory",
          "tests/flow", "tests/agent/test_public_steps.py", "tests/cli/test_managed_sessions.py",
          "tests/evaluation/test_t6_experiment.py", "tests/evaluation/test_t7_experiment.py",
          "tests/evaluation/test_native_experiment.py",
          "tests/evaluation/test_native_configuration_audit.py",
          "tests/evaluation/test_t4_t5_experiment.py"]


def check(root, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-v", "-o", "log_cli=false",
               "-o", "faulthandler_timeout=60", *SUITES]
    with (destination / "pytest.log").open("w") as stream:
        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    with tempfile.TemporaryDirectory() as tmp:
        source = freeze_source(root, Path(tmp))
    report = {"timestamp": now(), "command": command, "cwd": str(root),
              "return_code": result.returncode, "passed": result.returncode == 0,
              "source_digest": source["digest"], "source_files": source["files"],
              "test_files": {str(path.relative_to(root)): sha(path)
                             for path in (root / "tests").rglob("*.py")},
              "log": str(destination / "pytest.log"),
              "log_sha256": sha(destination / "pytest.log"),
              "provider_calls": 0}
    write_json(destination / "offline-check.json", report)
    return report


def validate_offline(root, report, manifest=None):
    if not report.get("passed") or report.get("return_code") != 0 or report.get("provider_calls") != 0:
        raise ValueError("T7 requires successful offline verification")
    for path, expected in {**report["source_files"], **report["test_files"]}.items():
        if sha(root / path) != expected:
            raise ValueError("Source/test changed after offline verification: " + path)
    if manifest and report["source_digest"] != manifest["source"]["digest"]:
        raise ValueError("Offline evidence belongs to a different frozen source")


def _job(output, job_id):
    jobs = read_json(Path(output) / "jobs.json")
    match = [job for job in jobs if job["job_id"] == job_id]
    if len(match) != 1:
        raise ValueError("Unknown or ambiguous job_id")
    return match[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "check", "prepare", "audit", "status", "lock", "provision-holdout",
        "prepare-holdout-indexes", "scan-holdout", "adjudicate", "run-holdout",
        "accept", "report-dev", "report-holdout", "policy-audit"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
        default=Path("experiments/results/T4_T5_bge_a0_40iter_1usd_20261016T074244Z"))
    parser.add_argument("--offline-report", type=Path,
        default=Path("experiments/results/T7_implementation_20261018/check/offline-check.json"))
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--index-settings", type=Path)
    parser.add_argument("--build-wheels", type=Path)
    parser.add_argument("--public-tools", type=Path)
    parser.add_argument("--analysis-plan", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--phase", choices=["dev", "holdout"])
    parser.add_argument("--winner")
    parser.add_argument("--confirm-user", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--note")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    output = args.output.resolve()

    if args.command == "check":
        report = check(root, output)
        print({"passed": report["passed"], "return_code": report["return_code"],
               "report": str(output / "offline-check.json")})
        raise SystemExit(report["return_code"])
    if args.command == "prepare":
        offline = read_json(args.offline_report)
        validate_offline(root, offline)
        manifest = t7.prepare(
            root, output, args.reference, environment_root=args.environment_root,
            index_settings=args.index_settings, build_wheels=args.build_wheels,
            public_tools=args.public_tools, analysis_plan=args.analysis_plan)
        validate_offline(root, offline, manifest)
        write_json(output / "offline-check.json", offline)
        print({"manifest_hash": manifest["manifest_hash"], "jobs": 670,
               "protocol": t7.PROTOCOL, "holdout_state": "locked"})
        return

    manifest = load_manifest(output)
    if args.command == "audit":
        audit = t7.job_audit(output, manifest)
        print({"total": audit["total"], "counts": audit["counts"],
               "holdout_state": audit["holdout_state"]})
        return
    if args.command == "status":
        print(t7.status(output))
        return
    if args.command == "report-dev":
        report = aggregate(output, "dev")
        print(report["groups"])
        return
    validate_execution_source(manifest, root)

    if args.command == "lock":
        if not args.winner:
            raise ValueError("lock requires --winner")
        lock = t7.create_lock(output, manifest, winner=args.winner,
                              confirm_user=args.confirm_user, note=args.note)
        print({"winner": lock["winner"], "control": lock["control"],
               "lock_hash": lock["lock_hash"]})
        return
    if args.command == "provision-holdout":
        print(t7.provision_holdout(output, manifest))
        return
    if args.command == "prepare-holdout-indexes":
        print(t7.prepare_holdout_indexes(output, manifest))
        return
    if args.command == "scan-holdout":
        print(t7.scan_holdout(output, manifest)["counts"])
        return
    if args.command == "adjudicate":
        print(t7.adjudicate(output, manifest, confirm_user=args.confirm_user))
        return
    if args.command == "run-holdout":
        job = _job(output, args.job_id)
        print(t7.run_holdout_job(output, manifest, job, resume=args.resume))
        return
    if args.command == "accept":
        job = _job(output, args.job_id)
        if args.phase and job["phase"] != args.phase:
            raise ValueError("Selected job is not in the requested phase")
        print(t7.accept_job(output, manifest, job, require_locked=job["phase"] == "holdout"))
        return
    if args.command == "report-holdout":
        report = t7.holdout_report(output, manifest)
        print({"holdout_state": report["holdout_state"],
               "arms": report["arms"], "executed": report["executed_arms"]})
        return
    if args.command == "policy-audit":
        report = t7.policy_audit(output, manifest)
        print({"passed": report["passed"], "variants": sorted(report["totals"])})
        return
    raise ValueError("Unhandled command")


if __name__ == "__main__":
    main()
