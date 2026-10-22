"""T4/T5 staged experiment CLI. No reference material is passed to generation workers."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from filelock import FileLock

from moatless.evaluation.experiment import aggregate, append_json, freeze_hypothesis, load_manifest, now, prepare, read_json, validate_execution_source, validate_gate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "preflight", "provision", "prepare-index", "contracts", "cli-boundaries", "isolation", "recovery", "gate", "run-phase", "generate", "own-tests", "baseline", "grade", "release", "report", "select"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["smoke", "dev", "cli_smoke"], default="smoke")
    parser.add_argument("--job-id", help="Select one frozen job for generation, grading or resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend", choices=["docker", "native"], help="Preparation backend; later commands infer it from the frozen manifest")
    parser.add_argument("--environment-root", type=Path, help="Native per-instance preparation directory (frozen at prepare)")
    parser.add_argument("--index-settings", type=Path, help="Explicit JSON index settings, frozen only when preparing a new candidate")
    parser.add_argument("--build-wheels", type=Path, help="Public offline build wheel directory, frozen at prepare")
    parser.add_argument("--public-tools", type=Path, help="Public native tool profile, frozen at prepare")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    if args.command == "prepare":
        from moatless.index.settings import IndexSettings
        settings = IndexSettings(**(read_json(args.index_settings) if args.index_settings else {}))
        embedding_service = None
        if args.backend == "native" and settings.embed_provider == "local":
            from moatless.evaluation.bge_identity import capture_identity
            embedding_service = capture_identity(root, settings.embed_base_url)
        manifest, jobs = prepare(root, args.output, execution_backend=args.backend or "docker",
                                 environment_root=args.environment_root, embedding_service=embedding_service,
                                 embedding_settings=settings.model_dump(mode="json"),
                                 build_wheels=args.build_wheels, public_tools=args.public_tools)
        print(f"Prepared {len(jobs)} jobs, manifest={manifest['manifest_hash']}; provider calls=0")
        return
    manifest = load_manifest(args.output)
    if args.index_settings or args.build_wheels or args.public_tools:
        raise ValueError("Index settings and native dependencies cannot change after preparation; create a new candidate")
    backend = manifest.get("execution_backend", "docker")
    if args.backend and args.backend != backend:
        raise ValueError("Requested backend differs from the frozen manifest")
    if args.command == "preflight":
        if backend == "native":
            from moatless.evaluation.experiment_native import preflight
        else:
            from scripts.context_memory_experiment import preflight
        checks = preflight(root)
        # New entrypoint provisions its own per-instance containers/indexes.
        checks["blockers"] = [item for item in checks["blockers"] if "Current batch entrypoint" not in item]
        checks["manifest_hash"] = manifest["manifest_hash"]
        write_json(args.output / "preflight.json", checks)
        print(checks)
        if checks["blockers"]:
            raise SystemExit(2)
        return
    if args.command == "report":
        report = aggregate(args.output, args.phase)
        print(report["groups"])
        return
    if args.command == "select":
        validate_gate(args.output, manifest)
        print(freeze_hypothesis(args.output))
        return
    if args.command == "gate":
        if backend != "native":
            raise ValueError("This gate builder requires the native backend")
        from moatless.evaluation.native_gate_scoped import build_scoped_native_gate
        print(build_scoped_native_gate(args.output, manifest)["local_research_acceptance"])
        return
    if args.phase == "dev":
        validate_gate(args.output, manifest)
    validate_execution_source(manifest, root)
    if backend == "native" and manifest["embedding_settings"]["embed_provider"] == "local":
        from moatless.evaluation.bge_identity import validate_identity
        health = validate_identity(root, manifest)
        append_json(args.output / "service-health.jsonl", {"timestamp": now(), "command": args.command,
                    "manifest_hash": manifest["manifest_hash"], "health": health})
    with FileLock(str(args.output / "scheduler.lock"), timeout=0):
        if backend == "native":
            from moatless.evaluation.experiment_native import baseline_job, grade_job, probe_recovery, provision, run_job, validate_agent_tests
        else:
            from moatless.evaluation.experiment_docker import baseline_job, grade_job, probe_recovery, provision, run_job, validate_agent_tests
        if args.command == "provision":
            if backend == "native":
                provision(args.output, manifest, args.phase, args.job_id)
            else:
                provision(args.output, manifest, args.phase)
            return
        if args.command == "recovery":
            probe_recovery(args.output, manifest)
            return
        if args.command == "contracts":
            if backend != "native":
                raise ValueError("The namespace worker contract command requires the native backend")
            from moatless.evaluation.experiment_native import probe_contracts
            probe_contracts(args.output, manifest)
            return
        if args.command == "cli-boundaries":
            if backend != "native":
                raise ValueError("Native CLI boundary evidence requires the native backend")
            from moatless.evaluation.experiment_native import probe_cli_boundaries
            probe_cli_boundaries(args.output, manifest)
            return
        if args.command == "isolation":
            if backend != "native":
                raise ValueError("Native isolation evidence requires the native backend")
            from moatless.evaluation.experiment_native import probe_isolation
            probe_isolation(args.output, manifest)
            return
        jobs = [job for job in read_json(args.output / "jobs.json") if job["phase"] == args.phase
                and (not args.job_id or job["job_id"] == args.job_id)]
        if not jobs:
            raise ValueError("No matching frozen job")
        if args.resume and len(jobs) != 1:
            raise ValueError("Resume requires exactly one --job-id")
        failures = []
        for job in jobs:
            stage = args.command
            try:
                if args.command == "run-phase":
                    if backend != "native":
                        raise ValueError("The staged native phase runner requires the native backend")
                    from moatless.evaluation.experiment_native import release_job_workspace
                    operations = [("generate", lambda: run_job(args.output, manifest, job, resume=args.resume)),
                                  ("own-tests", lambda: validate_agent_tests(args.output, manifest, job)),
                                  ("grade", lambda: grade_job(args.output, manifest, job)),
                                  ("release", lambda: release_job_workspace(args.output, manifest, job))]
                    for stage, operation in operations:
                        print({"job_id": job["job_id"], "instance_id": job["instance_id"], "stage": stage, "status": "started"}, flush=True)
                        append_json(args.output / f"stages-{args.phase}.jsonl", {"job_id": job["job_id"], "stage": stage, "status": "started", "timestamp": now()})
                        operation()
                        append_json(args.output / f"stages-{args.phase}.jsonl", {"job_id": job["job_id"], "stage": stage, "status": "completed", "timestamp": now()})
                    print({"job_id": job["job_id"], "status": "independently_graded"}, flush=True)
                elif args.command == "generate":
                    run_job(args.output, manifest, job, resume=args.resume)
                elif args.command == "prepare-index":
                    if backend != "native" or job["entrypoint"] != "batch":
                        raise ValueError("Separate index preparation requires a native batch job")
                    run_job(args.output, manifest, job, prepare_index_only=True)
                elif args.command == "own-tests":
                    validate_agent_tests(args.output, manifest, job)
                elif args.command == "baseline":
                    baseline_job(args.output, manifest, job)
                elif args.command == "release":
                    if backend != "native":
                        raise ValueError("This disposable-workspace release requires the native backend")
                    from moatless.evaluation.experiment_native import release_job_workspace
                    release_job_workspace(args.output, manifest, job)
                else:
                    grade_job(args.output, manifest, job)
            except Exception as exc:
                failure = {"job_id": job["job_id"], "stage": stage, "error_type": type(exc).__name__, "message": str(exc)}
                failures.append(failure)
                write_json(args.output / "runs" / job["job_id"] / f"host-failure-{args.command}.json", failure)
                append_json(args.output / f"stage-failures-{args.phase}.jsonl", {**failure, "timestamp": now()})
        write_json(args.output / f"{args.command}-{args.phase}-failures.json", failures)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
