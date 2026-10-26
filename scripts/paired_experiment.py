"""Independent A0/O2 experiment: freeze, run serial phases, inspect paired results."""

import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

from repoforge.evaluation import experiment as ex
from repoforge.evaluation import paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "preflight", "run", "report", "status"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=(*paired.PHASES, "all"), default="all")
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--conda", type=Path)
    parser.add_argument("--public-tools", type=Path)
    parser.add_argument("--build-wheels", type=Path)
    parser.add_argument("--resume-job", help="Explicitly reconcile/resume one failed in-flight job; no answer resampling")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    output = args.output.resolve()
    if args.command == "prepare":
        from repoforge.evaluation.bge_identity import capture_identity
        service = capture_identity(root, paired.INDEX_SETTINGS["embed_base_url"])
        manifest, jobs = paired.prepare(root, output, environment_root=args.environment_root,
            conda=args.conda, public_tools=args.public_tools, build_wheels=args.build_wheels,
            embedding_service=service)
        print(json.dumps({"manifest_hash": manifest["manifest_hash"], "jobs": len(jobs),
                          "source_digest": manifest["source"]["digest"]}, indent=2))
        return
    manifest = ex.load_manifest(output)
    if not manifest.get("paired"):
        raise ValueError("This command accepts only a new paired candidate")
    if args.command == "audit":
        jobs = paired.make_jobs(manifest)
        print(json.dumps({"counts": {phase: sum(j["phase"] == phase for j in jobs) for phase in paired.PHASES},
            "manifest_hash": manifest["manifest_hash"], "source_digest": manifest["source"]["digest"],
            "profiles": manifest["profiles"], "provider_calls": 0}, indent=2))
        return
    if args.command == "preflight":
        from repoforge.evaluation.experiment_native import preflight
        from repoforge.evaluation.bge_identity import validate_identity
        ex.validate_execution_source(manifest, root)
        checks = preflight(output)
        checks["embedding_health"] = validate_identity(root, manifest)
        checks["model_credential_present"] = bool(__import__("os").environ.get(manifest["cli_config"]["api_key_env"]))
        if not checks["model_credential_present"]:
            checks["blockers"].append("Missing model credential")
        else:
            from scripts.t4_cli_probe import model_metadata
            checks["model_metadata"] = model_metadata()
        ex.write_json(output / "preflight.json", checks)
        print(json.dumps(checks, indent=2))
        raise SystemExit(bool(checks["blockers"]))
    phases = paired.PHASES if args.phase == "all" else (args.phase,)
    if args.resume_job and args.resume_job not in {j["job_id"] for j in paired.make_jobs(manifest)}:
        raise ValueError("Unknown frozen resume job")
    if args.command == "run":
        for phase in phases:
            result = paired.run_phase(output, manifest, phase, resume_job=args.resume_job)
            print(json.dumps({"phase": phase, "groups": result["groups"], "paired": result["paired"]}, indent=2))
        return
    for phase in phases:
        result = paired.report(output, manifest, phase)
        if args.command == "report":
            # Status snapshots never overwrite immutable phase reports/locks.
            ex.write_json(output / f"status-{phase}.json", result)
        print(json.dumps({"phase": phase, "terminal": result["terminal"], "groups": result["groups"],
                          "paired": result["paired"]}, indent=2))


if __name__ == "__main__":
    main()
