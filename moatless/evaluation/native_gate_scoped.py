"""Independent gate builder with job-scoped answer paths.

Consumes immutable generation evidence. The original native verifier still owns
all acceptance conditions; neither raw answers nor source archives are changed.
"""
import argparse
from pathlib import Path
import shutil

from filelock import FileLock

from moatless.evaluation.experiment import load_manifest, now, read_json, sha, validate_execution_source, verify_answer, write_json
from moatless.evaluation.native_gate import configuration_evidence, validate_native_gate


def build_scoped_native_gate(output, manifest):
    """Publish only after all real component/job evidence validates."""
    output = Path(output).resolve()
    configuration = configuration_evidence(output, manifest)
    if not configuration["passed"]:
        raise ValueError("Actual loaded model/flow/tools/memory configuration is incomplete")
    patterns = {"contracts": "contracts-native/*/summary.json", "migration": "migration-native/*/summary.json",
                "isolation": "isolation-native/*/summary.json", "recovery": "recovery-native/*/summary.json",
                "cli_boundaries": "cli-boundaries-native/*/summary.json"}
    components = {"configuration": "configuration-native.json"}
    evidence = {"configuration-native.json": sha(output / "configuration-native.json"), **configuration["evidence"]}
    for name, pattern in patterns.items():
        paths = sorted(output.glob(pattern), key=lambda path: path.stat().st_mtime_ns)
        if not paths:
            raise ValueError(f"Missing actual native {name} evidence")
        path = paths[-1]  # A later failure must not be hidden by an older pass.
        summary = read_json(path)
        if not summary.get("passed") or summary.get("manifest_hash") != manifest["manifest_hash"]:
            raise ValueError(f"Latest native {name} evidence failed or has another identity")
        relative = str(path.relative_to(output))
        components[name] = relative
        evidence[relative] = sha(path)
        evidence.update(summary.get("evidence", {}))
    jobs = [job for job in read_json(output / "jobs.json") if job["phase"] in {"smoke", "cli_smoke"}]
    for job in jobs:
        directory = output / "runs" / job["job_id"]
        verify_answer(directory, job)
        for name in ("grading.json", "original-baseline.json", "own-tests.json", "answer.json"):
            path = directory / name
            row = read_json(path)
            evidence[str(path.relative_to(output))] = sha(path)
            recorded = row.get("evidence", {})
            if name == "answer.json":
                evidence.update({str((directory / relative).relative_to(output)): digest
                                 for relative, digest in recorded.items()})
            else:
                evidence.update(recorded)
            # Grading stores baseline logs separately from candidate logs.
            # Bind both copies, so editing raw original-failure evidence also
            # invalidates the gate rather than only hashing its summary.
            evidence.update(row.get("original_baseline", {}).get("evidence", {}))
    # Dev provisioning may add environments later without rewriting gate proof.
    receipts = read_json(output / "provision-native.json")
    snapshot = {name: receipts[name] for name in manifest["smoke_ids"]}
    write_json(output / "native-gate-provision.json", snapshot)
    evidence["native-gate-provision.json"] = sha(output / "native-gate-provision.json")
    auditor = output / "audit-source" / Path(__file__).name
    auditor.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, auditor)
    evidence[str(auditor.relative_to(output))] = sha(auditor)
    gate = {"manifest_hash": manifest["manifest_hash"], "execution_backend": "native", "created": now(),
            "components": components, "evidence": evidence, "gate_builder": "scoped-answer-evidence-v1", "provision_snapshot": "native-gate-provision.json",
            "local_research_acceptance": "complete", "official_harness": "not_run", "official_acceptance": "pending"}
    # Invalid drafts are explicit and never replace a previously accepted gate.
    write_json(output / "native-gate-draft.json", gate)
    if (output / "native-gate.json").exists():
        raise ValueError("Accepted native gate already exists; validate it instead of replacing it")
    write_json(output / "native-gate.json", gate)
    try:
        validate_native_gate(output, manifest)
    except Exception:
        (output / "native-gate.json").unlink()
        raise
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.output)
    if manifest.get('execution_backend') != 'native':
        raise ValueError('Scoped native gate requires a native manifest')
    validate_execution_source(manifest, Path(__file__).resolve().parents[2])
    with FileLock(str(args.output / 'scheduler.lock'), timeout=0):
        print(build_scoped_native_gate(args.output, manifest)['local_research_acceptance'])


if __name__ == '__main__':
    main()
