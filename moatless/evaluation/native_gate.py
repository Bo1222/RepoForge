"""Evidence-bound native research gate, independent of official completion."""

from pathlib import Path

from moatless.evaluation.experiment import read_json, sha, verify_answer
from moatless.evaluation.native_provision import NativeProvisioner
from moatless.evaluation.experiment import now, write_json


COMPONENTS = ("contracts", "migration", "isolation", "recovery", "cli_boundaries", "configuration")


def configuration_evidence(output, manifest):
    from moatless.message_history.policy import ContextPolicy
    from moatless.evaluation.configuration_bindings import tool_bindings
    output = Path(output).resolve()
    rows, evidence, bindings = [], {}, {}
    for job in read_json(output / "jobs.json"):
        if job["phase"] not in {"smoke", "cli_smoke"}:
            continue
        path = output / "runs" / job["job_id"] / "loaded-config.json"
        loaded = read_json(path)
        names = tool_bindings(manifest, job)
        bindings[job['job_id']] = names
        expected = manifest["cli_config"]
        managed = job["entrypoint"] == "cli" or manifest["profiles"][job["variant"]]["managed"]
        flow = "LangGraphFlow" if job["entrypoint"] == "cli" or manifest["profiles"][job["variant"]]["flow"] == "graph" else "AgenticLoop"
        checks = {
            "identity": loaded.get("manifest_hash") == manifest["manifest_hash"] and loaded.get("job_id") == job["job_id"],
            "flow": loaded.get("flow_type") == flow,
            "model": loaded.get("model") == expected["model"] and loaded.get("model_base_url") == expected["base_url"],
            "credential": loaded.get("key_env_matches") is True,
            "parameters": loaded.get("model_parameters") == expected["params"],
            "limits": all(loaded.get(key) == expected[key] for key in ("max_tokens", "temperature", "max_iterations", "max_cost", "model_timeout")),
            "memory": (loaded.get("memory_type") == "ManagedMessageHistoryGenerator") == managed,
            "tool_schema": bool(loaded.get("tool_names")) and {
                names.get(name, name) for name in loaded["tool_names"]
            } == set(loaded.get("schema_names", [])),
            "frozen_tool_bindings": len(loaded.get('tool_names', [])) == len(names)
                and set(loaded.get('tool_names', [])) == set(names)
                and len(loaded.get('schema_names', [])) == len(names)
                and set(loaded.get('schema_names', [])) == set(names.values()),
            "offline_tools": loaded.get("tool_isolation") == "native-offline-v1",
            "index": job["entrypoint"] == "cli" or loaded.get("index_settings") == manifest["embedding_settings"],
            "managed_policy": not managed or loaded.get("policy") == ContextPolicy(**expected["context_policy"]).model_dump(mode="json"),
        }
        rows.append({"job_id": job["job_id"], "checks": checks, "passed": all(checks.values())})
        evidence[str(path.relative_to(output))] = sha(path)
    summary = {"manifest_hash": manifest["manifest_hash"], "passed": len(rows) == 20 and all(row["passed"] for row in rows),
               "rows": rows, "evidence": evidence, "provider_calls": 0,
               "implementation_schema_bindings": bindings}
    write_json(output / "configuration-native.json", summary)
    return summary


def build_native_gate(output, manifest):
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
            evidence.update(row.get("evidence", {}))
            # Grading stores baseline logs separately from candidate logs.
            # Bind both copies, so editing raw original-failure evidence also
            # invalidates the gate rather than only hashing its summary.
            evidence.update(row.get("original_baseline", {}).get("evidence", {}))
    # Dev provisioning may add environments later without rewriting gate proof.
    receipts = read_json(output / "provision-native.json")
    snapshot = {name: receipts[name] for name in manifest["smoke_ids"]}
    write_json(output / "native-gate-provision.json", snapshot)
    evidence["native-gate-provision.json"] = sha(output / "native-gate-provision.json")
    gate = {"manifest_hash": manifest["manifest_hash"], "execution_backend": "native", "created": now(),
            "components": components, "evidence": evidence, "provision_snapshot": "native-gate-provision.json",
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


def validate_native_gate(output, manifest):
    from moatless.evaluation.generation import validate_generation_health

    output = Path(output).resolve()
    gate = read_json(output / "native-gate.json")
    if gate.get("manifest_hash") != manifest["manifest_hash"] or gate.get("execution_backend") != "native":
        raise ValueError("Native gate belongs to a different source/configuration/backend")
    evidence = gate.get("evidence", {})
    if not evidence or not all(name in gate.get("components", {}) for name in COMPONENTS):
        raise ValueError("Native gate requires actual contracts, migration, isolation, recovery and CLI evidence")
    for name, expected in evidence.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or sha(path) != expected:
            raise ValueError("Native gate evidence changed or escaped its experiment")
    for name in COMPONENTS:
        path = gate["components"][name]
        if path not in evidence:
            raise ValueError("Native gate component is not bound to an evidence hash")
        component = read_json(output / path)
        if not component.get("passed") or component.get("manifest_hash") != manifest["manifest_hash"]:
            raise ValueError(f"Native {name} evidence is incomplete or from another source")
        for recorded, expected in component.get("evidence", {}).items():
            if recorded not in evidence or evidence[recorded] != expected:
                raise ValueError("Gate omits a component's underlying evidence")
    recovery = read_json(output / gate["components"]["recovery"])
    expected_cases = {f"{mode}-{fault}" for mode in ("shadow", "disk")
                      for fault in ("after_generation", "before_tool", "after_effect", "after_commit")}
    if {row["case"] for row in recovery.get("cases", [])} != expected_cases or not all(
            row.get("passed") and row.get("first_exit_code") == 88 for row in recovery["cases"]):
        raise ValueError("Native recovery gate requires eight actual crash-window outcomes")
    provision_path = gate.get("provision_snapshot", "provision-native.json")
    if provision_path not in evidence:
        raise ValueError("Native environment receipts are not frozen into the gate")
    for prepared in read_json(output / provision_path).values():
        if prepared["source_sha256"] != manifest["source"]["sha256"]:
            raise ValueError("Native environment source identity differs from the gate")
        NativeProvisioner.verify(prepared["environment"])
    jobs = read_json(output / "jobs.json")
    smoke = [job for job in jobs if job["phase"] == "smoke"]
    cli = [job for job in jobs if job["phase"] == "cli_smoke"]
    if len(smoke) != 15 or len(cli) != 5:
        raise ValueError("Native gate requires the original 15 smoke and five CLI jobs")
    for job in [*smoke, *cli]:
        directory = output / "runs" / job["job_id"]
        answer = verify_answer(directory, job)
        validate_generation_health(directory)
        grade = read_json(directory / "grading.json")
        if (grade.get("answer_hash") != answer["answer_hash"] or grade.get("grading_backend") != "native"
                or grade.get("evaluation_protocol") != manifest["evaluation_protocol"]):
            raise ValueError("Native grading identity differs from its complete frozen answer")
        if grade.get("grader_source_sha256") != manifest["source"]["files"].get("moatless/evaluation/native_grading.py"):
            raise ValueError("Native grading adapter differs from the frozen gate source")
        if (grade.get("local_grading") not in {"completed", "patch_error"}
                or grade.get("local_resolved") not in (True, False)
                or not grade.get("original_baseline", {}).get("original_failure_reproduced")):
            raise ValueError("Unresolved native environment or original-failure reproduction blocks expansion")
        if grade.get("official_harness") != "not_run" or grade.get("official_resolved") is not None:
            raise ValueError("Native-only grading cannot manufacture official completion")
        for recorded, expected in grade.get("evidence", {}).items():
            if evidence.get(recorded) != expected:
                raise ValueError("Native gate omits independent grading evidence")
        baseline_path = str((directory / "original-baseline.json").relative_to(output))
        if baseline_path not in evidence or read_json(output / baseline_path) != grade["original_baseline"]:
            raise ValueError("Native gate omits or changes the original baseline report")
        baseline_evidence = grade["original_baseline"].get("evidence", {})
        if not baseline_evidence or any(evidence.get(name) != expected for name, expected in baseline_evidence.items()):
            raise ValueError("Native gate omits raw original-failure reproduction evidence")
        own = read_json(directory / "own-tests.json")
        if (own.get("answer_hash") != answer["answer_hash"] or own.get("hidden_tests") is not False
                or own.get("status") not in {"executed", "not_run"}):
            raise ValueError("Agent's own tests require independent public-only evidence or an explicit not_run record")
        for recorded, expected in own.get("evidence", {}).items():
            if evidence.get(recorded) != expected:
                raise ValueError("Native gate omits agent-test validation evidence")
    return gate
