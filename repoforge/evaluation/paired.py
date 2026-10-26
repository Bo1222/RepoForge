"""Frozen A0/O2 paired experiment; independent of historical T7 selection gates.

Only public preparation runs before answer freeze. Private tests are accessed
exclusively by NativeGrader after verify_answer. Reports never feed the agent.
"""

import asyncio
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from filelock import FileLock

from repoforge.evaluation import experiment as ex
from repoforge.policy.action import ActionPolicyConfig

PROTOCOL = "o2-a0-paired-v1"
ARMS = ("a0", "a5")
PHASES = ("smoke", "dev", "holdout")
SPLIT_HASHES = {
    "dev": "61a1977a507b06b10a1cd2f5637399405b4f442cbe8f501f832efb61bedb7a6f",
    "holdout": "74fa30e5a4bfb0432831456f342d9797a56f09a31cd639dcda230cf24dd63bbe",
}
INDEX_SETTINGS = {"embed_provider": "local", "embed_model": "BAAI/bge-m3",
    "embed_base_url": "http://127.0.0.1:8787/v1", "dimensions": 1024,
    "min_chunk_size": 100, "chunk_size": 500, "hard_token_limit": 750,
    "max_chunks": 1000, "preserve_chunk_content": True, "language": "python",
    "comment_strategy": "associate"}


def validate_design(manifest):
    design = manifest["paired"]
    if (design.get("protocol") != PROTOCOL or design.get("arms") != list(ARMS)
            or design.get("repeats") != 1 or manifest.get("t6") or manifest.get("t7")
            or manifest.get("exploration")):
        raise ValueError("Expected the independent off/a5, single-repeat paired protocol")
    if manifest["dataset"] != "princeton-nlp/SWE-bench_Verified":
        raise ValueError("Paired experiment requires SWE-bench Verified")
    split = manifest["split"]
    if (len(split["dev"]) != 30 or len(split["holdout"]) != 50
            or len(set(split["dev"] + split["holdout"])) != 80):
        raise ValueError("Expected disjoint dev30/holdout50")
    for phase, expected in SPLIT_HASHES.items():
        if hashlib.sha256(json.dumps(split[phase], sort_keys=True).encode()).hexdigest() != expected or split[f"{phase}_sha256"] != expected:
            raise ValueError("Frozen Wave2 instance selection changed")
    if list(manifest["smoke_ids"]) != list(ex.SMOKE_IDS):
        raise ValueError("Expected the original five smoke instances")
    if set(manifest["smoke_ids"]) & set(split["dev"] + split["holdout"]):
        raise ValueError("Smoke instances cannot enter the formal denominator")
    if set(manifest["profiles"]) != set(ARMS) or set(manifest["agent_configs"]) != set(ARMS):
        raise ValueError("Paired manifest must contain exactly A0 and O2")
    profiles = []
    for arm, variant in zip(ARMS, ("off", "a5")):
        profile = deepcopy(manifest["profiles"][arm])
        policy = profile.pop("action_policy")
        if policy != ActionPolicyConfig(variant=variant).model_dump(mode="json"):
            raise ValueError("Unexpected policy mapping or tuned thresholds")
        if (profile.get("flow") != "graph" or profile.get("managed") is not True
                or profile.get("submit_product_only") is not True):
            raise ValueError("Both arms require Graph, Managed Context and product-only export")
        profiles.append(profile)
    if profiles[0] != profiles[1] or manifest["agent_configs"]["a0"] != manifest["agent_configs"]["a5"]:
        raise ValueError("Arms may differ only in ActionPolicy")


def make_jobs(manifest):
    validate_design(manifest)
    jobs = []
    for phase in PHASES:
        ids = manifest["smoke_ids"] if phase == "smoke" else manifest["split"][phase]
        for position, identity in enumerate(ids):
            # Frozen convention: zero-based even=A0 first, odd=O2 first.
            for arm in ARMS if position % 2 == 0 else tuple(reversed(ARMS)):
                job = {"phase": phase, "variant": arm, "repeat": 0, "instance_id": identity,
                       "manifest_hash": manifest["manifest_hash"], "entrypoint": "batch"}
                jobs.append({**job, "job_id": ex.content_hash(job)[:24]})
    return jobs


def prepare(root, output, *, environment_root=None, conda=None, public_tools=None,
            build_wheels=None, embedding_service=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    frozen = ex.read_json(root / "repoforge/evaluation/profiles/wave2_split.json")
    old = ex.read_json(root / "repoforge/evaluation/profiles/frozen_split.json")
    if set(frozen["split"]["dev"] + frozen["split"]["holdout"]) & set(
            old["split"]["dev"] + old["split"]["holdout"]):
        raise ValueError("Wave2 overlaps historical T7")
    # Inspect actual generated jobs, not just predeclared historical manifests.
    used, selected = [], set(frozen["split"]["dev"] + frozen["split"]["holdout"])
    for path in (root / "experiments/results").glob("*/jobs.json"):
        previous = ex.read_json(path)
        if not isinstance(previous, list):
            continue
        for job in previous:
            if (job.get("instance_id") in selected and
                    (path.parent / "runs" / str(job.get("job_id")) / "generation.json").exists()):
                used.append({"candidate": str(path.parent), "job_id": job["job_id"],
                             "instance_id": job["instance_id"]})
    if used:
        raise ValueError("Wave2 has prior generated answers; do not silently replace the split: " + str(used))
    manifest, _ = ex.prepare(root, output, execution_backend="native",
        environment_root=environment_root, embedding_service=embedding_service,
        embedding_settings=INDEX_SETTINGS, public_tools=public_tools, build_wheels=build_wheels)
    manifest.pop("manifest_hash")
    manifest.pop("historical_split_manifest", None)
    manifest["split"] = frozen["split"]
    profile = deepcopy(ex.T7_PROFILES["a5"])
    manifest["profiles"] = {arm: {**deepcopy(profile), "action_policy":
        ActionPolicyConfig(variant="off" if arm == "a0" else "a5").model_dump(mode="json")} for arm in ARMS}
    agent = ex.agent_config(root, "a5", manifest["model_config"])
    manifest["agent_configs"] = {arm: deepcopy(agent) for arm in ARMS}
    manifest["native"]["conda"] = str(Path(conda).resolve()) if conda else None
    # Keep large new indexes off the user's nearly full repository volume.
    from repoforge.index.repository_store import builder_fingerprint
    manifest["native"]["index_store"] = {"root": str(output / "index-store"),
        "format": "repository-index-v1",
        "builder_sha256": builder_fingerprint((root / "repoforge/evaluation/generation.py").read_text())}
    manifest["paired"] = {"protocol": PROTOCOL, "arms": list(ARMS), "repeats": 1,
        "dataset_split": "test", "dataset_fingerprint": "1fdfd21ba2621130",
        "split_file_sha256": ex.sha(root / "repoforge/evaluation/profiles/wave2_split.json"),
        "order": "zero-based even A0 first; odd O2 first; serial",
        "bootstrap_samples": 10000, "bootstrap_seed": 20261023,
        "preparation_timeout": 1800, "grading_timeout": 600,
        "infrastructure_retries": 0, "resample_answers": False,
        "interpretation": "single-run exploratory; whole ActionPolicy contribution only"}
    manifest["paired"]["history_audit"] = {"generated_intersections": used,
        "scope": "repository experiments/results/*/jobs.json and runs/JOB/generation.json; human use and external runs not observable"}
    manifest["manifest_hash"] = ex.content_hash(manifest)
    jobs = make_jobs(manifest)
    ex.write_json(output / "manifest.json", manifest)
    ex.write_json(output / "jobs.json", jobs)
    ex.write_json(output / "provision-native.json", {})
    return ex.read_json(output / "manifest.json"), jobs


def error_text(exc):
    text = f"{type(exc).__name__}: {exc}"
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "JINA_API_KEY", "VOYAGE_API_KEY"):
        if os.environ.get(name):
            text = text.replace(os.environ[name], "[REDACTED]")
    return text


def verify_evidence(output, evidence):
    output = Path(output).resolve()
    for name, expected in evidence.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or ex.sha(path) != expected:
            raise ValueError("Evidence changed or escaped candidate: " + name)


def configuration_checks(manifest, job, loaded, generation, decisions):
    from repoforge.evaluation.configuration_bindings import tool_bindings
    from repoforge.message_history.policy import ContextPolicy
    config = manifest["cli_config"]
    tools = tool_bindings(manifest, job)
    expected_policy = manifest["profiles"][job["variant"]]["action_policy"]
    checks = {"job": loaded.get("job_id") == job["job_id"] and loaded.get("manifest_hash") == manifest["manifest_hash"],
        "graph": loaded.get("flow_type") == "LangGraphFlow",
        "managed": loaded.get("memory_type") == "ManagedMessageHistoryGenerator",
        "context": loaded.get("policy") == ContextPolicy(**config["context_policy"]).model_dump(mode="json"),
        "policy": loaded.get("action_policy") == expected_policy,
        "model": loaded.get("model") == config["model"] and loaded.get("model_base_url") == config["base_url"],
        "parameters": loaded.get("model_parameters") == config["params"],
        "credential": loaded.get("key_env_matches") is True,
        "tools": loaded.get("tool_names") == list(tools) and loaded.get("schema_names") == list(tools.values()),
        "index": loaded.get("index_settings") == manifest["embedding_settings"],
        "isolation": loaded.get("tool_isolation") == "native-offline-v1",
        "export": generation.get("submit_product_only") is True and generation.get("submitted_verified_patch") is False,
        "decisions": all(d.get("config") == expected_policy for d in decisions),
        "off_passthrough": job["variant"] != "a0" or not decisions,
        "enabled": job["variant"] != "a5" or bool(decisions) or not generation.get("policy_metrics", {}).get("actual_tool_steps")}
    for key in ("temperature", "max_tokens", "max_iterations", "max_cost", "model_timeout"):
        checks[key] = loaded.get(key) == config[key]
    return checks


def job_result(output, manifest, job):
    output = Path(output)
    directory = output / "runs" / job["job_id"]
    row = {**job, "status": "not_run", "resolved": None, "cost": None, "cost_complete": False,
           "official_harness": "not_run", "official_resolved": None, "evidence": {}}
    blocked = output / "preparation" / f"{job['instance_id']}.json"
    if blocked.exists() and ex.read_json(blocked).get("status") == "env_blocked":
        row.update(status="env_blocked", preparation=ex.read_json(blocked),
                   evidence={str(blocked.relative_to(output)): ex.sha(blocked)})
        return row
    if not (directory / "answer.json").exists():
        if (directory / "operation-error.json").exists():
            row.update(status="execution_error", error=ex.read_json(directory / "operation-error.json"))
        return row
    answer = ex.verify_answer(directory, job)
    generation = ex.read_json(directory / "generation.json")
    usage = ex.read_json(directory / "usage.json")
    loaded = ex.read_json(directory / "loaded-config.json")
    decisions = ex.read_json(directory / "policy-decisions.json")
    checks = configuration_checks(manifest, job, loaded, generation, decisions)
    if not all(checks.values()):
        raise ValueError(f"Actual paired configuration mismatch: {job['job_id']} {checks}")
    if usage.get("cost_complete") is not True or usage.get("reference_cost") is None:
        raise ValueError("Unknown generation cost; refusing continued execution")
    if generation.get("generation_status") != "completed":
        raise ValueError("Generation infrastructure failure requires reconciliation")
    row.update(status="generated", configuration_checks=checks, answer_hash=answer["answer_hash"],
        cost=usage["reference_cost"], cost_complete=True, finish_reason=generation.get("finish_reason"),
        completion_status=generation.get("completion_status"), metrics=generation.get("policy_metrics", {}),
        seconds=generation.get("generation_seconds"), flow_usage=generation.get("flow_usage"))
    files = ["answer.json", *answer["evidence"], "loaded-config.json"]
    if (directory / "grading.json").exists():
        grade = ex.read_json(directory / "grading.json")
        if grade.get("answer_hash") != answer["answer_hash"]:
            raise ValueError("Grade is not bound to the frozen answer")
        verify_evidence(output, grade.get("evidence", {}))
        baseline = grade.get("original_baseline", {})
        verify_evidence(output, baseline.get("evidence", {}))
        own = ex.read_json(directory / "own-tests.json")
        if own.get("answer_hash") != answer["answer_hash"]:
            raise ValueError("Own tests belong to another answer")
        verify_evidence(output, own.get("evidence", {}))
        row.update(status="graded" if grade.get("local_grading") in {"completed", "patch_error"} else "grading_error",
            resolved=grade.get("local_resolved"), patch_applicable=grade.get("patch_applicable"),
            original_failure_reproduced=baseline.get("original_failure_reproduced"),
            tests_status=grade.get("tests_status"), local_grading=grade.get("local_grading"))
        if baseline.get("original_failure_reproduced") is not True:
            row.update(status="grading_error", resolved=None)
        # Preserve raw grader output; distinguish patch-induced collection/hang
        # from installation failures using the successful matched baseline.
        cases = [output / path for path in grade.get("evidence", {}) if path.endswith("/case.json")]
        if row["status"] == "grading_error" and len(cases) == 1:
            candidate = ex.read_json(cases[0])
            checks_run = candidate.get("checks", [])
            last = checks_run[-1] if checks_run else {}
            if (baseline.get("status") == "completed" and row["original_failure_reproduced"]
                    and candidate.get("patch_applicable") is True
                    and (last.get("timed_out") is True or last.get("return_code") == 2)):
                row.update(status="graded", resolved=False,
                    adjudication="candidate test execution failed against a complete, reproducing baseline; raw grade retained")
        files += ["grading.json", "own-tests.json", "original-baseline.json"]
        row["evidence"].update(grade.get("evidence", {}))
        row["evidence"].update(baseline.get("evidence", {}))
        row["evidence"].update(own.get("evidence", {}))
    for name in files:
        path = directory / name
        row["evidence"][str(path.relative_to(output))] = ex.sha(path)
    prepared_path = output / "provision-native.json"
    if prepared_path.exists():
        prepared = ex.read_json(prepared_path).get(job["instance_id"])
        if prepared:
            row["environment_identity"] = ex.content_hash(prepared)
    # Read the identity recorded by this worker, not the mutable latest shared
    # index reference which could conceal cross-arm drift.
    runtime = ex.read_json(directory / "runtime-start.json")
    row["index_digest"] = runtime["index_digest"]
    row["runtime_dependencies"] = runtime.get("dependencies")
    row["runtime_packages"] = runtime.get("python_packages")
    return row


def paired_statistics(rows, identities, *, samples=10000, seed=20261023):
    by_id = {(r["instance_id"], r["variant"]): r for r in rows}
    complete, changes, missing = [], [], []
    for identity in identities:
        a, b = (by_id[identity, arm]["resolved"] for arm in ARMS)
        if isinstance(a, bool) and isinstance(b, bool):
            complete.append(int(b) - int(a))
            if a != b:
                changes.append({"instance_id": identity, "a0": a, "a5": b})
        else:
            missing.append(identity)
    interval = None
    if complete:
        rng = random.Random(seed)
        draws = sorted(sum(rng.choices(complete, k=len(complete))) / len(complete) for _ in range(samples))
        interval = [draws[max(0, math.ceil(.025 * samples) - 1)], draws[math.ceil(.975 * samples) - 1]]
    return {"pairable": len(complete), "missing_pairs": missing, "changes": changes,
        "fail_to_success": sum(x == 1 for x in complete), "success_to_fail": sum(x == -1 for x in complete),
        "delta_pairable": sum(complete) / len(complete) if complete else None,
        "bootstrap_95_ci_pairable": interval, "bootstrap_samples": samples, "bootstrap_seed": seed,
        "scope": "complete pairs only; all-selected success rates and missingness are reported separately"}


def report(output, manifest, phase):
    jobs = [j for j in make_jobs(manifest) if j["phase"] == phase]
    rows = [job_result(output, manifest, j) for j in jobs]
    identities = list(dict.fromkeys(j["instance_id"] for j in jobs))
    for identity in identities:
        pair = [r for r in rows if r["instance_id"] == identity and r["status"] == "graded"]
        if len(pair) == 2 and any(pair[0].get(k) != pair[1].get(k) for k in (
                "environment_identity", "index_digest", "runtime_dependencies", "runtime_packages")):
            raise ValueError("Paired arms used different environment or index identities")
    groups = {}
    for arm in ARMS:
        selected = [r for r in rows if r["variant"] == arm]
        attempted = [r for r in selected if r["cost_complete"]]
        solved = sum(r["resolved"] is True for r in selected)
        cost = sum(r["cost"] for r in attempted)
        groups[arm] = {"selected": len(selected), "generated": len(attempted),
            "graded": sum(r["status"] == "graded" for r in selected), "resolved": solved,
            "unknown": sum(r["resolved"] is None for r in selected),
            "resolved_over_selected": solved / len(selected) if attempted else None,
            "known_reference_cost": cost if attempted else None,
            "cost_per_resolved": cost / solved if solved and all(r["status"] in {"graded", "env_blocked"} for r in selected) else None}
    result = {"manifest_hash": manifest["manifest_hash"], "phase": phase, "groups": groups, "rows": rows,
        "terminal": all(r["status"] in {"graded", "env_blocked"} for r in rows),
        "paired": paired_statistics(rows, identities, samples=manifest["paired"]["bootstrap_samples"],
                                    seed=manifest["paired"]["bootstrap_seed"]),
        "official_harness": "not_run", "official_resolved": None,
        "interpretation": manifest["paired"]["interpretation"]}
    return result


def freeze_phase(output, manifest, phase):
    output = Path(output)
    result = report(output, manifest, phase)
    if not result["terminal"] or (phase == "smoke" and any(r["status"] != "graded" for r in result["rows"])):
        raise ValueError("Phase has incomplete execution/grading or unresolved smoke infrastructure")
    path = output / f"report-{phase}.json"
    lock = output / f"{phase}-lock.json"
    if lock.exists():
        validate_phase(output, manifest, phase)
        if ex.read_json(path) != result:
            raise ValueError("A frozen phase cannot be rewritten")
        return result
    ex.write_json(path, result)
    lines = [f"# A0 / O2 paired: {phase}", "", "Single-run native grading; official Harness not_run.", "",
             "| Arm | Selected | Graded | Resolved | Unknown | Known reference cost |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, group in result["groups"].items():
        lines.append(f"| {arm} | {group['selected']} | {group['graded']} | {group['resolved']} | "
                     f"{group['unknown']} | {group['known_reference_cost']} |")
    lines += ["", "Pairable results (missing pairs excluded from this interval):", "",
              "```json", json.dumps(result["paired"], ensure_ascii=False, indent=2), "```", "",
              "Full per-job evidence and F2P/P2P are in the adjacent JSON report.", ""]
    markdown = output / f"report-{phase}.md"
    markdown.write_text("\n".join(lines))
    evidence = {k: v for r in result["rows"] for k, v in r["evidence"].items()}
    evidence[markdown.name] = ex.sha(markdown)
    ex.write_json(lock, {"manifest_hash": manifest["manifest_hash"], "phase": phase,
        "report_sha256": ex.sha(path), "evidence": evidence, "created": ex.now()})
    return result


def validate_phase(output, manifest, phase):
    output = Path(output)
    lock = ex.read_json(output / f"{phase}-lock.json")
    if (lock["manifest_hash"] != manifest["manifest_hash"] or lock["phase"] != phase
            or ex.sha(output / f"report-{phase}.json") != lock["report_sha256"]):
        raise ValueError("Frozen phase identity changed")
    verify_evidence(output, lock["evidence"])
    result = ex.read_json(output / f"report-{phase}.json")
    if not result["terminal"] or result["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Previous phase is incomplete")
    return result


def require_previous(output, manifest, phase):
    if phase == "dev":
        return validate_phase(output, manifest, "smoke")
    if phase == "holdout":
        validate_phase(output, manifest, "smoke")
        return validate_phase(output, manifest, "dev")


async def prepare_environment(provisioner, public, setup_commit, deadline):
    return await asyncio.wait_for(provisioner.prepare(public, environment_setup_commit=setup_commit), deadline)


def provision_one(output, manifest, public, recipe):
    from repoforge.evaluation.native_provision import NativeProvisioner
    output = Path(output)
    path = output / "preparation" / f"{public['instance_id']}.json"
    receipts = ex.read_json(output / "provision-native.json")
    if path.exists():
        status = ex.read_json(path)
        if status["status"] == "env_blocked":
            return False
    if public["instance_id"] in receipts:
        prepared = receipts[public["instance_id"]]
        if prepared["public"] != public or prepared["source_sha256"] != manifest["source"]["sha256"]:
            raise ValueError("Prepared public/source identity changed")
        NativeProvisioner.verify(prepared["environment"])
        return True
    deadline = manifest["paired"]["preparation_timeout"]
    started = time.monotonic()
    print({"event": "environment_start", "instance_id": public["instance_id"],
           "timestamp": ex.now(), "deadline_seconds": deadline}, flush=True)

    class DeadlineProvisioner(NativeProvisioner):
        async def _run(self, *args, timeout=1800, **kwargs):
            remaining = deadline - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Per-instance preparation deadline reached")
            return await super()._run(*args, timeout=min(timeout, remaining), **kwargs)

    try:
        provisioner = DeadlineProvisioner(manifest["native"]["environment_root"],
            conda=manifest["native"].get("conda"), public_tools=manifest["native"].get("public_tools"))
        environment = asyncio.run(prepare_environment(provisioner, public,
            recipe.get("environment_setup_commit"), deadline))
        NativeProvisioner.verify(environment)
        receipts[public["instance_id"]] = {"environment": environment, "public": public,
                                          "source_sha256": manifest["source"]["sha256"]}
        ex.write_json(output / "provision-native.json", receipts)
        status = {"status": "ready", "environment_identity": ex.content_hash(receipts[public["instance_id"]])}
    except Exception as exc:
        status = {"status": "env_blocked", "error": error_text(exc)}
    ex.write_json(path, {**status, "manifest_hash": manifest["manifest_hash"],
        "instance_id": public["instance_id"], "seconds": time.monotonic() - started, "created": ex.now()})
    print({"event": "environment_end", "instance_id": public["instance_id"],
           "status": status["status"], "seconds": round(time.monotonic() - started, 2),
           "timestamp": ex.now()}, flush=True)
    return status["status"] == "ready"


def run_phase(output, manifest, phase, *, resume_job=None):
    from repoforge.evaluation import experiment_native as native
    from repoforge.evaluation.experiment_inputs import export_public_dataset
    root = Path(__file__).resolve().parents[2]
    output = Path(output)
    with FileLock(str(output / "paired-scheduler.lock"), timeout=0):
        ex.validate_execution_source(manifest, root)
        require_previous(output, manifest, phase)
        if (output / f"{phase}-lock.json").exists():
            return validate_phase(output, manifest, phase)
        inventory = export_public_dataset(output, manifest, phase)
        if ex.read_json(output / "dataset-receipt.json")["fingerprint"] != manifest["paired"]["dataset_fingerprint"]:
            raise ValueError("Dataset cache differs from the frozen reference fingerprint")
        recipes = ex.read_json(output / "host-install-recipes.json")
        print({"event": "phase_start", "phase": phase, "timestamp": ex.now()}, flush=True)
        for job in (j for j in make_jobs(manifest) if j["phase"] == phase):
            identity = job["instance_id"]
            if not provision_one(output, manifest, inventory[identity], recipes.get(identity, {})):
                continue
            directory = output / "runs" / job["job_id"]
            if ((directory / "operation-error.json").exists() and not (directory / "paired-result.json").exists()
                    and resume_job != job["job_id"]):
                raise ValueError("Previous operation failed; reconcile and explicitly resume the job: " + job["job_id"])
            try:
                print({"event": "job_start", "phase": phase, "instance_id": identity,
                       "arm": job["variant"], "job_id": job["job_id"], "timestamp": ex.now()}, flush=True)
                native.run_job(output, manifest, job, resume=resume_job == job["job_id"])
                # Validate live configuration and cost before any independent grading.
                if not (directory / "grading.json").exists():
                    job_result(output, manifest, job)
                if not (directory / "own-tests.json").exists():
                    print({"event": "own_test_validation", "job_id": job["job_id"], "timestamp": ex.now()}, flush=True)
                    native.validate_agent_tests(output, manifest, job)
                if not (directory / "grading.json").exists():
                    print({"event": "independent_grading", "job_id": job["job_id"], "timestamp": ex.now()}, flush=True)
                    native.grade_job(output, manifest, job)
                row = job_result(output, manifest, job)
                if row["status"] != "graded":
                    raise ValueError("Independent grading incomplete; retain evidence for reconciliation")
                ex.write_json(directory / "paired-result.json", row)
                # Release only confirmed disposable copies, never baseline or user files.
                if row["local_grading"] in {"completed", "patch_error"}:
                    native.release_job_workspace(output, manifest, job)
                print({"phase": phase, "instance_id": identity, "arm": job["variant"],
                       "resolved": row["resolved"], "cost": row["cost"]}, flush=True)
            except Exception as exc:
                ex.write_json(directory / "operation-error.json", {"job_id": job["job_id"],
                    "timestamp": ex.now(), "error": error_text(exc), "resume_required": True})
                raise
        return freeze_phase(output, manifest, phase)
