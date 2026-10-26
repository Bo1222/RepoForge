"""T7 policy ablation and locked holdout protocol.

The module owns preparation, the pre-registered dev matrix, the immutable
holdout lock, holdout preparation/scan/adjudication and the unified job
acceptance used by the controller.  It never reads benchmark answers before a
complete answer is frozen and it keeps the official Harness at not_run/null.
"""

from copy import deepcopy
import asyncio
import shutil
from pathlib import Path

from repoforge.evaluation.experiment import (
    PROFILES, SMOKE_IDS, SMOKE_VARIANTS, T7_DEV_VARIANTS, T7_HOLDOUT_VARIANTS,
    T7_PROFILES, T7_PROTOCOL, aggregate, agent_config, append_json, content_hash,
    freeze_source, load_manifest, make_jobs, now, read_json, sha, verify_answer, write_json,
)
from repoforge.evaluation.experiment_inputs import export_public_dataset, validate_embedding_service
from repoforge.evaluation.native_provision import NativeProvisioner

PROTOCOL = T7_PROTOCOL
LOCK_NAME = "t7-lock.json"
ANOMALY_NAME = "anomaly-jobs.json"
ENV_BLOCKED_NAME = "env-blocked-jobs.json"
HOLDOUT_PENDING_ANOMALY = "holdout-anomaly-jobs.json"
HOLDOUT_PENDING_ENV = "holdout-env-blocked-jobs.json"
HOLDOUT_SCAN = "holdout-scan.json"
HOLDOUT_ADJUDICATION = "holdout-adjudication.json"
ANALYSIS_PLAN = "docs/2026-10-18_T7_策略消融与留出预注册.md"
HOLDOUT_RECORD = "docs/2026-10-18_T7_留出环境预审登记.md"

SELECTION_RULE = (
    "dev local_resolved count is primary. A policy variant (a1-a4) replaces a0 only when its "
    "dev count is strictly greater than a0. Among eligible policy variants choose the highest "
    "count; tie-break by lower known reference cost (only when cost_complete), then higher "
    "effective verification rate, then variant name. If no policy variant exceeds a0, select a0 "
    "as a single-arm degraded fallback and record every policy variant as not improved."
)
FALSIFICATION_RULE = (
    "A policy variant is falsified when its paired dev local_resolved count does not exceed a0, "
    "or when it costs more without a resolved improvement. Falsification never changes the "
    "holdout composition: the winner is locked before any holdout job runs."
)
SMALL_SAMPLE_STATEMENT = (
    "One dev run per variant (n=30 per arm) is exploratory. Paired bootstrap intervals are "
    "descriptive only; they are not a significance test, a formal falsification threshold or a "
    "leaderboard claim. The formal metric remains local_resolved over all selected jobs; the "
    "official Harness stays not_run/null."
)

# Section 11.4 failure classes. The exact deterministic mapping is implemented in
# scripts/t7_analysis.py; these names are frozen into the manifest.
FAILURE_LABELS = ("context_omission", "invalid_repeat", "edit_error", "test_timing",
                  "unverified_finish", "budget", "environment", "model_protocol")


def expected_jobs(manifest):
    """Materialize the exact frozen matrix; make_jobs is the only source."""
    jobs = make_jobs(manifest)
    expected = []
    ordered_dev = [*SMOKE_IDS, *[identity for identity in manifest["split"]["dev"]
                                 if identity not in set(SMOKE_IDS)]]
    for variant in manifest["t7"]["smoke_variants"]:
        for instance_id in manifest["smoke_ids"]:
            expected.append(("smoke", variant, 0, instance_id, "batch"))
    for variant in manifest["t7"]["cli_variants"]:
        for instance_id in manifest["smoke_ids"]:
            expected.append(("cli_smoke", variant, 0, instance_id, "cli"))
    for variant in manifest["t7"]["dev_variants"]:
        for repeat in range(manifest["t7"]["dev_repeats"]):
            for instance_id in ordered_dev:
                expected.append(("dev", variant, repeat, instance_id, "batch"))
    for variant in manifest["t7"]["holdout_variants"]:
        for repeat in range(manifest["t7"]["holdout_repeats"]):
            for instance_id in manifest["split"]["holdout"]:
                expected.append(("holdout", variant, repeat, instance_id, "batch"))
    observed = [(job["phase"], job["variant"], job["repeat"], job["instance_id"], job["entrypoint"])
                for job in jobs]
    if observed != expected:
        raise ValueError("T7 frozen job order or composition changed")
    return jobs


def _check_reference_inputs(root, reference, manifest, *, environment_root, index_settings,
                            build_wheels, public_tools):
    """Every reused identity must equal the frozen reference candidate."""
    from repoforge.index.settings import IndexSettings
    from repoforge.evaluation.native_public_tools import freeze_profile
    from repoforge.index.repository_store import builder_fingerprint

    if reference != Path(manifest["native"]["environment_root"]).resolve():
        raise ValueError("T7 must reuse the reference candidate's frozen environment root")
    if environment_root is not None and Path(environment_root).resolve() != reference:
        raise ValueError("Requested environment root differs from the reference manifest")
    settings = IndexSettings(**(read_json(index_settings) if index_settings else
                                manifest["embedding_settings"])).model_dump(mode="json")
    if settings != manifest["embedding_settings"]:
        raise ValueError("T7 must reuse the reference candidate's frozen index settings")
    wheels = Path(build_wheels or manifest["native"]["build_wheels"]["path"]).resolve()
    frozen_wheels = manifest["native"]["build_wheels"]
    if sha(wheels / "manifest.json") != frozen_wheels["manifest_sha256"]:
        raise ValueError("Offline build wheel manifest differs from the reference")
    wheel_manifest = read_json(wheels / "manifest.json")
    for name, digest in wheel_manifest["files"].items():
        if sha(wheels / name) != digest:
            raise ValueError("Offline build dependency changed: " + name)
    if wheel_manifest != frozen_wheels["manifest"]:
        raise ValueError("Offline build wheel identities differ from the reference")
    profile_path = Path(public_tools or manifest["native"]["public_tools"]["path"]).resolve()
    profile = freeze_profile(profile_path)
    if profile["digest"] != manifest["native"]["public_tools"]["digest"]:
        raise ValueError("Public native tool profile differs from the reference")
    current_builder = builder_fingerprint((Path(root) / "repoforge/evaluation/generation.py").read_text())
    if current_builder != manifest["native"]["index_store"]["builder_sha256"]:
        raise ValueError("Index builder changed; the 30 reference indexes are not reusable")
    from repoforge.evaluation.bge_identity import validate_identity
    validate_identity(Path(root), manifest)
    return {"environment_root": str(reference), "index_settings": settings,
            "build_wheels_manifest_sha256": frozen_wheels["manifest_sha256"],
            "public_tools_digest": profile["digest"], "builder_sha256": current_builder}


def _verify_reused_indexes(root, output, manifest):
    """Prove all 30 dev indexes resolve under the new frozen profile."""
    from repoforge.evaluation.experiment_native import shared_index
    from repoforge.index.repository_store import EMPTY_PATCH, RepositoryIndexStore, index_profile
    receipts = read_json(output / "provision-native.json")
    seen, rows = set(), []
    for identity in manifest["split"]["dev"]:
        prepared = receipts.get(identity)
        if not prepared:
            raise ValueError("Missing reused environment receipt: " + identity)
        NativeProvisioner.verify(prepared["environment"])
        profile = index_profile(manifest, setup_patch_sha256=prepared["environment"].get(
            "setup_patch_sha256", EMPTY_PATCH))
        index = RepositoryIndexStore(manifest["native"]["index_store"]["root"]).find(
            prepared["public"], profile)
        if index is None:
            raise ValueError("Reference index is missing for " + identity)
        seen.add(identity)
        rows.append({"instance_id": identity, "index": str(index),
                     "profile_id": RepositoryIndexStore.profile_id(profile)})
    if len(seen) != 30:
        raise ValueError("All 30 development indexes must be reusable")
    return rows


def prepare(root, output, reference, *, environment_root=None, index_settings=None,
            build_wheels=None, public_tools=None, analysis_plan=None):
    """Create a new immutable T7 candidate from the frozen reference candidate."""
    root, output, reference = Path(root).resolve(), Path(output).resolve(), Path(reference).resolve()
    old = load_manifest(reference)
    if output.exists():
        raise ValueError("Use a new T7 candidate; identities are immutable")
    if old.get("execution_backend") != "native" or old.get("grading_backend") != "native":
        raise ValueError("T7 requires the native execution and grading backend")
    if old.get("t6") or old.get("t7"):
        raise ValueError("The reference must be the historical T4/T5 candidate, not T6/T7")
    split = old["split"]
    if len(split["dev"]) != 30 or len(split["holdout"]) != 100 or set(split["dev"]) & set(split["holdout"]):
        raise ValueError("T7 requires the frozen dev30/holdout100 without overlap")
    if not set(SMOKE_IDS) <= set(split["dev"]):
        raise ValueError("T7 smoke must stay inside the development set")
    inputs = _check_reference_inputs(
        root, Path(old["native"]["environment_root"]).resolve(), old,
        environment_root=environment_root, index_settings=index_settings,
        build_wheels=build_wheels, public_tools=public_tools)
    from repoforge.evaluation.t6 import query_compatibility
    compatibility = query_compatibility(root, reference)
    plan_path = Path(analysis_plan or root / ANALYSIS_PLAN)
    if not plan_path.is_file():
        raise ValueError("The pre-registered analysis plan is required")
    plan_path = plan_path.resolve()
    analysis_sha = sha(plan_path)
    try:
        plan_relative = str(plan_path.relative_to(root))
    except ValueError:
        plan_relative = str(plan_path)

    output.mkdir(parents=True)
    manifest = deepcopy(old)
    manifest.pop("manifest_hash")
    manifest["created"] = now()
    manifest["source"] = freeze_source(root, output)
    manifest["baseline_commit"] = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest["diff"] = __import__("subprocess").check_output(
        ["git", "diff", "HEAD", "--", "repoforge", "scripts"], cwd=root, text=True)
    manifest["environment"]["python"] = __import__("sys").version
    import importlib.metadata
    manifest["environment"]["dependencies"] = {
        dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]}
    manifest["profiles"] = {**deepcopy(PROFILES), **deepcopy(T7_PROFILES)}
    variants = list(dict.fromkeys([*SMOKE_VARIANTS, *T7_DEV_VARIANTS]))
    manifest["agent_configs"] = {variant: agent_config(root, variant, manifest["model_config"])
                                 for variant in variants}
    manifest["smoke_ids"] = list(SMOKE_IDS)
    manifest.pop("t6", None)
    manifest["t7"] = {
        "protocol": PROTOCOL,
        "reference_candidate": str(reference),
        "historical_manifest": old["manifest_hash"],
        "smoke_variants": list(SMOKE_VARIANTS),
        "cli_variants": ["a0"],
        "dev_variants": list(T7_DEV_VARIANTS),
        "dev_repeats": 1,
        "dev_order": "smoke-first-v1",
        "holdout_variants": list(T7_HOLDOUT_VARIANTS),
        "holdout_repeats": 1,
        "selection_rule": SELECTION_RULE,
        "falsification_rule": FALSIFICATION_RULE,
        "small_sample_statement": SMALL_SAMPLE_STATEMENT,
        "failure_labels": list(FAILURE_LABELS),
        "analysis_plan": plan_relative,
        "analysis_plan_sha256": analysis_sha,
    }
    manifest["holdout_state"] = "locked"
    manifest["official_harness"] = "not_run"
    manifest["native"]["index_store"] = deepcopy(old["native"]["index_store"])
    manifest["native"]["index_store"]["query_only_compatibility"] = compatibility
    manifest["native"]["build_wheels"] = deepcopy(old["native"]["build_wheels"])
    manifest["native"]["public_tools"] = deepcopy(old["native"]["public_tools"])
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "jobs.json", expected_jobs(manifest))

    # Reuse verified immutable public environments. Receipts are re-signed to
    # the new source identity; environments are never rebuilt here.
    receipts = {}
    for identity in split["dev"]:
        item = deepcopy(read_json(reference / "provision-native.json")[identity])
        NativeProvisioner.verify(item["environment"])
        item["source_sha256"] = manifest["source"]["sha256"]
        receipts[identity] = item
    write_json(output / "provision-native.json", receipts)
    for name in ("dataset-receipt.json", "host-install-recipes.json"):
        source = reference / name
        if source.exists():
            write_json(output / name, read_json(source))
    cache = reference / "host-dataset-cache"
    if cache.exists():
        shutil.copytree(cache, output / "host-dataset-cache", symlinks=True)
    write_json(output / "query-compatibility.json", compatibility)
    indexes = _verify_reused_indexes(root, output, manifest)

    # Copy the evidence-backed dev registrations and their raw evidence.
    registrations = {}
    for name in (ANOMALY_NAME, ENV_BLOCKED_NAME):
        source = reference / name
        data = read_json(source) if source.exists() else {}
        for identity, entry in data.items():
            entry["phase"] = "dev"
            entry["_copied_from"] = str(source)
            record = entry.get("record")
            if record:
                if not (root / record).exists():
                    raise ValueError("Registration record is missing: " + record)
                entry["record_sha256"] = sha(root / record)
        write_json(output / name, data)
        registrations[name] = {"sha256": sha(output / name), "entries": sorted(data)}
    evidence_source = reference / "anomaly-evidence"
    if evidence_source.exists():
        shutil.copytree(evidence_source, output / "anomaly-evidence")
        registrations["anomaly-evidence"] = {
            "files": {str(path.relative_to(output)): sha(path)
                      for path in sorted((output / "anomaly-evidence").rglob("*")) if path.is_file()}}
    # Freeze the controller and analysis tools that will run inside the candidate.
    controller_source = root / "scripts/t7_controller_template.py"
    analysis_source = root / "scripts/t7_analysis.py"
    shutil.copy2(controller_source, output / "controller.py")
    tools = output / "analysis-tools"
    tools.mkdir(parents=True)
    shutil.copy2(analysis_source, tools / "t7_analysis.py")
    write_json(output / "controller-revision.json", {
        "created": now(), "manifest_hash": manifest["manifest_hash"],
        "controller": {"source": "scripts/t7_controller_template.py",
                       "sha256": sha(output / "controller.py")},
        "analysis": {"source": "scripts/t7_analysis.py",
                     "sha256": sha(tools / "t7_analysis.py")},
        "source_files": {name: manifest["source"]["files"][name] for name in (
            "scripts/t7_controller_template.py", "scripts/t7_analysis.py")},
    })
    write_json(output / "t7-prepare-audit.json", {
        "created": now(), "manifest_hash": manifest["manifest_hash"],
        "reference": {"path": str(reference), "manifest_hash": old["manifest_hash"],
                      "source_sha256": old["source"]["sha256"]},
        "inputs": inputs, "query_compatibility": compatibility,
        "indexes": indexes, "registrations": registrations,
        "analysis_plan": {"path": plan_relative, "sha256": analysis_sha},
        "provider_calls": 0, "official_harness": "not_run",
    })
    job_audit(output, manifest)
    return manifest


def job_audit(output, manifest=None):
    """Audit the full 15/5/150/500 matrix before any real run."""
    output = Path(output)
    manifest = manifest or load_manifest(output)
    t7 = manifest.get("t7") or {}
    if t7.get("protocol") != PROTOCOL:
        raise ValueError("T7 protocol block missing or changed")
    jobs = expected_jobs(manifest)
    counts = {phase: sum(job["phase"] == phase for job in jobs)
              for phase in ("smoke", "cli_smoke", "dev", "holdout")}
    if counts != {"smoke": 15, "cli_smoke": 5, "dev": 150, "holdout": 500} or len(jobs) != 670:
        raise ValueError("T7 job matrix is not 15/5/150/500")
    if manifest.get("holdout_state") != "locked" or manifest.get("official_harness") != "not_run":
        raise ValueError("T7 freezes the holdout as locked and the official Harness as not_run")
    expected_variants = {
        "smoke": list(SMOKE_VARIANTS), "cli_smoke": ["a0"],
        "dev": list(T7_DEV_VARIANTS), "holdout": list(T7_HOLDOUT_VARIANTS)}
    for phase, variants in expected_variants.items():
        if sorted({job["variant"] for job in jobs if job["phase"] == phase}) != sorted(variants):
            raise ValueError("T7 variant composition changed in " + phase)
    ordered_dev = [*SMOKE_IDS, *[identity for identity in manifest["split"]["dev"]
                                 if identity not in set(SMOKE_IDS)]]
    for variant in T7_DEV_VARIANTS:
        rows = [job for job in jobs if job["phase"] == "dev" and job["variant"] == variant]
        if [job["instance_id"] for job in rows] != ordered_dev or {job["repeat"] for job in rows} != {0}:
            raise ValueError("T7 dev order is not smoke-first-v1 for " + variant)
        if any(manifest["profiles"][variant].get("action_policy") is None for _ in [0]):
            raise ValueError("T7 variant lacks its action policy: " + variant)
    if (manifest["cli_config"]["max_iterations"] != 40
            or manifest["cli_config"]["max_cost"] != 1.0):
        raise ValueError("T7 requires the frozen 40-round/$1 soft budget")
    for variant in T7_DEV_VARIANTS:
        expected = {"version": 1, "variant": "off" if variant == "a0" else variant}
        if manifest["profiles"][variant].get("action_policy") != expected:
            raise ValueError("T7 action policy differs for " + variant)
    if manifest["source"]["files"].get("scripts/t7_analysis.py") != sha(
            Path(__file__).resolve().parents[2] / "scripts/t7_analysis.py"):
        raise ValueError("T7 analysis tool changed after freezing")
    registration = {}
    anomaly_path, blocked_path = output / ANOMALY_NAME, output / ENV_BLOCKED_NAME
    if anomaly_path.exists() and blocked_path.exists() and (
            set(read_json(anomaly_path)) & set(read_json(blocked_path))):
        raise ValueError("A job cannot be both a grading anomaly and environment-blocked")
    for name in (ANOMALY_NAME, ENV_BLOCKED_NAME):
        path = output / name
        data = read_json(path) if path.exists() else {}
        for identity, entry in data.items():
            if entry.get("phase") != "dev":
                raise ValueError("Copied T7 dev registrations must stay phase=dev")
            record = entry.get("record")
            if not record or not (Path(__file__).resolve().parents[2] / record).exists():
                raise ValueError("Registration record is missing: " + str(record))
        registration[name] = {"entries": sorted(data),
                              "sha256": sha(path) if path.exists() else None}
    audit = {"created": now(), "manifest_hash": manifest["manifest_hash"], "protocol": PROTOCOL,
             "total": len(jobs), "counts": counts, "holdout_state": "locked",
             "official_harness": "not_run", "dev_order": "smoke-first-v1",
             "budget": {"max_iterations": 40, "max_cost": 1.0},
             "variants": {phase: list(variants) for phase, variants in expected_variants.items()},
             "registrations": registration,
             "controller": read_json(output / "controller-revision.json")
             if (output / "controller-revision.json").exists() else None,
             "provider_calls": 0}
    write_json(output / "jobs-audit.json", audit)
    return audit


def status(output):
    output = Path(output)
    manifest = load_manifest(output)
    jobs = expected_jobs(manifest)
    rows = {}
    for phase in ("smoke", "cli_smoke", "dev", "holdout"):
        selected = [job for job in jobs if job["phase"] == phase]
        rows[phase] = {
            "selected": len(selected),
            "generated": sum((output / "runs" / job["job_id"] / "answer.json").exists()
                             for job in selected),
            "graded": sum((output / "runs" / job["job_id"] / "grading.json").exists()
                          for job in selected),
        }
    lock = read_json(output / LOCK_NAME) if (output / LOCK_NAME).exists() else None
    adjudicated = (output / HOLDOUT_ADJUDICATION).exists()
    return {"protocol": PROTOCOL, "manifest_hash": manifest["manifest_hash"],
            "holdout_state": manifest.get("holdout_state"), "phases": rows,
            "lock": {"present": lock is not None,
                     "winner": lock.get("winner") if lock else None,
                     "control": lock.get("control") if lock else None,
                     "lock_hash": lock.get("lock_hash") if lock else None},
            "holdout_adjudicated": adjudicated,
            "analysis_plan_sha256": manifest["t7"]["analysis_plan_sha256"],
            "official_harness": "not_run"}


# ---------------------------------------------------------------------------
# Immutable holdout lock


def _lock_hash(lock):
    return content_hash({key: value for key, value in lock.items() if key != "lock_hash"})


def _analysis_files(output):
    root = Path(output) / "analysis/t7"
    return {name: sha(root / name) for name in
            ("labels.jsonl", "stats.json", "paired.json", "recommendation.json", "report.md")
            if (root / name).exists()}


def validate_lock(output, manifest=None):
    output = Path(output)
    path = output / LOCK_NAME
    if not path.exists():
        raise ValueError("T7 holdout requires a frozen lock; no lock exists")
    lock = read_json(path)
    if lock.get("protocol") != PROTOCOL or lock.get("version") != 1:
        raise ValueError("Unknown T7 lock protocol")
    if lock.get("lock_hash") != _lock_hash(lock):
        raise ValueError("T7 lock changed after it was written")
    manifest = manifest or load_manifest(output)
    if lock.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("T7 lock belongs to another candidate")
    if manifest.get("holdout_state") != "locked":
        raise ValueError("A T7 manifest must keep holdout_state=locked")
    if lock.get("control") != "a0" or lock.get("winner") not in T7_DEV_VARIANTS:
        raise ValueError("T7 lock requires a known winner and the a0 control")
    if lock.get("analysis_plan_sha256") != manifest["t7"].get("analysis_plan_sha256"):
        raise ValueError("T7 lock and manifest cite different pre-registration documents")
    report = output / "report-dev.json"
    if not report.exists() or sha(report) != lock.get("dev_report_sha256"):
        raise ValueError("Development report changed after locking")
    recorded = lock.get("analysis") or {}
    actual = _analysis_files(output)
    if recorded != actual:
        raise ValueError("Development analysis changed after locking")
    if not lock.get("user_confirmation", {}).get("confirmed"):
        raise ValueError("T7 lock requires explicit user confirmation")
    if lock.get("holdout_runs_at_lock") != 0:
        raise ValueError("T7 lock must be written before any holdout run")
    if not lock.get("selection") or not lock.get("falsification"):
        raise ValueError("T7 lock must record selection and falsification conclusions")
    return lock


def _holdout_effects(output, jobs):
    return [job["job_id"] for job in jobs if job["phase"] == "holdout"
            and any((output / "runs" / job["job_id"] / name).exists()
                    for name in ("generation.json", "generation-worker-started.json", "answer.json"))]


def create_lock(output, manifest, *, winner, confirm_user, note=None):
    """Write the immutable lock after every dev job and analysis artifact verifies."""
    output = Path(output)
    if not confirm_user:
        raise ValueError("Locking the holdout requires --confirm-user")
    path = output / LOCK_NAME
    if path.exists():
        raise ValueError("T7 lock is immutable; create a new candidate to revise it")
    dev_jobs = [job for job in expected_jobs(manifest) if job["phase"] == "dev"]
    if len(dev_jobs) != 150:
        raise ValueError("T7 lock requires the complete 150-job dev matrix")
    effects = _holdout_effects(output, expected_jobs(manifest))
    if effects:
        raise ValueError("Holdout effects exist before locking: " + ",".join(effects))
    for job in dev_jobs:
        accept_job(output, manifest, job, require_locked=False)
    report_path = output / "report-dev.json"
    if not report_path.exists():
        raise ValueError("Run the dev report before locking")
    report = read_json(report_path)
    if report.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("Development report belongs to another candidate")
    analysis = output / "analysis/t7"
    recommendation = read_json(analysis / "recommendation.json")
    stats = read_json(analysis / "stats.json")
    if (recommendation.get("manifest_hash") != manifest["manifest_hash"]
            or recommendation.get("analysis_plan_sha256") != manifest["t7"]["analysis_plan_sha256"]):
        raise ValueError("Recommendation does not belong to this pre-registered candidate")
    computed = select_winner(stats)
    if recommendation.get("winner") != computed["winner"]:
        raise ValueError("Recommendation winner does not follow the pre-registered rule")
    if winner != computed["winner"]:
        raise ValueError(f"Requested winner {winner} differs from the pre-registered winner {computed['winner']}")
    if computed["degraded_fallback"] and winner != "a0":
        raise ValueError("Only a0 may be locked as the degraded fallback")
    lock = {
        "version": 1, "protocol": PROTOCOL, "created": now(),
        "manifest_hash": manifest["manifest_hash"], "winner": winner, "control": "a0",
        "dev_report_sha256": sha(report_path), "analysis": _analysis_files(output),
        "analysis_plan_sha256": manifest["t7"]["analysis_plan_sha256"],
        "selection": {"rule": manifest["t7"]["selection_rule"],
                      "ranking": computed["ranking"], "winner": winner,
                      "degraded_fallback": computed["degraded_fallback"],
                      "falsified_variants": computed["falsified_variants"]},
        "falsification": {"rule": manifest["t7"]["falsification_rule"],
                          "conclusion": recommendation["falsification"]},
        "small_sample": manifest["t7"]["small_sample_statement"],
        "holdout_runs_at_lock": 0,
        "user_confirmation": {"confirmed": True, "confirmed_at": now(),
                              "statement": "User confirmed the pre-registered winner before holdout",
                              "note": note},
        "holdout_state": "locked", "official_harness": "not_run",
    }
    lock["lock_hash"] = _lock_hash(lock)
    # Exclusive creation: a concurrent or repeated lock attempt cannot replace it.
    with path.open("x", encoding="utf-8") as stream:
        import json
        json.dump(lock, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        __import__("os").fsync(stream.fileno())
    return lock


def assert_holdout_job_allowed(output, manifest, job, lock=None):
    output = Path(output)
    lock = lock or validate_lock(output, manifest)
    if job["phase"] != "holdout":
        raise ValueError("Only frozen holdout jobs may use the locked runner")
    if job["variant"] not in {lock["winner"], lock["control"]}:
        raise ValueError("Holdout arm is not locked: " + job["variant"])
    frozen = {item["job_id"]: item for item in expected_jobs(manifest)}
    if frozen.get(job["job_id"]) != job:
        raise ValueError("Holdout job is not part of the frozen matrix")
    jobs = expected_jobs(manifest)
    leaked = _holdout_effects(output, [item for item in jobs if item["variant"] not in {
        lock["winner"], lock["control"]}])
    if leaked:
        raise ValueError("Unlocked holdout arm has run effects: " + ",".join(leaked))
    return lock


def select_winner(stats):
    """Deterministic pre-registered selection; shared by analysis and locking."""
    variants = stats.get("variants") or {}
    missing = [variant for variant in T7_DEV_VARIANTS if variant not in variants]
    if missing:
        raise ValueError("Variant statistics are incomplete: " + ",".join(missing))
    counts = {variant: variants[variant].get("resolved", 0) for variant in T7_DEV_VARIANTS}
    base = counts["a0"]
    eligible = [variant for variant in ("a1", "a2", "a3", "a4") if counts[variant] > base]

    def cost(variant):
        value = variants[variant].get("known_reference_cost")
        complete = variants[variant].get("cost_complete") is True
        return value if complete and value is not None else float("inf")

    def verification(variant):
        value = variants[variant].get("effective_verification_rate")
        return value if isinstance(value, (int, float)) else -1.0

    if eligible:
        winner = sorted(eligible, key=lambda variant: (-counts[variant], cost(variant),
                                                       -verification(variant), variant))[0]
        degraded = False
        falsified = [variant for variant in ("a1", "a2", "a3", "a4") if variant not in eligible]
    else:
        winner, degraded = "a0", True
        falsified = ["a1", "a2", "a3", "a4"]
    ranking = sorted(T7_DEV_VARIANTS, key=lambda variant: (-counts[variant], cost(variant),
                                                          -verification(variant), variant))
    return {"winner": winner, "degraded_fallback": degraded, "falsified_variants": falsified,
            "counts": counts, "ranking": [{"variant": variant, "resolved": counts[variant],
                                           "known_reference_cost": variants[variant].get("known_reference_cost"),
                                           "cost_complete": variants[variant].get("cost_complete"),
                                           "effective_verification_rate": variants[variant].get(
                                               "effective_verification_rate")}
                                          for variant in ranking]}


# ---------------------------------------------------------------------------
# Acceptance (shared by dev and holdout)


def _registration(output, name, job):
    path = Path(output) / name
    data = read_json(path) if path.exists() else {}
    entry = data.get(job["instance_id"])
    if not entry or entry.get("phase") != job["phase"]:
        return None
    arms = entry.get("arms")
    if arms and job["variant"] not in arms:
        return None
    return entry


def _record_exists(root, entry):
    record = entry.get("record")
    return bool(record) and (Path(root) / record).exists()


def accept_job(output, manifest, job, *, require_locked=False, root=None):
    """Unified acceptance for smoke/cli/dev/holdout jobs.

    A policy stop is a completed business result.  Registered environment-blocked
    instances may end either with environment_error or with a normal completed
    grade; the latter follows the ordinary terminal checks.
    """
    output = Path(output)
    root = Path(root or Path(__file__).resolve().parents[2])
    if require_locked:
        assert_holdout_job_allowed(output, manifest, job)
    from repoforge.evaluation.configuration_bindings import tool_bindings
    from repoforge.evaluation.generation import validate_generation_health
    from repoforge.message_history.policy import ContextPolicy
    from repoforge.policy import ActionPolicyConfig

    directory = output / "runs" / job["job_id"]
    answer = verify_answer(directory, job)
    validate_generation_health(directory)
    grade = read_json(directory / "grading.json")
    own = read_json(directory / "own-tests.json")
    usage = read_json(directory / "usage.json")
    loaded = read_json(directory / "loaded-config.json")
    profile = manifest["profiles"][job["variant"]]
    cli = manifest["cli_config"]
    managed = job["entrypoint"] == "cli" or profile["managed"]
    expected_flow = ("LangGraphFlow" if job["entrypoint"] == "cli" or profile["flow"] == "graph"
                     else "AgenticLoop")
    expected_policy = ActionPolicyConfig(**profile.get(
        "action_policy", {"variant": "off"})).model_dump(mode="json")
    # Legacy loop has no ActionPolicy object; its loader records the minimal
    # disabled marker. It can only run with the policy off.
    loaded_policy = loaded.get("action_policy")
    policy_ok = (loaded_policy == expected_policy if expected_flow == "LangGraphFlow"
                 else loaded_policy in ({"variant": "off"}, expected_policy))
    expected_memory = ("ManagedMessageHistoryGenerator" if managed
                       else "MessageHistoryGenerator")
    expected_context = (ContextPolicy(**cli["context_policy"]).model_dump(mode="json")
                        if managed else None)
    tools = tool_bindings(manifest, job)
    checks = {
        "answer": grade.get("answer_hash") == answer["answer_hash"],
        "grade": grade.get("grading_backend") == "native"
                 and grade.get("evaluation_protocol") == manifest["evaluation_protocol"]
                 and grade.get("grader_source_sha256") == manifest["source"]["files"].get(
                     "repoforge/evaluation/native_grading.py"),
        "own": own.get("answer_hash") == answer["answer_hash"]
               and own.get("hidden_tests") is False
               and own.get("status") in {"executed", "not_run"},
        "cost": usage.get("cost_complete") is True,
        "official": grade.get("official_harness") == "not_run"
                    and grade.get("official_resolved") is None,
        "budget": loaded.get("max_iterations") == cli["max_iterations"]
                  and loaded.get("max_cost") == cli["max_cost"],
        "action_policy": policy_ok,
        "identity": loaded.get("manifest_hash") == manifest["manifest_hash"]
                    and loaded.get("job_id") == job["job_id"],
        "flow": loaded.get("flow_type") == expected_flow,
        "memory": loaded.get("memory_type") == expected_memory,
        "context": loaded.get("policy") == expected_context,
        "tools": set(loaded.get("tool_names", [])) == set(tools)
                 and set(loaded.get("schema_names", [])) == set(tools.values()),
        "isolation": loaded.get("tool_isolation") == "native-offline-v1",
    }
    anomaly = _registration(output, ANOMALY_NAME, job)
    blocked = _registration(output, ENV_BLOCKED_NAME, job)
    if anomaly and blocked:
        raise ValueError("Job is registered as both anomaly and environment-blocked")
    if anomaly:
        checks["anomaly_grading"] = grade.get("local_grading") in {"completed", "patch_error"} and \
            grade.get("original_baseline", {}).get("original_failure_reproduced") is False
        checks["anomaly_record"] = _record_exists(root, anomaly)
    elif blocked:
        checks["env_blocked_record"] = _record_exists(root, blocked)
        if grade.get("local_grading") == "environment_error":
            # Documented upstream/environment blockage: retained in the denominator,
            # never counted as resolved and never re-provisioned silently.
            checks["env_blocked_grading"] = True
        else:
            # The same registered instance may also finish normally (for example a
            # variant that avoids a candidate-side evaluator crash).
            checks["original_failure"] = grade.get("original_baseline", {}).get(
                "original_failure_reproduced") is True
            checks["terminal"] = grade.get("local_grading") in {"completed", "patch_error"} and \
                type(grade.get("local_resolved")) is bool
    else:
        checks["original_failure"] = grade.get("original_baseline", {}).get(
            "original_failure_reproduced") is True
        checks["terminal"] = grade.get("local_grading") in {"completed", "patch_error"} and \
            type(grade.get("local_resolved")) is bool
    for relative, expected in {**grade.get("evidence", {}),
                               **grade.get("original_baseline", {}).get("evidence", {}),
                               **own.get("evidence", {})}.items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or sha(path) != expected:
            raise ValueError("Raw evidence changed: " + relative)
    if not all(checks.values()):
        import json
        raise ValueError("Job acceptance failed: " + json.dumps(checks))
    return {"job_id": job["job_id"], "phase": job["phase"], "variant": job["variant"],
            "instance_id": job["instance_id"], "answer_hash": answer["answer_hash"],
            "generation_status": read_json(directory / "generation.json").get("generation_status"),
            "finish_reason": read_json(directory / "generation.json").get("finish_reason"),
            "local_resolved": grade.get("local_resolved"), "reference_cost": usage.get("reference_cost"),
            "anomaly": anomaly.get("record") if anomaly else None,
            "env_blocked": blocked.get("record") if blocked else None,
            "checks": checks, "timestamp": now()}


# ---------------------------------------------------------------------------
# Holdout provisioning, scanning and adjudication


def provision_holdout(output, manifest):
    """Provision all 100 holdout instances; failures stay on record, never replaced."""
    output = Path(output)
    lock = validate_lock(output, manifest)
    inventory = export_public_dataset(output, manifest, "holdout")
    recipes = read_json(output / "host-install-recipes.json") if (
        output / "host-install-recipes.json").exists() else {}
    path = output / "provision-native.json"
    receipts = read_json(path) if path.exists() else {}
    provisioner = NativeProvisioner(manifest["native"]["environment_root"],
                                    public_tools=manifest["native"].get("public_tools"))
    failures, done = {}, 0
    for identity in manifest["split"]["holdout"]:
        if identity in receipts:
            NativeProvisioner.verify(receipts[identity]["environment"])
            done += 1
            continue
        try:
            receipt = asyncio.run(provisioner.prepare(
                inventory[identity], environment_setup_commit=recipes.get(identity, {}).get(
                    "environment_setup_commit")))
        except Exception as exc:  # preserve the failed preparation directory and continue
            failures[identity] = {"type": type(exc).__name__, "message": str(exc), "timestamp": now()}
            write_json(output / "holdout-provision-failures.json", failures)
            append_json(output / "holdout-provision.jsonl", {"instance_id": identity,
                                                              "status": "failed", **failures[identity]})
            continue
        receipts[identity] = {"environment": receipt, "source_sha256": manifest["source"]["sha256"],
                              "public": inventory[identity]}
        write_json(path, receipts)
        done += 1
        append_json(output / "holdout-provision.jsonl", {"instance_id": identity,
                                                          "status": "ready", "timestamp": now()})
    summary = {"manifest_hash": manifest["manifest_hash"], "lock_hash": lock["lock_hash"],
               "selected": len(manifest["split"]["holdout"]), "ready": done,
               "failed": sorted(failures), "provider_calls": 0, "timestamp": now()}
    write_json(output / "holdout-provision-summary.json", summary)
    return summary


def _wheel_gap(output, result):
    for relative in result.get("evidence", {}):
        if relative.endswith("test-output.txt"):
            path = Path(output) / relative
            if path.exists():
                text = path.read_text(errors="replace")
                if ("No matching distribution found" in text
                        or "Could not find a version that satisfies" in text
                        or "PIP_NO_INDEX" in text):
                    return True
    return False


def classify_scan(result, output):
    """Deterministic pre-admission classification; human adjudication follows."""
    if result.get("provision_failed"):
        return "provision_failed"
    case = result.get("case") or {}
    ftp = result.get("FAIL_TO_PASS") or []
    if not ftp:
        return "empty_or_corrupt_ftp"
    if case.get("status") != "completed":
        return "build_wheel_gap" if _wheel_gap(output, result) else "environment_error"
    if result.get("original_failure_reproduced"):
        return "clean"
    statuses = case.get("test_statuses") or {}
    observed = result.get("observed_ftp_failures") or []
    missing = [name for name in ftp if name not in statuses]
    raw = ""
    for relative in result.get("evidence", {}):
        if relative.endswith("test-output.txt"):
            path = Path(output) / relative
            if path.exists():
                raw = path.read_text(errors="replace")
    if missing and any(name in raw for name in missing):
        return "parse_suspect"
    return "ftp_not_reproduced"


def scan_holdout(output, manifest):
    """Baseline-only scan of all 100 holdout environments; answers are never read."""
    output = Path(output)
    lock = validate_lock(output, manifest)
    from repoforge.evaluation.native_grading import NativeGrader
    grader = NativeGrader(output, manifest)
    receipts = read_json(output / "provision-native.json")
    rows, pending_anomaly, pending_env = [], {}, {}
    for identity in manifest["split"]["holdout"]:
        prepared = receipts.get(identity)
        if not prepared:
            result = {"instance_id": identity, "provision_failed": True, "case": {},
                      "FAIL_TO_PASS": [], "observed_ftp_failures": [],
                      "original_failure_reproduced": False, "evidence": {}}
        else:
            try:
                NativeProvisioner.verify(prepared["environment"])
                result = grader.diagnose(identity)
            except Exception as exc:
                result = {"instance_id": identity, "case": {"status": "environment_error"},
                          "FAIL_TO_PASS": [], "observed_ftp_failures": [],
                          "original_failure_reproduced": False,
                          "evidence": {}, "error": {"type": type(exc).__name__, "message": str(exc)}}
        classification = classify_scan(result, output)
        row = {"instance_id": identity, "classification": classification,
               "case_status": (result.get("case") or {}).get("status"),
               "ftp": len(result.get("FAIL_TO_PASS") or []),
               "observed_ftp_failures": result.get("observed_ftp_failures"),
               "original_failure_reproduced": result.get("original_failure_reproduced"),
               "evidence": result.get("evidence", {}),
               "error": result.get("error")}
        rows.append(row)
        entry = {"record": HOLDOUT_RECORD, "phase": "holdout",
                 "arms": [lock["winner"], lock["control"]],
                 "classification": classification, "scan_sha256": None,
                 "approved_by_user": None, "approved_at": None,
                 "summary": f"Holdout pre-admission scan classified {identity} as {classification}"}
        if classification in {"ftp_not_reproduced", "parse_suspect", "empty_or_corrupt_ftp"}:
            pending_anomaly[identity] = entry
        elif classification in {"environment_error", "build_wheel_gap", "provision_failed"}:
            pending_env[identity] = entry
    scan = {"manifest_hash": manifest["manifest_hash"], "lock_hash": lock["lock_hash"],
            "created": now(), "classifier": "t7-holdout-scan-v1",
            "counts": {name: sum(row["classification"] == name for row in rows) for name in (
                "clean", "ftp_not_reproduced", "parse_suspect", "empty_or_corrupt_ftp",
                "build_wheel_gap", "environment_error", "provision_failed")},
            "rows": rows, "adjudication_required": bool(pending_anomaly or pending_env),
            "provider_calls": 0, "official_harness": "not_run"}
    write_json(output / HOLDOUT_SCAN, scan)
    scan_sha = sha(output / HOLDOUT_SCAN)
    for entry in [*pending_anomaly.values(), *pending_env.values()]:
        entry["scan_sha256"] = scan_sha
    write_json(output / HOLDOUT_PENDING_ANOMALY, pending_anomaly)
    write_json(output / HOLDOUT_PENDING_ENV, pending_env)
    write_json(output / "holdout-adjudication-request.json", {
        "created": now(), "manifest_hash": manifest["manifest_hash"],
        "lock_hash": lock["lock_hash"], "scan_sha256": scan_sha,
        "counts": scan["counts"], "anomaly": sorted(pending_anomaly),
        "environment_blocked": sorted(pending_env),
        "note": "No task is replaced. Every holdout instance stays in the denominator; "
                "registered instances only relax the terminal condition for both locked arms."})
    return scan


def adjudicate(output, manifest, *, confirm_user):
    """Merge the approved holdout registrations; no task selection changes."""
    output = Path(output)
    if not confirm_user:
        raise ValueError("Holdout adjudication requires --confirm-user")
    lock = validate_lock(output, manifest)
    if not (output / HOLDOUT_SCAN).exists():
        raise ValueError("Run scan-holdout before adjudication")
    scan_sha = sha(output / HOLDOUT_SCAN)
    request = read_json(output / "holdout-adjudication-request.json")
    if request.get("scan_sha256") != scan_sha or request.get("lock_hash") != lock["lock_hash"]:
        raise ValueError("Scan changed after the adjudication request; re-scan first")
    anomaly = read_json(output / HOLDOUT_PENDING_ANOMALY)
    blocked = read_json(output / HOLDOUT_PENDING_ENV)
    if set(anomaly) & set(blocked):
        raise ValueError("An instance cannot be both a grading anomaly and environment-blocked")
    now_value = now()
    for name, pending in ((ANOMALY_NAME, anomaly), (ENV_BLOCKED_NAME, blocked)):
        merged = read_json(output / name) if (output / name).exists() else {}
        for identity, entry in pending.items():
            existing = merged.get(identity)
            if existing and existing.get("phase") != "holdout":
                raise ValueError("An instance cannot be registered in two phases: " + identity)
            merged[identity] = {**entry, "approved_by_user": now_value,
                                "approved_at": now_value, "lock_hash": lock["lock_hash"],
                                "scan_sha256": scan_sha}
        write_json(output / name, merged)
    result = {"created": now_value, "manifest_hash": manifest["manifest_hash"],
              "lock_hash": lock["lock_hash"], "scan_sha256": scan_sha,
              "user_confirmation": {"confirmed": True, "confirmed_at": now_value},
              "anomaly": sorted(anomaly), "environment_blocked": sorted(blocked),
              "tasks_replaced": 0, "denominator": len(manifest["split"]["holdout"]),
              "official_harness": "not_run"}
    write_json(output / HOLDOUT_ADJUDICATION, result)
    return result


def prepare_holdout_indexes(output, manifest):
    """Prepare/reuse one shared index per holdout instance before the two arms."""
    output = Path(output)
    lock = validate_lock(output, manifest)
    from repoforge.evaluation.experiment_native import run_job, shared_index, verify_index_receipt
    jobs = [job for job in expected_jobs(manifest) if job["phase"] == "holdout"
            and job["variant"] in {lock["winner"], lock["control"]}]
    chosen = {}
    for identity in manifest["split"]["holdout"]:
        arms = [job for job in jobs if job["instance_id"] == identity]
        if not arms:
            raise ValueError("No locked holdout job for " + identity)
        chosen[identity] = next(job for job in arms if job["variant"] == lock["winner"])
    receipts = read_json(output / "provision-native.json")
    rows, failures = [], {}
    for identity in manifest["split"]["holdout"]:
        prepared = receipts.get(identity)
        if not prepared:
            failures[identity] = "provision_failed"
            rows.append({"instance_id": identity, "status": "provision_failed"})
            continue
        try:
            store, profile, index = shared_index(output, manifest, prepared)
            if index is None:
                run_job(output, manifest, chosen[identity], prepare_index_only=True)
                store, profile, index = shared_index(output, manifest, prepared)
            if index is None:
                raise ValueError("Index preparation published no shared index")
            receipt = verify_index_receipt(index, manifest, prepared["public"],
                                           setup_patch_sha256=prepared["environment"].get(
                                               "setup_patch_sha256"))
            rows.append({"instance_id": identity, "status": "ready", "index": str(index),
                         "digest": receipt["digest"]})
        except Exception as exc:
            failures[identity] = f"{type(exc).__name__}: {exc}"
            rows.append({"instance_id": identity, "status": "failed", "error": failures[identity]})
        write_json(output / "holdout-index-coverage.json", {
            "manifest_hash": manifest["manifest_hash"], "lock_hash": lock["lock_hash"],
            "selected": len(rows), "ready": sum(row.get("status") == "ready" for row in rows),
            "failed": sorted(failures), "rows": rows, "timestamp": now()})
    coverage = read_json(output / "holdout-index-coverage.json")
    blocked = sum(row.get("status") == "provision_failed" for row in rows)
    if coverage["ready"] + blocked != 100:
        raise ValueError("All 100 holdout indexes must be ready before generation")
    coverage["provision_failed"] = blocked
    write_json(output / "holdout-index-coverage.json", coverage)
    return coverage


def run_holdout_job(output, manifest, job, *, resume=False):
    """Run one locked holdout job end to end under the scheduler lock."""
    output = Path(output)
    lock = assert_holdout_job_allowed(output, manifest, job)
    from filelock import FileLock
    from repoforge.evaluation.experiment import validate_gate, validate_execution_source
    root = Path(__file__).resolve().parents[2]
    validate_execution_source(manifest, root)
    validate_gate(output, manifest)
    from repoforge.evaluation.experiment_native import (
        grade_job, release_job_workspace, run_job, validate_agent_tests)
    directory = output / "runs" / job["job_id"]
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        operations = [("generate", lambda: run_job(output, manifest, job, resume=resume)),
                      ("own-tests", lambda: validate_agent_tests(output, manifest, job)),
                      ("grade", lambda: grade_job(output, manifest, job)),
                      ("release", lambda: release_job_workspace(output, manifest, job))]
        for stage, operation in operations:
            append_json(output / "holdout-operations.jsonl", {
                "timestamp": now(), "job_id": job["job_id"], "variant": job["variant"],
                "stage": stage, "status": "started"})
            operation()
            append_json(output / "holdout-operations.jsonl", {
                "timestamp": now(), "job_id": job["job_id"], "variant": job["variant"],
                "stage": stage, "status": "completed"})
        row = accept_job(output, manifest, job, require_locked=True)
        row["lock_hash"] = lock["lock_hash"]
        append_json(output / "verification-holdout.jsonl", row)
        return row


def policy_audit(output, manifest):
    """Verify the actual action policy identity and decision evidence per variant."""
    output = Path(output)
    root = Path(__file__).resolve().parents[2]
    from repoforge.policy import ActionPolicyConfig
    rows, totals = [], {}
    for job in expected_jobs(manifest):
        if job["phase"] not in {"dev", "holdout"}:
            continue
        directory = output / "runs" / job["job_id"]
        if not (directory / "generation.json").exists():
            continue
        expected = ActionPolicyConfig(**manifest["profiles"][job["variant"]].get(
            "action_policy", {"variant": "off"})).model_dump(mode="json")
        loaded = read_json(directory / "loaded-config.json").get("action_policy")
        decisions = read_json(directory / "policy-decisions.json") if (
            directory / "policy-decisions.json").exists() else []
        generation = read_json(directory / "generation.json")
        checks = {
            "configured": loaded == expected,
            "decisions_configured": all(row.get("config") == expected for row in decisions),
            "disabled_has_no_decisions": job["variant"] != "a0" or not decisions,
            "enabled_observed": job["variant"] == "a0" or bool(decisions) or not (
                generation.get("policy_metrics", {}).get("actual_tool_steps")),
        }
        rows.append({"job_id": job["job_id"], "phase": job["phase"], "variant": job["variant"],
                     "instance_id": job["instance_id"], "expected": expected, "loaded": loaded,
                     "decisions": len(decisions), "checks": checks, "passed": all(checks.values())})
        metrics = totals.setdefault(job["variant"], {"jobs": 0, "decisions": 0, "interventions": 0,
                                                     "skipped_proposals": 0, "policy_tool_steps": 0})
        metrics["jobs"] += 1
        metrics["decisions"] += len(decisions)
        values = generation.get("policy_metrics") or {}
        for name in ("interventions", "skipped_proposals", "policy_tool_steps"):
            metrics[name] += values.get(name, 0)
    report = {"manifest_hash": manifest["manifest_hash"], "created": now(),
              "passed": bool(rows) and all(row["passed"] for row in rows),
              "rows": rows, "totals": totals, "provider_calls": 0,
              "official_harness": "not_run",
              "evidence": {str((output / "runs" / row["job_id"] / "loaded-config.json").relative_to(output)):
                           sha(output / "runs" / row["job_id"] / "loaded-config.json")
                           for row in rows if (output / "runs" / row["job_id"] / "loaded-config.json").exists()}}
    write_json(output / "policy-audit.json", report)
    if not report["passed"]:
        raise ValueError("Actual action policy evidence is incomplete")
    return report


def holdout_report(output, manifest):
    output = Path(output)
    lock = validate_lock(output, manifest)
    if not (output / HOLDOUT_ADJUDICATION).exists():
        raise ValueError("Holdout report requires an adjudicated scan")
    result = aggregate(output, "holdout", holdout_unlock=lock)
    result["lock_sha256"] = content_hash(lock)
    result["arms"] = [lock["winner"], lock["control"]]
    executed = {}
    for variant in {lock["winner"], lock["control"]}:
        rows = [row for row in result["rows"] if row["variant"] == variant]
        executed[variant] = {"selected": len(rows),
                             "attempted": sum(row["generation_status"] != "not_run" for row in rows),
                             "graded": sum(row.get("local_grading") == "completed" for row in rows),
                             "resolved": sum(row["resolved"] is True for row in rows)}
    result["executed_arms"] = executed
    result["holdout_state"] = "unlocked_by_t7_lock"
    write_json(output / "report-holdout.json", result)
    return result
