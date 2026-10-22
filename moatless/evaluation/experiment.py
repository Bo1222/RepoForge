"""Frozen T4/T5 identities and evidence, independent of model or Docker execution."""

import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

from moatless.message_history.policy import content_hash

PUBLIC_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement", "version")
SMOKE_IDS = ("pytest-dev__pytest-7490", "django__django-10914", "sympy__sympy-12481",
             "scikit-learn__scikit-learn-14053", "astropy__astropy-8872")
PROFILES = {
    "loop_legacy": {"flow": "loop", "managed": False, "correct_test_prompt": False},
    "graph_legacy": {"flow": "graph", "managed": False, "correct_test_prompt": False},
    "a0": {"flow": "graph", "managed": True, "correct_test_prompt": True},
}
# The native gate requires exactly 3 smoke variants x 5 tasks and 5 CLI A0 jobs.
# T7 policy variants must therefore never enter the smoke job selection.
SMOKE_VARIANTS = ("loop_legacy", "graph_legacy", "a0")
DEV_REPEATS = 3
T7_PROTOCOL = "t7-policy-ablation-v1"
T7_DEV_VARIANTS = ("a0", "a1", "a2", "a3", "a4")
T7_HOLDOUT_VARIANTS = ("a0", "a1", "a2", "a3", "a4")
T7_PROFILES = {
    "a0": {"flow": "graph", "managed": True, "correct_test_prompt": True,
           "action_policy": {"version": 1, "variant": "off"}},
    **{f"a{i}": {"flow": "graph", "managed": True, "correct_test_prompt": True,
                 "action_policy": {"version": 1, "variant": f"a{i}"}} for i in (1, 2, 3, 4)},
    # O2: a3 thresholds + verification-first policy + product-only answer patch.
    "a5": {"flow": "graph", "managed": True, "correct_test_prompt": True,
           "submit_product_only": True,
           "action_policy": {"version": 1, "variant": "a5"}},
}
A6_PROFILES = {"a6": {"flow": "graph", "managed": True, "correct_test_prompt": True,
                       "submit_product_only": True, "action_policy": {"version": 1, "variant": "a6"}}}
# A7: one O2 controller with A6 evidence/obligation/completion gates.  It is a
# new batch protocol; historical a6 results and checkpoints never mix with it.
A7_PROFILES = {"a7": {"flow": "graph", "managed": True, "correct_test_prompt": True,
                       "submit_product_only": True, "action_policy": {"version": 1, "variant": "a7"}}}
ALL_PROFILES = {**PROFILES, **T7_PROFILES, **A6_PROFILES, **A7_PROFILES}
LABELS = {"invalid_repeat", "test_timing", "unverified_finish", "context_omission",
          "edit_error", "budget", "environment", "model_protocol", "unknown"}
PRICES = {"date": "2026-10-12", "currency": "USD", "unit": "per_1M_tokens",
          "reference": {"input": .30, "cache": .006, "output": 1.20},
          "embedding": {"text-embedding-3-small": .02, "jina-embeddings-v3": .05, "BAAI/bge-m3": 0},
          "local_embedding_cost_basis": "external API fee is zero; GPU/CPU hardware cost is unpriced",
          "source": "https://api-docs.deepseek.com/quick_start/pricing/",
          "embedding_source": "https://developers.openai.com/api/docs/models/text-embedding-3-small",
          "embedding_sources": {"jina-embeddings-v3": "https://jina.ai/en-US/embeddings/"},
          "jina_reference_basis": "project reference USD 0.05/1M tokens, based on indexed official standard package; actual key tariff, taxes and bill unknown",
          "basis": "frozen peak reference for budgets; provider bill not available"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication; interrupted writes cannot masquerade as complete answers.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(Path(path).read_text())


def public_instance(row):
    missing = set(PUBLIC_FIELDS) - set(row)
    if missing:
        raise ValueError(f"Missing public fields: {sorted(missing)}")
    public = {key: row[key] for key in PUBLIC_FIELDS}
    if not all(isinstance(value, str) and value for value in public.values()):
        raise ValueError("Public instance fields must be nonempty strings")
    return public


def agent_config(root, variant, model):
    """Freeze actual tool definitions and prompt, without importing any answer data."""
    from copy import deepcopy
    configs = read_json(Path(root) / "moatless/agent/agents.json")
    config = deepcopy(next(item for item in configs if item["agent_id"] == "code_and_test"))
    config.pop("completion", None)
    if ALL_PROFILES[variant]["correct_test_prompt"]:
        for text in ("Tests execute after code changes", "Tests execute after test modifications",
                     "Tests run automatically after each code change."):
            config["system_prompt"] = config["system_prompt"].replace(text, "Run tests explicitly using RunTests.")
        config["system_prompt"] = config["system_prompt"].replace("Tests Run Automatically", "Explicit Test Verification")
    # Identification is JSON rather than the action generator's tool-call schema.
    auxiliary = {**model, "response_format": "json", "max_tokens": 8000,
                 "completion_model_class": "moatless.completion.json.JsonCompletionModel"}
    for action in config["actions"]:
        if "max_identify_tokens" in action:
            action["completion_model"] = {**auxiliary, "max_tokens": action["max_identify_tokens"]}
    if variant in {"a6", "a7"}:
        config["actions"].extend([
            {"action_class": "moatless.actions.register_acceptance_plan.RegisterAcceptancePlan"},
            {"action_class": "moatless.actions.check_syntax.CheckSyntax"},
            {"action_class": "moatless.actions.replan_repair.ReplanRepair"},
            {"action_class": "moatless.actions.explain_test_conflict.ExplainTestConflict"}])
        config["system_prompt"] += ("\nRegisterAcceptancePlan before editing: extract observable criteria from public issue quotes. "
            "Use one explicit test module per invocation and separate mandatory checks for additional modules. "
            "Issue behavior requires base failure/current pass with identical tests. "
            "Never mark criteria passed yourself. A passed check only finishes the task when every obligation is satisfied. "
            "Tests are an observation barrier: wait for results before proposing further edits. "
            "After repeated failed repairs, read new evidence and call ReplanRepair before editing again. "
            "An original assertion conflicting with proven public issue behavior may be recorded with ExplainTestConflict; "
            "this only permits partial delivery and never waives mandatory checks. "
            "When all requirements pass the controller finalizes automatically; hidden resolution is unknown.")
    config["completion_model"] = model
    config["model_timeout"] = model["timeout"]
    config["tool_timeout"] = 120
    return config


def freeze_source(root, output):
    """Archive code/config only; never mount reports, .env or benchmark answers."""
    root, output = Path(root), Path(output)
    paths = [*root.glob("moatless/**/*.py"), *root.glob("moatless/**/*.scm"), *root.glob("scripts/*.py")]
    paths += [path for path in (root / "moatless/utils/_static/tiktoken_cache").glob("*") if path.is_file()]
    paths += [root / name for name in ("pyproject.toml", "uv.lock", "README.md",
                                      "moatless/agent/agents.json", "moatless/completion/model_config.json",
                                      "moatless/evaluation/profiles/flash_nonthinking_a0.json")]
    paths = sorted(set(paths))
    hashes = {str(path.relative_to(root)): sha(path) for path in paths}
    with tarfile.open(output / "source.tar", "w") as archive:
        for path in paths:
            body = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(root)))
            info.size = len(body)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(body))
    return {"path": "source.tar", "sha256": sha(output / "source.tar"),
            "files": hashes, "digest": content_hash(hashes)}


def prepare(root, output, *, execution_backend="docker", environment_root=None, exploration=None,
            embedding_service=None, embedding_settings=None, build_wheels=None, public_tools=None,
            repair_profile=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if execution_backend not in {"docker", "native"}:
        raise ValueError("Unknown execution backend")
    if repair_profile is not None and (
            repair_profile not in {"a6", "a7"} or execution_backend != "native" or exploration):
        raise ValueError("Repair profiles use a new native batch manifest; no CLI or shared-snapshot exploration")
    if output.exists():
        raise ValueError("Use a new experiment directory; historical identities cannot be overwritten")
    output.mkdir(parents=True)
    # The living split definition first: frozen_split.json carries the dev-cohort
    # revision record (2026-10-16 matplotlib replacement and provenance), while the
    # E1 manifest stays the untouched historical record (its hash is cited inside
    # frozen_split.json).
    historical_path = root / "moatless/evaluation/profiles/frozen_split.json"
    if not historical_path.exists():
        historical_path = root / "context_memory_evidence/e1/manifest.json"
    historical = read_json(historical_path)
    split = historical["split"]
    if len(split["dev"]) != 30 or len(split["holdout"]) != 100 or set(split["dev"]) & set(split["holdout"]):
        raise ValueError("Expected frozen dev30/holdout100 without overlap")
    if not set(SMOKE_IDS) <= set(split["dev"]):
        raise ValueError("Smoke must use only the development set")
    if exploration is not None:
        ids = exploration["instance_ids"]
        if execution_backend != "native" or not ids or len(ids) != len(set(ids)):
            raise ValueError("Exploration requires distinct native instances")
        if set(ids) & set(split["holdout"]):
            raise ValueError("Exploration cannot unlock the historical holdout")
    cli = read_json(root / "moatless/evaluation/profiles/flash_nonthinking_a0.json")
    from moatless.index.settings import IndexSettings
    embedding_settings = IndexSettings(**(embedding_settings or {})).model_dump(mode="json")
    model = {"model": cli["model"], "model_base_url": cli["base_url"], "response_format": "tool_call",
             "completion_model_class": "moatless.completion.tool_call.ToolCallCompletionModel",
             "max_tokens": cli["max_tokens"], "temperature": cli["temperature"],
             "timeout": cli["model_timeout"], "params": cli["params"], "message_cache": False,
             "thoughts_in_action": False}
    manifest = {"version": 1, "created": now(), "dataset": historical["dataset"], "split": split,
                "historical_split_manifest": historical["manifest_hash"],
                "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "diff": subprocess.check_output(["git", "diff", "HEAD", "--", "moatless", "scripts", "pyproject.toml", "uv.lock"], cwd=root, text=True),
                "source": freeze_source(root, output), "cli_config": cli, "model_config": model,
                "embedding_settings": embedding_settings,
                "agent_configs": {variant: agent_config(root, variant, model) for variant in PROFILES},
                "expected_model": {"id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash", "context_window": 1048576},
                "profiles": PROFILES, "prices": PRICES, "generation_fields": PUBLIC_FIELDS,
                "environment": {"python": sys.version, "dependencies": {
                    dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]}},
                "smoke_ids": SMOKE_IDS, "holdout_state": "locked", "official_harness": "not_run"}
    if execution_backend == "native":
        import platform
        manifest.update(version=2, execution_backend="native", grading_backend="native",
                        evaluation_protocol="native-swebench-compatible-v1")
        manifest["native"] = {"environment_root": str(Path(environment_root or output / "environments").resolve()),
            "framework_prefix": str(Path(sys.prefix).resolve()), "framework_base_prefix": str(Path(sys.base_prefix).resolve()),
            "host_os": platform.platform(), "machine": platform.machine(),
            "environment_receipts": "provision-native.json", "initial_concurrency": 1}
        if build_wheels and not (Path(build_wheels) / "manifest.json").is_file():
            raise ValueError("Explicit offline build wheels require a manifest")
        build_wheels = Path(build_wheels).resolve() if build_wheels else root / "experiments/runtime/native_build_wheels"
        if (build_wheels / "manifest.json").exists():
            wheel_manifest = read_json(build_wheels / "manifest.json")
            if any(sha(build_wheels / name) != digest for name, digest in wheel_manifest["files"].items()):
                raise ValueError("Public offline build wheels changed")
            manifest["native"]["build_wheels"] = {"path": str(build_wheels), "manifest_sha256": sha(build_wheels / "manifest.json"),
                                                 "manifest": wheel_manifest}
        if public_tools:
            from moatless.evaluation.native_public_tools import freeze_profile
            manifest["native"]["public_tools"] = freeze_profile(public_tools)
        manifest['embedding_execution'] = {'jina_tokens_per_minute': 80000,
            'jina_max_retries': 4, 'minute_limit_retry_seconds': 60,
            'public_preparation_cache': 'per-text; preparation-only mount; query cache disabled',
            'source_fetch': 'exact-base HTTP/1.1 for fresh preparations; cached receipts retain actual recipes'}
        if embedding_settings["embed_provider"] == "local":
            manifest['embedding_execution'] = {'provider': 'local', 'device': 'service GPU 0',
                'model_revision': '5617a9f61b028005a4858fdac845db406aefb181',
                'precision': 'float16', 'batch_size': 8, 'max_input_tokens': 2048,
                'overlength': 'reject; no truncation', 'worker': 'HTTP only; no GPU/model mount',
                'external_api_cost_usd': 0, 'hardware_cost': 'unpriced'}
            model_manifest = root / "experiments/runtime/local_bge/model-manifest.json"
            if model_manifest.exists():
                manifest['embedding_execution']['model_artifacts'] = read_json(model_manifest)
                manifest['embedding_execution']['host_files'] = {
                    str(path.relative_to(root)): sha(path) for path in (
                        model_manifest, root / 'scripts/local_bge/requirements.lock',
                        root / 'scripts/local_bge/server.py')}
    if exploration is not None:
        manifest.update(exploration=exploration, smoke_ids=exploration["instance_ids"],
                        evaluation_protocol="native-swebench-shared-snapshot-exploration-v1")
    if embedding_service is not None:
        manifest["embedding_service"] = embedding_service
    if execution_backend == "native" and embedding_settings["embed_provider"] == "local" and embedding_service:
        from moatless.index.repository_store import builder_fingerprint
        manifest["native"]["index_store"] = {
            "root": str(root / "experiments/vector_store"), "format": "repository-index-v1",
            "builder_sha256": builder_fingerprint((root / "moatless/evaluation/generation.py").read_text()),
        }
    if repair_profile == "a6":
        manifest["profiles"] = A6_PROFILES
        manifest["agent_configs"] = {"a6": agent_config(root, "a6", model)}
        manifest["repair"] = {"protocol": "acceptance-repair-v2", "variant": "a6"}
    elif repair_profile == "a7":
        manifest["profiles"] = A7_PROFILES
        manifest["agent_configs"] = {"a7": agent_config(root, "a7", model)}
        manifest["repair"] = {"protocol": "acceptance-repair-v3", "variant": "a7"}
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(output / "manifest.json", manifest)
    jobs = make_jobs(manifest)
    write_json(output / "jobs.json", jobs)
    return manifest, jobs


def _make_t7_jobs(manifest):
    t7 = manifest["t7"]
    if t7.get("protocol") != T7_PROTOCOL:
        raise ValueError("T7 requires the t7-policy-ablation-v1 protocol")
    if list(t7.get("smoke_variants") or []) != list(SMOKE_VARIANTS):
        raise ValueError("T7 smoke composition is frozen at the three legacy variants")
    if list(t7.get("cli_variants") or []) != ["a0"]:
        raise ValueError("T7 CLI stays A0 with the policy off")
    if list(t7.get("dev_variants") or []) != list(T7_DEV_VARIANTS) or t7.get("dev_repeats") != 1:
        raise ValueError("T7 dev matrix is frozen at A0-A4 x dev30 x 1")
    if (list(t7.get("holdout_variants") or []) != list(T7_HOLDOUT_VARIANTS)
            or t7.get("holdout_repeats") != 1):
        raise ValueError("T7 holdout matrix is frozen at A0-A4 x holdout100 x 1")
    if t7.get("dev_order") != "smoke-first-v1":
        raise ValueError("T7 dev order is frozen at smoke-first-v1")
    if not t7.get("selection_rule") or not t7.get("falsification_rule"):
        raise ValueError("T7 requires frozen selection and falsification rules")
    if not t7.get("analysis_plan_sha256") or not t7.get("small_sample_statement"):
        raise ValueError("T7 requires a pre-registered analysis plan with a small-sample statement")
    if manifest.get("holdout_state") != "locked" or manifest.get("official_harness") != "not_run":
        raise ValueError("T7 freezes the holdout as locked and the official Harness as not_run")
    for variant in {*SMOKE_VARIANTS, *t7["dev_variants"], *t7["holdout_variants"]}:
        if variant not in (manifest.get("profiles") or {}):
            raise ValueError("T7 profile is missing: " + variant)
    split = manifest["split"]
    if len(split["dev"]) != 30 or len(split["holdout"]) != 100 or set(split["dev"]) & set(split["holdout"]):
        raise ValueError("T7 requires the frozen dev30/holdout100 without overlap")
    ordered_dev = [*SMOKE_IDS, *[identity for identity in split["dev"] if identity not in set(SMOKE_IDS)]]

    def emit(phase, variant, repeat, instance_id, entrypoint="batch"):
        job = {"phase": phase, "variant": variant, "repeat": repeat, "instance_id": instance_id,
               "manifest_hash": manifest["manifest_hash"], "entrypoint": entrypoint}
        return {**job, "job_id": content_hash(job)[:24]}

    jobs = []
    for variant in SMOKE_VARIANTS:
        for instance_id in manifest["smoke_ids"]:
            jobs.append(emit("smoke", variant, 0, instance_id))
    for variant in t7["cli_variants"]:
        for instance_id in manifest["smoke_ids"]:
            jobs.append(emit("cli_smoke", variant, 0, instance_id, entrypoint="cli"))
    for variant in t7["dev_variants"]:
        for repeat in range(t7["dev_repeats"]):
            for instance_id in ordered_dev:
                jobs.append(emit("dev", variant, repeat, instance_id))
    # The full holdout matrix for every variant is frozen before any dev result
    # exists; only the locked arms are executed later (see t7-lock.json).
    for variant in t7["holdout_variants"]:
        for repeat in range(t7["holdout_repeats"]):
            for instance_id in split["holdout"]:
                jobs.append(emit("holdout", variant, repeat, instance_id))
    return jobs


def make_jobs(manifest):
    jobs = []
    if manifest.get("repair"):
        if manifest.get("t6") or manifest.get("t7") or manifest.get("exploration"):
            raise ValueError("Repair protocol cannot reuse another experiment protocol")
        allowed = ({"protocol": "acceptance-repair-v1", "variant": "a6"},
                   {"protocol": "acceptance-repair-v2", "variant": "a6"},
                   {"protocol": "acceptance-repair-v3", "variant": "a7"})
        if manifest["repair"] not in allowed:
            raise ValueError("Unknown repair protocol")
        variant = manifest["repair"]["variant"]
        expected = A7_PROFILES["a7"] if variant == "a7" else A6_PROFILES["a6"]
        if manifest.get("profiles", {}).get(variant) != expected:
            raise ValueError(f"Repair manifest requires the explicit {variant} profile")
        for instance_id in manifest["split"]["dev"]:
            job = {"phase": "dev", "variant": variant, "repeat": 0, "instance_id": instance_id,
                   "manifest_hash": manifest["manifest_hash"], "entrypoint": "batch"}
            jobs.append({**job, "job_id": content_hash(job)[:24]})
        return jobs
    if manifest.get("t6") and manifest.get("t7"):
        raise ValueError("A manifest cannot be both T6 and T7")
    if manifest.get("t6"):
        if (manifest["t6"].get("protocol") != "t6-a0-smoke-v1"
                or list(manifest["smoke_ids"]) != list(SMOKE_IDS)):
            raise ValueError("T6 requires the original five A0 smoke instances")
        for instance_id in manifest["smoke_ids"]:
            job = {"phase": "smoke", "variant": "a0", "repeat": 0, "instance_id": instance_id,
                   "manifest_hash": manifest["manifest_hash"], "entrypoint": "batch"}
            jobs.append({**job, "job_id": content_hash(job)[:24]})
        return jobs
    if manifest.get("t7"):
        return _make_t7_jobs(manifest)
    for phase in (("smoke",) if manifest.get("exploration") else ("smoke", "dev", "cli_smoke")):
        variants = ["a0"] if manifest.get("exploration") else list(SMOKE_VARIANTS) if phase == "smoke" else ["a0"]
        for variant in variants:
            for repeat in range(DEV_REPEATS if phase == "dev" else 1):
                for instance_id in (manifest["split"]["dev"] if phase == "dev" else manifest["smoke_ids"]):
                    job = {"phase": phase, "variant": variant, "repeat": repeat, "instance_id": instance_id,
                           "manifest_hash": manifest["manifest_hash"], "entrypoint": "cli" if phase == "cli_smoke" else "batch"}
                    job["job_id"] = content_hash(job)[:24]
                    jobs.append(job)
    return jobs


def load_manifest(output):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if content_hash({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("Manifest changed after freezing")
    if sha(output / manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("Frozen source archive changed")
    if read_json(output / "jobs.json") != make_jobs(manifest):
        raise ValueError("Frozen job selection changed")
    return manifest


def validate_execution_source(manifest, root):
    """Host orchestration must match the frozen code; reports remain readable later."""
    for name, expected in manifest["source"]["files"].items():
        if name == "README.md":
            continue
        if sha(Path(root) / name) != expected:
            raise ValueError(f"Execution source changed after freeze: {name}; prepare a new experiment")
    dependencies = ("swebench", "datasets") if manifest.get("execution_backend") == "native" else ("swebench", "docker", "datasets")
    for name in dependencies:
        if importlib.metadata.version(name) != manifest["environment"]["dependencies"][name]:
            raise ValueError(f"Host evaluation dependency changed after freeze: {name}")
    for name, expected in manifest.get('embedding_execution', {}).get('host_files', {}).items():
        if sha(Path(root) / name) != expected:
            raise ValueError(f"Local embedding identity changed after freeze: {name}")


def freeze_answer(directory, job):
    directory = Path(directory)
    if (directory / "answer.json").exists():
        return verify_answer(directory, job)
    result = read_json(directory / "generation.json")
    if result["job_id"] != job["job_id"] or result["manifest_hash"] != job["manifest_hash"]:
        raise ValueError("Generation result belongs to another experiment")
    required = ("patch.diff", "trajectory.json", "usage.json", "events.jsonl", "sdk.jsonl", "generation.json",
                "agent-tests.json", "test-requests.json")
    evidence = {name: sha(directory / name) for name in required}
    for name in ("model-metadata.json", "runtime-ready.json", "index-provenance.json", "policy-decisions.json"):
        if (directory / name).exists():
            evidence[name] = sha(directory / name)
    answer = {"job": job, "evidence": evidence, "frozen": now(), "selected_node_id": result["selected_node_id"]}
    answer["answer_hash"] = content_hash(answer)
    write_json(directory / "answer.json", answer)
    return answer


def verify_answer(directory, job):
    directory = Path(directory)
    answer = read_json(directory / "answer.json")
    if answer["job"] != job or content_hash({k: v for k, v in answer.items() if k != "answer_hash"}) != answer["answer_hash"]:
        raise ValueError("Answer identity changed")
    for name, expected in answer["evidence"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"Frozen answer evidence changed: {name}")
    return answer


def validate_gate(output, manifest):
    """No hand-entered 'pass' flag can substitute for the smoke evidence."""
    if manifest.get("exploration"):
        raise ValueError("Exploration is not a T4 gate and cannot authorize development or holdout runs")
    if manifest.get("execution_backend") == "native":
        from moatless.evaluation.native_gate import validate_native_gate
        return validate_native_gate(output, manifest)
    output = Path(output)
    gate = read_json(output / "t4-gate.json")
    if gate.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("T4 gate belongs to a different source/configuration")
    if not gate.get("local_contracts_passed") or not gate.get("real_recovery_passed") or not gate.get("cli_verified"):
        raise ValueError("T4 local, real recovery and CLI evidence is incomplete")
    for name, expected in gate.get("evidence", {}).items():
        if sha(output / name) != expected:
            raise ValueError("T4 gate evidence changed")
    if not gate.get("evidence"):
        raise ValueError("T4 gate must reference actual evidence files")
    jobs = read_json(output / "jobs.json")
    for job in (j for j in jobs if j["phase"] == "smoke"):
        directory = output / "runs" / job["job_id"]
        answer = verify_answer(directory, job)
        grade = read_json(directory / "grading.json")
        if grade.get("answer_hash") != answer["answer_hash"] or grade.get("official_harness") not in {"completed", "empty_patch"}:
            raise ValueError("All 15 smoke answers require independent completed Harness grading")
        if grade.get("original_baseline", {}).get("status") != "completed":
            raise ValueError("Smoke requires a completed independent original-baseline diagnostic")
    for job in (j for j in jobs if j["phase"] == "cli_smoke"):
        directory = output / "runs" / job["job_id"]
        answer = verify_answer(directory, job)
        validation = read_json(directory / "own-tests.json")
        if validation.get("answer_hash") != answer["answer_hash"] or validation.get("status") != "executed":
            raise ValueError("All five CLI smoke answers require independent agent-test execution")
        if not validation.get("evidence"):
            raise ValueError("CLI validation must reference actual evidence files")
        for name, expected in validation["evidence"].items():
            if sha(output / name) != expected:
                raise ValueError("CLI validation evidence changed")


def aggregate(output, phase="dev", *, repeats=None, holdout_unlock=None):
    """Missing/failed jobs stay in the denominator; CLI and smoke stay separate."""
    output = Path(output)
    manifest = load_manifest(output)
    native = manifest.get("grading_backend") == "native"
    rows = []
    for job in read_json(output / "jobs.json"):
        if job["phase"] != phase:
            continue
        if repeats is not None and job["repeat"] not in repeats:
            continue
        directory = output / "runs" / job["job_id"]
        row = {**job, "generation_status": "not_run", "official_harness": "not_run", "resolved": None}
        if (directory / "generation.json").exists():
            row.update(read_json(directory / "generation.json"))
        elif (directory / "host-failure-generate.json").exists():
            row.update(generation_status="environment_error", error_kind="generation_host",
                       host_failure=read_json(directory / "host-failure-generate.json"))
        if (directory / "usage.json").exists():
            row["usage"] = read_json(directory / "usage.json")
        if (directory / "answer.json").exists():
            answer = verify_answer(directory, job)
            if (directory / "grading.json").exists():
                grade = read_json(directory / "grading.json")
                if grade.get("answer_hash") != answer["answer_hash"]:
                    raise ValueError("Grading result does not match frozen answer")
                row.update(grade)
        if (directory / "labels.json").exists():
            labels = read_json(directory / "labels.json")
            if labels.get("answer_hash") != verify_answer(directory, job)["answer_hash"]:
                raise ValueError("Failure labels do not match frozen answer")
            for item in labels.get("labels", []):
                if item["category"] not in LABELS or not item.get("evidence"):
                    raise ValueError("Failure labels need a known category and trace evidence")
            if sum(bool(item.get("primary")) for item in labels.get("labels", [])) > 1:
                raise ValueError("Each failed run requires at most one primary failure label")
            row["labels"] = labels["labels"]
        if native:
            row["resolved"] = row.get("local_resolved")
            row.setdefault("local_grading", "not_run")
            row.setdefault("official_resolved", None)
        rows.append(row)
    groups = {}
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        resolved = sum(row["resolved"] is True for row in selected)
        costs = [row["usage"]["reference_cost"] for row in selected if row.get("usage") and row["usage"].get("reference_cost") is not None]
        known_cost = sum(row["usage"].get("known_reference_cost", row["usage"].get("reference_cost")) or 0
                         for row in selected if row.get("usage"))
        groups[variant] = {"selected": len(selected), "attempted": sum(row["generation_status"] != "not_run" for row in selected),
                           "generated": sum(row["generation_status"] in {"completed", "failed"} for row in selected),
                           "host_errors": sum(row["generation_status"] == "environment_error" for row in selected),
                           "graded": sum(row.get("local_grading") == "completed" if native else row["official_harness"] == "completed" for row in selected),
                           "resolved": resolved,
                           "resolved_over_selected": resolved / len(selected) if any(row["generation_status"] != "not_run" for row in selected) else None,
                           "known_reference_cost": known_cost, "cost_complete": len(costs) == len(selected),
                           "cost_per_resolved": sum(costs) / resolved if resolved and len(costs) == len(selected) else None,
                           "by_repeat": {str(repeat): {"selected": sum(row["repeat"] == repeat for row in selected),
                               "resolved": (sum(row["repeat"] == repeat and row["resolved"] is True for row in selected)
                                   if any(row["repeat"] == repeat and row['generation_status'] != 'not_run' for row in selected) else None)}
                               for repeat in sorted({row["repeat"] for row in selected})}}
        if native:
            if not any(row.get("local_resolved") is True or row.get("local_resolved") is False for row in selected):
                groups[variant]["resolved_over_selected"] = None
            groups[variant].update(local_resolved=resolved,
                                  local_resolved_over_selected=groups[variant]["resolved_over_selected"],
                                  official_resolved=None, official_resolved_over_selected=None,
                                  grading_backend="native", evaluation_protocol=manifest["evaluation_protocol"])
    # A single primary label per failed instance, counted once despite three repetitions.
    categories = {}
    for identity in sorted({row["instance_id"] for row in rows}):
        primary = [label["category"] for row in rows if row["instance_id"] == identity and row["resolved"] is False
                   for label in row.get("labels", []) if label.get("primary")]
        if primary:
            from collections import Counter
            counts = Counter(primary)
            # Repetition ties must not let lexical category order prefer B.
            action_categories = {"invalid_repeat", "test_timing", "unverified_finish"}
            category = sorted(counts, key=lambda value: (-counts[value], value not in action_categories, value))[0]
            categories[category] = categories.get(category, 0) + 1
    # Variant-scoped summaries: a multi-variant phase (T7) must not merge failure
    # categories or policy metrics across configurations.
    from collections import Counter
    failure_instances_by_variant, finish_reasons_by_variant, policy_metrics_by_variant = {}, {}, {}
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        variant_categories = {}
        for identity in sorted({row["instance_id"] for row in subset}):
            primary = [label["category"] for row in subset if row["instance_id"] == identity and row["resolved"] is False
                       for label in row.get("labels", []) if label.get("primary")]
            if primary:
                counts = Counter(primary)
                action_categories = {"invalid_repeat", "test_timing", "unverified_finish"}
                category = sorted(counts, key=lambda value: (-counts[value], value not in action_categories, value))[0]
                variant_categories[category] = variant_categories.get(category, 0) + 1
        failure_instances_by_variant[variant] = variant_categories
        finish_reasons_by_variant[variant] = dict(Counter(row["finish_reason"] for row in subset if row.get("finish_reason")))
        metrics = {}
        for row in subset:
            for key, value in (row.get("policy_metrics") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key] = metrics.get(key, 0) + value
        policy_metrics_by_variant[variant] = metrics
    action = sum(categories.get(key, 0) for key in ("invalid_repeat", "test_timing", "unverified_finish"))
    context = categories.get("context_omission", 0)
    infrastructure = categories.get("environment", 0) + categories.get("model_protocol", 0)
    labels_complete = all(row["resolved"] is not False or any(label.get("primary") for label in row.get("labels", []))
                          for row in rows)
    grading_complete = all(row.get("local_grading") in {"completed", "patch_error"} and row["resolved"] is not None
                           if native else row["official_harness"] in {"completed", "empty_patch"} for row in rows)
    ready = phase == "dev" and bool(rows) and labels_complete and grading_complete
    selection = None
    if ready and max(action, context) > 0 and max(infrastructure, categories.get("unknown", 0)) < max(action, context):
        selection = "A" if action >= context else "B"
    result = {"manifest_hash": manifest["manifest_hash"], "phase": phase, "rows": rows, "groups": groups,
              "primary_failure_instances": categories, "selected_direction": selection,
              "failure_instances_by_variant": failure_instances_by_variant,
              "finish_reasons_by_variant": finish_reasons_by_variant,
              "policy_metrics_by_variant": policy_metrics_by_variant,
              "failure_labels_complete": labels_complete,
              "hypothesis_status": "ready_for_documented_hypothesis" if selection else "insufficient_evidence",
              "holdout_state": "unlocked_by_t7_lock" if holdout_unlock else "locked", "provider_bill": "not_available"}
    if holdout_unlock:
        result["lock_sha256"] = content_hash(holdout_unlock)
    if native:
        result.update(execution_backend="native", grading_backend="native", evaluation_protocol=manifest["evaluation_protocol"],
                      official_harness="not_run", metric="local_resolved_over_selected")
    write_json(output / f"report-{phase}.json", result)
    return result


def freeze_hypothesis(output):
    output = Path(output)
    report = aggregate(output, "dev")
    direction = report["selected_direction"]
    if direction is None:
        raise ValueError("Complete A0 grading and evidence-backed labels before choosing an algorithm")
    result = {"manifest_hash": report["manifest_hash"], "direction": direction,
              "failure_evidence": report["primary_failure_instances"],
              "control": "a0; identical source/model/non-thinking/prompt/tools/context policy/budget",
              "primary_metric": ("local resolved over all selected development jobs" if report.get("grading_backend") == "native"
                                 else "official resolved over all selected development jobs"),
              "secondary_metrics": ["reference cost", "SDK calls", "effective tests", "generation wall time"],
              "intervention": "before ordered ActionStep execution; no extra candidate generation" if direction == "A"
                              else "between retrieval candidates and FileContext span selection; action policy unchanged",
              "hypothesis": "Execution feedback reduces invalid repeats/unverified termination at the same budget" if direction == "A"
                             else "Budgeted retrieval span selection reduces observable context omissions at the same budget",
              "falsification": "No paired resolved improvement, higher cost without resolved improvement, or regressions attributable to intervention",
              "holdout_state": "locked", "implementation_phase": "T6", "frozen": now()}
    result["analysis_snapshot_hash"] = content_hash(report)
    result["hypothesis_hash"] = content_hash(result)
    if (output / "hypothesis.json").exists():
        raise ValueError("Hypothesis is already frozen; create a new experiment version to revise it")
    write_json(output / "hypothesis-analysis.json", report)
    write_json(output / "hypothesis.json", result)
    return result
