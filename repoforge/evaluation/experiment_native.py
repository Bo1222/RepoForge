"""Public generation coordination for the optional native execution backend."""

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from repoforge.evaluation.experiment import (
    append_json, content_hash, now, read_json, sha, verify_answer, write_json,
)
from repoforge.evaluation.experiment_inputs import bundle_for, embedding_credentials, embedding_service_identity, export_public_dataset, index_settings_for, validate_embedding_service
from repoforge.evaluation.experiment_docker import safe_extract
from repoforge.evaluation.native_execution import NativeExecutor, NativeMount
from repoforge.evaluation.native_provision import NativeProvisioner


def preflight(root):
    checks = {"execution_backend": "native", "grading_backend": "native", "provider_calls": 0,
              "official_harness": "not_run", "blockers": []}
    for executable in ("conda", "bwrap", "git"):
        checks[executable] = shutil.which(executable)
        if not checks[executable]:
            checks["blockers"].append(f"Missing {executable}")
    checks["free_bytes"] = shutil.disk_usage(root).free
    return checks


def source_directory(output, manifest):
    directory = Path(output) / "native-source"
    if not directory.exists():
        safe_extract(Path(output) / manifest["source"]["path"], directory)
    for name, expected in manifest["source"]["files"].items():
        if sha(directory / name) != expected:
            raise ValueError("Frozen native worker source changed")
    return directory


def provision(output, manifest, phase, job_id=None):
    output = Path(output).resolve()
    inventory = export_public_dataset(output, manifest, phase)
    jobs = [job for job in read_json(output / "jobs.json") if job["phase"] == phase
            and (not job_id or job["job_id"] == job_id)]
    if not jobs:
        raise ValueError("No matching frozen native job")
    recipes = read_json(output / "host-install-recipes.json")
    provisioner = NativeProvisioner(manifest["native"]["environment_root"], public_tools=manifest["native"].get("public_tools"))
    path = output / "provision-native.json"
    receipts = read_json(path) if path.exists() else {}
    for identity in sorted({job["instance_id"] for job in jobs}):
        receipt = asyncio.run(provisioner.prepare(inventory[identity],
            environment_setup_commit=recipes.get(identity, {}).get("environment_setup_commit")))
        receipts[identity] = {"environment": receipt, "source_sha256": manifest["source"]["sha256"],
                              "public": inventory[identity]}
        write_json(path, receipts)
    source_directory(output, manifest)
    return receipts


def executor_for(output, manifest, prepared, repo, inputs, destination, *, index=None,
                 index_writable=False, credentials=False, extra_environment=None,
                 runtime_prefix=None, runtime_writable=False, network=None, embedding_cache=None):
    """Mount only this public task, its frozen framework/index and its outputs."""
    output, repo = Path(output).resolve(), Path(repo).resolve()
    environment = prepared["environment"]
    NativeProvisioner.verify(environment)
    prefix, original_repo = Path(environment["prefix"]), Path(environment["repo"])
    if runtime_prefix and runtime_writable:
        changes = NativeProvisioner.relocate_editable(runtime_prefix, original_repo, "/testbed")
        if changes:
            append_json(Path(destination) / "editable-relocation.jsonl", {"timestamp": now(), "changes": changes})
    framework = Path(manifest["native"]["framework_prefix"])
    base = Path(manifest["native"]["framework_base_prefix"])
    mounts = [NativeMount(source_directory(output, manifest), "/agent"),
              NativeMount(framework, str(framework)), NativeMount(base / "bin", str(base / "bin")),
              NativeMount(base / "lib", str(base / "lib")),
              NativeMount(Path(runtime_prefix).resolve() if runtime_prefix else prefix, str(prefix), runtime_writable),
              NativeMount(repo, "/testbed", True),
              NativeMount(repo, str(original_repo), True),
              NativeMount(Path(inputs).resolve(), "/input"),
              NativeMount(Path(destination).resolve(), "/out", True)]
    if index is not None:
        mounts.append(NativeMount(Path(index).resolve(), "/index", index_writable))
    if embedding_cache is not None:
        mounts.append(NativeMount(Path(embedding_cache).resolve(), "/embedding-cache", True))
    env = {"PATH": f"{prefix / 'bin'}:/usr/bin:/bin", "PYTHONPATH": "/agent",
           "SKIP_INSTALL": "true", "SKIP_CONDA_ACTIVATE": "true", "PYTEST_ADDOPTS": "",
           "GENERATION_EXECUTION_BACKEND": "native", "GENERATION_NATIVE_PREFIX": str(prefix),
           "GENERATION_IMAGE_ID": manifest["source"]["sha256"],
           "GENERATION_TOOL_ISOLATION": "native-offline-v1",
           "GENERATION_NATIVE_SOURCE_PATH": "/testbed",
           "GENERATION_INDEX_DIGEST": "none", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    from repoforge.evaluation.native_public_tools import runtime_tools, extend_environment
    tool_mounts, tool_environment = runtime_tools(manifest["native"].get("public_tools"), prepared["public"]["repo"])
    mounts.extend(tool_mounts)
    extend_environment(env, tool_environment)
    if tool_mounts:
        digest = manifest["native"]["public_tools"]["digest"]
        if environment.get("public_tools_digest") != digest:
            raise ValueError("Prepared environment belongs to another public tool profile")
        env["GENERATION_PUBLIC_TOOLS_DIGEST"] = digest
    if credentials:
        from repoforge.index.embed_model import embedding_key_env
        for name in ("DEEPSEEK_API_KEY", embedding_key_env(index_settings_for(manifest).embed_provider)):
            if name and os.environ.get(name):
                env[name] = os.environ[name]
    if credentials or network:
        # A shared network namespace still needs the host's explicit transport
        # settings. Do not inherit the rest of the host environment or home.
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
                     "https_proxy", "http_proxy", "all_proxy", "no_proxy",
                     "OPENAI_API_BASE", "OPENAI_BASE_URL"):
            if os.environ.get(name):
                env[name] = os.environ[name]
    env.update(extra_environment or {})
    return NativeExecutor(mounts, environment=env, network=credentials if network is None else network)


def worker_command(manifest, *, build_index=False, resume=False):
    command = [str(Path(manifest["native"]["framework_prefix"]) / "bin/python"),
               "-m", "repoforge.evaluation.generation", "--bundle", "/input/bundle.json", "--output", "/out"]
    if build_index:
        command.append("--build-index")
    else:
        command.extend(["--index", "/index"])
    if resume:
        command.append("--resume")
    return command


def run_worker(runner, command, log, *, timeout=None):
    result = asyncio.run(runner.run(command, cwd="/agent", timeout=timeout,
                                   redact=[os.environ.get(name, "") for name in (
                                       "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "JINA_API_KEY", "VOYAGE_API_KEY")]))
    Path(log).write_text(result.output)
    return result.return_code


def shared_index(output, manifest, prepared):
    """Resolve one repository snapshot; do not expose the store to the worker."""
    config = manifest.get("native", {}).get("index_store")
    if not config:
        return None, None, Path(output) / "indexes" / prepared["public"]["instance_id"]
    from repoforge.index.repository_store import EMPTY_PATCH, RepositoryIndexStore, index_profile
    profile = index_profile(manifest, setup_patch_sha256=prepared["environment"].get("setup_patch_sha256", EMPTY_PATCH))
    store = RepositoryIndexStore(config["root"])
    return store, profile, store.find(prepared["public"], profile)


def verify_index_receipt(index, manifest, public, *, setup_patch_sha256=None):
    index = Path(index)
    receipt = read_json(index / "index-receipt.json")
    if receipt.get("format") == "repository-index-v1":
        from repoforge.index.repository_store import EMPTY_PATCH, RepositoryIndexStore, index_profile
        profile = index_profile(manifest, setup_patch_sha256=setup_patch_sha256 or EMPTY_PATCH)
        return RepositoryIndexStore(manifest["native"]["index_store"]["root"]).verify(index, public=public, profile=profile)
    if receipt["instance"] != public and not manifest.get("exploration"):
        raise ValueError("Native index belongs to a different public task")
    settings = index_settings_for(manifest).model_dump(mode="json")
    if receipt["settings"] != settings:
        raise ValueError("Native index settings differ from the frozen candidate")
    if settings["embed_provider"] == "local" and (
            embedding_service_identity(manifest) is None or
            receipt.get("embedding_service_identity") != embedding_service_identity(manifest)):
        raise ValueError("Native index BGE deployment identity differs from the frozen candidate")
    for name, expected in receipt["files"].items():
        path = (index / name).resolve()
        if not path.is_relative_to(index.resolve()) or sha(path) != expected:
            raise ValueError("Frozen native index changed or escaped its directory")
    if content_hash(receipt["files"]) != receipt["digest"]:
        raise ValueError("Frozen native index digest changed")
    return receipt


def audit_bge_tokens(output, manifest, prepared, repo, index):
    """Host preparation uses the service tokenizer; workers stay HTTP-only."""
    root = Path(__file__).resolve().parents[2]
    destination = Path(output) / "token-audits" / prepared["public"]["instance_id"] / uuid.uuid4().hex
    destination.mkdir(parents=True)
    settings = destination / "settings.json"
    chunks = destination / "chunks.json"
    report = destination / "report.json"
    write_json(settings, manifest["embedding_settings"])
    script = source_directory(output, manifest) / "scripts/bge_token_audit.py"
    runtime = root / "experiments/runtime/local_bge"
    commands = [
        [str(Path(manifest["native"]["framework_prefix"]) / "bin/python"), str(script), "export",
         "--repo", str(repo), "--settings", str(settings), "--chunks", str(chunks)],
        [str(runtime / "venv/bin/python"), str(script), "count", "--chunks", str(chunks),
         "--model", str(runtime / "model"), "--output", str(report)],
    ]
    env = {key: value for key, value in os.environ.items() if "KEY" not in key and "TOKEN" not in key}
    env.update(PYTHONPATH=str(source_directory(output, manifest)), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    steps = []
    for i, command in enumerate(commands):
        started, clock = now(), time.monotonic()
        with (destination / f"step-{i}.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        steps.append({"command": command, "exit_code": result.returncode, "started": started,
                      "ended": now(), "wall_seconds": time.monotonic() - clock})
        write_json(destination / "steps.json", steps)
        if result.returncode:
            raise ValueError(f"BGE token precheck failed (exit {result.returncode}); inspect {destination}")
    receipt = {"manifest_hash": manifest["manifest_hash"], "instance": prepared["public"],
               "report": read_json(report),
               "evidence": {str(p.relative_to(output)): sha(p) for p in destination.iterdir() if p.is_file()}}
    write_json(Path(index) / "token-audit.json", receipt)
    return receipt


def copy_public_workspace(source, destination):
    """Faithful copy for a job workspace.

    Symlinks must stay symlinks: dereferencing turns tracked links into
    typechanges, so git status no longer matches the base commit and
    clean-source consumers such as the CLI worktree entrypoint refuse to run.
    """
    return shutil.copytree(source, destination, symlinks=True,
                           ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))


def apply_testbed_setup(repo, prepared):
    """Repair generated test-runner metadata without touching the answer patch.

    Exact-base shallow fetches deliberately have no tags.  ``setuptools-scm``
    then writes ``0.1.dev1`` into pytest's ignored ``_version.py``, while the
    public ``pyproject.toml`` requires ``minversion = 2.0``.  Every local
    RunTests would abort with rc=4 before collecting tests, which hides real
    product regressions from the agent.  The provisioning contract uses
    ``SETUPTOOLS_SCM_PRETEND_VERSION`` for the same reason; old reused pilot
    environments predate it.  Apply the equivalent value in the disposable job
    workspace and record it as environment evidence, never as a product edit.
    """
    import re
    public = prepared.get("public") or {}
    if public.get("repo") != "pytest-dev/pytest":
        return None
    repo = Path(repo)
    version_file = repo / "src/_pytest/_version.py"
    pyproject = repo / "pyproject.toml"
    if not version_file.is_file() or not pyproject.is_file():
        return None
    match = re.search(r'^\s*minversion\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    if not match:
        return None
    before = version_file.read_text()
    current = re.search(r"__version__\s*=\s*version\s*=\s*'([^']+)'", before)
    if not current:
        return None

    def numeric(value):
        return tuple(int(part) for part in re.findall(r"\d+", value)[:3])

    target = str(public.get("version") or "6.0")
    if numeric(current.group(1)) >= numeric(match.group(1)):
        return None
    version_tuple = numeric(target)
    if len(version_tuple) == 1:
        version_tuple = (version_tuple[0], 0)
    after = re.sub(r"__version__\s*=\s*version\s*=\s*'[^']*'",
                   f"__version__ = version = '{target}'", before)
    after = re.sub(r"__version_tuple__\s*=\s*version_tuple\s*=\s*\([^)]*\)",
                   f"__version_tuple__ = version_tuple = {version_tuple}", after)
    if after == before:
        return None
    version_file.write_text(after)
    return {"version": 1, "kind": "testbed-version-metadata",
            "path": str(version_file.relative_to(repo)),
            "before": current.group(1), "after": target,
            "minversion": match.group(1),
            "reason": "exact-base source has no release tags; setuptools-scm generated a version below pyproject minversion",
            "in_answer_patch": False}


def run_job(output, manifest, job, resume=False, prepare_index_only=False):
    from repoforge.evaluation.generation import validate_generation_health

    output = Path(output).resolve()
    if manifest.get("exploration"):
        from repoforge.evaluation.exploration import verify_shared_index
        if prepare_index_only:
            raise ValueError("New index preparation is disabled for this exploration")
        verify_shared_index(output, manifest, job)
    directory = output / "runs" / job["job_id"]
    if (directory / "answer.json").exists():
        answer = verify_answer(directory, job)
        validate_generation_health(directory)
        return answer
    service_health = validate_embedding_service(manifest)
    if service_health is not None:
        append_json(output / "embedding-service-checks.jsonl", {
            "timestamp": now(), "job_id": job["job_id"], "manifest_hash": manifest["manifest_hash"],
            "health": service_health})
    prepared = read_json(output / "provision-native.json")[job["instance_id"]]
    if prepared["source_sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Native provision belongs to another frozen worker source")
    if not os.environ.get(manifest["cli_config"]["api_key_env"]):
        raise ValueError("Required model credential is not configured")
    receipt_path = directory / "workspace.json"
    repo = output / "workspaces" / job["job_id"]
    runtime_prefix = output / "runtime-prefixes" / job["job_id"]
    prepared_only = receipt_path.exists() and not any((directory / name).exists() for name in (
        "generation-worker-started.json", "model-metadata.json", "sdk.jsonl", "events.jsonl", "attempts.jsonl"))
    if resume or prepared_only:
        receipt = read_json(receipt_path)
        if not receipt.get("workspace_prepared") or not repo.is_dir():
            raise ValueError("Original native workspace is missing or was never prepared; refusing automatic recovery")
        stat = repo.stat()
        if receipt["workspace_device"] != stat.st_dev or receipt["workspace_inode"] != stat.st_ino:
            raise ValueError("Original native workspace identity changed")
        if not receipt.get("runtime_prepared") or not runtime_prefix.is_dir():
            raise ValueError("Original private runtime is missing; refusing silent dependency replacement")
        if prepared_only and not resume:
            actual = subprocess.check_output(["git", "diff", "--binary", prepared["public"]["base_commit"]], cwd=repo)
            original = (Path(prepared["environment"]["repo"]).parent / "setup.patch").read_bytes()
            if actual != original or NativeProvisioner.metadata(runtime_prefix) != NativeProvisioner.metadata(Path(prepared["environment"]["prefix"])):
                raise ValueError("Index-only workspace/runtime changed before generation")
            if NativeProvisioner.package_metadata(runtime_prefix) != NativeProvisioner.package_metadata(Path(prepared["environment"]["prefix"])):
                raise ValueError("Index-only Python packages changed before generation")
    else:
        if receipt_path.exists() or repo.exists():
            raise ValueError("Job has prior effects; use explicit resume or a new experiment")
        directory.mkdir(parents=True, exist_ok=True)
        write_json(receipt_path, {"workspace_prepared": False, "manifest_hash": manifest["manifest_hash"]})
        repo.parent.mkdir(parents=True, exist_ok=True)
        copy_public_workspace(prepared["environment"]["repo"], repo)
        setup = apply_testbed_setup(repo, prepared)
        if setup:
            write_json(directory / "testbed-setup.json", setup)
        # Tool-driven installs need a writable private prefix. Hard links would
        # allow in-place writes to change the shared prepared environment.
        runtime_prefix.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prepared["environment"]["prefix"], runtime_prefix, symlinks=True)
        subprocess.run(["git", "update-ref", "refs/heads/native-baseline", prepared["public"]["base_commit"]],
                       cwd=repo, check=True, capture_output=True)
        stat = repo.stat()
        receipt = {"workspace_prepared": True, "manifest_hash": manifest["manifest_hash"],
                   "runtime_prepared": True, "runtime_prefix": str(runtime_prefix),
                   "workspace_id": uuid.uuid4().hex, "workspace_device": stat.st_dev, "workspace_inode": stat.st_ino}
        write_json(receipt_path, receipt)
    if not (directory / "testbed-setup.json").exists():
        setup = apply_testbed_setup(repo, prepared)
        if setup:
            write_json(directory / "testbed-setup.json", setup)
    inputs = output / "inputs" / job["job_id"]
    write_json(inputs / "bundle.json", bundle_for(manifest, job, prepared["public"]))
    store, profile, index = shared_index(output, manifest, prepared)
    if manifest.get("t6") and (index is None or not (index / "index-receipt.json").is_file()):
        raise ValueError("T6 permits only verified existing indexes, never an automatic rebuild")
    if index is None:
        index = output / "index-builds" / job["instance_id"]
    index.mkdir(parents=True, exist_ok=True)
    built_here = job["entrypoint"] == "batch" and not (index / "index-receipt.json").exists()
    if job["entrypoint"] == "batch" and not (index / "index-receipt.json").exists():
        embedding_credentials(manifest)
        if index_settings_for(manifest).embed_provider == "local":
            audit_bge_tokens(output, manifest, prepared, repo, index)
        cache = output / "public-embedding-cache"
        cache.mkdir(parents=True, exist_ok=True)
        runner = executor_for(output, manifest, prepared, repo, inputs, index, credentials=True,
                              embedding_cache=cache,
                              extra_environment={"GENERATION_WORKSPACE_ID": receipt["workspace_id"],
                                                 "JINA_EMBEDDING_CACHE_DIR": "/embedding-cache"})
        code = run_worker(runner, worker_command(manifest, build_index=True), output / f"index-{job['instance_id']}.log")
        if code:
            raise ValueError(f"Native index preparation failed (exit {code}); preserve its SDK ledger before retrying")
    digest = "none"
    if job["entrypoint"] == "batch":
        if store is not None and not (index / "index-receipt.json").exists():
            raise ValueError("Shared index preparation did not publish a complete receipt")
        if store is not None and index.is_relative_to(output / "index-builds"):
            source_index = index
            timings = source_index / "index-stage-timings.json"
            provenance = {"manifest_hash": manifest["manifest_hash"], "job_id": job["job_id"]}
            if timings.exists():
                target = output / "index-stage-timings" / (job["instance_id"] + ".json")
                write_json(target, read_json(timings))
                provenance["timing_evidence"] = {"path": str(target), "sha256": sha(target)}
            index = store.publish(source_index, prepared["public"], profile,
                                  provenance=provenance)
            from repoforge.index.repository_store import retire_index_data
            retire_index_data(source_index, reason="Published into shared repository store", replacement=index)
        index_receipt = verify_index_receipt(index, manifest, prepared["public"],
                                            setup_patch_sha256=prepared["environment"].get("setup_patch_sha256"))
        digest = index_receipt["digest"]
        if store is not None:
            write_json(output / "index-references" / (job["instance_id"] + ".json"),
                       {"index": str(index), "digest": digest, "repository": prepared["public"]["repo"],
                        "base_commit": prepared["public"]["base_commit"], "profile_id": store.profile_id(profile),
                        "current_job_embedding_build": "built" if built_here else "reused"})
    if prepare_index_only:
        return {"index_digest": digest, "generation_started": False}
    runner = executor_for(output, manifest, prepared, repo, inputs, directory, index=index, credentials=True,
        runtime_prefix=runtime_prefix, runtime_writable=True,
        extra_environment={"GENERATION_WORKSPACE_ID": receipt["workspace_id"], "GENERATION_INDEX_DIGEST": digest})
    write_json(directory / "generation-worker-started.json", {"timestamp": now(), "resume": resume,
               "manifest_hash": manifest["manifest_hash"]})
    code = run_worker(runner, worker_command(manifest, resume=resume), directory / f"worker-{now().replace(':', '-')}.log")
    append_json(directory / "attempts.jsonl", {"timestamp": now(), "resume": resume, "exit_code": code})
    if code:
        raise ValueError(f"Native generation failed (exit {code}); retain the original workspace and reconcile before resuming")
    answer = verify_answer(directory, job)
    validate_generation_health(directory)
    return answer


def baseline_job(output, manifest, job):
    from repoforge.evaluation.native_grading import NativeGrader
    return NativeGrader(output, manifest).baseline(job)


def grade_job(output, manifest, job):
    from repoforge.evaluation.native_grading import NativeGrader
    return NativeGrader(output, manifest).grade(job)


def release_job_workspace(output, manifest, job):
    """Release only our disposable copies after frozen independent verification."""
    output = Path(output).resolve()
    directory = output / "runs" / job["job_id"]
    answer = verify_answer(directory, job)
    grade, own = read_json(directory / "grading.json"), read_json(directory / "own-tests.json")
    if (grade.get("answer_hash") != answer["answer_hash"] or own.get("answer_hash") != answer["answer_hash"]
            or grade.get("local_grading") not in {"completed", "patch_error"}
            or own.get("status") not in {"executed", "not_run"}):
        raise ValueError("Frozen answer and independent grade/test evidence are required before release")
    for relative in (f"workspaces/{job['job_id']}", f"runtime-prefixes/{job['job_id']}"):
        path = output / relative
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError("Disposable workspace path escapes this experiment")
        append_json(directory / "disposable-release.jsonl", {"path": relative, "answer_hash": answer["answer_hash"],
                    "timestamp": now(), "status": "already_absent" if not path.exists() else "release_started"})
        if path.exists():
            shutil.rmtree(path)
            append_json(directory / "disposable-release.jsonl", {"path": relative, "status": "released", "timestamp": now()})
    # Checkpoint databases serve in-flight crash recovery only; the frozen answer,
    # grading and trajectory live outside them. Delete them (including CLI jobs'
    # nested sessions/ copies) once the acceptance checks above have passed.
    checkpoint_files, checkpoint_bytes = 0, 0
    for path in sorted(directory.rglob("checkpoint.sqlite*")):
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint path escapes this job directory")
        size = path.stat().st_size
        append_json(directory / "disposable-release.jsonl", {
            "path": str(path.relative_to(directory)), "answer_hash": answer["answer_hash"],
            "timestamp": now(), "status": "release_started", "bytes": size})
        path.unlink()
        checkpoint_files += 1
        checkpoint_bytes += size
        append_json(directory / "disposable-release.jsonl", {
            "path": str(path.relative_to(directory)), "status": "released", "timestamp": now()})
    if checkpoint_files:
        append_json(directory / "disposable-release.jsonl", {
            "path": "checkpoint.sqlite*", "status": "released", "files": checkpoint_files,
            "bytes": checkpoint_bytes, "timestamp": now()})


def validate_agent_tests(output, manifest, job):
    output = Path(output).resolve()
    directory = output / "runs" / job["job_id"]
    answer = verify_answer(directory, job)
    prepared = read_json(output / "provision-native.json")[job["instance_id"]]
    destination = output / "own-tests-native" / content_hash([answer["answer_hash"], now()])[:24]
    destination.mkdir(parents=True)
    repo, prefix = destination / "repo", destination / "runtime"
    shutil.copytree(prepared["environment"]["repo"], repo, symlinks=True)
    shutil.copytree(prepared["environment"]["prefix"], prefix, symlinks=True)
    inputs, own_input = destination / "input", destination / "answer-input"
    own_input.mkdir()
    write_json(inputs / "bundle.json", bundle_for(manifest, job, prepared["public"]))
    for name in ("patch.diff", "test-requests.json"):
        shutil.copyfile(directory / name, own_input / name)
    try:
        runner = executor_for(output, manifest, prepared, repo, inputs, destination,
                              runtime_prefix=prefix, runtime_writable=True, network=False)
        runner = NativeExecutor([*runner.mounts, NativeMount(own_input, "/answer")],
                                environment=runner.environment(), network=False)
        command = [str(Path(manifest["native"]["framework_prefix"]) / "bin/python"),
                   "-m", "repoforge.evaluation.generation", "--bundle", "/input/bundle.json",
                   "--output", "/out", "--own-tests", "/answer"]
        code = run_worker(runner, command, destination / "worker.log")
        write_json(destination / "result.json", {"answer_hash": answer["answer_hash"], "exit_code": code,
                    "official_harness": "not_run", "hidden_tests": False, "provider_calls": 0})
        if code:
            raise ValueError(f"Native own-test validation failed (exit {code})")
        validation = read_json(destination / "own-test-validation.json")
        write_json(directory / "own-tests.json", {"answer_hash": answer["answer_hash"],
            "status": validation["status"], "hidden_tests": False,
            "evidence": {str(p.relative_to(output)): sha(p) for p in destination.iterdir() if p.is_file()}})
        return validation
    finally:
        for path in (repo, prefix):
            if path.exists():
                shutil.rmtree(path)


def probe_recovery(output, manifest):
    """Actual historical runtime, eight process-crash cases, zero model calls."""
    output = Path(output).resolve()
    identity = "pytest-dev__pytest-7490"
    prepared = read_json(output / "provision-native.json")[identity]
    destination = output / "recovery-native" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    job = {"phase": "recovery", "variant": "a0", "instance_id": identity,
           "manifest_hash": manifest["manifest_hash"], "entrypoint": "recovery"}
    inputs = destination / "input"
    write_json(inputs / "bundle.json", bundle_for(manifest, job, prepared["public"]))
    index = shared_index(output, manifest, prepared)[2] if index_settings_for(manifest).embed_provider == "local" else None
    digest = "none"
    if index_settings_for(manifest).embed_provider == "local" and index is None:
        raise ValueError("Prepare the shared repository index before recovery probes")
    if index is not None:
        receipt = verify_index_receipt(index, manifest, prepared["public"],
                                       setup_patch_sha256=prepared["environment"].get("setup_patch_sha256"))
        digest = receipt["digest"]
    runner = executor_for(output, manifest, prepared, prepared["environment"]["repo"], inputs, destination,
                          index=index, extra_environment={"GENERATION_WORKSPACE_ID": destination.name,
                                                          "GENERATION_INDEX_DIGEST": digest})
    rows = []
    try:
        for shadow in (False, True):
            for fault in ("after_generation", "before_tool", "after_effect", "after_commit"):
                name = ("shadow" if shadow else "disk") + "-" + fault
                command = [str(Path(manifest["native"]["framework_prefix"]) / "bin/python"),
                    "-m", "scripts.recovery_smoke", "--directory", f"/out/{name}",
                    "--swe-bundle", "/input/bundle.json"]
                if shadow:
                    command.append("--shadow")
                first = run_worker(runner, command + ["--fault", fault], destination / f"{name}-fault.log", timeout=90)
                calls = (destination / name / "calls.jsonl").read_text() if (destination / name / "calls.jsonl").exists() else ""
                resumed = run_worker(runner, command, destination / f"{name}-resume.log", timeout=90)
                after = (destination / name / "calls.jsonl").read_text() if (destination / name / "calls.jsonl").exists() else ""
                uncertain = fault in {"before_tool", "after_effect"}
                passed = first == 88 and ((resumed != 0 and calls == after and (destination / name / "conflict.json").exists())
                                         if uncertain else resumed == 0)
                if not uncertain and resumed == 0:
                    state = read_json(destination / name / "result.json")
                    kinds = [__import__("json").loads(line)["kind"] for line in after.splitlines()]
                    passed = passed and kinds == ["model", "tool:0", "tool:1", "tool:2", "tool:3"] and state["finish_reason"] == "terminal"
                rows.append({"case": name, "first_exit_code": first, "resume_exit_code": resumed,
                             "unknown_effect_refused": uncertain, "passed": passed,
                             "commands": [command + ["--fault", fault], command]})
    finally:
        summary = {"manifest_hash": manifest["manifest_hash"], "execution_backend": "native", "cases": rows,
                   "index_digest": digest,
                   "passed": len(rows) == 8 and all(row["passed"] for row in rows),
                   "provider_calls": 0, "official_harness": "not_run",
                   "evidence": {str(p.relative_to(output)): sha(p) for p in destination.rglob("*") if p.is_file()}}
        write_json(destination / "summary.json", summary)
    if not summary["passed"]:
        raise ValueError("Native recovery probe failed; inspect retained crash-window evidence")
    return summary


def probe_contracts(output, manifest):
    """Actual native worker for three variants, without provider/index calls."""
    from scripts.migration_smoke import SOURCE, TEST
    output = Path(output).resolve()
    identity = "pytest-dev__pytest-7490"
    prepared = read_json(output / "provision-native.json")[identity]
    destination = output / "contracts-native" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    rows = []
    for variant in ("loop_legacy", "graph_legacy", "loop_managed", "a0"):
        case = destination / variant
        repo = case / "repo"
        shutil.copytree(prepared["environment"]["repo"], repo, symlinks=True)
        (repo / "calc.py").write_text(SOURCE)
        (repo / "test_calc.py").write_text(TEST)
        for command in (["git", "add", "calc.py", "test_calc.py"],
                        ["git", "-c", "user.name=NativeContract", "-c", "user.email=contract@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "public arithmetic contract fixture"],
                        ["git", "update-ref", "refs/heads/native-contract", "HEAD"]):
            subprocess.run(command, cwd=repo, check=True, capture_output=True,
                           env={**os.environ, "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
                                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000"})
        job = {"phase": "contracts", "variant": variant, "instance_id": identity,
               "manifest_hash": manifest["manifest_hash"], "entrypoint": "batch", "repeat": 0,
               "job_id": content_hash([manifest["manifest_hash"], variant, "contracts"])}
        public = {**prepared["public"], "problem_statement": "Fix the public arithmetic fixture: add(2,3) returns -1."}
        inputs, results, index = case / "input", case / "results", case / "index"
        results.mkdir()
        index.mkdir()
        bundle = bundle_for(manifest, {**job, "variant": "a0"} if variant == "loop_managed" else job, public)
        if variant == "loop_managed":
            bundle["job"] = job
            bundle["profile"] = {**bundle["profile"], "flow": "loop"}
        write_json(inputs / "bundle.json", bundle)
        runner = executor_for(output, manifest, prepared, repo, inputs, results, index=index,
                              extra_environment={"GENERATION_WORKSPACE_ID": case.name})
        command = [str(Path(manifest["native"]["framework_prefix"]) / "bin/python"),
                   "-m", "scripts.native_scripted_worker", "--bundle", "/input/bundle.json",
                   "--output", "/out", "--index", "/index"]
        code = run_worker(runner, command, case / "worker.log", timeout=120)
        contract = read_json(results / "contract.json") if (results / "contract.json").exists() else {"passed": False}
        rows.append({"variant": variant, "return_code": code, "command": command, "contract": contract})
        if code:
            break
    summary = {"manifest_hash": manifest["manifest_hash"], "execution_backend": "native", "rows": rows,
               "passed": len(rows) == 4 and all(row["return_code"] == 0 and row["contract"]["passed"] for row in rows),
               "provider_calls": 0, "synthetic_workflow": True, "official_harness": "not_run",
               "evidence": {str(p.relative_to(output)): sha(p) for p in destination.rglob("*") if p.is_file()}}
    write_json(destination / "summary.json", summary)
    if not summary["passed"]:
        raise ValueError("Native worker contract failed; preserve its original logs")
    return summary


def probe_cli_boundaries(output, manifest):
    """Use the actual historical pytest runtime for synthetic CLI boundaries."""
    output = Path(output).resolve()
    identity = "pytest-dev__pytest-7490"
    prepared = read_json(output / "provision-native.json")[identity]
    destination = output / "cli-boundaries-native" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    inputs = destination / "input"
    job = {"phase": "cli_boundaries", "variant": "a0", "instance_id": identity,
           "manifest_hash": manifest["manifest_hash"], "entrypoint": "cli"}
    write_json(inputs / "bundle.json", bundle_for(manifest, job, prepared["public"]))
    runner = executor_for(output, manifest, prepared, prepared["environment"]["repo"], inputs, destination,
                          extra_environment={"GENERATION_WORKSPACE_ID": destination.name,
                                             "DEEPSEEK_API_KEY": "native-offline-fixture"})
    command = [str(Path(manifest["native"]["framework_prefix"]) / "bin/python"),
               "-m", "scripts.native_cli_boundaries", "--bundle", "/input/bundle.json", "--output", "/out"]
    code = run_worker(runner, command, destination / "worker.log", timeout=180)
    result = read_json(destination / "result.json") if (destination / "result.json").exists() else {}
    summary = {**result, "manifest_hash": manifest["manifest_hash"], "execution_backend": "native",
               "command": command, "return_code": code,
               "passed": code == 0 and result.get("passed") is True,
               "evidence": {str(p.relative_to(output)): sha(p) for p in destination.rglob("*") if p.is_file()}}
    write_json(destination / "summary.json", summary)
    if not summary["passed"]:
        raise ValueError("Native CLI boundaries failed; inspect retained fixture evidence")
    return summary


def probe_isolation(output, manifest):
    output = Path(output).resolve()
    prepared = read_json(output / "provision-native.json")["pytest-dev__pytest-7490"]
    destination = output / "isolation-native" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    inputs = destination / "input"
    marker = output / "private-grading-sentinel.txt"
    if not marker.exists():
        marker.write_text("host-only grading sentinel\n")
    root = Path(__file__).resolve().parents[2]
    raw = list((output / "host-dataset-cache").rglob("*.arrow"))
    cache_marker = output / "public-embedding-cache" / "isolation-sentinel.txt"
    cache_marker.parent.mkdir(parents=True, exist_ok=True)
    cache_marker.write_text("public preparation cache sentinel\n")
    blocked = [str(marker), str(cache_marker), "/embedding-cache/isolation-sentinel.txt",
               str(root / ".env"), str(Path.home() / ".ssh"),
               str(Path.home() / ".aws"), "/var/run/docker.sock", *map(str, raw)]
    local = index_settings_for(manifest).embed_provider == "local"
    if local:
        blocked.extend([str(root / "experiments/runtime/local_bge/model"),
                        str(root / "experiments/runtime/local_bge/venv"),
                        str(root / "repoforge/evaluation/native_grading.py")])
    index = shared_index(output, manifest, prepared)[2] if local else None
    if local and index is None:
        raise ValueError("Prepare the shared repository index before isolation probes")
    public_paths = ["/testbed/testing/test_skipping.py", "/input/isolation.json"]
    if index is not None:
        if not (index / "index-receipt.json").exists():
            raise ValueError("Local BGE isolation requires the actual prepared public index")
        public_paths.append("/index/settings.json")
    write_json(inputs / "isolation.json", {"manifest_hash": manifest["manifest_hash"],
               "blocked_paths": blocked, "public_paths": public_paths,
               "embedding_service": ({"url": index_settings_for(manifest).embed_base_url.removesuffix('/v1') + '/health',
                   "model": manifest['embedding_settings']['embed_model'],
                   "revision": manifest['embedding_execution']['model_revision']} if local else None)})
    runner = executor_for(output, manifest, prepared, prepared["environment"]["repo"], inputs, destination,
        network=True, index=index, extra_environment={"DEEPSEEK_API_KEY": "native-offline-fixture"})
    command = [str(Path(manifest["native"]["framework_prefix"]) / "bin/python"),
               "-m", "scripts.native_isolation_probe", "--config", "/input/isolation.json", "--output", "/out"]
    code = run_worker(runner, command, destination / "worker.log", timeout=60)
    result = read_json(destination / "result.json") if (destination / "result.json").exists() else {}
    summary = {**result, "execution_backend": "native", "manifest_hash": manifest["manifest_hash"],
               "command": command, "return_code": code, "passed": code == 0 and result.get("passed") is True,
               "evidence": {str(p.relative_to(output)): sha(p) for p in destination.rglob("*") if p.is_file()}}
    write_json(destination / "summary.json", summary)
    if not summary["passed"]:
        raise ValueError("Native filesystem/network/PID isolation failed; inspect retained evidence")
    return summary
