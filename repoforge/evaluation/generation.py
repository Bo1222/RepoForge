"""Container worker. Its bundle and filesystem contain public generation data only."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

from repoforge.evaluation.experiment import append_json, content_hash, freeze_answer, now, public_instance, read_json, sha, write_json
from repoforge.evaluation.experiment_usage import SDKRecorder


def validate_generation_health(directory):
    """Infrastructure failures are not ordinary unsuccessful repair attempts."""
    usage = read_json(Path(directory) / "usage.json")
    if usage.get("cost_complete") is not True:
        raise ValueError("Unknown generation usage/cost; expansion is blocked")
    result = read_json(Path(directory) / "generation.json")
    if result.get("generation_status") == "failed" and (
        result.get("selected_node_id") == 0 or result.get("error_type") in {
            "IsADirectoryError", "FileNotFoundError", "PermissionError", "OSError",
            "RecoveryConflict", "OperationalError", "DatabaseError", "IntegrityError",
        }
    ):
        raise ValueError(f"Generation infrastructure failure: {result.get('error_type')}; expansion is blocked")
    return result


def environment_identity(image_id):
    """Fresh probe on each process start; never trust a previous ready flag alone."""
    if os.environ.get("GENERATION_EXECUTION_BACKEND") == "native":
        from repoforge.evaluation.native_provision import NativeProvisioner
        prefix = Path(os.environ["GENERATION_NATIVE_PREFIX"])
        stat = Path("/testbed").stat()
        return {"execution_backend": "native", "source_id": image_id,
                "workspace_id": os.environ["GENERATION_WORKSPACE_ID"],
                "workspace_device": stat.st_dev, "workspace_inode": stat.st_ino,
                "dependencies": content_hash(NativeProvisioner.metadata(prefix)),
                "python_packages": content_hash(NativeProvisioner.package_metadata(prefix)),
                "index_digest": os.environ.get("GENERATION_INDEX_DIGEST"),
                "public_tools_digest": os.environ.get("GENERATION_PUBLIC_TOOLS_DIGEST")}
    dependencies = subprocess.check_output(
        ["/opt/miniconda3/bin/conda", "list", "-n", "testbed", "--json"], text=True, timeout=15)
    return {"image_id": image_id, "container_id": os.environ["GENERATION_CONTAINER_ID"],
            "dependencies": content_hash(json.loads(dependencies)),
            "index_digest": os.environ.get("GENERATION_INDEX_DIGEST")}


def load_bundle(path):
    bundle = read_json(path)
    if set(bundle["instance"]) != set(public_instance(bundle["instance"])):
        raise ValueError("Worker bundle contains non-public instance fields")
    for name, expected in bundle["source_files"].items():
        if sha(Path("/agent") / name) != expected:
            raise ValueError("Worker source changed after manifest freeze")
    return bundle


def submit_product_only(profile) -> bool:
    """Whether the frozen answer must exclude test-file edits.

    SWE-bench applies the official test patch and ignores candidate test edits.
    O2 (a5) therefore submits the product patch only.  Older profiles keep
    their historical full-patch shape.
    """
    return bool(profile.get("submit_product_only",
                            profile.get("action_policy", {}).get("variant") in {"a5"}))


def select_submitted_patch(patch, tests):
    """Export current product patch. Recovery is always an explicit operation."""
    return patch, False


def copy_setup_artifacts(source, worktree):
    """Retain ignored build products needed by historical editable installs."""
    import shutil
    source, worktree = Path(source).resolve(), Path(worktree).resolve()
    ignored = subprocess.check_output([
        "git", "-C", str(source), "ls-files", "--others", "--ignored", "--exclude-standard", "-z"])
    copied = []
    for raw in ignored.decode().split("\0"):
        if not raw:
            continue
        relative = Path(raw)
        if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".git"} for part in relative.parts):
            continue
        path, target = source / relative, worktree / relative
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("Generated setup artifact escapes public source")
        if path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append({"path": str(relative), "sha256": sha(path)})
    return copied


def record_loaded_config(bundle, flow, output, index=None):
    from repoforge.cli.service import safe_config
    from repoforge.index.settings import IndexSettings
    agent = flow.agent
    model = agent.completion_model
    settings = getattr(index, "_settings", None)
    write_json(Path(output) / "loaded-config.json", {
        "manifest_hash": bundle["job"]["manifest_hash"], "job_id": bundle["job"]["job_id"],
        "entrypoint": bundle["job"]["entrypoint"], "flow_type": type(flow).__name__,
        "agent_type": type(agent).__name__, "memory_type": type(agent.memory).__name__,
        "policy": agent.memory.policy.model_dump(mode="json") if hasattr(agent.memory, "policy") else None,
        "action_policy": flow.action_policy.model_dump(mode="json") if hasattr(flow, "action_policy") else {"variant": "off"},
        "tool_names": [action.name for action in agent.actions],
        "schema_names": [schema.name for schema in model._response_schema or []],
        "model": model.model, "model_base_url": model.model_base_url,
        "key_env_matches": bool(model.model_api_key) and model.model_api_key == os.environ.get(bundle["cli_config"]["api_key_env"]),
        "model_parameters": safe_config(model.params), "max_tokens": model.max_tokens,
        "temperature": model.temperature, "model_timeout": model.timeout,
        "max_iterations": flow.max_iterations, "max_cost": flow.max_cost,
        "index_settings": settings.model_dump(mode="json") if isinstance(settings, IndexSettings) else None,
        "tool_isolation": os.environ.get("GENERATION_TOOL_ISOLATION"),
        "index_provenance": bundle.get("index_provenance"),
    })


async def run(bundle, output, index_path, resume=False):
    from repoforge.agent.agent import ActionAgent
    from repoforge.cli.service import SessionService, safe_config
    from repoforge.flow import AgenticLoop, LangGraphFlow
    from repoforge.index.code_index import CodeIndex
    from repoforge.message_history.managed import ManagedMessageHistoryGenerator
    from repoforge.message_history.policy import ContextPolicy
    from repoforge.repository.git import GitRepository
    from repoforge.runtime.local import SweBenchLocalEnvironment
    from repoforge.storage.file_storage import FileStorage
    from repoforge.workspace import Workspace

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "answer.json").exists():
        raise ValueError("Frozen answers cannot be regenerated; create a new job/experiment")
    job, config = bundle["job"], bundle["cli_config"]
    from scripts.t4_cli_probe import model_metadata
    write_json(output / "model-metadata.json", model_metadata())
    secrets = [os.environ.get(name, "") for name in (
        "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "JINA_API_KEY", "VOYAGE_API_KEY")]
    def redact(value):
        for secret in secrets:
            value = safe_config(value, secret)
        return value
    for name in ("events.jsonl", "sdk.jsonl"):
        (output / name).touch(exist_ok=True)
    recorder = SDKRecorder(output, bundle["prices"], config["max_cost"])
    generation_start = time.monotonic()
    failed = None
    selected = None
    completion = {"status": "unverified", "official_harness": "not_run", "resolved": None}

    if job["entrypoint"] == "cli":
        service = SessionService(output / "sessions")
        if resume:
            session = service.load(read_json(output / "session-id.json")["session_id"])
        else:
            session = await service.create("/testbed", bundle["instance"]["problem_statement"],
                                           {**config, "instance_id": job["instance_id"],
                                            "env": {"PATH": os.environ.get("GENERATION_NATIVE_PREFIX", "/opt/miniconda3/envs/testbed") + "/bin:" + os.environ["PATH"],
                                                    "PYTHONPATH": "", "PYTEST_ADDOPTS": ""}})
            worktree = Path(session.info["worktree"])
            write_json(output / "cli-setup-artifacts.json", copy_setup_artifacts("/testbed", worktree))
            session.info["config"]["env"]["PYTHONPATH"] = str(worktree / "src") + os.pathsep + str(worktree)
            if os.environ.get("GENERATION_EXECUTION_BACKEND") == "native":
                from repoforge.evaluation.native_provision import NativeProvisioner
                changes = NativeProvisioner.relocate_editable(os.environ["GENERATION_NATIVE_PREFIX"],
                    os.environ["GENERATION_NATIVE_SOURCE_PATH"], str(worktree))
                write_json(output / "cli-editable-relocation.json", changes)
            session.workspace.environment.env = session.info["config"]["env"]
            from repoforge.flow.durable import workspace_fingerprint
            session.info["initial_fingerprint"] = workspace_fingerprint(session.workspace, session.flow.checkpoint_path)
            service.save(session)
            write_json(output / "session-id.json", {"session_id": session.id})
        service.listeners.append(lambda identity, event: append_json(output / "events.jsonl", redact(event)))
        flow = session.flow
        record_loaded_config(bundle, flow, output)
        try:
            with recorder.capture():
                await service.start(session)
            selected = flow.root.get_last_node()
        except Exception as exc:
            failed = type(exc).__name__
        finally:
            await service.close()
        patch = service.diff(session)
        tests = service.tests(session)
    else:
        identity = environment_identity(os.environ["GENERATION_IMAGE_ID"])
        readiness = output / "runtime-ready.json"
        if resume and (not readiness.exists() or read_json(readiness) != identity):
            raise ValueError("Live generation environment differs from the checked ready runtime")
        def refresh_runtime_identity():
            actual = environment_identity(os.environ["GENERATION_IMAGE_ID"])
            write_json(readiness, actual)
            return actual

        runtime = SweBenchLocalEnvironment(Path("/testbed"), bundle["instance"], FileStorage(output / "tool-logs"),
                                          generation_only=True, recovery_identity=identity, resume_ready=resume,
                                          recovery_probe=refresh_runtime_identity)
        await runtime._wait_for_install()
        identity = environment_identity(os.environ["GENERATION_IMAGE_ID"])
        runtime._recovery_identity = identity
        write_json(readiness, identity)
        initial_runtime = output / "runtime-start.json"
        if not initial_runtime.exists():
            if resume and bundle.get("experiment_protocol") == "o2-a0-paired-v1":
                raise ValueError("Initial runtime identity is missing; refusing to invent resume provenance")
            if not resume:
                write_json(initial_runtime, identity)
        repository = GitRepository(repo_path="/testbed", shadow_mode=True)
        index = CodeIndex.from_persist_dir(str(index_path), file_repo=repository)
        workspace = Workspace(repository=repository, runtime=runtime, environment=runtime, code_index=index)
        cfg = json.loads(json.dumps(bundle["agent_config"]))
        cfg["completion_model"]["model_api_key"] = os.environ[config["api_key_env"]]
        for action in cfg["actions"]:
            if action.get("completion_model"):
                action["completion_model"]["model_api_key"] = os.environ[config["api_key_env"]]
        if bundle["profile"]["managed"]:
            # Choose memory before loading the Agent. Its constructor seals the
            # model's tool schema; reconstructing with an already initialized
            # Legacy model cannot add Managed tools or change its system prompt.
            cfg["memory"] = ManagedMessageHistoryGenerator(
                policy=ContextPolicy(**config["context_policy"])).model_dump(mode="json")
        agent = ActionAgent.from_dict(cfg)
        agent.model_timeout = config["model_timeout"]
        agent.tool_timeout = config["tool_timeout"]
        implementation = LangGraphFlow if bundle["profile"]["flow"] == "graph" else AgenticLoop
        options = {key: config[key] for key in ("max_iterations", "max_cost")}
        if implementation is LangGraphFlow:
            options.update(checkpoint_path=str(output / "checkpoint.sqlite"), checkpoint_run_id=job["job_id"],
                           model_timeout=config["model_timeout"], tool_timeout=config["tool_timeout"],
                           action_policy=bundle["profile"].get("action_policy", {"variant": "off"}))
        elif bundle["profile"].get("action_policy", {}).get("variant", "off") != "off":
            raise ValueError("ActionPolicy is supported only by the graph batch entrypoint")
        elif resume:
            raise ValueError("Legacy Loop does not claim durable automatic recovery")
        flow = implementation.create(message=bundle["instance"]["problem_statement"], agent=agent,
                                     project_id="t4-t5", trajectory_id=job["job_id"], **options)
        if implementation is LangGraphFlow:
            flow._budget_provider = recorder.totals
        record_loaded_config(bundle, flow, output, index)
        async def event_callback(event):
            append_json(output / "events.jsonl", redact(event.model_dump(mode="json")))
        async def snapshot(current):
            write_json(output / "trajectory.json", redact(current.get_trajectory_data()))
        flow._on_event = event_callback
        if implementation is LangGraphFlow:
            flow._on_snapshot = snapshot
        try:
            with recorder.capture():
                selected = await flow.run(workspace=workspace)
        except Exception as exc:
            failed = type(exc).__name__
        # O2 (a5) submits only the product patch so an agent cannot make its
        # own modified test part of the frozen answer.  Older profiles keep
        # their historical patch shape.
        product_only = submit_product_only(bundle["profile"])
        from repoforge.execution_memory import test_records
        tests = test_records(flow.root.get_last_node(), workspace) if bundle["profile"]["managed"] else []
        patch = flow.root.get_last_node().file_context.generate_git_patch(ignore_tests=product_only)
        submitted_verified_patch = False
        if product_only:
            patch, submitted_verified_patch = select_submitted_patch(patch, tests)
        completion = flow.root.get_last_node().completion_status or {"status": "unverified"}
    selected = selected or flow.root.get_last_node()
    steps = [step for node in selected.get_trajectory() for step in node.action_steps if step.observation is not None]
    actual = [step for step in steps if (step.observation.properties or {}).get("execution_status") != "skipped"]
    decisions = [decision for node in selected.get_trajectory() for decision in (node.policy_decisions or [])]
    write_json(output / "policy-decisions.json", decisions)
    (output / "patch.diff").write_text(patch)
    write_json(output / "trajectory.json", redact(flow.get_trajectory_data()))
    write_json(output / "agent-tests.json", redact(tests))
    write_json(output / "test-requests.json", redact([
        {"source_id": f"{node.node_id}:{index}", "arguments": step.action.model_dump(exclude={"thoughts"})}
        for node in selected.get_trajectory() for index, step in enumerate(node.action_steps)
        if step.action.name == "RunTests" and step.observation is not None
        and (step.observation.properties or {}).get("execution_status") != "skipped"
    ]))
    write_json(output / "generation.json", {
        "job_id": job["job_id"], "manifest_hash": job["manifest_hash"],
        "bundle_hash": content_hash(bundle),
        "generation_status": "failed" if failed or selected.error else "completed",
        "error_type": failed, "error_kind": (getattr(flow, "graph_state", None) or {}).get("error_kind"),
        "selected_node_id": selected.node_id, "finish_reason": (getattr(flow, "graph_state", None) or {}).get("finish_reason"),
        "generation_seconds": time.monotonic() - generation_start, "test_records": len(tests),
        "submit_product_only": product_only,
        "submitted_verified_patch": submitted_verified_patch,
        "submitted_patch_hash": content_hash(patch), "completion_status": completion,
        "flow_usage": flow.total_usage().model_dump(), "official_harness": "not_run",
        "policy_metrics": {"actual_tool_steps": len(actual), "skipped_proposals": len(steps) - len(actual),
            "policy_tool_steps": sum(step.origin == "policy" for step in actual),
            "decisions": len(decisions), "interventions": sum(d['decision']['kind'] != 'allow' for d in decisions),
            "test_processes": sum(len((step.observation.properties or {}).get('test_execution', [])) + (
                len((step.observation.properties or {}).get('public_baseline', {}).get('execution', []))
                if not (step.observation.properties or {}).get('public_baseline_cached') else 0) for step in actual)},
    })
    # A recovery conflict is not an independently frozen answer.
    if failed == "RecoveryConflict":
        raise ValueError("Recovery conflict; preserve evidence and reconcile before freezing")
    freeze_answer(output, job)


def build_index(bundle, output):
    from repoforge.index.code_index import CodeIndex
    from repoforge.index.settings import IndexSettings
    from repoforge.repository.git import GitRepository
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "index-receipt.json").exists():
        raise ValueError("Frozen indexes cannot be overwritten")
    # Preparation retries are separate attempts. Retain their unknown charges
    # and successful cache entries; never reset a generation SDK ledger.
    if (output / "sdk.jsonl").exists():
        import uuid
        archive = output / "index-preparation-attempts" / uuid.uuid4().hex
        archive.mkdir(parents=True)
        for name in ("sdk.jsonl", "usage.json"):
            if (output / name).exists():
                (output / name).rename(archive / name)
    recorder = SDKRecorder(output, bundle["prices"])
    settings = IndexSettings(**bundle.get("embedding_settings", {}))
    with recorder.capture():
        index = CodeIndex(file_repo=GitRepository(repo_path="/testbed", shadow_mode=True), settings=settings)
        index.run_ingestion(repo_path="/testbed", num_workers=1)
        index.persist(str(output))
    files = {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file() and path.name != "index-receipt.json"}
    histories = [read_json(path) for path in output.glob("index-preparation-attempts/*/usage.json")]
    histories.append(read_json(output / "usage.json"))
    write_json(output / "index-receipt.json", {"instance": bundle["instance"], "settings": settings.model_dump(mode="json"),
        "embedding_service_identity": bundle.get("embedding_service_identity"),
        "files": files, "digest": content_hash(files),
        "preparation_known_reference_cost": sum(row.get("known_reference_cost", 0) for row in histories),
        "preparation_cost_complete": all(row.get("cost_complete") for row in histories),
        "preparation_attempts": len(histories)})


async def validate_own_tests(bundle, output, answer_directory):
    """Fresh public-only container; no hidden test patch and no model credentials."""
    from repoforge.runtime.local import SweBenchLocalEnvironment
    from repoforge.storage.file_storage import FileStorage
    runtime = SweBenchLocalEnvironment(Path("/testbed"), bundle["instance"], FileStorage(output), generation_only=True)
    requests = read_json(Path(answer_directory) / "test-requests.json")
    patch = (Path(answer_directory) / "patch.diff").read_text()
    results = []
    await runtime._wait_for_install()
    for index, request in enumerate(requests):
        arguments = request["arguments"]
        source = arguments.get("test_source", "workspace")
        effective_patch = patch
        if source == "base_public":
            from unidiff import PatchSet
            from repoforge.utils.file import is_test
            effective_patch = "".join(str(file) for file in PatchSet(patch) if not is_test(file.path))
        async with runtime._temporary_patch(effective_patch):
            if "command" in arguments:
                code_output = await runtime._execute_command(arguments["command"], timeout=bundle["cli_config"]["tool_timeout"])
                stdout, code = code_output
                (Path(output) / f"own-test-{index}.txt").write_text(stdout)
                results.append({"source_id": request["source_id"], "return_code": code,
                                "output": f"own-test-{index}.txt", "command": arguments["command"]})
            else:
                tests = await runtime._run_tests_applied(test_files=arguments["test_files"], timeout=bundle["cli_config"]["tool_timeout"])
                results.append({"source_id": request["source_id"], "targets": arguments["test_files"],
                                "test_source": source, "tested_patch_hash": content_hash(effective_patch),
                                "execution": runtime.last_test_execution,
                                "results": [test.model_dump(mode="json") for test in tests]})
    write_json(Path(output) / "own-test-validation.json", {"status": "executed" if results else "not_run",
                                                            "hidden_tests": False, "model_calls": 0, "results": results})


def timed_index_build(bundle, output):
    """Observe stage boundaries without changing the frozen indexing algorithm."""
    from contextlib import ExitStack
    from functools import wraps
    from unittest.mock import patch
    from repoforge.index.code_index import CodeIndex

    rows = []

    def observe(name, original):
        @wraps(original)
        def measured(*args, **kwargs):
            started, clock = now(), time.monotonic()
            status = "failed"
            try:
                result = original(*args, **kwargs)
                status = "completed"
                return result
            finally:
                rows.append({"stage": name, "started": started, "ended": now(),
                             "wall_seconds": time.monotonic() - clock, "status": status})
        return measured

    try:
        with ExitStack() as stack:
            for method, stage in (("run_ingestion", "ingestion_including_embedding"), ("persist", "index_persistence")):
                stack.enter_context(patch.object(CodeIndex, method, observe(stage, getattr(CodeIndex, method))))
            build_index(bundle, output)
    finally:
        write_json(Path(output) / "index-stage-timings.json", {"stages": rows,
            "embedding_timing": "SDK request intervals are recorded separately in sdk.jsonl; ingestion includes parsing"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--own-tests", type=Path)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    if args.own_tests:
        asyncio.run(validate_own_tests(bundle, args.output, args.own_tests))
    elif args.build_index:
        timed_index_build(bundle, args.output)
    else:
        asyncio.run(run(bundle, args.output, args.index, args.resume))


if __name__ == "__main__":
    main()
