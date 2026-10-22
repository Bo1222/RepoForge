"""Public-only provisioning/generation and separate official Harness grading."""

import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

from moatless.evaluation.experiment import (
    append_json, content_hash, now, public_instance, read_json, sha, verify_answer, write_json,
)

from moatless.evaluation.experiment_inputs import bundle_for, embedding_credentials, export_public_dataset, index_settings_for
from moatless.index.embed_model import embedding_key_env


def docker_client():
    import docker
    client = docker.from_env()
    client.ping()
    return client


def safe_extract(archive, destination):
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError("Unexpected source archive member")
        source.extractall(destination)


def provision(output, manifest, phase):
    from docker.errors import ImageNotFound
    from swebench.harness.docker_build import build_instance_images
    from swebench.harness.test_spec.test_spec import make_test_spec
    output = Path(output).resolve()
    client = docker_client()  # fail before downloads or provider calls
    inventory = export_public_dataset(output, manifest, phase)
    started = time.monotonic()
    recipes = read_json(output / "host-install-recipes.json")
    # Placeholder official fields contain no answers, tests, hints or labels.
    specs = [make_test_spec({**row, **recipes.get(row["instance_id"], {}), "test_patch": "", "FAIL_TO_PASS": [], "PASS_TO_PASS": []})
             for row in inventory.values()]
    build_instance_images(client, specs, max_workers=1, tag="latest")
    config = output / "provision.json"
    receipts = read_json(config) if config.exists() else {}
    for spec in specs:
        baseline_image = client.images.get(spec.instance_image_key)
        tag = "moatless-generation:" + content_hash([baseline_image.id, manifest["source"]["sha256"]])[:24]
        try:
            image = client.images.get(tag)
        except ImageNotFound:
            with tempfile.TemporaryDirectory(prefix="moatless-generation-image-") as temporary:
                context = Path(temporary)
                safe_extract(output / "source.tar", context / "agent")
                # Framework Python is separate from the repository's historical testbed Python.
                # No dataset cache, grader output, Docker socket or user home is mounted.
                dockerfile = (
                    f"FROM {baseline_image.id}\n"
                    "USER root\nCOPY agent /agent\nWORKDIR /agent\n"
                    "RUN /opt/miniconda3/bin/conda create -n moatless-agent python=3.10.9 pip -y\n"
                    "RUN /opt/miniconda3/envs/moatless-agent/bin/pip install uv==0.8.15\n"
                    "RUN UV_CACHE_DIR=/tmp/uv-cache /opt/miniconda3/envs/moatless-agent/bin/uv sync --frozen --no-dev --python /opt/miniconda3/envs/moatless-agent/bin/python\n"
                    # Drop refs/reflogs to future code in this disposable generation image only.
                    "RUN cd /testbed && git checkout --detach HEAD && git for-each-ref --format='delete %(refname)' | git update-ref --stdin && git reflog expire --expire=now --all && git gc --prune=now\n"
                    "ENV SKIP_INSTALL=true\nCMD [\"sleep\", \"infinity\"]\n"
                )
                (context / "Dockerfile").write_text(dockerfile)
                image, logs = client.images.build(path=str(context), tag=tag, rm=True)
                for entry in logs:
                    append_json(output / "image-build.jsonl", {"instance_id": spec.instance_id, **entry})
        receipts[spec.instance_id] = {"image_id": image.id, "baseline_image_id": baseline_image.id,
                                      "source_sha256": manifest["source"]["sha256"], "public": inventory[spec.instance_id]}
        write_json(config, receipts)
    write_json(output / f"provision-{phase}.json", {"seconds": time.monotonic() - started, "instances": list(inventory),
                                                   "provider_calls": 0, "official_harness": "not_run"})
    return receipts


def worker_command(build_index=False, resume=False):
    argv = ["/agent/.venv/bin/python", "-m", "moatless.evaluation.generation",
            "--bundle", "/input/bundle.json", "--output", "/out"]
    if build_index:
        argv.append("--build-index")
    else:
        argv.extend(["--index", "/index"])
    if resume:
        argv.append("--resume")
    return argv


def exec_worker(container, command, log, environment=None):
    result = container.exec_run(command, environment=environment or {}, workdir="/agent", demux=False)
    # Output can include echoed task source; redact credentials before recording.
    output = result.output.decode(errors="replace")
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "JINA_API_KEY", "VOYAGE_API_KEY"):
        if os.environ.get(name):
            output = output.replace(os.environ[name], "[REDACTED]")
    Path(log).write_text(output)
    return result.exit_code


def run_job(output, manifest, job, resume=False):
    output = Path(output).resolve()
    directory = output / "runs" / job["job_id"]
    if (directory / "answer.json").exists():
        return verify_answer(directory, job)
    client = docker_client()
    provisioned = read_json(output / "provision.json")[job["instance_id"]]
    if provisioned["source_sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Generation image belongs to another frozen source")
    cli = manifest["cli_config"]
    if not os.environ.get(cli["api_key_env"]):
        raise ValueError("Required model credential is not configured")
    inputs = output / "inputs" / job["job_id"]
    inputs.mkdir(parents=True, exist_ok=True)
    write_json(inputs / "bundle.json", bundle_for(manifest, job, provisioned["public"]))
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "container.json"
    index_path = output / "indexes" / job["instance_id"]
    index_path.mkdir(parents=True, exist_ok=True)
    if resume:
        prior = read_json(receipt)
        container = client.containers.get(prior["container_id"])
        container.reload()
        if container.status != "running" or container.image.id != provisioned["image_id"]:
            raise ValueError("Original generation container is unavailable or changed; no silent reprovisioning")
        worker_resume = prior.get("phase") != "provisioned"
    else:
        if receipt.exists():
            raise ValueError("Job has prior effects; use --resume or a new experiment, never silent rerun")
        container = client.containers.run(provisioned["image_id"], detach=True,
            volumes={str(inputs): {"bind": "/input", "mode": "ro"},
                     str(directory): {"bind": "/out", "mode": "rw"},
                     str(index_path): {"bind": "/index", "mode": "ro"}},
            environment={name: os.environ[name] for name in (
                "DEEPSEEK_API_KEY", embedding_key_env(index_settings_for(manifest).embed_provider)) if name and os.environ.get(name)},
            labels={"moatless.experiment": manifest["manifest_hash"], "moatless.job": job["job_id"]})
        write_json(receipt, {"container_id": container.id, "image_id": container.image.id, "phase": "provisioned"})
        worker_resume = False
    if not (index_path / "index-receipt.json").exists() and job["entrypoint"] == "batch":
        credentials = embedding_credentials(manifest)
        setup = client.containers.run(provisioned["image_id"], detach=True,
            volumes={str(inputs): {"bind": "/input", "mode": "ro"}, str(index_path): {"bind": "/out", "mode": "rw"}},
            environment=credentials)
        try:
            code = exec_worker(setup, worker_command(build_index=True), output / f"index-{job['instance_id']}.log")
            if code:
                raise ValueError(f"Index preparation failed with exit code {code}")
        finally:
            setup.stop()
            setup.remove()
    index_digest = None
    if job["entrypoint"] == "batch":
        index_receipt = read_json(index_path / "index-receipt.json")
        if index_receipt["instance"] != provisioned["public"]:
            raise ValueError("Index belongs to another instance/base commit")
        for name, expected in index_receipt["files"].items():
            if sha(index_path / name) != expected:
                raise ValueError("Frozen index changed")
        index_digest = index_receipt["digest"]
    write_json(receipt, {"container_id": container.id, "image_id": container.image.id, "phase": "worker_started"})
    code = exec_worker(container, worker_command(resume=worker_resume), directory / f"worker-{now().replace(':', '-')}.log",
        {"GENERATION_CONTAINER_ID": container.id, "GENERATION_IMAGE_ID": container.image.id,
         "GENERATION_INDEX_DIGEST": index_digest or "none"})
    append_json(directory / "attempts.jsonl", {"timestamp": now(), "resume": resume, "exit_code": code})
    if code:
        # Keep container/workspace/checkpoints so uncertain effects can be reconciled.
        raise ValueError(f"Generation failed with exit code {code}; inspect retained worker evidence")
    answer = verify_answer(directory, job)
    container.stop()  # terminal container remains available for separate own-test validation
    return answer


def baseline_job(output, manifest, job):
    """After answer freeze, run official tests on the unmodified product in isolation.

    This is an original-failure diagnostic using the official TestSpec/parser,
    not a candidate Harness verdict. No gold product patch is ever applied.
    """
    from datasets import load_dataset
    from pathlib import PurePosixPath
    from swebench.harness.docker_utils import copy_to_container, exec_run_with_timeout
    from swebench.harness.constants import EvalType, FAIL_ONLY_REPOS
    from swebench.harness.grading import get_logs_eval, get_eval_tests_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    output = Path(output).resolve()
    answer = verify_answer(output / "runs" / job["job_id"], job)
    client = docker_client()
    dataset = load_dataset(manifest["dataset"], split="test", cache_dir=str(output / "host-dataset-cache"))
    receipt = read_json(output / "dataset-receipt.json")
    if dataset._fingerprint != receipt["fingerprint"]:
        raise ValueError("Grading dataset differs from frozen public dataset")
    row = next(row for row in dataset if row["instance_id"] == job["instance_id"])
    provisioned = read_json(output / "provision.json")[job["instance_id"]]
    if public_instance(row) != provisioned["public"]:
        raise ValueError("Baseline instance differs from generation instance")
    identity = content_hash([job["instance_id"], receipt["fingerprint"], provisioned["baseline_image_id"]])
    directory = output / "baseline-diagnostics" / identity
    prior = directory / "result.json"
    if prior.exists() and read_json(prior).get("status") == "completed":
        result = read_json(prior)
        if sha(directory / "test-output.txt") != result["log_sha256"]:
            raise ValueError("Original baseline diagnostic evidence changed")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    # Official tests/labels are confined to this post-freeze grader container.
    spec = make_test_spec(row)
    script = directory / "eval.sh"
    script.write_text(spec.eval_script)
    container = client.containers.run(provisioned["baseline_image_id"], detach=True,
                                       command=["sleep", "infinity"], environment={}, volumes={})
    result = {"identity": identity, "instance_id": job["instance_id"], "status": "error",
              "official_harness": "not_run", "kind": "official_TestSpec_original_failure_diagnostic",
              "product_patch_applied": False, "provider_calls": 0, "answer_frozen_before_diagnostic": answer["answer_hash"],
              "image_id": container.image.id, "container_id": container.id, "dataset_fingerprint": dataset._fingerprint}
    try:
        check = container.exec_run(["bash", "-lc", "git rev-parse HEAD && git status --porcelain"], workdir="/testbed")
        clean = check.output.decode().strip()
        if check.exit_code or clean != row["base_commit"]:
            raise ValueError("Original baseline is not the clean frozen base commit")
        copy_to_container(container, script, PurePosixPath("/eval.sh"))
        output_text, timed_out, seconds = exec_run_with_timeout(container, "/bin/bash /eval.sh", 1800)
        log = directory / "test-output.txt"
        log.write_text(output_text)
        statuses, parsed = get_logs_eval(spec, str(log))
        tests = get_eval_tests_report(statuses, {"FAIL_TO_PASS": spec.FAIL_TO_PASS, "PASS_TO_PASS": spec.PASS_TO_PASS},
                    eval_type=EvalType.FAIL_ONLY if spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL) if parsed else {}
        observed_failures = [name for name in spec.FAIL_TO_PASS if statuses.get(name) in {"FAILED", "ERROR"}]
        result.update(status="completed" if parsed and not timed_out else "error", timed_out=timed_out,
                      test_seconds=seconds, log_sha256=sha(log), tests_status=tests,
                      observed_FAIL_TO_PASS_failures=observed_failures,
                      failure_reproduced=bool(observed_failures) if parsed and any(name in statuses for name in spec.FAIL_TO_PASS) else None,
                      process_exit_code=None, exit_code_note="Official timeout helper does not expose process exit code")
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(prior, result)
        container.stop()
        container.remove()
        verify_answer(output / "runs" / job["job_id"], job)
    return result


def grade_job(output, manifest, job):
    output = Path(output).resolve()
    directory = output / "runs" / job["job_id"]
    answer = verify_answer(directory, job)
    baseline = baseline_job(output, manifest, job)
    attempt = content_hash([answer["answer_hash"], now()])[:24]
    grading = output / "grading" / attempt
    grading.mkdir(parents=True)
    predictions = grading / "predictions.jsonl"
    predictions.write_text(json.dumps({"instance_id": job["instance_id"],
        "model_name_or_path": job["variant"], "model_patch": (directory / "patch.diff").read_text()}) + "\n")
    argv = [sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", manifest["dataset"],
            "--predictions_path", str(predictions), "--instance_ids", job["instance_id"],
            "--max_workers", "1", "--run_id", attempt, "--timeout", "1800"]
    started = time.monotonic()
    with (grading / "harness.log").open("w") as stream:
        result = subprocess.run(argv, cwd=grading, stdout=stream, stderr=subprocess.STDOUT, timeout=7200)
    reports = list(grading.glob("logs/evaluation/**/report.json"))
    report = read_json(reports[0]).get(job["instance_id"]) if len(reports) == 1 else None
    summary = {"answer_hash": answer["answer_hash"], "official_harness": "completed" if report else "error",
               "resolved": report["resolved"] if report else None, "exit_code": result.returncode,
               "command": argv, "cwd": str(grading), "grading_seconds": time.monotonic() - started,
               "patch_successfully_applied": report.get("patch_successfully_applied") if report else None,
               "FAIL_TO_PASS": (report or {}).get("tests_status", {}).get("FAIL_TO_PASS"),
               "PASS_TO_PASS": (report or {}).get("tests_status", {}).get("PASS_TO_PASS"),
               "report": report, "evidence": str(grading.relative_to(output))}
    summary["original_baseline"] = {"identity": baseline["identity"], "status": baseline["status"],
                                    "failure_reproduced": baseline.get("failure_reproduced")}
    if not (directory / "patch.diff").read_text().strip() and result.returncode == 0:
        # Official Harness skips empty patches; do not manufacture test results.
        summary.update(official_harness="empty_patch", resolved=False, patch_successfully_applied=False)
    append_json(directory / "grading-attempts.jsonl", summary)
    write_json(directory / "grading.json", summary)
    verify_answer(directory, job)
    return summary


def probe_recovery(output, manifest):
    """Four crash windows in real SWE runtime, shadow and disk; no model API."""
    output = Path(output).resolve()
    client = docker_client()
    job = {"phase": "recovery", "variant": "a0", "instance_id": "pytest-dev__pytest-7490",
           "manifest_hash": manifest["manifest_hash"], "entrypoint": "recovery"}
    instance = read_json(output / "provision.json")[job["instance_id"]]
    if instance["source_sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Recovery image belongs to another frozen source")
    destination = output / "recovery" / content_hash([manifest["manifest_hash"], now()])[:24]
    destination.mkdir(parents=True)
    inputs = destination / "input"
    write_json(inputs / "bundle.json", bundle_for(manifest, job, instance["public"]))
    container = client.containers.run(instance["image_id"], detach=True, environment={},
        volumes={str(inputs): {"bind": "/input", "mode": "ro"}, str(destination): {"bind": "/out", "mode": "rw"}})
    rows = []
    environment = {"GENERATION_CONTAINER_ID": container.id, "GENERATION_IMAGE_ID": container.image.id,
                   "GENERATION_INDEX_DIGEST": "none"}
    try:
        for shadow in (False, True):
            for fault in ("after_generation", "before_tool", "after_effect", "after_commit"):
                name = ("shadow" if shadow else "disk") + "-" + fault
                command = ["/agent/.venv/bin/python", "-m", "scripts.recovery_smoke",
                           "--directory", f"/out/{name}", "--swe-bundle", "/input/bundle.json"]
                if shadow:
                    command.append("--shadow")
                first = exec_worker(container, command + ["--fault", fault], destination / f"{name}-fault.log", environment)
                calls = (destination / name / "calls.jsonl").read_text()
                resumed = exec_worker(container, command, destination / f"{name}-resume.log", environment)
                after = (destination / name / "calls.jsonl").read_text()
                uncertain = fault in {"before_tool", "after_effect"}
                passed = first == 88 and ((resumed != 0 and calls == after and (destination / name / "conflict.json").exists())
                                         if uncertain else resumed == 0)
                if not uncertain and resumed == 0:
                    state = read_json(destination / name / "result.json")
                    kinds = [json.loads(line)["kind"] for line in after.splitlines()]
                    passed = passed and kinds == ["model", "tool:0", "tool:1", "tool:2", "tool:3"] and state["finish_reason"] == "terminal"
                rows.append({"case": name, "first_exit_code": first, "resume_exit_code": resumed,
                             "unknown_effect_refused": uncertain, "passed": passed, "commands": [command + ["--fault", fault], command]})
    finally:
        write_json(destination / "summary.json", {"manifest_hash": manifest["manifest_hash"], "cases": rows,
            "passed": len(rows) == 8 and all(row["passed"] for row in rows), "provider_calls": 0, "official_harness": "not_run",
            "evidence": {str(path.relative_to(destination)): sha(path) for path in destination.rglob("*") if path.is_file()}})
        container.stop()
        container.remove()
    summary = read_json(destination / "summary.json")
    if not summary["passed"]:
        raise ValueError("Real SWE recovery probe failed; inspect preserved case evidence")
    return summary


def validate_agent_tests(output, manifest, job):
    output = Path(output).resolve()
    client = docker_client()
    directory = output / "runs" / job["job_id"]
    answer = verify_answer(directory, job)
    instance = read_json(output / "provision.json")[job["instance_id"]]
    destination = output / "own-tests" / content_hash([answer["answer_hash"], now()])[:24]
    destination.mkdir(parents=True)
    inputs = output / "inputs" / job["job_id"]
    own_input = destination / "answer-input"
    own_input.mkdir()
    for name in ("patch.diff", "test-requests.json"):
        shutil.copyfile(directory / name, own_input / name)
    container = client.containers.run(instance["image_id"], detach=True, environment={},
        volumes={str(inputs): {"bind": "/input", "mode": "ro"},
                 str(own_input): {"bind": "/answer", "mode": "ro"},
                 str(destination): {"bind": "/out", "mode": "rw"}})
    try:
        command = ["/agent/.venv/bin/python", "-m", "moatless.evaluation.generation", "--bundle", "/input/bundle.json",
                   "--output", "/out", "--own-tests", "/answer"]
        code = exec_worker(container, command, destination / "worker.log")
        write_json(destination / "result.json", {"answer_hash": answer["answer_hash"], "exit_code": code,
                                                 "official_harness": "not_run", "hidden_tests": False})
        if code:
            raise ValueError(f"Own-test validation failed with exit code {code}")
        validation = read_json(destination / "own-test-validation.json")
        write_json(directory / "own-tests.json", {"answer_hash": answer["answer_hash"],
            "status": validation["status"], "evidence": {
                str((destination / name).relative_to(output)): sha(destination / name)
                for name in ("result.json", "own-test-validation.json")}})
    finally:
        container.stop()
        container.remove()
    return read_json(destination / "result.json")
