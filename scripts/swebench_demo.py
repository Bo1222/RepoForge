"""Run a real SWE-bench task through the CLI service, with independent local grading.

Default: pytest-dev__pytest-7490. No golden patch is used. No Docker or remote PR writes.
Large repositories, environments and private grading inputs stay below /tmp.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import time

import litellm
from dotenv import load_dotenv
import requests
import pyarrow.parquet as parquet

from moatless.cli.service import SessionService, SYSTEM_PROMPT, api_key, error_message, safe_config
from moatless.cli import worktree
from swebench.harness.constants.python import MAP_REPO_VERSION_TO_SPECS_PY
from swebench.harness.log_parsers.python import parse_log_pytest
from swebench.harness.test_spec.python import get_test_directives

DATASET = "princeton-nlp/SWE-bench_Lite"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def append_json(path, value, secret=None):
    with path.open("a") as stream:
        stream.write(json.dumps(safe_config(value, secret), ensure_ascii=False, default=str) + "\n")


def sequence(value):
    return json.loads(value) if isinstance(value, str) else value


def command(argv, cwd, log, env=None, timeout=900, check=True, input=None):
    started = time.monotonic()
    try:
        result = subprocess.run(argv, cwd=cwd, env=env, input=input, capture_output=True, text=True, timeout=timeout)
        text = result.stdout + result.stderr
        log.write_text(text)
        append_json(
            log.parent / "commands.jsonl",
            {
                "timestamp": now(),
                "argv": argv,
                "cwd": str(cwd),
                "return_code": result.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "output": log.name,
            },
        )
        if check and result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
        return result.returncode, text
    except subprocess.TimeoutExpired as exc:
        log.write_text(str(exc))
        raise RuntimeError(f"Command timed out; see {log}") from exc


def fetch_instance(identity):
    # Read official Parquet columns excluding the reference product patch and hints.
    listing = requests.get(f"https://huggingface.co/api/datasets/{DATASET}/tree/main/data", timeout=(15, 30))
    listing.raise_for_status()
    shards = [item for item in listing.json() if item["path"].startswith("data/test-")]
    matches = []
    for item in shards:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{item['path']}"
        response = requests.get(url, params={"download": "true", "nocache": str(time.time_ns())}, timeout=(15, 60))
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()
        if digest != item["lfs"]["oid"]:
            raise ValueError("Official dataset shard hash differs from its published LFS hash")
        source = parquet.ParquetFile(io.BytesIO(response.content))
        columns = [name for name in source.schema_arrow.names if name not in {"patch", "hints_text"}]
        for row in source.read(columns=columns).to_pylist():
            if row["instance_id"] == identity:
                matches.append({**row, "dataset_file": item["path"], "dataset_sha256": digest})
    if len(matches) != 1:
        raise ValueError("Expected exactly one official dataset instance")
    return matches[0]


def classify(instance, baseline, repaired):
    rows = []
    for group in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        for test in sequence(instance[group]):
            before, after = baseline.get(test, "NOT_RUN"), repaired.get(test, "NOT_RUN")
            expected = "FAILED" if group == "FAIL_TO_PASS" else "PASSED"
            rows.append(
                {
                    "group": group,
                    "test": test,
                    "baseline": before,
                    "repaired": after,
                    "baseline_valid": before == expected,
                    "passed": after == "PASSED",
                }
            )
    return {
        "cases": rows,
        "baseline_valid": all(r["baseline_valid"] for r in rows),
        "local_validation_passed": bool(rows) and all(r["baseline_valid"] and r["passed"] for r in rows),
        "official_harness": "not_run",
    }


def prepare_generated_files(source, destination):
    # setuptools-scm creates this ignored build artifact during installation.
    generated = source / "src/_pytest/_version.py"
    if generated.exists():
        shutil.copy2(generated, destination / "src/_pytest/_version.py")


def validate_copy(source, destination, instance, python, product_patch, logs, label):
    command(["git", "clone", "--shared", "--quiet", str(source), str(destination)], logs, logs / f"{label}-clone.txt")
    command(["git", "checkout", "--detach", instance["base_commit"]], destination, logs / f"{label}-checkout.txt")
    prepare_generated_files(source, destination)
    if product_patch.strip():
        command(["git", "apply", "--check", "-"], destination, logs / f"{label}-product-check.txt", input=product_patch)
        command(["git", "apply", "-"], destination, logs / f"{label}-product-apply.txt", input=product_patch)
    command(
        ["git", "apply", "--check", "-"], destination, logs / f"{label}-tests-check.txt", input=instance["test_patch"]
    )
    command(["git", "apply", "-"], destination, logs / f"{label}-tests-apply.txt", input=instance["test_patch"])
    env = {**os.environ, "PYTHONPATH": str(destination / "src"), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    argv = [str(python), "-m", "pytest", "-rA", *get_test_directives(instance)]
    code, output = command(argv, destination, logs / f"{label}-test-output.txt", env=env, check=False)
    statuses = parse_log_pytest(output, None)
    write_json(logs / f"{label}-test-statuses.json", statuses)
    return code, statuses


def product_only_patch(session, instance):
    # Official test files are restored to baseline before applying the hidden test patch.
    # Generate a temporary-index patch excluding all test files; test edits remain in full export.
    repo = session.info["worktree"]
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        worktree.git(repo, "read-tree", session.info["base_commit"], env=env)
        worktree.git(repo, "add", "-A", "--", ".", env=env)
        return worktree.git(
            repo,
            "diff",
            "--cached",
            "--binary",
            session.info["base_commit"],
            "--",
            ".",
            ":(exclude)testing/**",
            ":(exclude)tests/**",
            env=env,
        )


async def run_agent(source, python, instance, output, runtime, args):
    service = SessionService(runtime / "sessions")
    config = {
        "model": args.model,
        "max_iterations": args.max_iterations,
        "max_cost": args.max_cost,
        "model_timeout": 180,
        "tool_timeout": 900,
        "thoughts_in_action": True,
        "env": {
            "PATH": str(python.parent) + os.pathsep + os.environ["PATH"],
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "",
        },
        "system_prompt": SYSTEM_PROMPT
        + "\nFor each action, use the thoughts field ONLY for a brief public action rationale: objective, observed evidence (file/line or prior tool result), and expected check. Do not provide private chain-of-thought. Do not read outside the task worktree or access upstream fixes or benchmark answers. Use RunTests for verification. Finish only after explaining the reproduced issue, fix, actual verification and any unresolved limitations.",
    }
    task = (
        instance["problem_statement"]
        + "\n\nTask: Fix this real issue. Read the existing implementation, reproduce the regression before editing product code, add appropriate regression tests, and run the relevant existing tests. Do not weaken tests. Do not commit or push. The task's Python environment is on PATH; use python -m pytest. Official hidden grading tests will be run independently after your answer is frozen."
    )
    write_json(output / "agent-config.json", safe_config(config))
    (output / "task.md").write_text(task)
    session = await service.create(str(source), task, config)
    prepare_generated_files(source, Path(session.info["worktree"]))
    # Explicit PYTHONPATH prevents the editable base checkout from masking worktree changes.
    session.info["config"]["env"]["PYTHONPATH"] = str(Path(session.info["worktree"]) / "src")
    session.workspace.environment.env = session.info["config"]["env"]
    from moatless.flow.durable import workspace_fingerprint

    session.info["initial_fingerprint"] = workspace_fingerprint(session.workspace, session.flow.checkpoint_path)
    service.save(session)
    secret = api_key(config)
    initial_snapshot = session.flow._on_snapshot
    sdk = litellm.acompletion
    requests_count = 0
    patches = output / "action-patches"
    patches.mkdir()

    async def snapshot(flow):
        await initial_snapshot(flow)
        state = flow.graph_state or {}
        append_json(
            output / "phases.jsonl",
            {
                "timestamp": now(),
                "node_id": state.get("current_node_id"),
                "snapshot_version": state.get("snapshot_version"),
                "phase": state.get("phase"),
                "action_cursor": state.get("action_cursor"),
                "finish_reason": state.get("finish_reason"),
                "error_kind": state.get("error_kind"),
            },
            secret,
        )

    session.flow._on_snapshot = snapshot

    def listener(identity, event):
        data = event.get("data") or {}
        if event.get("scope") == "tool":
            print(f"{now()} Node{data['node_id']} step{data['step']} {data['name']} {event['event_type']}", flush=True)
            if event["event_type"] in {"running", "completed", "failed", "cancelled"}:
                name = f"node-{data['node_id']}-step-{data['step']}-{event['event_type']}.diff"
                (patches / name).write_text(service.diff(session))

    service.listeners.append(listener)

    async def audited_sdk(*positional, **kwargs):
        nonlocal requests_count
        requests_count += 1
        call = requests_count
        node = session.flow.root.get_last_node().node_id
        append_json(
            output / "model-calls.jsonl",
            {
                "call": call,
                "node_id": node,
                "timestamp": now(),
                "event": "request",
                "request": {k: v for k, v in kwargs.items() if k not in {"api_key", "extra_headers", "headers"}},
            },
            secret,
        )
        started = time.monotonic()
        try:
            response = await sdk(*positional, **kwargs)  # Real provider call, never scripted.
            append_json(
                output / "model-calls.jsonl",
                {
                    "call": call,
                    "node_id": node,
                    "timestamp": now(),
                    "event": "response",
                    "elapsed_seconds": time.monotonic() - started,
                    "response": response.model_dump(mode="json"),
                },
                secret,
            )
            return response
        except BaseException as exc:
            append_json(
                output / "model-calls.jsonl",
                {
                    "call": call,
                    "node_id": node,
                    "timestamp": now(),
                    "event": "error",
                    "error": error_message(exc, config),
                },
                secret,
            )
            raise

    litellm.acompletion = audited_sdk
    failure = None
    try:
        await service.start(session)
    except Exception as exc:
        failure = error_message(exc, config)
        print("Agent stopped: " + failure, flush=True)
    finally:
        litellm.acompletion = sdk
        await service.close()
        service.export(session, str(output / "agent-export"))
        (output / "product.patch").write_text(product_only_patch(session, instance))
    return (
        session,
        service,
        {
            "session_id": session.id,
            "status": session.info["status"],
            "error": failure,
            "sdk_calls": requests_count,
            "usage": service.usage(session),
        },
    )


def report(output, instance, result, session):
    events = [json.loads(line) for line in (output / "agent-export/events.jsonl").read_text().splitlines()]
    rows = [e for e in events if e.get("scope") == "tool" and e["event_type"] in {"completed", "failed", "cancelled"}]
    assistant_excerpts = {}
    for line in (output / "model-calls.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["event"] == "response":
            text = record["response"]["choices"][0]["message"].get("content") or ""
            if text.strip():
                assistant_excerpts[record["node_id"]] = text.strip().split("\n\n", 1)[0][:300]
    lines = [
        f"# SWE-bench 真实修复：{instance['instance_id']}",
        "",
        "本报告基于真实模型与工具输出。行动说明由模型提供；它是可审阅的简短依据，不代表内部思考过程。",
        "",
        f"- 基准提交：{instance['base_commit']}",
        f"- 运行状态：{result['agent']['status']}；SDK 请求数：{result['agent']['sdk_calls']}",
        f"- Agent 声称完成：{result['agent_claimed_complete']}",
        f"- 独立本地验证通过：{result['local_validation_passed']}",
        "- 官方 Docker Harness：not_run（当前用户不能访问 Docker）",
        "",
        "## 逐步轨迹",
        "",
    ]
    for event in rows:
        d = event["data"]
        args = dict(d.get("arguments") or {})
        reason = args.pop("thoughts", None)
        if not reason and d["node_id"] in assistant_excerpts:
            reason = "同轮模型回复原文节选（非工具参数中的说明）：" + assistant_excerpts[d["node_id"]]
        obs = d.get("observation") or {}
        ref = f"Node{d['node_id']} / step{d['step']}"
        lines += [
            f"### {ref}：{d['name']}",
            "",
            f"模型行动说明：{reason or '缺失；不补造调用理由。'}",
            "",
            "参数：",
            "```json",
            json.dumps(args, ensure_ascii=False, indent=2),
            "```",
            "",
            f"实际状态：{event['event_type']}；耗时：{d.get('elapsed', 'unknown')} 秒；结果字段：{json.dumps(obs.get('properties', {}), ensure_ascii=False)}",
            "",
            "实际输出（完整）：",
            "```text",
            obs.get("message") or d.get("error", ""),
            "```",
            "",
        ]
    lines += [
        "## 状态与完成判定",
        "",
        "Agent 按 issue、历史动作与 Observation 选择下一工具。Flow 的 generate/policy/execute/record/commit/end 是执行阶段，不是 issue 正确性的证明。完整阶段记录见 phases.jsonl。",
        "",
        "Finish 是模型结束声明；费用/轮数耗尽、工具错误和模型错误是其他停止原因。独立验证只在冻结补丁后运行，不把隐藏测试反馈给 Agent。",
        "",
        "## 独立测试结果",
        "",
        "| 类型 | 测试 | 基准 | 修复后 | 合格 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in result["cases"]:
        lines.append(
            f"| {case['group']} | {case['test'].replace('|', chr(92) + '|')} | {case['baseline']} | {case['repaired']} | {case['baseline_valid'] and case['passed']} |"
        )
    lines += [
        "",
        "本地独立验证不等于官方 resolved。若有缺失测试、环境错误或失败，结果保持未通过。官方参考产品补丁未使用。",
        "",
        "## 复现与证据",
        "",
        "启动命令见 invocation.json；环境安装与验证命令见 commands.jsonl。模型请求与响应见 model-calls.jsonl；原始事件、轨迹、补丁、用量与标准 predictions 见 agent-export/。每个动作前后补丁见 action-patches/。",
        "",
        "原始任务见 task.md。在相同基准与环境下可通过 CLI 的 --repo、--config、--task-file 重现任务；CLI 内使用 /tests、/diff、/usage、/architecture 查看。实时模型再次运行可能产生不同轨迹。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", default="pytest-dev__pytest-7490")
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--max-iterations", type=int, default=80)
    args = parser.parse_args()
    load_dotenv()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "runner-snapshot.py").write_text(Path(__file__).read_text())
    runtime = Path(tempfile.mkdtemp(prefix="moatless-swebench-", dir="/tmp"))
    write_json(
        output / "invocation.json", {"args": vars(args), "runtime": runtime, "started": now(), "cwd": Path.cwd()}
    )
    logging.basicConfig(filename=output / "cli.log", level=logging.WARNING, force=True)
    credential = api_key({"model": args.model})

    class RedactedFormatter(logging.Formatter):
        def format(self, record):
            rendered = super().format(record)
            return rendered.replace(credential, "[API key redacted]") if credential else rendered

    for handler in logging.getLogger().handlers:
        handler.setFormatter(RedactedFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger("LiteLLM").handlers.clear()
    logging.getLogger("LiteLLM").propagate = True
    try:
        print("Fetching official instance", flush=True)
        instance = fetch_instance(args.instance_id)
        write_json(runtime / "grading-input.json", instance)
        public = {
            k: instance[k]
            for k in (
                "instance_id",
                "repo",
                "base_commit",
                "version",
                "problem_statement",
                "dataset_file",
                "dataset_sha256",
            )
        }
        write_json(output / "instance.json", {"dataset": DATASET, **public})
        source = runtime / "source"
        command(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/" + instance["repo"] + ".git",
                str(source),
            ],
            output,
            output / "source-clone.txt",
        )
        command(["git", "checkout", "--detach", instance["base_commit"]], source, output / "source-checkout.txt")
        specs = MAP_REPO_VERSION_TO_SPECS_PY[instance["repo"]][instance["version"]]
        write_json(output / "environment-spec.json", specs)
        command(
            ["uv", "venv", "--python", specs["python"], str(runtime / "testenv")], output, output / "venv-create.txt"
        )
        python = runtime / "testenv/bin/python"
        command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "setuptools==68.0.0",
                "setuptools-scm==7.1.0",
                "pip",
                "wheel",
                *specs.get("pip_packages", []),
            ],
            output,
            output / "dependencies-install.txt",
        )
        command(
            ["uv", "pip", "install", "--python", str(python), "--no-build-isolation", "-e", str(source)],
            output,
            output / "project-install.txt",
        )
        command([str(python), "-m", "pip", "freeze"], output, output / "environment-freeze.txt", check=False)
        baseline_code, baseline = validate_copy(source, runtime / "baseline", instance, python, "", output, "baseline")
        gate = classify(instance, baseline, {})
        write_json(output / "baseline-check.json", gate)
        if not gate["baseline_valid"]:
            raise RuntimeError("Baseline does not reproduce every required official test; see baseline-check.json")
        print("Baseline verified; starting real Agent", flush=True)
        session, service, agent = asyncio.run(run_agent(source, python, instance, output, runtime, args))
        product_patch = (output / "product.patch").read_text()
        repaired_code, repaired = validate_copy(
            source, runtime / "repaired", instance, python, product_patch, output, "repaired"
        )
        env = {**os.environ, **session.info["config"]["env"]}
        agent_tests_code, _ = command(
            [str(python), "-m", "pytest", "-rA", "testing/test_skipping.py"],
            Path(session.info["worktree"]),
            output / "agent-independent-tests.txt",
            env=env,
            check=False,
        )
        result = classify(instance, baseline, repaired)
        result.update(
            {
                "agent": agent,
                "agent_claimed_complete": any(
                    s.action.name == "Finish" and s.observation and s.observation.terminal
                    for n in session.flow.root.get_all_nodes()
                    for s in n.action_steps
                ),
                "baseline_return_code": baseline_code,
                "repaired_return_code": repaired_code,
                "agent_tests_return_code": agent_tests_code,
                "finished": now(),
                "runtime": str(runtime),
            }
        )
        result["local_validation_passed"] = (
            result["local_validation_passed"]
            and bool(product_patch.strip())
            and repaired_code == 0
            and agent_tests_code == 0
        )
        write_json(output / "result.json", result)
        report(output, instance, result, session)
        print(
            json.dumps(
                {k: result[k] for k in ("agent_claimed_complete", "local_validation_passed", "official_harness")}
            ),
            flush=True,
        )
    except Exception as exc:
        message = error_message(exc, {"model": args.model})
        write_json(output / "run-error.json", {"error": message, "timestamp": now(), "runtime": str(runtime)})
        print(message, flush=True)
        raise SystemExit(1) from None
    finally:
        files = sorted(p for p in output.rglob("*") if p.is_file())
        write_json(
            output / "manifest.json",
            {
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
                if p.name != "manifest.json"
            },
        )


if __name__ == "__main__":
    main()
