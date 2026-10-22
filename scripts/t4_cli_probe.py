"""Real Flash non-thinking read/edit/test/follow-up probe on an isolated arithmetic repository.

Synthetic local task, not a SWE-bench instance or official grading result.
"""

import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import httpx
import tempfile

from dotenv import load_dotenv

from moatless.cli.service import SessionService
from moatless.evaluation.experiment import PRICES, now, read_json, write_json
from moatless.evaluation.experiment_usage import SDKRecorder


def model_metadata():
    # Use the same verified CA bundle/transport as the model SDK. Conda's
    # stdlib default CA directory is outside the native framework allowlist.
    response = httpx.get("https://api.deepseek.com/models", headers={
        "Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]}, timeout=30)
    response.raise_for_status()
    models = response.json()["data"]
    model = next(row for row in models if row["id"] == "deepseek-flash")
    if model.get("name") != "DeepSeek-V4.1-Flash" or model.get("context_window") != 1048576:
        raise ValueError("Live canonical model changed; freeze a new configuration")
    return model


async def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory to preserve prior attempts")
    output.mkdir(parents=True)
    write_json(output / "model-metadata.json", {"timestamp": now(), "model": model_metadata(),
                                              "method": "GET /models", "inference_calls": 0})
    source = output / "source"
    source.mkdir()
    (source / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (source / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    (source / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    for args in (["git", "init", "--quiet", str(source)], ["git", "-C", str(source), "add", "."],
                 ["git", "-C", str(source), "-c", "user.name=T4 probe", "-c", "user.email=t4@example.invalid", "commit", "-qm", "Synthetic baseline"]):
        subprocess.run(args, check=True, capture_output=True)
    baseline = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    config = read_json(Path(__file__).resolve().parents[1] / "moatless/evaluation/profiles/flash_nonthinking_a0.json")
    config.update(max_iterations=12, max_cost=.1, env={"PATH": str(Path(os.sys.executable).parent) + os.pathsep + os.environ["PATH"],
                                                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""})
    write_json(output / "config.json", config)
    service = SessionService(output / "sessions")
    task = "修复 calc.py 中 add(2,3) 返回 -1 而不是5的问题。先读取源码并用 RunTests 执行 python -m pytest -q 复现，再仅修正 add 的运算符，重新运行同一测试，最后说明实际验证结果。不要提交或推送，不要访问工作区外的文件。"
    session = await service.create(str(source), task, config)
    recorder = SDKRecorder(output, PRICES, .1)
    try:
        with recorder.capture():
            await service.start(session)
            await service.start(session, "继续：读取 calc.py 并再次用 RunTests 执行 python -m pytest -q，确认修复仍然通过；无需修改代码。")
        service.export(session, str(output / "export"))
        tests = service.tests(session)
        write_json(output / "result.json", {"task": "synthetic local arithmetic", "baseline_commit": baseline,
            "session_id": session.id, "status": session.info["status"], "tests": tests,
            "completion_status": session.info.get("completion_status"), "official_harness": "not_run",
            "source_unchanged": (source / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"})
        if session.info["status"] not in {"completed", "terminal"} or not tests or tests[-1].get("return_code") != 0:
            raise ValueError("Real CLI probe did not complete with a recorded passing test")
    finally:
        await service.close()


def verify_export(output):
    """Recheck original evidence and full exported patch, without another model call."""
    output = Path(output).resolve()
    result = read_json(output / "result.json")
    if result["status"] not in {"completed", "terminal"} or not result["source_unchanged"]:
        raise ValueError("Probe did not terminate normally with unchanged source checkout")
    if result.get("completion_status", {}).get("status") != "verified":
        raise ValueError("No current-version task-local verification")
    with tempfile.TemporaryDirectory(prefix="moatless-t4-independent-") as temporary:
        target = Path(temporary) / "repo"
        subprocess.run(["git", "clone", "--quiet", str(output / "source"), str(target)], check=True)
        env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}
        before = subprocess.run([os.sys.executable, "-m", "pytest", "-q"], cwd=target, env=env, capture_output=True, text=True)
        # Separate copy avoids baseline-generated caches colliding with artifacts
        # in a historical full patch; the original answer is never filtered.
        target = Path(temporary) / "fixed"
        subprocess.run(["git", "clone", "--quiet", str(output / "source"), str(target)], check=True)
        patch = (output / "export/patch.diff").read_text()
        subprocess.run(["git", "apply", "--check", "-"], cwd=target, input=patch, text=True, check=True)
        subprocess.run(["git", "apply", "-"], cwd=target, input=patch, text=True, check=True)
        after = subprocess.run([os.sys.executable, "-m", "pytest", "-q"], cwd=target, env=env, capture_output=True, text=True)
    write_json(output / "independent-validation.json", {"baseline_exit_code": before.returncode,
        "patched_exit_code": after.returncode, "baseline_output": before.stdout, "patched_output": after.stdout,
        "model_calls": 0, "official_harness": "not_run", "passed": before.returncode == 1 and after.returncode == 0})
    if before.returncode != 1 or after.returncode != 0:
        raise ValueError("Independent baseline/fixed checks failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--output", type=Path)
    options.add_argument("--verify-export", type=Path)
    args = parser.parse_args()
    if args.verify_export:
        verify_export(args.verify_export)
    else:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
