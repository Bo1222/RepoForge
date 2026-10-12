"""Real provider smoke through the installed CLI; exports evidence without credentials."""

import argparse
import json
import os
import signal
import time
from pathlib import Path
import subprocess
import sys


def command(args, cwd=None, timeout=300):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = root / "source"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n")
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    for operation in (
        ["init", "-q"],
        ["config", "user.email", "smoke@example.invalid"],
        ["config", "user.name", "Smoke"],
        ["add", "."],
        ["commit", "-qm", "baseline"],
    ):
        r = command(["git", "-C", str(repo), *operation])
        assert r.returncode == 0, r.stderr
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {"model": args.model, "max_iterations": 15, "max_cost": 0.1, "model_timeout": 90, "tool_timeout": 30}
        )
    )
    cli = [sys.executable, "-m", "moatless.cli", "--session-dir", str(root / "sessions")]
    task = "读取 calc.py 和 test_calc.py。先用 RunTests 执行 python -m pytest -q 确认失败；修复 add 函数，再用 RunTests 测试通过。不要提交。"
    r = command([*cli, "run", "--repo", str(repo), "--config", str(config), task, "--export", str(root / "first-turn")])
    (root / "first-command.txt").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, f"first turn failed; see {root / 'first-command.txt'}"
    info_path = next((root / "sessions").glob("*/session.json"))
    info = json.loads(info_path.read_text())
    followup = "继续在同一个工作区给 calc.py 增加 multiply(a,b) 返回乘积。在 test_calc.py 增加对应测试并用 RunTests 执行 python -m pytest -q。保留上轮 add 修复。"
    r = command([*cli, "resume", info["id"], "--headless", followup, "--export", str(root / "followup")])
    (root / "followup-command.txt").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, f"follow-up failed; see {root / 'followup-command.txt'}"
    worktree = Path(info["worktree"])
    check = command([sys.executable, "-m", "pytest", "-q"], cwd=worktree)
    (root / "independent-tests.txt").write_text(check.stdout + check.stderr)
    assert check.returncode == 0
    assert "return a - b" in (repo / "calc.py").read_text(), "source modified automatically"
    assert "multiply" in (worktree / "calc.py").read_text()
    tests = json.loads((root / "followup" / "tests.json").read_text())
    assert tests[0]["test_status"] == "failed" and tests[-1]["test_status"] == "passed"
    # Cancel a real model-selected Bash command, then restart the CLI and prove no replay.
    interrupt_task = (
        "只执行一次 Bash 命令：echo $$ > sleeping-pid; touch interrupted; sleep 120。不要编辑其他文件，不要运行测试。"
    )
    with (root / "interrupt-command.txt").open("w") as log:
        process = subprocess.Popen(
            [*cli, "run", "--repo", str(repo), "--config", str(config), interrupt_task], stdout=log, stderr=log
        )
        interrupted = None
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and process.poll() is None:
            for path in (root / "sessions").glob("*/session.json"):
                candidate = json.loads(path.read_text())
                if Path(candidate["worktree"], "interrupted").exists():
                    interrupted = candidate
                    break
            if interrupted:
                break
            time.sleep(0.1)
        if not interrupted:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
            raise AssertionError("Real interruption tool did not start")
        tool_pid = int(Path(interrupted["worktree"], "sleeping-pid").read_text())
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=15) == 130
    try:
        os.kill(tool_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("Cancelled tool process survived")
    resumed = command([*cli, "resume", interrupted["id"], "--headless"])
    (root / "interrupt-resume.txt").write_text(resumed.stdout + resumed.stderr)
    assert resumed.returncode == 1 and "RecoveryConflict" in resumed.stdout
    from moatless.cli.service import SessionService

    interrupted_service = SessionService(root / "sessions")
    interrupted_service.export(interrupted_service.load(interrupted["id"]), str(root / "interrupted"))
    summary = {
        "model": args.model,
        "session_id": info["id"],
        "first_turn": "passed",
        "followup": "passed",
        "independent_pytest": "passed",
        "source_unchanged": True,
        "tests": [t["test_status"] for t in tests],
        "usage": json.loads((root / "followup" / "usage.json").read_text()),
        "interrupt_session": interrupted["id"],
        "cancelled_process_terminated": True,
        "unknown_side_effect_resume": "recovery_conflict_no_replay",
        "official_harness": "not_run",
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
