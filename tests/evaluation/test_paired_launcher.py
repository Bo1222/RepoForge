"""Exercise the real shell runner with a fake backend; no benchmark/model calls."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def launcher(tmp_path):
    repository = tmp_path / "repository with spaces"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "scripts/run_o2_a0_dev_holdout.sh"
    script = scripts / source.name
    shutil.copyfile(source, script)
    executable = repository / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text(f"#!{sys.executable}\n" + '''
import json,os,sys
from pathlib import Path
if sys.argv[1] == '-c':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
arguments=sys.argv[1:]
with open(os.environ['PAIRED_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(arguments)+'\\n')
if '--phase' in arguments:
    phase=arguments[arguments.index('--phase')+1]
    if phase == os.environ.get('PAIRED_TEST_FAIL_PHASE'):
        raise SystemExit(17)
    output=Path(arguments[arguments.index('--output')+1])
    (output/(phase+'-lock.json')).write_text('{}')
''')
    executable.chmod(0o755)
    output = tmp_path / "candidate with spaces"
    output.mkdir()
    (output / "manifest.json").write_text("{}")
    (output / "jobs.json").write_text("[]")
    log = tmp_path / "commands.jsonl"
    environment = {**os.environ, "PAIRED_TEST_LOG": str(log)}
    return script, output, log, environment


def test_launcher_runs_all_phases_once_and_records_completion(launcher):
    script, output, log, environment = launcher
    result = subprocess.run(["bash", str(script), str(output)], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 4
    assert calls[0][3] == "preflight"
    assert [call[call.index("--phase") + 1] for call in calls[1:]] == ["smoke", "dev", "holdout"]
    assert all("status" not in call and "report" not in call for call in calls)
    receipt = json.loads((output / "dev-holdout-exit.json").read_text())
    assert receipt["exit_code"] == 0
    assert receipt["phase_locks"] == {"smoke": True, "dev": True, "holdout": True}
    assert [json.loads(line) for line in (output / "dev-holdout-exits.jsonl").read_text().splitlines()] == [receipt]


def test_launcher_preserves_failure_and_does_not_start_holdout_or_retry(launcher):
    script, output, log, environment = launcher
    result = subprocess.run(["bash", str(script), str(output)],
                            env={**environment, "PAIRED_TEST_FAIL_PHASE": "dev"}, capture_output=True, text=True)
    assert result.returncode == 17
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[call.index("--phase") + 1] for call in calls[1:]] == ["smoke", "dev"]
    receipt = json.loads((output / "dev-holdout-exit.json").read_text())
    assert receipt["phase"] == "dev" and receipt["exit_code"] == 17
    assert not receipt["phase_locks"]["holdout"]
    assert json.loads((output / "dev-holdout-exits.jsonl").read_text())["exit_code"] == 17


def test_launcher_rejects_an_active_scheduler_without_backend_calls(launcher):
    script, output, log, environment = launcher
    with (output / "paired-scheduler.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(script), str(output)], env=environment, capture_output=True, text=True)
    assert result.returncode == 75
    assert "existing experiment scheduler" in result.stderr
    assert not log.exists()
    assert not (output / "dev-holdout-exit.json").exists()
