"""Meaningful checks for demo grading and real-call audit preservation."""

import argparse
import json
from pathlib import Path
import sys

import pytest

from moatless.cli.service import SessionService
from moatless.exceptions import CompletionRuntimeError
from scripts.swebench_demo import append_json, classify, run_agent
from swebench.harness.log_parsers.python import parse_log_pytest


@pytest.mark.parametrize(
    "after, expected", [("PASSED", True), ("FAILED", False), ("SKIPPED", False), ("NOT_RUN", False)]
)
def test_independent_grading_requires_every_required_case(after, expected):
    instance = {"FAIL_TO_PASS": '["test_regression"]', "PASS_TO_PASS": ["test_existing"]}
    baseline = parse_log_pytest("FAILED test_regression\nPASSED test_existing", None)
    repaired = parse_log_pytest(f"{after} test_regression\nPASSED test_existing", None)
    result = classify(instance, baseline, repaired)
    assert result["local_validation_passed"] is expected
    assert result["official_harness"] == "not_run"
    assert len(result["cases"]) == 2
    invalid = classify(instance, {"test_regression": "PASSED", "test_existing": "PASSED"}, repaired)
    assert not invalid["baseline_valid"] and not invalid["local_validation_passed"]
    regression = classify(instance, baseline, {"test_regression": "PASSED", "test_existing": "FAILED"})
    assert not regression["local_validation_passed"]


def test_audit_redacts_credentials_and_output(tmp_path):
    path = tmp_path / "audit.jsonl"
    append_json(path, {"api_key": "secret-demo", "output": "failed with secret-demo"}, "secret-demo")
    assert "secret-demo" not in path.read_text()
    assert json.loads(path.read_text())["api_key"] is None


@pytest.mark.asyncio
async def test_action_rationale_is_opt_in(tmp_path):
    service = SessionService(tmp_path / "sessions")
    normal = await service.create(None, "ordinary")
    audited = await service.create(None, "audited", {"thoughts_in_action": True})
    assert not normal.flow.agent.completion_model.thoughts_in_action
    assert audited.flow.agent.completion_model.thoughts_in_action
    tool = audited.flow.agent.completion_model._get_completion_params([audited.flow.agent.actions[0].args_schema])
    assert "thoughts" in tool["tools"][0]["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_sdk_failure_preserves_export_and_audit(repo, tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    secret = "offline-demo-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    calls = []

    async def sdk(**kwargs):
        calls.append(kwargs)
        raise CompletionRuntimeError("provider failure " + secret)

    monkeypatch.setattr("litellm.acompletion", sdk)
    args = argparse.Namespace(model="deepseek/deepseek-chat", max_iterations=5, max_cost=2)
    instance = {"problem_statement": "Fix the regression", "test_patch": ""}
    session, _, result = await run_agent(repo, Path(sys.executable), instance, output, runtime, args)
    assert len(calls) == 1
    assert result["status"] == "failed"
    assert "provider failure" in result["error"] and secret not in result["error"]
    assert (output / "agent-export/trajectory.json").exists()
    assert (output / "phases.jsonl").exists()
    audit = [json.loads(line) for line in (output / "model-calls.jsonl").read_text().splitlines()]
    assert [entry["event"] for entry in audit] == ["request", "error"]
    assert all(secret not in p.read_text() for p in output.rglob("*.json*"))
    assert session.info["status"] == "failed"


@pytest.mark.asyncio
async def test_test_pipeline_keeps_failure_exit_status(repo):
    from moatless.actions.local_coding import LocalRunTests, LocalTestArgs
    from moatless.cli.environment import SessionEnvironment
    from moatless.repository.file import FileRepository
    from moatless.workspace import Workspace

    workspace = Workspace(
        repository=FileRepository(repo_path=str(repo), shadow_mode=False), environment=SessionEnvironment(str(repo))
    )
    action = LocalRunTests()
    action._workspace = workspace
    observation = await action.execute(LocalTestArgs(command="printf 'test failure\\n'; bash -c 'exit 7' | tail -n 2"))
    assert observation.properties["return_code"] == 7
    assert observation.properties["test_status"] == "failed"
    assert "test failure" in observation.message
    await workspace.environment.close()
