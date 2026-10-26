"""Execute the actual batch worker with real tools and SDK fixtures, without Docker/API."""

import json
from pathlib import Path
import subprocess

import pytest

from repoforge.actions.verified_finish import VerifiedFinishArgs
from repoforge.actions.run_tests import RunTestsArgs
from repoforge.actions.string_replace import StringReplaceArgs
from repoforge.actions.view_code import CodeSpan, ViewCodeArgs
from repoforge.evaluation.experiment import prepare, read_json, verify_answer
from repoforge.evaluation.experiment_docker import bundle_for
from scripts.migration_smoke import SOURCE, TEST

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["loop_legacy", "graph_legacy", "a0"])
async def test_actual_batch_worker_reads_reproduces_edits_tests_and_freezes(tmp_path, monkeypatch, variant):
    from litellm.types.utils import ModelResponse
    import repoforge.evaluation.generation as worker

    repo = tmp_path / "testbed"
    repo.mkdir()
    (repo / "calc.py").write_text(SOURCE)
    (repo / "test_calc.py").write_text(TEST)
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    for command in (["git", "init", "-q"], ["git", "add", "."],
                    ["git", "-c", "user.name=WorkerTest", "-c", "user.email=worker@example.invalid", "commit", "-qm", "synthetic baseline"]):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest, jobs = prepare(ROOT, tmp_path / "experiment")
    job = next(job for job in jobs if job["phase"] == "smoke" and job["variant"] == variant)
    public = {"instance_id": job["instance_id"], "repo": "pytest-dev/pytest", "version": "6.0", "base_commit": base,
              "problem_statement": "Fix add(2, 3) returning -1"}
    bundle = bundle_for(manifest, job, public)
    # Override filesystem/container boundary only; actual Agent/Flow/runtime/tools/SDK ledger run.
    monkeypatch.setattr(worker, "Path", lambda value: repo if str(value) == "/testbed" else Path(value))
    from repoforge.repository.git import GitRepository
    def repository(**kwargs):
        if kwargs.get("repo_path") == "/testbed":
            kwargs["repo_path"] = str(repo)
        return GitRepository(**kwargs)
    monkeypatch.setattr("repoforge.repository.git.GitRepository", repository)
    monkeypatch.setattr(worker, "environment_identity", lambda image: {"image_id": image, "container_id": "test-container", "dependencies": "test-env"})
    monkeypatch.setattr("scripts.t4_cli_probe.model_metadata", lambda: {"id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash"})
    from repoforge.index.code_index import CodeIndex
    from unittest.mock import MagicMock
    monkeypatch.setattr("repoforge.index.code_index.CodeIndex.from_persist_dir", lambda *a, **k: MagicMock(spec=CodeIndex))
    monkeypatch.setenv("GENERATION_IMAGE_ID", "test-image")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "worker-offline-key")
    monkeypatch.setenv("SKIP_INSTALL", "true")
    monkeypatch.setenv("SKIP_CONDA_ACTIVATE", "true")
    actions = [ViewCodeArgs(files=[CodeSpan(file_path="calc.py", span_ids=["add"]),
                                   CodeSpan(file_path="test_calc.py", span_ids=["test_add"])]),
               RunTestsArgs(test_files=["test_calc.py"]),
               StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b"),
               RunTestsArgs(test_files=["test_calc.py"]),
               VerifiedFinishArgs(finish_reason="Actual tests pass", test_verification="Existing test_add fails before and passes after the arithmetic fix")]
    calls = []
    async def sdk(**kwargs):
        index = len(calls)
        assert index < len(actions), "Unexpected additional model call"
        calls.append(kwargs)
        action = actions[index]
        return ModelResponse(model="deepseek-flash", usage={"prompt_tokens": 100, "completion_tokens": 10},
            choices=[{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": f"call_{index}", "type": "function",
                    "function": {"name": action.name, "arguments": action.model_dump_json()}}]}}])
    monkeypatch.setattr("litellm.acompletion", sdk)
    output = tmp_path / "answer"
    await worker.run(bundle, output, tmp_path / "unused-index")
    answer = verify_answer(output, job)
    generation = read_json(output / "generation.json")
    assert generation["generation_status"] == "completed", generation
    assert generation["official_harness"] == "not_run"
    assert len(calls) == 5
    assert all(call["extra_body"]["thinking"]["type"] == "disabled" for call in calls)
    assert read_json(output / "usage.json")["sdk_calls"] == 5
    assert "return a + b" in (output / "patch.diff").read_text()
    assert (repo / "calc.py").read_text() == SOURCE  # shadow patch never contaminates baseline
    requests = read_json(output / "test-requests.json")
    assert len(requests) == 2
    nodes = read_json(output / "trajectory.json")["nodes"]
    tests = [step["observation"] for node in nodes for step in node.get("action_steps", [])
             if step["action"]["action_args_class"] == "repoforge.actions.run_tests.RunTestsArgs"]
    assert len(tests) == 2
    assert tests[0]["properties"]["return_code"] == 1
    assert tests[1]["properties"]["return_code"] == 0
    assert "worker-offline-key" not in json.dumps(read_json(output / "trajectory.json"))
    assert answer["selected_node_id"] == 5


def test_index_timing_preserves_original_algorithm_identity_and_failure(tmp_path, monkeypatch):
    from repoforge.evaluation import generation
    from repoforge.evaluation.experiment import read_json
    from repoforge.index.code_index import CodeIndex
    from repoforge.index.repository_store import builder_fingerprint
    import inspect

    before = builder_fingerprint(inspect.getsource(generation))
    original_persist = CodeIndex.persist
    def failing_persist(self, *args, **kwargs):
        raise RuntimeError("disk failure")
    monkeypatch.setattr(CodeIndex, "persist", failing_persist)
    def attempt(bundle, output):
        CodeIndex.persist(None, str(output))
    monkeypatch.setattr(generation, "build_index", attempt)
    with pytest.raises(RuntimeError, match="disk failure"):
        generation.timed_index_build({}, tmp_path)
    rows = read_json(tmp_path / "index-stage-timings.json")["stages"]
    assert rows[0]["stage"] == "index_persistence" and rows[0]["status"] == "failed"
    assert rows[0]["wall_seconds"] >= 0
    assert CodeIndex.persist is failing_persist
    monkeypatch.undo()
    assert CodeIndex.persist is original_persist
    assert builder_fingerprint(inspect.getsource(generation)) == before


def test_submit_product_only_is_opt_in_for_a5():
    from repoforge.evaluation.generation import submit_product_only
    assert submit_product_only({"action_policy": {"variant": "a5"}}) is True
    assert submit_product_only({"submit_product_only": True}) is True
    assert submit_product_only({"action_policy": {"variant": "a3"}}) is False
    assert submit_product_only({}) is False


def test_select_submitted_patch_keeps_current_version_even_after_historical_public_pass():
    from repoforge.evaluation.generation import select_submitted_patch
    from repoforge.message_history.policy import content_hash
    old, new = "old-diff", "new-diff"
    record = {"superseded": False, "tested_patch_hash": content_hash(old),
              "product_patch_hash": content_hash(old), "targets": ["tests/test_code.py"],
              "test_source": "base_public", "test_origins": {"tests/test_code.py": "original"},
              "effective_status": "stale", "introduced_failures": [], "status": "passed",
              "product_patch": old}
    patch, used = select_submitted_patch(new, [record])
    assert used is False and patch == new
    current = dict(record, tested_patch_hash=content_hash(new),
                   product_patch_hash=content_hash(new), effective_status="passed",
                   product_patch=new)
    patch, used = select_submitted_patch(new, [current])
    assert used is False and patch == new
