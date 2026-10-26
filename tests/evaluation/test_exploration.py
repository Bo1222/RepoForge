from pathlib import Path
import json

import pytest

from repoforge.evaluation.experiment import content_hash, load_manifest, prepare, sha, validate_gate, write_json
from repoforge.evaluation.exploration import install_shared_index, snapshot_policy, verify_shared_index
from repoforge.index.settings import IndexSettings

ROOT = Path(__file__).resolve().parents[2]
IDS = ["pytest-dev__pytest-7490", "pytest-dev__pytest-7982", "pytest-dev__pytest-8399"]


def experiment(tmp_path):
    source = tmp_path / "source-index"
    source.mkdir()
    (source / "vector_index.faiss").write_bytes(b"offline fixture")
    files = {"vector_index.faiss": sha(source / "vector_index.faiss")}
    public = {"instance_id": IDS[0], "repo": "pytest-dev/pytest", "base_commit": "old",
              "version": "6.0", "problem_statement": "public fixture"}
    write_json(source / "index-receipt.json", {"instance": public, "settings": IndexSettings().model_dump(mode="json"),
                                             "files": files, "digest": content_hash(files)})
    selection = {"instance_ids": IDS, "index_policy": snapshot_policy(source)}
    output = tmp_path / "experiment"
    manifest, jobs = prepare(ROOT, output, execution_backend="native", exploration=selection)
    install_shared_index(output, manifest)
    write_json(output / "provision-native.json", {job["instance_id"]: {"public": {**public,
               "instance_id": job["instance_id"], "base_commit": "actual-target"}} for job in jobs})
    return output, manifest, jobs


def test_exploration_selects_three_only_and_cannot_authorize_dev(tmp_path):
    output, manifest, jobs = experiment(tmp_path)
    assert len(jobs) == 3
    assert {job["variant"] for job in jobs} == {"a0"}
    assert {job["phase"] for job in jobs} == {"smoke"}
    assert load_manifest(output)["manifest_hash"] == manifest["manifest_hash"]
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_gate(output, manifest)


def test_missing_existing_index_refuses_before_workspace_or_worker(tmp_path, monkeypatch):
    from repoforge.evaluation import experiment_native
    output, manifest, jobs = experiment(tmp_path)
    (output / "indexes" / IDS[0] / "index-receipt.json").unlink()
    monkeypatch.setattr(experiment_native, "run_worker", lambda *a, **kw: pytest.fail("New ingestion must not run"))
    with pytest.raises(ValueError, match="new index preparation is disabled"):
        experiment_native.run_job(output, manifest, jobs[0])
    assert not (output / "workspaces").exists()


def test_shared_snapshot_retains_old_commit_and_detects_mutation(tmp_path):
    output, manifest, jobs = experiment(tmp_path)
    proof = verify_shared_index(output, manifest, jobs[1])
    assert proof["source_base_commit"] == "old"
    assert proof["target_base_commit"] == "actual-target" and proof["version_mismatch"]
    (output / "indexes" / IDS[1] / "vector_index.faiss").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        verify_shared_index(output, manifest, jobs[1])


def test_shared_snapshot_rejects_other_repository_and_prepare_index(tmp_path):
    from repoforge.evaluation import experiment_native
    output, manifest, jobs = experiment(tmp_path)
    write_json(output / "provision-native.json", {IDS[0]: {"public": {"repo": "django/django"}}})
    with pytest.raises(ValueError, match="same repository"):
        verify_shared_index(output, manifest, jobs[0])
    with pytest.raises(ValueError, match="disabled"):
        experiment_native.run_job(output, manifest, jobs[0], prepare_index_only=True)


def test_exploration_keeps_holdout_locked(tmp_path):
    historical = json.loads((ROOT / "repoforge/evaluation/profiles/frozen_split.json").read_text())
    with pytest.raises(ValueError, match="holdout"):
        prepare(ROOT, tmp_path / "rejected", execution_backend="native",
                exploration={"instance_ids": [historical["split"]["holdout"][0]]})
