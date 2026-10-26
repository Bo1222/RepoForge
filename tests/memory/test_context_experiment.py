import pytest
import json

from scripts.context_memory_experiment import freeze_sources, profiles, split_instances


def test_experiment_split_is_disjoint_stratified_and_deterministic_without_answers():
    ids = [f"owner__large-{i}" for i in range(400)] + [f"owner__small-{i}" for i in range(100)]
    first = split_instances(ids)
    assert first == split_instances(list(reversed(ids)))
    assert len(first["dev"]) == 30 and len(first["holdout"]) == 100
    assert not set(first["dev"]).intersection(first["holdout"])
    assert first["counts"]["dev"] == {"owner/large": 24, "owner/small": 6}
    assert first["counts"]["holdout"] == {"owner/large": 80, "owner/small": 20}
    assert set(first["dev"] + first["holdout"]).issubset(ids)
    with pytest.raises(ValueError):
        split_instances(ids[:50])
    with pytest.raises(ValueError):
        split_instances(ids + [ids[0]])


def test_ablation_profiles_separate_notes_and_repository_rules():
    configs = profiles(65536)
    assert configs["frozen_baseline"]["context_policy"] is None
    assert configs["integrity_only"]["context_policy"] is None
    before = configs["managed_facts"]["context_policy"]
    after = configs["managed_task"]["context_policy"]
    assert {key for key in before if before[key] != after[key]} == {"enable_task_state"}
    rules = configs["managed_rules"]["context_policy"]
    assert {key for key in after if after[key] != rules[key]} == {"load_repo_instructions"}


def test_freeze_refuses_executed_job_before_mutating_artifacts(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"candidate_source_hashes": {}}))
    (tmp_path / "dev-jobs.jsonl").write_text(json.dumps({"generation_status": "completed"}) + "\n")
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="executed job"):
        freeze_sources(tmp_path, tmp_path)
    assert manifest.read_bytes() == original
    assert not (tmp_path / "baseline-source.tar").exists()


def test_freeze_refuses_changed_candidate_before_creating_snapshot(tmp_path):
    (tmp_path / "repoforge").mkdir()
    (tmp_path / "repoforge" / "changed.py").write_text("changed = True\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"candidate_source_hashes": {}}))
    (tmp_path / "dev-jobs.jsonl").write_text(json.dumps({"generation_status": "not_run"}) + "\n")
    with pytest.raises(ValueError, match="changed after preparation"):
        freeze_sources(tmp_path, tmp_path)
    assert not (tmp_path / "candidate-source.tar").exists()
