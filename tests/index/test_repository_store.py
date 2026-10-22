from copy import deepcopy
from pathlib import Path

import pytest

from moatless.evaluation.experiment import content_hash, sha, write_json
from moatless.index.repository_store import (
    CORE_FILES, RepositoryIndexStore, builder_fingerprint, index_profile, retire_index_data,
)


def producer(tmp_path):
    from moatless.index.repository_store import PIPELINE_FILES
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "producer"
    source.mkdir()
    settings = {"embed_provider": "local", "embed_model": "BAAI/bge-m3", "dimensions": 1024}
    manifest = {"source": {"files": {name: sha(root / name) for name in PIPELINE_FILES}},
                "embedding_settings": settings,
                "embedding_service": {"specification": {"revision": "fixed"}, "files": {}, "dependencies": {}},
                "native": {"index_store": {"builder_sha256": "fixed-builder"}}}
    profile = index_profile(manifest)
    public = {"repo": "example/repository", "base_commit": "a" * 40,
              "instance_id": "one", "problem_statement": "public issue"}
    for name in CORE_FILES:
        write_json(source / name, {})
    write_json(source / "settings.json", settings)
    write_json(source / "vector_index.json", {"vector_id_to_text_id": {"0": "node"}})
    (source / "sdk.jsonl").write_text("original public request ledger\n")
    files = {name: sha(source / name) for name in (*CORE_FILES, "sdk.jsonl")}
    write_json(source / "index-receipt.json", {"instance": public, "settings": settings,
               "embedding_service_identity": profile["embedding_service_identity"],
               "files": files, "digest": content_hash(files)})
    return source, public, manifest, profile


def test_same_snapshot_reuses_across_experiments_and_issue_ids(tmp_path):
    source, public, manifest, profile = producer(tmp_path)
    store = RepositoryIndexStore(tmp_path / "shared")
    index = store.publish(source, public, profile, provenance={"run": "r2"})
    another = {**public, "instance_id": "different", "problem_statement": "another issue"}
    assert store.find(another, profile) == index
    assert store.publish(source, another, profile, provenance={"run": "r4"}) == index
    assert len(store.refresh_catalog()) == 1
    assert len(list((index.parent / "provenance").glob("*.json"))) == 2
    assert "sdk.jsonl" not in {p.name for p in index.iterdir()}  # Only current public index mounted.


def test_native_driver_resolves_shared_index_in_another_experiment(tmp_path):
    from moatless.evaluation.experiment_native import shared_index, verify_index_receipt
    source, public, manifest, profile = producer(tmp_path)
    store = RepositoryIndexStore(tmp_path / "shared")
    index = store.publish(source, public, profile, provenance={"run": "old"})
    manifest["native"]["index_store"]["root"] = str(store.root)
    prepared = {"public": {**public, "instance_id": "a-new-job"}, "environment": {}}
    _, actual_profile, actual = shared_index(tmp_path / "new-run", manifest, prepared)
    assert actual == index and actual_profile == profile
    assert verify_index_receipt(actual, manifest, prepared["public"])["repository_snapshot"]["base_commit"] == public["base_commit"]
    with pytest.raises(ValueError, match="source"):
        verify_index_receipt(actual, manifest, {**public, "base_commit": "b" * 40})


@pytest.mark.parametrize("change", ["commit", "settings", "revision", "pipeline", "setup"])
def test_other_snapshot_or_embedding_pipeline_is_never_reused(tmp_path, change):
    source, public, manifest, profile = producer(tmp_path)
    store = RepositoryIndexStore(tmp_path / "shared")
    store.publish(source, public, profile, provenance={})
    changed_public, changed_profile = deepcopy(public), deepcopy(profile)
    if change == "commit":
        changed_public["base_commit"] = "b" * 40
    elif change == "settings":
        changed_profile["settings"]["dimensions"] = 1536
    elif change == "revision":
        changed_profile["embedding_service_identity"] = "changed"
    elif change == "pipeline":
        changed_profile["builder_sha256"] = "changed"
    else:
        changed_profile["setup_patch_sha256"] = "changed"
    assert store.find(changed_public, changed_profile) is None


def test_corrupt_or_incomplete_existing_index_cannot_trigger_silent_rebuild(tmp_path):
    source, public, _, profile = producer(tmp_path)
    store = RepositoryIndexStore(tmp_path / "shared")
    index = store.publish(source, public, profile, provenance={})
    (index / "docstore.json").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        store.find(public, profile)
    (index / "index-receipt.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        store.find(public, profile)


def test_retirement_removes_only_verified_data_and_keeps_cost_evidence(tmp_path):
    source, public, _, profile = producer(tmp_path)
    store = RepositoryIndexStore(tmp_path / "shared")
    index = store.publish(source, public, profile, provenance={})
    record = retire_index_data(source, reason="deduplicated", replacement=index)
    assert record["status"] == "completed"
    assert all(not (source / name).exists() for name in CORE_FILES)
    assert (source / "sdk.jsonl").read_text() == "original public request ledger\n"
    assert (source / "index-receipt.json").exists()
    assert store.find(public, profile) == index


def test_corrupt_or_escaped_producer_is_not_imported_or_deleted(tmp_path):
    source, public, _, profile = producer(tmp_path)
    store = RepositoryIndexStore(tmp_path / "shared")
    target = source / "docstore.json"
    target.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="escaped"):
        store.publish(source, public, profile, provenance={})
    with pytest.raises(ValueError, match="escaped"):
        retire_index_data(source, reason="must fail")
    assert (source / "vector_index.faiss").exists()
    assert outside.read_text() == "{}"


def test_builder_fingerprint_ignores_unrelated_grader_changes():
    code = "def build_index(x):\n return x\n"
    assert builder_fingerprint(code) == builder_fingerprint(code + "\ndef grader():\n return 2\n")
    assert builder_fingerprint(code) != builder_fingerprint(code.replace("return x", "return 3"))


@pytest.mark.parametrize("repo,commit", [("../bad", "a" * 40), ("good/repo", "short"), ("good/../bad", "b" * 40)])
def test_repository_paths_require_full_commit_and_cannot_escape(tmp_path, repo, commit):
    store = RepositoryIndexStore(tmp_path)
    with pytest.raises(ValueError):
        store.directory({"repo": repo, "base_commit": commit}, {})
