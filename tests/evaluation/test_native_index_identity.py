from copy import deepcopy

import pytest

from repoforge.evaluation.experiment import content_hash, sha, write_json
from repoforge.evaluation.experiment_inputs import embedding_service_identity
from repoforge.evaluation.experiment_native import verify_index_receipt
from repoforge.index.settings import IndexSettings


def fixture_index(tmp_path):
    manifest = {"embedding_settings": IndexSettings().model_dump(mode="json"),
                "embedding_service": {"specification": {"revision": "fixed"},
                                      "files": {"model": "hash"}, "dependencies": {"torch": "version"}}}
    public = {"base_commit": "full-base"}
    (tmp_path / "settings.json").write_text("public index fixture")
    files = {"settings.json": sha(tmp_path / "settings.json")}
    receipt = {"instance": public, "settings": manifest["embedding_settings"], "files": files,
               "digest": content_hash(files), "embedding_service_identity": embedding_service_identity(manifest)}
    write_json(tmp_path / "index-receipt.json", receipt)
    return manifest, public, receipt


@pytest.mark.parametrize("change", ["base", "settings", "service", "content", "digest"])
def test_index_from_other_source_settings_or_service_cannot_start_generation(tmp_path, change):
    manifest, public, receipt = fixture_index(tmp_path)
    assert verify_index_receipt(tmp_path, manifest, public) == receipt
    if change == "base":
        public = {"base_commit": "different-base"}
    elif change == "settings":
        manifest = deepcopy(manifest)
        manifest["embedding_settings"]["hard_token_limit"] = 750
    elif change == "service":
        manifest["embedding_service"]["specification"]["revision"] = "changed"
    elif change == "content":
        (tmp_path / "settings.json").write_text("changed vectors/settings")
    else:
        receipt["digest"] = "changed"
        write_json(tmp_path / "index-receipt.json", receipt)
    with pytest.raises(ValueError):
        verify_index_receipt(tmp_path, manifest, public)


def test_receipt_cannot_bind_a_file_outside_its_index(tmp_path):
    index = tmp_path / "index"
    index.mkdir()
    manifest, public, receipt = fixture_index(index)
    outside = tmp_path / "outside"
    outside.write_text("fixture")
    receipt["files"] = {"../outside": sha(outside)}
    receipt["digest"] = content_hash(receipt["files"])
    write_json(index / "index-receipt.json", receipt)
    with pytest.raises(ValueError, match="escaped"):
        verify_index_receipt(index, manifest, public)
