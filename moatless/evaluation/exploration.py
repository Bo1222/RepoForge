"""Explicit small-sample reuse of an older public index; never a formal T4 gate."""

from pathlib import Path
import shutil

from moatless.evaluation.experiment import content_hash, read_json, sha, write_json
from moatless.index.settings import IndexSettings


def snapshot_policy(source):
    source = Path(source).resolve()
    receipt = read_json(source / "index-receipt.json")
    for name, expected in receipt["files"].items():
        if sha(source / name) != expected:
            raise ValueError("Source index changed")
    if receipt["digest"] != content_hash(receipt["files"]):
        raise ValueError("Source index digest changed")
    return {"mode": "reuse-public-snapshot", "new_indexes_allowed": False,
            "source_index": str(source), "source_instance": receipt["instance"],
            "receipt_sha256": sha(source / "index-receipt.json"),
            "index_digest": receipt["digest"],
            "settings": IndexSettings(**receipt["settings"]).model_dump(mode="json"),
            "limitations": "Older retrieval snapshot may miss new files/spans; target source and grading use each instance's actual base commit."}


def install_shared_index(output, manifest):
    policy = manifest["exploration"]["index_policy"]
    if snapshot_policy(policy["source_index"]) != policy:
        raise ValueError("Source index policy changed")
    for identity in manifest["exploration"]["instance_ids"]:
        target = Path(output) / "indexes" / identity
        shutil.copytree(policy["source_index"], target)


def verify_shared_index(output, manifest, job):
    selection = manifest["exploration"]
    policy = selection["index_policy"]
    if (policy["mode"] != "reuse-public-snapshot" or policy["new_indexes_allowed"] is not False
            or job["instance_id"] not in selection["instance_ids"] or job["phase"] != "smoke"
            or job["variant"] != "a0" or job["entrypoint"] != "batch"):
        raise ValueError("Invalid exploration job or index policy")
    target = Path(output) / "indexes" / job["instance_id"]
    receipt_path = target / "index-receipt.json"
    if not receipt_path.exists():
        raise ValueError("Existing index required; new index preparation is disabled")
    if sha(receipt_path) != policy["receipt_sha256"]:
        raise ValueError("Shared snapshot receipt changed")
    receipt = read_json(receipt_path)
    if receipt["instance"] != policy["source_instance"] or receipt["digest"] != policy["index_digest"]:
        raise ValueError("Shared snapshot provenance changed")
    if IndexSettings(**receipt["settings"]).model_dump(mode="json") != manifest["embedding_settings"]:
        raise ValueError("Shared snapshot embedding settings differ")
    public = read_json(Path(output) / "provision-native.json")[job["instance_id"]]["public"]
    if public["repo"] != policy["source_instance"]["repo"]:
        raise ValueError("Shared snapshot must use the same repository")
    for name, expected in receipt["files"].items():
        if sha(target / name) != expected:
            raise ValueError("Shared snapshot file changed")
    provenance = {"job_id": job["job_id"], "target_base_commit": public["base_commit"],
                  "source_base_commit": policy["source_instance"]["base_commit"],
                  "version_mismatch": public["base_commit"] != policy["source_instance"]["base_commit"],
                  "new_index_provider_calls": 0, "index_digest": policy["index_digest"],
                  "receipt_sha256": policy["receipt_sha256"], "limitations": policy["limitations"]}
    write_json(Path(output) / "runs" / job["job_id"] / "index-provenance.json", provenance)
    return provenance
