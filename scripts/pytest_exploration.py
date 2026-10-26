"""Prepare three public pytest tasks using one existing index, with no ingestion."""

import argparse
from pathlib import Path
import shutil

from datasets import Dataset
from dotenv import load_dotenv

from repoforge.evaluation.experiment import prepare, public_instance, read_json, write_json
from repoforge.evaluation.exploration import install_shared_index, snapshot_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    ids = ["pytest-dev__pytest-7490", "pytest-dev__pytest-7982", "pytest-dev__pytest-8399"]
    source = args.source_output.resolve()
    policy = snapshot_policy(source / "indexes" / ids[0])
    arrows = list((source / "host-dataset-cache").rglob("*.arrow"))
    if len(arrows) != 1:
        raise ValueError("Expected one already downloaded Verified dataset")
    dataset = Dataset.from_file(str(arrows[0]))
    public = {row["instance_id"]: public_instance(row) for row in dataset if row["instance_id"] in ids}
    if set(public) != set(ids) or any(row["repo"] != "pytest-dev/pytest" for row in public.values()):
        raise ValueError("Selection requires three public pytest tasks")
    exploration = {"purpose": "three-instance exploratory evaluation; not T4/T5 completion",
                   "instance_ids": ids, "index_policy": policy,
                   "selection_basis": "existing6.0 task plus later6.2/6.3 public tasks outside locked holdout; no grade-based selection",
                   "repeat": 1, "variant": "a0", "concurrency": 1}
    manifest, jobs = prepare(root, args.output, execution_backend="native",
                             environment_root=args.environment_root, exploration=exploration)
    if manifest["embedding_settings"] != policy["settings"]:
        raise ValueError("Configured embedding differs from the existing index")
    output = args.output.resolve()
    shutil.copytree(source / "host-dataset-cache", output / "host-dataset-cache")
    write_json(output / "dataset-receipt.json", read_json(source / "dataset-receipt.json"))
    write_json(output / "selection-public.json", public)
    install_shared_index(output, manifest)
    print({"jobs": len(jobs), "manifest_hash": manifest["manifest_hash"], "new_index_provider_calls": 0})


if __name__ == "__main__":
    main()
