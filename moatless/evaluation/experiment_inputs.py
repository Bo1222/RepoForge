"""Host-only public input export shared by execution backends."""

from pathlib import Path
import os
import json
import urllib.request

from moatless.evaluation.experiment import content_hash, public_instance, read_json, write_json


def index_settings_for(manifest):
    from moatless.index.settings import IndexSettings
    # Historical manifests predate provider selection and used OpenAI 1536d.
    return IndexSettings(**manifest.get("embedding_settings", {
        "embed_model": "text-embedding-3-small", "dimensions": 1536}))


def embedding_credentials(manifest):
    from moatless.index.embed_model import embedding_key_env
    name = embedding_key_env(index_settings_for(manifest).embed_provider)
    if name is None:
        return {}
    if not os.environ.get(name):
        raise ValueError(f"The frozen index embedding model requires {name}")
    return {name: os.environ[name]}


def validate_embedding_service(manifest):
    """Check the live local service before preparing or using a frozen index."""
    settings = index_settings_for(manifest)
    if settings.embed_provider != "local":
        return None
    execution = manifest["embedding_execution"]
    endpoint = settings.embed_base_url.removesuffix("/v1").rstrip("/") + "/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(endpoint, timeout=5) as response:
        health = json.load(response)
    expected = {"status": "ok", "model": settings.embed_model,
                "revision": execution["model_revision"], "dimensions": settings.dimensions,
                "max_input_tokens": execution["max_input_tokens"],
                "batch_size": execution["batch_size"], "precision": execution["precision"]}
    if any(health.get(key) != value for key, value in expected.items()):
        raise ValueError("Live local embedding service differs from the frozen identity")
    return health


def bundle_for(manifest, job, instance):
    bundle = {"job": job, "instance": public_instance(instance), "cli_config": manifest["cli_config"],
            "agent_config": manifest["agent_configs"][job["variant"]], "profile": manifest["profiles"][job["variant"]],
            "prices": manifest["prices"], "source_files": manifest["source"]["files"],
            "embedding_settings": index_settings_for(manifest).model_dump(mode="json"),
            "embedding_service_identity": embedding_service_identity(manifest)}
    if manifest.get("exploration"):
        policy = manifest["exploration"]["index_policy"]
        bundle["index_provenance"] = {"source_base_commit": policy["source_instance"]["base_commit"],
                                      "target_base_commit": instance["base_commit"],
                                      "limitations": policy["limitations"]}
    return bundle


def embedding_service_identity(manifest):
    service = manifest.get("embedding_service")
    return content_hash({key: service[key] for key in ("specification", "files", "dependencies")}) if service else None


def export_public_dataset(output, manifest, phase):
    """Private dataset is read only in the host preparation process, never mounted."""
    from datasets import load_dataset
    output = Path(output)
    if phase == "holdout":
        ids = set(manifest["split"]["holdout"])
    elif phase == "dev":
        ids = set(manifest["split"]["dev"])
    else:
        ids = set(manifest["smoke_ids"])
    dataset = load_dataset(manifest["dataset"], split="test", cache_dir=str(output / "host-dataset-cache"))
    selected = [row for row in dataset if row["instance_id"] in ids]
    receipt_path = output / "dataset-receipt.json"
    if receipt_path.exists() and read_json(receipt_path)["fingerprint"] != dataset._fingerprint:
        raise ValueError("Dataset fingerprint changed after preparation")
    inventory = {row["instance_id"]: public_instance(row) for row in selected}
    if set(inventory) != ids:
        raise ValueError("Dataset does not contain all frozen instance IDs")
    path = output / "public" / "instances.json"
    existing = read_json(path) if path.exists() else {}
    if any(identity in existing and existing[identity] != row for identity, row in inventory.items()):
        raise ValueError("Public dataset content changed after preparation")
    write_json(path, {**existing, **inventory})
    recipes_path = output / "host-install-recipes.json"
    recipes = read_json(recipes_path) if recipes_path.exists() else {}
    additions = {row["instance_id"]: {"environment_setup_commit": row["environment_setup_commit"]}
                 for row in selected if row.get("environment_setup_commit")}
    if any(identity in recipes and recipes[identity] != recipe for identity, recipe in additions.items()):
        raise ValueError("Installation recipe changed after preparation")
    recipes.update(additions)
    write_json(recipes_path, recipes)
    receipt = {"dataset": manifest["dataset"], "fingerprint": dataset._fingerprint,
               "public_instances_digest": content_hash({**existing, **inventory})}
    write_json(receipt_path, receipt)
    return inventory


