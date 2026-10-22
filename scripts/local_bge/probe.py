"""Small real-service acceptance probe; FAISS artifacts only live in /tmp."""

import asyncio
import json
import math
from pathlib import Path
import tempfile
import time

import httpx
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery

from moatless.evaluation.experiment_usage import SDKRecorder
from moatless.index.embed_model import get_embed_model
from moatless.index.settings import IndexSettings
from moatless.index.simple_faiss import SimpleFaissVectorStore

CODES = [
    'def verify_password(password, stored_hash):\n    """Verify a login password against its hash."""\n    return hash_password(password) == stored_hash\n',
    'def add_numbers(a, b):\n    """Add two numeric values."""\n    return a + b\n',
    'def read_json_file(path):\n    """Read a JSON configuration file from disk."""\n    with open(path) as stream:\n        return json.load(stream)\n',
]


def check(vectors):
    for vector in vectors:
        assert len(vector) == 1024 and all(math.isfinite(v) for v in vector)
        assert math.isclose(sum(v * v for v in vector), 1, abs_tol=1e-4)


def main(output):
    import importlib.util

    heavy_dependencies = {
        name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "sentence_transformers")
    }
    assert not any(heavy_dependencies.values()), heavy_dependencies
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = IndexSettings(provider="local", embed_model="BAAI/bge-m3", base_url="http://127.0.0.1:8787/v1")
    model = get_embed_model(
        settings.embed_model,
        provider=settings.embed_provider,
        base_url=settings.embed_base_url,
        dimensions=settings.dimensions,
    )
    times = {}
    with SDKRecorder(output, {"embedding": {"BAAI/bge-m3": 0}}).capture() as recorder:
        started = time.monotonic()
        single = model.get_text_embedding(CODES[0])
        times["single"] = time.monotonic() - started
        started = time.monotonic()
        batch = model.get_text_embedding_batch(CODES)
        times["batch"] = time.monotonic() - started
        started = time.monotonic()
        query = model.get_query_embedding("Verify the password of a user logging in")
        times["query"] = time.monotonic() - started
        check([single, *batch, query])
        assert sum(a * b for a, b in zip(single, batch[0])) > 0.999
        # More than one service inference batch; repeated inputs identify order.
        many = model.get_text_embedding_batch([CODES[i % 3] for i in range(10)])
        check(many)
        assert all(sum(a * b for a, b in zip(v, batch[i % 3])) > 0.999 for i, v in enumerate(many))
        async_vector = asyncio.run(model.aget_query_embedding("Read JSON configuration from disk"))
        check([async_vector])
        with tempfile.TemporaryDirectory(prefix="local-bge-faiss-") as temporary:
            store = SimpleFaissVectorStore.from_defaults(d=1024)
            store.add(
                [
                    TextNode(id_=str(i), text=text, embedding=vector)
                    for i, (text, vector) in enumerate(zip(CODES, batch))
                ]
            )
            store.persist(temporary)
            settings.persist(temporary)
            loaded = SimpleFaissVectorStore.from_persist_dir(temporary)
            assert IndexSettings.from_persist_dir(temporary) == settings
            hit = loaded.query(VectorStoreQuery(query_embedding=query, similarity_top_k=3))
            assert hit.ids[0] == "0", hit.ids
        assert not Path(temporary).exists()
        usage = recorder.totals()
    with httpx.Client(trust_env=False, timeout=120) as client:
        raw = client.post(
            settings.embed_base_url + "/embeddings", json={"model": settings.embed_model, "input": CODES[0]}
        )
        raw.raise_for_status()
        check([raw.json()["data"][0]["embedding"]])
        # Verified pinned tokenizer: stripped repeated "test" + two special tokens.
        exact = client.post(
            settings.embed_base_url + "/embeddings",
            json={"model": settings.embed_model, "input": [("test " * 2046).strip()] * 8},
        )
        exact.raise_for_status()
        check([row["embedding"] for row in exact.json()["data"]])
        assert exact.json()["usage"]["prompt_tokens"] == 8 * 2048
        too_long = client.post(
            settings.embed_base_url + "/embeddings",
            json={"model": settings.embed_model, "input": [("test " * 2047).strip()]},
        )
        assert too_long.status_code == 400 and "no truncation" in too_long.text
        assert too_long.json()["detail"]["token_lengths"] == [2049]
        health = client.get("http://127.0.0.1:8787/health").json()
    result = {
        "status": "passed",
        "settings": settings.model_dump(mode="json"),
        "times_seconds": times,
        "retrieval_ids": hit.ids,
        "usage": usage,
        "health": health,
        "single_raw_usage": raw.json()["usage"],
        "boundary_usage": exact.json()["usage"],
        "overlength": too_long.json(),
        "temporary_faiss_deleted": True,
        "llm_calls": 0,
        "swebench": "not_run",
        "worker_heavy_dependencies": heavy_dependencies,
        "nvidia_device_visible": Path("/dev/nvidia0").exists(),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": "passed", "output": str(output), "times": times}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
