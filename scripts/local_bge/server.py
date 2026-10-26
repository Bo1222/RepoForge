"""One-process, serial CUDA BGE-M3 dense service. No cloud fallback."""

import asyncio
from contextlib import asynccontextmanager
import json
import hashlib
import os
from pathlib import Path
import time
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, StrictStr

MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MAX_TOKENS = 2048
BATCH_SIZE = 8


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["BAAI/bge-m3"]
    input: StrictStr | list[StrictStr]
    dimensions: Literal[1024] = 1024
    encoding_format: Literal["float"] = "float"


def check_inputs(tokenizer, texts):
    if not texts or len(texts) > 64:
        raise HTTPException(400, "input must contain 1 to 64 strings")
    if any(len(text) > 100_000 for text in texts):
        raise HTTPException(413, "input character limit exceeded (100000 per item)")
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]]
    too_long = [i for i, length in enumerate(lengths) if length > MAX_TOKENS]
    if too_long:
        raise HTTPException(
            400,
            {"message": "input exceeds 2048 BGE tokens; no truncation", "indices": too_long, "token_lengths": lengths},
        )
    return lengths


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; local BGE service requires GPU, no CPU fallback")
    path = Path(os.environ.get("BGE_MODEL_DIR", "experiments/runtime/local_bge/model")).resolve()
    manifest = json.loads((path.parent / "model-manifest.json").read_text())
    if manifest["revision"] != REVISION or manifest["model"] != MODEL:
        raise RuntimeError("Unexpected model identity")
    for name, expected in manifest["files"].items():
        file = path / name
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if file.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise RuntimeError(f"Model file hash mismatch: {name}")
    started = time.monotonic()
    model = SentenceTransformer(
        str(path),
        device="cuda:0",
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.float16},
    )
    if model.get_sentence_embedding_dimension() != 1024 or model.max_seq_length < MAX_TOKENS:
        raise RuntimeError("Unexpected original model configuration")
    model.eval()
    torch.cuda.synchronize()
    return model, time.monotonic() - started


def create_app(loader=load_model):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model, app.state.load_seconds = loader()
        app.state.lock = asyncio.Lock()
        app.state.inference_lock = threading.Lock()
        app.state.request_count = 0
        app.state.request_seconds = []
        print(
            json.dumps({"event": "model_ready", "load_seconds": app.state.load_seconds, "revision": REVISION}),
            flush=True,
        )
        yield
        app.state.model = None

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        import resource
        import torch

        return {
            "status": "ok",
            "model": MODEL,
            "revision": REVISION,
            "dimensions": 1024,
            "max_input_tokens": MAX_TOKENS,
            "batch_size": BATCH_SIZE,
            "precision": "float16",
            "load_seconds": app.state.load_seconds,
            "request_count": app.state.request_count,
            "request_seconds": app.state.request_seconds[-100:],
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "external_embedding_api_cost_usd": 0,
            "hardware_cost": "unpriced",
        }

    def infer_serial(texts):
        import torch

        lengths = check_inputs(app.state.model.tokenizer, texts)
        started = time.monotonic()
        with torch.inference_mode():
            vectors = app.state.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=False,
            )
            # Normalize in float32 so FP16 rounding does not affect stored norms.
            vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1)
        if (
            vectors.shape != (len(texts), 1024)
            or not torch.isfinite(vectors).all()
            or not torch.allclose(
                torch.linalg.vector_norm(vectors, dim=1), torch.ones(len(texts), device=vectors.device), atol=1e-5
            )
        ):
            raise RuntimeError("Invalid BGE embeddings")
        result = vectors.cpu().tolist()
        app.state.request_count += 1
        app.state.request_seconds.append(time.monotonic() - started)
        app.state.request_seconds = app.state.request_seconds[-100:]
        return {
            "object": "list",
            "model": MODEL,
            "data": [{"object": "embedding", "index": i, "embedding": vector} for i, vector in enumerate(result)],
            "usage": {"prompt_tokens": sum(lengths), "total_tokens": sum(lengths)},
        }

    def infer(texts):
        # Also serialize threads if a cancelled HTTP coroutine releases its
        # asyncio lock before its ongoing GPU operation has finished.
        with app.state.inference_lock:
            return infer_serial(texts)

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingRequest):
        texts = [request.input] if isinstance(request.input, str) else request.input
        async with app.state.lock:
            return await asyncio.to_thread(infer, texts)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8787, workers=1)
