import json
import math

import httpx
from openai import APIConnectionError, AsyncOpenAI, OpenAI
import pytest

from moatless.index.embed_model import embedding_key_env, get_embed_model
from moatless.index.local import LocalEmbedding
from moatless.index.settings import IndexSettings
from moatless.evaluation.experiment_inputs import embedding_credentials, index_settings_for


def test_local_settings_credentials_and_roundtrip(monkeypatch, tmp_path):
    for key in ("OPENAI_API_KEY", "JINA_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    settings = IndexSettings(provider="local", embed_model="BAAI/bge-m3", base_url="http://127.0.0.1:8787/v1")
    assert settings.dimensions == 1024
    settings.persist(tmp_path)
    assert IndexSettings.from_persist_dir(tmp_path) == settings
    manifest = {"embedding_settings": settings.model_dump(mode="json")}
    assert index_settings_for(manifest) == settings
    assert embedding_credentials(manifest) == {}
    assert embedding_key_env("local") is None
    assert isinstance(get_embed_model(settings.embed_model), LocalEmbedding)
    assert IndexSettings(embed_model="jina-embeddings-v3").embed_provider == "jina"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimensions": 1536},
        {"embed_model": "other"},
        {"base_url": "https://api.openai.com/v1"},
        {"base_url": "http://localhost:8787/v1?redirect=x"},
    ],
)
def test_local_bad_settings_rejected(kwargs):
    params = {"provider": "local", "embed_model": "BAAI/bge-m3"}
    params.update(kwargs)
    with pytest.raises(ValueError):
        IndexSettings(**params)


def response(request, requests):
    body = json.loads(request.content)
    requests.append(body)
    assert request.url.host == "127.0.0.1"
    assert set(body) == {"model", "input", "dimensions", "encoding_format"}
    data = []
    for i in reversed(range(len(body["input"]))):
        v = [0.0] * 1024
        v[i] = 1.0
        data.append({"index": i, "embedding": v, "object": "embedding"})
    return httpx.Response(
        200,
        json={"model": "BAAI/bge-m3", "object": "list", "data": data, "usage": {"prompt_tokens": 8, "total_tokens": 8}},
    )


def test_sync_order_query_and_no_task():
    requests = []
    model = LocalEmbedding()
    with httpx.Client(transport=httpx.MockTransport(lambda r: response(r, requests))) as client:
        model._client = OpenAI(api_key="local-no-key", base_url=model.api_base, http_client=client, max_retries=0)
        vectors = model.get_text_embedding_batch(["first", "second"])
        assert vectors[0][0] == vectors[1][1] == 1.0
        assert model.get_query_embedding("unchanged query")[0] == 1.0
    assert requests[-1]["input"] == ["unchanged query"]


@pytest.mark.asyncio
async def test_async_order_query_and_no_task():
    requests = []
    model = LocalEmbedding()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: response(r, requests))) as client:
        model._aclient = AsyncOpenAI(api_key="local-no-key", base_url=model.api_base, http_client=client, max_retries=0)
        vectors = await model.aget_text_embedding_batch(["first", "second"])
        assert vectors[0][0] == vectors[1][1] == 1.0
        assert (await model.aget_query_embedding("unchanged query"))[0] == 1.0


@pytest.mark.parametrize("reason", ["dimension", "indices", "non-finite", "unnormalized", "model"])
def test_invalid_local_response_rejected(reason):
    model = LocalEmbedding()
    v = [0.0] * 1024
    v[0] = 1.0
    row = {"index": 0, "embedding": v}
    result = {"model": model.model_name, "data": [row]}
    if reason == "dimension":
        row["embedding"] = [1.0]
    if reason == "indices":
        row["index"] = 1
    if reason == "non-finite":
        v[1] = math.inf
    if reason == "unnormalized":
        v[0] = 0.0
    if reason == "model":
        result["model"] = "other"
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=result))) as client:
        model._client = OpenAI(api_key="local-no-key", base_url=model.api_base, http_client=client, max_retries=0)
        # JSON cannot encode infinity; send it via OpenAI's parsed response instead.
        if reason == "non-finite":
            from openai.types import CreateEmbeddingResponse

            parsed = CreateEmbeddingResponse(
                model=model.model_name,
                object="list",
                data=[{"object": "embedding", **row}],
                usage={"prompt_tokens": 1, "total_tokens": 1},
            )
            with pytest.raises(ValueError, match=reason):
                model._vectors(parsed, 1)
        else:
            with pytest.raises(ValueError, match=reason):
                model.get_query_embedding("q")


def test_service_failure_never_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-be-used")
    hosts = []

    def fail(request):
        hosts.append(request.url.host)
        raise httpx.ConnectError("offline service", request=request)

    model = LocalEmbedding()
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        model._client = OpenAI(api_key="local-no-key", base_url=model.api_base, http_client=client, max_retries=0)
        with pytest.raises(APIConnectionError):
            model.get_query_embedding("q")
    assert hosts == ["127.0.0.1"]


def test_bge_estimate_never_loads_hf(monkeypatch):
    from moatless.utils.tokenizer import count_tokens, _hf_tokenizers
    import sys
    from types import SimpleNamespace

    # Framework intentionally has no transformers dependency.
    fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: pytest.fail("HF download")))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert count_tokens("def add(a,b): return a+b", "BAAI/bge-m3") > 0
    assert "BAAI/bge-m3" not in _hf_tokenizers


@pytest.mark.parametrize("failed", [False, True])
def test_local_ledger_api_cost_zero_hardware_unpriced(tmp_path, failed):
    from moatless.evaluation.experiment_usage import SDKRecorder

    recorder = SDKRecorder(tmp_path, {"embedding": {}})
    identity, start = recorder.begin("embedding", {"model": "BAAI/bge-m3"})
    recorder.end(
        identity,
        start,
        "embedding",
        "BAAI/bge-m3",
        response=None if failed else {"usage": {"prompt_tokens": 9}},
        error=RuntimeError("offline") if failed else None,
    )
    assert recorder.totals()["reference_cost"] == 0
    assert recorder.rows[-1]["hardware_cost"] == "unpriced"
    recorder.begin("embedding", {"model": "BAAI/bge-m3"})


def test_native_executor_local_credentials_keep_only_deepseek(monkeypatch, tmp_path):
    from moatless.evaluation import experiment_native as native

    paths = {}
    for name in ("agent", "framework", "base", "prefix", "repo", "inputs", "output"):
        paths[name] = tmp_path / name
        paths[name].mkdir()
    for name in ("bin", "lib"):
        (paths["base"] / name).mkdir()
    monkeypatch.setattr(native.NativeProvisioner, "verify", lambda environment: None)
    monkeypatch.setattr(native, "source_directory", lambda output, manifest: paths["agent"])
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "JINA_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.setenv(name, "offline-fixture")
    manifest = {
        "source": {"sha256": "offline-fixture-source"},
        "native": {"framework_prefix": str(paths["framework"]), "framework_base_prefix": str(paths["base"])},
        "embedding_settings": IndexSettings(provider="local", embed_model="BAAI/bge-m3").model_dump(mode="json"),
    }
    prepared = {"public": {"repo": "pytest-dev/pytest"},
                "environment": {"prefix": str(paths["prefix"]), "repo": str(paths["repo"])}}
    runner = native.executor_for(
        paths["output"], manifest, prepared, paths["repo"], paths["inputs"], paths["output"], credentials=True
    )
    assert runner.environment()["DEEPSEEK_API_KEY"] == "offline-fixture"
    assert not any(name in runner.environment() for name in ("OPENAI_API_KEY", "JINA_API_KEY", "VOYAGE_API_KEY"))
    assert runner.network is True
