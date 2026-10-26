import json

import httpx
from openai import AsyncOpenAI, AuthenticationError, OpenAI
import pytest

from repoforge.evaluation.experiment_inputs import embedding_credentials, index_settings_for
from repoforge.index.embed_model import get_embed_model
from repoforge.index.jina import JinaEmbedding
from repoforge.index.settings import IndexSettings


def test_local_defaults_and_legacy_index_roundtrip(tmp_path):
    current = IndexSettings()
    assert (current.embed_provider, current.embed_model, current.embed_base_url, current.dimensions) == (
        "local", "BAAI/bge-m3", "http://127.0.0.1:8787/v1", 1024)
    current.persist(tmp_path)
    assert IndexSettings.from_persist_dir(tmp_path) == current
    # Old persisted files contained only model/dimensions; never relabel them Jina.
    (tmp_path / "settings.json").write_text(json.dumps({"embed_model": "text-embedding-3-small", "dimensions": 1536}))
    old = IndexSettings.from_persist_dir(tmp_path)
    assert old.embed_provider == "openai" and old.dimensions == 1536
    assert old.embed_base_url is None


def test_explicit_provider_and_model_selection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-openai")
    monkeypatch.setenv("VOYAGE_API_KEY", "offline-voyage")
    openai = IndexSettings(provider="openai", embed_model="text-embedding-3-small")
    assert openai.dimensions == 1536
    assert get_embed_model(openai.embed_model).model_name == "text-embedding-3-small"
    assert get_embed_model("voyage-code-3").model_name == "voyage-code-3"
    custom = IndexSettings(provider="jina", base_url="https://api.jina.ai/v1", dimensions=512)
    assert custom.embed_base_url == "https://api.jina.ai/v1" and custom.dimensions == 512
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embed_model("x", provider="unsupported")


def test_jina_credentials_never_fall_back_to_openai(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-other-provider")
    with pytest.raises(ValueError, match="JINA_API_KEY"):
        get_embed_model("jina-embeddings-v3")
    monkeypatch.setenv("JINA_API_KEY", "offline-jina")
    model = get_embed_model("jina-embeddings-v3")
    assert isinstance(model, JinaEmbedding)
    assert model.api_key.get_secret_value() == "offline-jina"
    assert "offline-jina" not in json.dumps(model.model_dump(mode="json"))
    assert "offline-jina" not in repr(model)


def handler(requests):
    def respond(request):
        body = json.loads(request.content)
        assert str(request.url) == "https://api.jina.ai/v1/embeddings"
        assert request.headers["authorization"] == "Bearer offline-jina"
        assert body["model"] == "jina-embeddings-v3" and body["dimensions"] == 4
        assert body["normalized"] is True and body["late_chunking"] is False
        requests.append(body)
        # Jina/OpenAI data order is not assumed to match input order.
        data = [{"object": "embedding", "index": i, "embedding": [float(i)] * 4}
                for i in reversed(range(len(body["input"])))]
        return httpx.Response(200, json={"object": "list", "model": body["model"], "data": data,
                                        "usage": {"prompt_tokens": 5, "total_tokens": 5}})
    return respond


def test_jina_sync_batch_and_query_use_distinct_tasks():
    requests = []
    model = JinaEmbedding(api_key="offline-jina", dimensions=4)
    with httpx.Client(transport=httpx.MockTransport(handler(requests))) as client:
        model._client = OpenAI(api_key="offline-jina", base_url=model.api_base, http_client=client, max_retries=0)
        assert model.get_text_embedding_batch(["first", "second"]) == [[0.0] * 4, [1.0] * 4]
        assert model.get_query_embedding("query") == [0.0] * 4
    assert [request["task"] for request in requests] == ["retrieval.passage", "retrieval.query"]


@pytest.mark.asyncio
async def test_jina_async_batch_and_query_use_distinct_tasks():
    requests = []
    model = JinaEmbedding(api_key="offline-jina", dimensions=4)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(requests))) as client:
        model._aclient = AsyncOpenAI(api_key="offline-jina", base_url=model.api_base, http_client=client, max_retries=0)
        assert await model.aget_text_embedding_batch(["first", "second"]) == [[0.0] * 4, [1.0] * 4]
        assert await model.aget_query_embedding("query") == [0.0] * 4
    assert [request["task"] for request in requests] == ["retrieval.passage", "retrieval.query"]


@pytest.mark.parametrize("data,reason", [
    ([{"index": 0, "embedding": [0.0]}], "dimension"),
    ([{"index": 1, "embedding": [0.0] * 4}], "indices"),
])
def test_invalid_response_rejected_before_faiss(data, reason):
    model = JinaEmbedding(api_key="offline-jina", dimensions=4)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "model": model.model_name, "data": data, "usage": {"total_tokens": 1}}))) as client:
        model._client = OpenAI(api_key="offline-jina", base_url=model.api_base, http_client=client, max_retries=0)
        with pytest.raises(ValueError, match=reason):
            model.get_query_embedding("query")


def test_http_failure_does_not_switch_providers():
    requests = []
    def fail(request):
        requests.append(request.url.host)
        return httpx.Response(401, json={"error": {"message": "offline invalid key"}})
    model = JinaEmbedding(api_key="offline-jina", max_retries=0)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        model._client = OpenAI(api_key="offline-jina", base_url=model.api_base, http_client=client, max_retries=0)
        with pytest.raises(AuthenticationError):
            model.get_query_embedding("query")
    assert requests == ["api.jina.ai"]


@pytest.mark.parametrize("dimension", [1024, 1536])
def test_faiss_dimension_survives_persistence(tmp_path, dimension):
    from repoforge.index.simple_faiss import SimpleFaissVectorStore
    store = SimpleFaissVectorStore.from_defaults(d=dimension)
    assert store.d == store.client.d == dimension
    store.persist(str(tmp_path))
    reloaded = SimpleFaissVectorStore.from_persist_dir(str(tmp_path))
    assert reloaded.d == reloaded.client.d == dimension


def test_jina_token_estimate_works_offline():
    from repoforge.utils.tokenizer import _hf_tokenizers, count_tokens
    assert count_tokens("def add(a, b): return a + b", "jina-embeddings-v3") > 0
    assert "jina-embeddings-v3" not in _hf_tokenizers


def test_experiment_credentials_follow_frozen_provider(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "offline-jina")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manifest = {"embedding_settings": IndexSettings(provider="jina").model_dump(mode="json")}
    assert embedding_credentials(manifest) == {"JINA_API_KEY": "offline-jina"}
    assert index_settings_for({}).embed_provider == "openai"
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        embedding_credentials({})


def test_jina_refuses_old_openai_dimensions():
    with pytest.raises(ValueError, match="at most 1024"):
        IndexSettings(provider="jina", dimensions=1536)


def test_jina_zero_dimensions_are_rejected(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "offline-jina")
    with pytest.raises(ValueError):
        get_embed_model("jina-embeddings-v3", dimensions=0)


def test_jina_total_tokens_recorded_without_inventing_a_price(tmp_path):
    from repoforge.evaluation.experiment_usage import SDKRecorder
    recorder = SDKRecorder(tmp_path, {"embedding": {}})
    identity, start = recorder.begin("embedding", {"model": "jina-embeddings-v3", "input": "synthetic"})
    recorder.end(identity, start, "embedding", "jina-embeddings-v3", response={
        "model": "jina-embeddings-v3", "usage": {"prompt_tokens": None, "total_tokens": 249}})
    assert recorder.totals()["prompt_tokens"] == 249
    assert recorder.totals()["reference_cost"] is None
    assert recorder.totals()["cost_complete"] is False
    with pytest.raises(Exception, match="Unknown"):
        recorder.begin("embedding", {"model": "jina-embeddings-v3"})
