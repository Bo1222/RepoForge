"""Real HTTP boundary fixtures for pacing, minute retries and durable batch reuse."""
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI, RateLimitError

from repoforge.index.jina import JinaEmbedding, JinaOpenAI


def response(request, tokens=1000):
    texts = json.loads(request.content)['input']
    return httpx.Response(200, json={'object': 'list', 'model': 'jina-embeddings-v3',
        'data': [{'object': 'embedding', 'index': i, 'embedding': [float(len(text))] * 4}
                 for i, text in enumerate(texts)], 'usage': {'prompt_tokens': tokens, 'total_tokens': tokens}})


def test_successful_batch_survives_failure_and_new_client_without_rebilling(tmp_path):
    calls = []
    def initial(request):
        calls.append(json.loads(request.content)['input'])
        return response(request, 0) if len(calls) == 1 else httpx.Response(429, json={'code': 'RATE_TOKEN_LIMIT_EXCEEDED'})
    model = JinaEmbedding(api_key='offline', dimensions=4, cache_dir=str(tmp_path), max_retries=0)
    with httpx.Client(transport=httpx.MockTransport(initial)) as client:
        model._client = OpenAI(api_key='offline', base_url=model.api_base, http_client=client, max_retries=0)
        assert model._get_text_embeddings(['one', 'longer']) == [[3.] * 4, [6.] * 4]
        with pytest.raises(RateLimitError):
            model._get_text_embeddings(['uncached'])
    recovered = JinaEmbedding(api_key='offline', dimensions=4, cache_dir=str(tmp_path))
    def resumed(request):
        calls.append(json.loads(request.content)['input'])
        return response(request, 0)
    with httpx.Client(transport=httpx.MockTransport(resumed)) as client:
        recovered._client = OpenAI(api_key='offline', base_url=model.api_base, http_client=client, max_retries=0)
        assert recovered._get_text_embeddings(['longer', 'one', 'uncached']) == [[6.] * 4, [3.] * 4, [8.] * 4]
        assert calls[-1] == ['uncached']
        recovered._get_query_embedding('one')
        assert calls[-1] == ['one']  # Passage cache cannot replace query embeddings.
    assert len(calls) == 4


def test_actual_usage_paces_next_request(monkeypatch):
    clock = [10.0]
    sleeps = []
    monkeypatch.setattr('repoforge.index.jina.time.monotonic', lambda: clock[0])
    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay
    monkeypatch.setattr('repoforge.index.jina.time.sleep', sleep)
    model = JinaEmbedding(api_key='offline', dimensions=4, tokens_per_minute=80000)
    model._client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.] * 4)],
                        usage=SimpleNamespace(prompt_tokens=8000, total_tokens=8000))))
    model._get_text_embedding('first')
    model._get_text_embedding('second')
    assert sleeps == [0, 6]


def test_only_token_minute_limit_gets_long_sdk_retry():
    client = JinaOpenAI(api_key='offline')
    limited = httpx.Response(429, json={'code': 'RATE_TOKEN_LIMIT_EXCEEDED'})
    assert client._should_retry(limited)
    assert limited.headers['retry-after'] == '60'
    unknown = httpx.Response(429, json={'code': 'OTHER_LIMIT'})
    client._should_retry(unknown)
    assert 'retry-after' not in unknown.headers
    client.close()
