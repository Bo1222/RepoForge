"""Jina retrieval embeddings over its OpenAI-compatible HTTP endpoint."""

import asyncio
import time
from typing import Any, Optional

from llama_index.core.embeddings import BaseEmbedding
from openai import AsyncOpenAI, OpenAI
from pydantic import Field, PrivateAttr, SecretStr

from moatless.index.embedding_cache import EmbeddingCache, cache_key


class MinuteLimitRetry:
    def _should_retry(self, response):
        if response.status_code == 429:
            response.read()
            try:
                token_limit = response.json().get('code') == 'RATE_TOKEN_LIMIT_EXCEEDED'
            except ValueError:
                token_limit = False
            if token_limit and not response.headers.get('retry-after'):
                # SDK retries remain inside one recorded logical request. The
                # default subsecond delay cannot clear Jina's minute window.
                response.headers['retry-after'] = '60'
        return super()._should_retry(response)


class JinaOpenAI(MinuteLimitRetry, OpenAI):
    pass


class JinaAsyncOpenAI(MinuteLimitRetry, AsyncOpenAI):
    pass


class JinaEmbedding(BaseEmbedding):
    model_name: str = "jina-embeddings-v3"
    api_base: str = "https://api.jina.ai/v1"
    api_key: SecretStr = Field(exclude=True, repr=False)
    dimensions: int = Field(default=1024, gt=0, le=1024)
    timeout: float = Field(default=60, gt=0)
    max_retries: int = Field(default=4, ge=0)
    tokens_per_minute: int = Field(default=80000, gt=0)
    cache_dir: Optional[str] = Field(default=None, exclude=True)
    _next_request_at: float = PrivateAttr(default=0)
    _async_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _client: Optional[OpenAI] = PrivateAttr(default=None)
    _aclient: Optional[AsyncOpenAI] = PrivateAttr(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "JinaEmbedding"

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = JinaOpenAI(api_key=self.api_key.get_secret_value(), base_url=self.api_base,
                                  timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:
        if self._aclient is None:
            self._aclient = JinaAsyncOpenAI(api_key=self.api_key.get_secret_value(), base_url=self.api_base,
                                       timeout=self.timeout, max_retries=self.max_retries)
        return self._aclient

    def _request(self, texts: list[str], task: str) -> dict[str, Any]:
        return {"model": self.model_name, "input": texts, "dimensions": self.dimensions,
                "encoding_format": "float",
                "extra_body": {"task": task, "normalized": True, "late_chunking": False}}

    def _vectors(self, response, count: int) -> list[list[float]]:
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(count)):
            raise ValueError("Jina returned missing or duplicate embedding indices")
        vectors = [item.embedding for item in data]
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError("Jina returned a different embedding dimension than the configured index")
        return vectors

    def _cached(self, texts, task):
        cache = EmbeddingCache(self.cache_dir) if self.cache_dir and task == 'retrieval.passage' else None
        settings = [self.model_name, self.api_base, self.dimensions, task, True, False]
        keys = [cache_key(settings, text) for text in texts]
        found = cache.read(keys) if cache else {}
        if any(len(vector) != self.dimensions for vector in found.values()):
            raise ValueError('Embedding cache dimension changed')
        missing = list(dict.fromkeys(key for key in keys if key not in found))
        return cache, keys, found, missing, [texts[keys.index(key)] for key in missing]

    def _pace_next(self, response):
        usage = getattr(response, 'usage', None)
        tokens = (getattr(usage, 'prompt_tokens', None) or getattr(usage, 'total_tokens', None) or 0)
        self._next_request_at = time.monotonic() + tokens * 60 / self.tokens_per_minute

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        if not texts:
            return []
        cache, keys, found, missing, inputs = self._cached(texts, task)
        if missing:
            time.sleep(max(0, self._next_request_at - time.monotonic()))
            response = self._get_client().embeddings.create(**self._request(inputs, task))
            self._pace_next(response)
            rows = dict(zip(missing, self._vectors(response, len(inputs))))
            if cache:
                cache.write(rows)
            found.update(rows)
        return [found[key] for key in keys]

    async def _aembed(self, texts: list[str], task: str) -> list[list[float]]:
        if not texts:
            return []
        async with self._async_lock:
            cache, keys, found, missing, inputs = self._cached(texts, task)
            if missing:
                await asyncio.sleep(max(0, self._next_request_at - time.monotonic()))
                response = await self._get_aclient().embeddings.create(**self._request(inputs, task))
                self._pace_next(response)
                rows = dict(zip(missing, self._vectors(response, len(inputs))))
                if cache:
                    cache.write(rows)
                found.update(rows)
            return [found[key] for key in keys]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed([query], "retrieval.query")[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return (await self._aembed([query], "retrieval.query"))[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text], "retrieval.passage")[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._aembed([text], "retrieval.passage"))[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "retrieval.passage")

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._aembed(texts, "retrieval.passage")
