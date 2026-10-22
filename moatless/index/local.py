"""Dense BGE embeddings from an explicitly local OpenAI-compatible service."""

import math
from typing import Optional
from urllib.parse import urlsplit

from llama_index.core.embeddings import BaseEmbedding
from openai import AsyncOpenAI, OpenAI
from pydantic import Field, PrivateAttr, model_validator


MODEL = "BAAI/bge-m3"
BASE_URL = "http://127.0.0.1:8787/v1"


def validate_local_config(model, base_url, dimensions):
    if model != MODEL or dimensions != 1024:
        raise ValueError("Local BGE-M3 requires model BAAI/bge-m3 and dimensions=1024; rebuild incompatible indexes")
    url = urlsplit(base_url or "")
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost", "::1")
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/") != "/v1"
    ):
        raise ValueError("Local embedding base_url must be a loopback HTTP /v1 endpoint")


class LocalEmbedding(BaseEmbedding):
    model_name: str = MODEL
    api_base: str = BASE_URL
    dimensions: int = 1024
    embed_batch_size: int = 8
    timeout: float = Field(default=120, gt=0)
    _client: Optional[OpenAI] = PrivateAttr(default=None)
    _aclient: Optional[AsyncOpenAI] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_config(self):
        validate_local_config(self.model_name, self.api_base, self.dimensions)
        return self

    @classmethod
    def class_name(cls):
        return "LocalEmbedding"

    def _get_client(self):
        if self._client is None:
            import httpx

            # The SDK requires a placeholder string. No cloud credential is read.
            # Ignore proxy environment variables so loopback stays on this host.
            self._client = OpenAI(
                api_key="local-no-key",
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=0,
                http_client=httpx.Client(trust_env=False, timeout=self.timeout),
            )
        return self._client

    def _get_aclient(self):
        if self._aclient is None:
            import httpx

            self._aclient = AsyncOpenAI(
                api_key="local-no-key",
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=0,
                http_client=httpx.AsyncClient(trust_env=False, timeout=self.timeout),
            )
        return self._aclient

    def _request(self, texts):
        return {"model": self.model_name, "input": texts, "dimensions": self.dimensions, "encoding_format": "float"}

    def _vectors(self, response, count):
        if response.model != self.model_name:
            raise ValueError("Local service returned an unexpected model")
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(count)):
            raise ValueError("Local service returned missing or duplicate embedding indices")
        vectors = [item.embedding for item in data]
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError("Local service returned a different embedding dimension")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Local service returned non-finite embedding values")
            if not math.isclose(sum(value * value for value in vector), 1, abs_tol=0.01):
                raise ValueError("Local service returned an unnormalized embedding")
        return vectors

    def _embed(self, texts):
        if not texts:
            return []
        return self._vectors(self._get_client().embeddings.create(**self._request(texts)), len(texts))

    async def _aembed(self, texts):
        if not texts:
            return []
        return self._vectors(await self._get_aclient().embeddings.create(**self._request(texts)), len(texts))

    def _get_query_embedding(self, query):
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query):
        return (await self._aembed([query]))[0]

    def _get_text_embedding(self, text):
        return self._embed([text])[0]

    async def _aget_text_embedding(self, text):
        return (await self._aembed([text]))[0]

    def _get_text_embeddings(self, texts):
        return self._embed(texts)

    async def _aget_text_embeddings(self, texts):
        return await self._aembed(texts)
