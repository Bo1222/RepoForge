import os
from typing import Optional

from llama_index.core.embeddings import BaseEmbedding


def embedding_provider(model_name: str) -> str:
    if model_name == "BAAI/bge-m3":
        return "local"
    if model_name.startswith("jina-"):
        return "jina"
    if model_name.startswith("voyage"):
        return "voyage"
    return "openai"


def embedding_key_env(provider: str) -> Optional[str]:
    return {"local": None, "jina": "JINA_API_KEY", "voyage": "VOYAGE_API_KEY", "openai": "OPENAI_API_KEY"}[provider]


def get_embed_model(model_name: str, *, provider: Optional[str] = None,
                    base_url: Optional[str] = None, dimensions: Optional[int] = None) -> BaseEmbedding:
    provider = provider or embedding_provider(model_name)
    if provider == "local":
        from moatless.index.local import LocalEmbedding

        return LocalEmbedding(model_name=model_name, api_base=base_url or "http://127.0.0.1:8787/v1",
                              dimensions=dimensions if dimensions is not None else 1024)
    if provider == "jina":
        from moatless.index.jina import JinaEmbedding

        key = os.environ.get("JINA_API_KEY")
        if not key:
            raise ValueError("JINA_API_KEY environment variable is not set")
        return JinaEmbedding(model_name=model_name, api_key=key,
                             api_base=base_url or "https://api.jina.ai/v1",
                             cache_dir=os.environ.get("JINA_EMBEDDING_CACHE_DIR"),
                             dimensions=dimensions if dimensions is not None else 1024)
    if provider == "voyage":
        try:
            from llama_index.embeddings.voyageai import VoyageEmbedding
        except ImportError as e:
            raise ImportError(
                "llama-index-embeddings-voyageai is not installed. Please install it using `pip install llama-index-embeddings-voyageai`"
            ) from e

        if "VOYAGE_API_KEY" not in os.environ:
            raise ValueError("VOYAGE_API_KEY environment variable is not set. Please set it to your Voyage API key.")

        return VoyageEmbedding(
            model_name=model_name,
            voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
            truncation=True,
            embed_batch_size=60,
        )
    elif provider == "openai":
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as e:
            raise ImportError(
                "llama-index-embeddings-openai is not installed. Please install it using `pip install llama-index-embeddings-openai`"
            ) from e

        kwargs = {}
        if base_url:
            kwargs["api_base"] = base_url
        if dimensions is not None and model_name.startswith("text-embedding-3-"):
            kwargs["dimensions"] = dimensions
        return OpenAIEmbedding(model_name=model_name, **kwargs)
    raise ValueError(f"Unsupported embedding provider: {provider}")
