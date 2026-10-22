import json
import os
from enum import Enum
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, model_validator

from moatless.index.embed_model import embedding_provider


class CommentStrategy(Enum):
    # Keep comments
    INCLUDE = "include"

    # Always associate comments before a code block with the code block
    ASSOCIATE = "associate"

    # Exclude comments in parsed chunks
    EXCLUDE = "exclude"


class IndexSettings(BaseModel):
    embed_model: str = Field(default="BAAI/bge-m3", description="The embedding model to use.")
    embed_provider: Optional[Literal["local", "jina", "openai", "voyage"]] = Field(
        default=None, validation_alias=AliasChoices("embed_provider", "provider")
    )
    embed_base_url: Optional[str] = Field(default=None, validation_alias=AliasChoices("embed_base_url", "base_url"))
    dimensions: int = Field(default=1024, gt=0, description="The number of dimensions of the vectors.")

    @model_validator(mode="before")
    @classmethod
    def embedding_defaults(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        explicit_provider = value.get("embed_provider") or value.get("provider")
        default_models = {
            "local": "BAAI/bge-m3",
            "jina": "jina-embeddings-v3",
            "openai": "text-embedding-3-small",
            "voyage": "voyage-code-3",
        }
        model = value.get("embed_model", default_models.get(explicit_provider, "BAAI/bge-m3"))
        if not isinstance(model, str):
            return value  # Let Pydantic report the invalid model field.
        value["embed_model"] = model
        provider = value.get("embed_provider") or value.get("provider") or embedding_provider(model)
        value["embed_provider"] = provider
        if not value.get("embed_base_url") and not value.get("base_url") and provider == "jina":
            value["embed_base_url"] = "https://api.jina.ai/v1"
        if provider == "local":
            value["embed_base_url"] = value.get("embed_base_url") or value.get("base_url") or "http://127.0.0.1:8787/v1"
            value.setdefault("preserve_chunk_content", True)
        if "dimensions" not in value:
            value["dimensions"] = (
                1024 if provider in ("jina", "local") else (3072 if model == "text-embedding-3-large" else 1536)
            )
        return value

    @model_validator(mode="after")
    def validate_jina_dimensions(self):
        if self.embed_provider == "local":
            from moatless.index.local import validate_local_config

            validate_local_config(self.embed_model, self.embed_base_url, self.dimensions)
        if self.embed_provider == "jina" and self.dimensions > 1024:
            raise ValueError("Jina embeddings v3 dimensions must be at most 1024; rebuild the index")
        return self

    language: str = Field(default="python", description="The language of the code.")
    min_chunk_size: int = Field(default=100, description="The minimum chunk size.")
    chunk_size: int = Field(default=750, description="The soft max chunk size.")
    hard_token_limit: int = Field(default=2000, description="The hard token limit.")
    max_chunks: int = Field(default=200, description="The maximum number of chunks for one file.")
    preserve_chunk_content: bool = Field(default=False, description="Split oversized content without truncation or generated-file omission.")
    comment_strategy: CommentStrategy = Field(
        default=CommentStrategy.ASSOCIATE,
        description="Strategy on how comments will be indexed.",
    )

    def to_serializable_dict(self):
        data = self.dict()
        data["comment_strategy"] = data["comment_strategy"].value
        return data

    def persist(self, persist_dir: str):
        with open(os.path.join(persist_dir, "settings.json"), "w") as f:
            json.dump(self.to_serializable_dict(), f, indent=4)

    @classmethod
    def from_persist_dir(cls, persist_dir: str):
        with open(os.path.join(persist_dir, "settings.json")) as f:
            data = json.load(f)
        return cls(**data)
