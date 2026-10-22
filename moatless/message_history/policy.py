"""Versioned, opt-in request budgeting with an explicitly labelled estimator."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

from moatless.exceptions import CompletionRuntimeError
from moatless.utils.tokenizer import count_tokens


class ContextBudgetExceeded(CompletionRuntimeError):
    error_kind = "context_budget_exceeded"


class ContextPolicy(BaseModel):
    version: Literal[1] = 1
    context_window: int = Field(gt=0)
    max_input_tokens: int = Field(default=32768, gt=0)
    max_observation_tokens: int = Field(default=4000, ge=128)
    recent_interactions: int = Field(default=2, ge=0)
    load_repo_instructions: bool = False
    enable_task_state: bool = True
    retry_prompt_tokens: int = Field(default=1024, ge=256)

    def input_budget(self, output_tokens: int | None) -> int:
        if output_tokens is None or output_tokens < 0:
            raise ValueError("A finite model output token limit is required for context budgeting")
        safety = max(self.retry_prompt_tokens, (self.context_window + 19) // 20)
        available = min(self.max_input_tokens, self.context_window - output_tokens - safety)
        if available <= 0:
            raise ContextBudgetExceeded("context_budget_exceeded: no room after output and safety reserves")
        return available


def canonical_json(value) -> str:
    def encode(item):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        return str(item)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=encode)


def content_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class TokenEstimator:
    def __init__(self, model: str):
        import tiktoken

        self.model = model.split("/", 1)[-1]
        try:
            tiktoken.encoding_name_for_model(self.model)
            self.label = f"tiktoken:{self.model}; serialized-request estimate"
        except KeyError:
            self.model = "gpt-3.5-turbo"
            self.label = "tiktoken:cl100k_base fallback; serialized-request estimate"

    def count(self, value) -> int:
        text = value if isinstance(value, str) else canonical_json(value)
        return count_tokens(text, model=self.model)

    def shorten(self, text: str, limit: int, marker: str) -> str:
        if self.count(text) <= limit:
            return text
        if self.count(marker) > limit:
            raise ContextBudgetExceeded("context_budget_exceeded: truncation reference cannot fit")
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            head = middle // 2
            candidate = text[:head] + marker + (text[-(middle - head):] if middle - head else "")
            if self.count(candidate) <= limit:
                low = middle
            else:
                high = middle - 1
        head = low // 2
        return text[:head] + marker + (text[-(low - head):] if low - head else "")


def request_envelope(messages: list[dict], model) -> dict:
    # Authentication and headers are never part of exported request context.
    params = model._completion_params or {}
    envelope = {"model": model.model, "messages": messages}
    for key in ("tools", "response_format"):
        if key in params:
            envelope[key] = params[key]
    return envelope


def output_limit(model) -> int | None:
    params = model._completion_params or {}
    limits = [params.get("max_tokens", model.max_tokens), params.get("max_completion_tokens")]
    finite = [int(x) for x in limits if x is not None]
    return max(finite) if finite else None
