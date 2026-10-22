"""Experiment-only SDK evidence and a shared budget for action/auxiliary calls."""

from contextlib import contextmanager
from pathlib import Path
import threading
import time
from uuid import uuid4
from datetime import datetime

from moatless.completion.stats import Usage
from moatless.evaluation.experiment import append_json, now, write_json
from moatless.exceptions import CompletionRuntimeError
from moatless.message_history.policy import content_hash


class SDKRecorder:
    def __init__(self, directory, prices, max_cost=None):
        self.directory = Path(directory)
        self.prices = prices
        self.max_cost = max_cost
        self.rows = []
        self.lock = threading.Lock()
        path = self.directory / "sdk.jsonl"
        if path.exists():
            import json
            self.rows = [json.loads(line) for line in path.read_text().splitlines()]
            started = {row["call_id"] for row in self.rows if row["status"] == "started"}
            ended = {row["call_id"] for row in self.rows if row["status"] != "started"}
            if started - ended:
                raise CompletionRuntimeError("Unconfirmed SDK call; reconcile result/cost before recovery")

    def add(self, row):
        with self.lock:
            self.rows.append(row)
            append_json(self.directory / "sdk.jsonl", row)

    def totals(self):
        ended = [row for row in self.rows if row["status"] != "started"]
        pending = {row["call_id"] for row in self.rows if row["status"] == "started"} - {row["call_id"] for row in ended}
        known = not pending and all(row.get("reference_cost") is not None for row in ended)
        provider_estimate_complete = known and all(row.get("provider_estimated_cost") is not None for row in ended)
        return {"sdk_calls": sum(row["status"] == "started" for row in self.rows),
                "chat_calls": sum(row["status"] == "started" and row["kind"] == "chat" for row in self.rows),
                "embedding_calls": sum(row["status"] == "started" and row["kind"] == "embedding" for row in self.rows),
                "reference_cost": sum(row["reference_cost"] for row in ended) if known else None,
                "known_reference_cost": sum(row.get("reference_cost") or 0 for row in ended),
                "provider_estimated_cost": sum(row["provider_estimated_cost"] for row in ended) if provider_estimate_complete else None,
                "provider_estimated_cost_range": [sum(row["provider_cost_range"][i] for row in ended)
                                                   for i in (0, 1)] if provider_estimate_complete else None,
                "provider_estimate_basis": "frozen rates and SDK start UTC; Chinese public holiday status unverified; not a bill",
                "cost_complete": known, "provider_bill": "not_available",
                "prompt_tokens": sum(row.get("usage", {}).get("prompt_tokens", 0) for row in ended),
                "completion_tokens": sum(row.get("usage", {}).get("completion_tokens", 0) for row in ended),
                "cache_read_tokens": sum(row.get("usage", {}).get("cache_read_tokens", 0) for row in ended),
                "sdk_seconds": sum(row.get("seconds", 0) for row in ended),
                "http_attempts": "not_observed; SDK-internal transport retries may differ"}

    def begin(self, kind, kwargs):
        from moatless.context_data import current_node_id, current_action_step
        totals = self.totals()
        if not totals["cost_complete"]:
            raise CompletionRuntimeError("Unknown SDK usage/cost; budget cannot be checked")
        if self.max_cost is not None and totals["reference_cost"] >= self.max_cost:
            raise CompletionRuntimeError("Experiment SDK cost budget exhausted")
        if kind == "chat" and (kwargs.get("model") != "openai/deepseek-flash" or
                               kwargs.get("extra_body", {}).get("thinking", {}).get("type") != "disabled"):
            raise CompletionRuntimeError("Experiment requires Flash with explicit thinking disabled on every call")
        selected = {key: kwargs[key] for key in ("model", "messages", "tools", "response_format",
                    "max_tokens", "temperature", "extra_body", "input") if key in kwargs}
        identity = uuid4().hex
        self.add({"call_id": identity, "kind": kind, "status": "started", "timestamp": now(),
                  "node_id": current_node_id.get(), "action_step": current_action_step.get(),
                  "request_hash": content_hash(selected), "model": kwargs.get("model"),
                  "max_tokens": kwargs.get("max_tokens"), "temperature": kwargs.get("temperature"),
                  "thinking": kwargs.get("extra_body", {}).get("thinking"),
                  "input_characters": len(str(kwargs.get("input", "")))})
        return identity, time.monotonic()

    def end(self, identity, start, kind, model, response=None, error=None):
        usage, cost, response_model = {}, None, None
        local_embedding = kind == "embedding" and model == "BAAI/bge-m3"
        if response is not None:
            raw = response.model_dump() if hasattr(response, "model_dump") else response
            response_model = raw.get("model")
            if raw.get("usage"):
                if kind == "chat":
                    parsed = Usage.from_completion_response(raw, model)
                    usage = parsed.model_dump()
                    cost = parsed.completion_cost
                else:
                    tokens = raw["usage"].get("prompt_tokens")
                    if tokens is None:
                        tokens = raw["usage"].get("total_tokens")
                    rate = self.prices["embedding"].get(model)
                    if tokens is not None:
                        usage = {"prompt_tokens": tokens}
                        if rate is not None:
                            cost = tokens * rate / 1_000_000
        if local_embedding:
            # Even failed loopback requests incur no external embedding API fee.
            # This says nothing about hardware/energy costs.
            cost = 0
        estimate, price_range = None, None
        if cost is not None:
            started = next(row for row in self.rows if row["call_id"] == identity and row["status"] == "started")
            utc = datetime.fromisoformat(started["timestamp"])
            peak = utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)
            estimate = cost if kind == "embedding" or peak else cost / 2
            price_range = [cost / 2, cost] if kind == "chat" and peak else [estimate, estimate]
        self.add({"call_id": identity, "kind": kind, "status": "failed" if error else "completed",
                  "timestamp": now(), "model": model, "seconds": time.monotonic() - start,
                  "usage": usage, "reference_cost": cost, "response_model": response_model,
                  "provider_estimated_cost": estimate, "provider_cost_range": price_range,
                  **({"external_embedding_api_cost_usd": 0, "hardware_cost": "unpriced"} if local_embedding else {}),
                  "error_type": type(error).__name__ if error else None})

    @contextmanager
    def capture(self):
        import litellm
        from openai.resources.embeddings import Embeddings, AsyncEmbeddings
        original_chat, original_embedding, original_async_embedding = litellm.acompletion, Embeddings.create, AsyncEmbeddings.create

        async def chat(**kwargs):
            identity, start = self.begin("chat", kwargs)
            try:
                result = await original_chat(**kwargs)
            except BaseException as exc:
                self.end(identity, start, "chat", kwargs["model"], error=exc)
                raise
            self.end(identity, start, "chat", kwargs["model"], response=result)
            message = result.choices[0].message
            if getattr(message, "reasoning_content", None):
                raise CompletionRuntimeError("Provider returned native reasoning despite disabled thinking")
            return result

        def embedding(resource, **kwargs):
            identity, start = self.begin("embedding", kwargs)
            try:
                result = original_embedding(resource, **kwargs)
            except BaseException as exc:
                self.end(identity, start, "embedding", kwargs["model"], error=exc)
                raise
            self.end(identity, start, "embedding", kwargs["model"], response=result)
            return result

        async def async_embedding(resource, **kwargs):
            identity, start = self.begin("embedding", kwargs)
            try:
                result = await original_async_embedding(resource, **kwargs)
            except BaseException as exc:
                self.end(identity, start, "embedding", kwargs["model"], error=exc)
                raise
            self.end(identity, start, "embedding", kwargs["model"], response=result)
            return result

        litellm.acompletion, Embeddings.create, AsyncEmbeddings.create = chat, embedding, async_embedding
        try:
            yield self
        finally:
            litellm.acompletion, Embeddings.create, AsyncEmbeddings.create = original_chat, original_embedding, original_async_embedding
            write_json(self.directory / "usage.json", self.totals())
