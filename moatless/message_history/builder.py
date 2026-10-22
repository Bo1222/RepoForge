"""Shared request-level assembly, separate from history formats and stored facts."""

from pydantic import BaseModel, Field
from typing import Literal

from moatless.message_history.assembly import ContextUnit
from moatless.message_history.policy import ContextBudgetExceeded, ContextPolicy, TokenEstimator, output_limit, request_envelope


class ContextBuildRecord(BaseModel):
    version: Literal[1] = 1
    policy: ContextPolicy
    model: str
    estimator: str
    input_budget: int
    estimated_input_tokens: int
    included_sources: list[str]
    omitted_sources: list[str]
    shortened_sources: list[str]
    prepared_requests: list[dict] = Field(default_factory=list)
    workspace_version: str | None = None
    stale_sources: list[str] = Field(default_factory=list)
    deduplicated_sources: list[str] = Field(default_factory=list)
    instruction_sources: list[dict] = Field(default_factory=list)


class ContextBuilder:
    def __init__(self, model, policy: ContextPolicy):
        self.model = model
        self.policy = policy
        self.estimator = TokenEstimator(model.model)
        self.budget = policy.input_budget(output_limit(model))

    def request_tokens(self, messages: list[dict]) -> int:
        prepared = self.model._prepare_messages(messages, self.model._system_prompt)
        return self.estimator.count(request_envelope(prepared, self.model))

    def select(self, units: list[ContextUnit], shortened: list[str]) -> set[int]:
        def cost(indices):
            return self.request_tokens([m for i, unit in enumerate(units) if i in indices for m in unit.messages])

        selected = {i for i, unit in enumerate(units) if unit.required}
        if cost(selected) > self.budget:
            raise ContextBudgetExceeded("context_budget_exceeded: required task context and request envelope cannot fit")
        interactions = [i for i, unit in enumerate(units) if unit.messages[0].get("tool_calls")]
        recent = interactions[-self.policy.recent_interactions:] if self.policy.recent_interactions else []
        selected.update(recent)
        if cost(selected) > self.budget:
            for i in recent:
                for result in units[i].messages[1:]:
                    _, node_id, _, step_index = result["tool_call_id"].split("_")
                    source = f"{node_id}:{step_index}"
                    marker = f"\n[ReadObservation source_id={source}; original output retained.]\n"
                    result["content"] = self.estimator.shorten(result["content"], 128, marker)
                    if source not in shortened:
                        shortened.append(source)
            # Recent interactions are preferred optional units. Drop complete
            # groups if compressed bodies still cannot fit; required units were
            # checked above, and assistant/tool pairings must remain intact.
            while cost(selected) > self.budget and recent:
                selected.discard(recent.pop(0))
        for i in range(len(units) - 1, -1, -1):
            if i not in selected and cost(selected | {i}) <= self.budget:
                selected.add(i)
        return selected
