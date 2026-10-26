"""Model-maintained task notes, separate from system-owned execution facts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: int = Field(ge=0)
    step_index: int = Field(ge=0)


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    status: Literal["pending", "supported", "rejected"] = "pending"
    evidence: list[EvidenceRef] = Field(default_factory=list)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: Literal["pending", "in_progress", "done"] = "pending"
    evidence: list[EvidenceRef] = Field(default_factory=list)


TaskPhase = Literal["understanding", "investigation", "modification", "testing", "validation", "blocked"]


class TaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    revision: int = Field(default=0, ge=0)
    actual_behavior: str | None = None
    expected_behavior: str | None = None
    acceptance_criteria: list[str] = Field(default_factory=list)
    phase: TaskPhase = "understanding"
    plan: list[PlanStep] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    next_step: str | None = None
    related_files: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)


TASK_MEMORY_PROMPT = """\n# Task context and memory
When requirements, investigation direction or plans change, use UpdateTaskState to maintain concise task notes.
Distinguish pending hypotheses from evidence-supported or rejected hypotheses; cite completed observations by node_id and step_index.
Task notes are model-maintained claims. Tool outcomes, file changes and test results are system-owned facts and cannot be overwritten by notes.
Use ReadObservation with a source_id to inspect original outputs omitted from the projected context.
Report whether the work is verified, unverified or blocked using tests for the current code version. Never present old tests or unexecuted checks as current passes.
"""
