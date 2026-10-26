"""Explicit task-note updates recorded as ordinary actions in the trajectory."""

from pydantic import ConfigDict

from repoforge.actions.action import Action
from repoforge.actions.schema import ActionArguments, Observation
from repoforge.task_state import Hypothesis, PlanStep, TaskPhase, TaskState


class UpdateTaskStateArgs(ActionArguments):
    model_config = ConfigDict(title="UpdateTaskState", extra="forbid")
    actual_behavior: str | None = None
    expected_behavior: str | None = None
    acceptance_criteria: list[str] | None = None
    phase: TaskPhase | None = None
    plan: list[PlanStep] | None = None
    hypotheses: list[Hypothesis] | None = None
    next_step: str | None = None
    related_files: list[str] | None = None
    blocked_by: list[str] | None = None


class UpdateTaskState(Action):
    args_schema = UpdateTaskStateArgs

    async def execute_for_node(self, args: UpdateTaskStateArgs, node) -> Observation:
        ancestors = {n.node_id: n for n in node.get_trajectory()}
        for item in [*(args.plan or []), *(args.hypotheses or [])]:
            if isinstance(item, Hypothesis) and item.status != "pending" and not item.evidence:
                return Observation.create("Supported/rejected hypotheses require completed evidence references.",
                                          properties={"fail_reason": "invalid_task_evidence"})
            for ref in item.evidence:
                source = ancestors.get(ref.node_id)
                if source is None or ref.step_index >= len(source.action_steps) or source.action_steps[ref.step_index].observation is None:
                    return Observation.create("Evidence must refer to a completed action in this trajectory.",
                                              properties={"fail_reason": "invalid_task_evidence"})
        for items in (args.plan or [], args.hypotheses or []):
            if len({item.id for item in items}) != len(items):
                return Observation.create("Task item IDs must be unique.", properties={"fail_reason": "duplicate_task_id"})
        state = node.task_state or next((n.task_state for n in reversed(node.get_trajectory()[:-1]) if n.task_state), None)
        state = state.model_copy(deep=True) if state is not None else TaskState()
        updates = args.model_dump(exclude_none=True, exclude={"thoughts"})
        merged = state.model_dump()
        merged.update(updates)
        merged["revision"] = state.revision + 1
        node.task_state = TaskState.model_validate(merged)
        return Observation.create(f"Task notes updated to revision {node.task_state.revision}; evidence references are valid, claims are not independently proven.",
                                  properties={"task_state_revision": node.task_state.revision})
