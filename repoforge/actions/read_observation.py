"""Read original observations by stable ancestor-node/step reference."""

from pydantic import ConfigDict, Field, model_validator

from repoforge.actions.action import Action
from repoforge.actions.schema import ActionArguments, Observation


class ReadObservationArgs(ActionArguments):
    model_config = ConfigDict(title="ReadObservation")
    source_id: str = Field(pattern=r"^\d+:\d+$", description="Observation reference: node_id:step_index")
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class ReadObservation(Action):
    args_schema = ReadObservationArgs

    async def execute_for_node(self, args: ReadObservationArgs, node) -> Observation:
        node_id, step_index = map(int, args.source_id.split(":"))
        source = next((n for n in node.get_trajectory() if n.node_id == node_id), None)
        if source is None or step_index >= len(source.action_steps) or source.action_steps[step_index].observation is None:
            return Observation.create("No completed observation at this reference in the current trajectory.",
                                      properties={"fail_reason": "observation_not_found"})
        original = source.action_steps[step_index].observation
        text = original.message or original.summary or ""
        lines = text.splitlines()
        end = min(args.end_line or args.start_line + 199, args.start_line + 199)
        content = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines)
                            if args.start_line <= i + 1 <= end)
        return Observation.create(content, properties={"source_id": args.source_id, "total_lines": len(lines),
                                                       "start_line": args.start_line, "end_line": min(end, len(lines))})
