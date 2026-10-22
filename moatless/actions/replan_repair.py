from pydantic import ConfigDict, Field
from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation


class ReplanRepairArgs(ActionArguments):
    """Submit a revised diagnosis after reading new evidence for a blocked failure family.

    The controller validates evidence references before allowing the final repair.
    """

    model_config = ConfigDict(title="ReplanRepair", extra="forbid")
    family_id: str
    hypothesis: str = Field(min_length=12, max_length=2000)
    evidence_sources: list[str] = Field(min_length=1, max_length=10)
    next_steps: str = Field(min_length=12, max_length=2000)


class ReplanRepair(Action):
    args_schema = ReplanRepairArgs

    async def execute_for_node(self, args, node):
        return Observation.create(
            "Replan submitted; execution evidence must be accepted by the controller.",
            properties={"replan_proposal": args.model_dump(exclude={"thoughts"})},
        )
