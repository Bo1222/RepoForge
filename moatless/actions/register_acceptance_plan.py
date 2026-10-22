from pydantic import ConfigDict

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.policy.repair import AcceptancePlan, validate_revision


class RegisterAcceptancePlanArgs(ActionArguments):
    """Register public task obligations and mandatory checks, before editing product code.

    Cite exact public issue quotes. Issue behavior checks must demonstrate a
    failing public reproduction on base and a pass with the same tests on the
    current product. Only the controller can assess satisfaction.
    """
    model_config = ConfigDict(title="RegisterAcceptancePlan", extra="forbid")
    plan: AcceptancePlan


class RegisterAcceptancePlan(Action):
    args_schema = RegisterAcceptancePlanArgs

    async def execute_for_node(self, args, node):
        try:
            validate_revision(node, args.plan, getattr(self, "_workspace", None))
        except ValueError as exc:
            return Observation.create(str(exc), properties={"fail_reason": "invalid_args"})
        return Observation.create("Acceptance plan registered. Status is derived from execution receipts.",
                                  properties={"accepted_plan": args.plan.model_dump(mode="json")})
