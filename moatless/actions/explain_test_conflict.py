"""Record a public, execution-backed conflict for partial delivery only."""

from pydantic import ConfigDict, Field
from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation


class ExplainTestConflictArgs(ActionArguments):
    """Cite an original assertion conflicting with a proven public issue behavior.

    This can only support partially_verified delivery. It never turns failing
    tests into passes or exempts a mandatory acceptance check.
    """

    model_config = ConfigDict(title="ExplainTestConflict", extra="forbid")
    criterion_id: str
    candidate_source: str
    failing_case: str
    public_quote: str = Field(min_length=3)
    original_assertion: str = Field(min_length=6)
    explanation: str = Field(min_length=20, max_length=2000)


class ExplainTestConflict(Action):
    args_schema = ExplainTestConflictArgs

    async def execute_for_node(self, args, node):
        from moatless.policy.conflicts import validate_conflict

        claim = args.model_dump(exclude={"thoughts"})
        error = validate_conflict(node, self._workspace, claim)
        if error:
            return Observation.create(error, properties={"fail_reason": "invalid_args"})
        return Observation.create(
            "Public conflict evidence recorded; only partial delivery is eligible after all task checks pass.",
            properties={"expectation_conflict": claim},
        )
