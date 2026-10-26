"""Historical configurations stay readable after the package rename."""

from copy import deepcopy

from repoforge.actions.action import Action
from repoforge.actions.finish import Finish, FinishArgs
from repoforge.actions.schema import ActionArguments
from repoforge.compat import canonical_classpath


def test_load_historical_component_without_mutating_record():
    record = {"action_class": "moatless.actions.finish.Finish"}
    original = deepcopy(record)
    action = Action.model_validate(record)
    assert isinstance(action, Finish)
    assert record == original
    assert action.model_dump()["action_class"] == "repoforge.actions.finish.Finish"


def test_load_historical_action_arguments_without_mutating_record():
    record = {
        "action_args_class": "moatless.actions.finish.FinishArgs",
        "finish_reason": "Saved historical result",
    }
    original = deepcopy(record)
    action = ActionArguments.model_validate(record)
    assert isinstance(action, FinishArgs)
    assert action.finish_reason == record["finish_reason"]
    assert record == original


def test_current_component_round_trip():
    record = Finish().model_dump()
    assert record["action_class"] == "repoforge.actions.finish.Finish"
    assert isinstance(Action.model_validate(record), Finish)


def test_custom_component_namespace_is_preserved():
    assert canonical_classpath("custom.moatless.Finish") == "custom.moatless.Finish"
    assert canonical_classpath("moatless_extension.actions.Finish") == "moatless_extension.actions.Finish"
