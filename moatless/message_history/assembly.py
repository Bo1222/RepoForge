"""Select complete interactions without mutating the canonical trajectory."""

from dataclasses import dataclass

from moatless.utils.tokenizer import count_tokens


@dataclass
class ContextUnit:
    messages: list[dict]
    required: bool = False

    @property
    def tokens(self) -> int:
        return sum(count_tokens(str(message)) for message in self.messages)


def select_units(units: list[ContextUnit], max_tokens: int | None) -> list[ContextUnit]:
    if max_tokens is None:
        return units
    required_tokens = sum(unit.tokens for unit in units if unit.required)
    if required_tokens > max_tokens:
        raise RuntimeError("context_budget_exceeded: user requirements exceed the history budget")
    remaining = max_tokens - required_tokens
    selected = {i for i, unit in enumerate(units) if unit.required}
    for i in range(len(units) - 1, -1, -1):
        if i in selected:
            continue
        cost = units[i].tokens
        if cost <= remaining:
            selected.add(i)
            remaining -= cost
    # Use positions, never equality-based list.index (duplicate messages are valid).
    return [unit for i, unit in enumerate(units) if i in selected]


def limit_messages(messages: list[dict], max_tokens: int | None, required_users: set[int] | None = None) -> list[dict]:
    units = []
    i = 0
    while i < len(messages):
        message = messages[i]
        group = [message]
        if message.get("tool_calls"):
            expected = [call["id"] for call in message["tool_calls"]]
            following = messages[i + 1:i + 1 + len(expected)]
            if [m.get("tool_call_id") for m in following] != expected:
                raise ValueError("Unbalanced tool call/result history")
            group.extend(following)
        elif message.get("role") == "tool":
            raise ValueError("Orphan tool result in history")
        elif (message.get("role") == "assistant" and i + 1 < len(messages)
              and messages[i + 1].get("role") == "user"
              and isinstance(messages[i + 1].get("content"), str)
              and messages[i + 1]["content"].startswith("Observations:")):
            group.append(messages[i + 1])
        required = message.get("role") == "user" and (required_users is None or i in required_users)
        units.append(ContextUnit(group, required))
        i += len(group)
    return [message for unit in select_units(units, max_tokens) for message in unit.messages]
