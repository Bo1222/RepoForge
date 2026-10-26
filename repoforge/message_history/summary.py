"""One-message rendering of completed history; no additional model call."""

from repoforge.message_history.assembly import ContextUnit, select_units
from repoforge.message_history.compact import CompactMessageHistoryGenerator
from repoforge.node import Node
from repoforge.workspace import Workspace


class SummaryMessageHistoryGenerator(CompactMessageHistoryGenerator):
    async def generate_messages(self, node: Node, workspace: Workspace) -> list[dict]:
        units = []
        for group in await self.get_node_messages(node):
            if group.user_message:
                units.append(ContextUnit([{"role": "user", "content": group.user_message}], required=True))
            text = group.assistant_message or ""
            for action, observation in zip(group.actions, group.observations, strict=True):
                text += "\n" + action.to_prompt() + "\nObservation: " + observation
            if group.state_content:
                text += "\n" + group.state_content
            if text:
                units.append(ContextUnit([{"role": "assistant", "content": text}]))
        selected = select_units(units, self.max_tokens)
        return [{"role": "user", "content": "\n\n".join(
            message["content"] for unit in selected for message in unit.messages
        )}]
