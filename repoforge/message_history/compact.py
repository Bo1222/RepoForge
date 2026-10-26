"""Compact completed interactions plus an explicitly labelled workspace projection."""

from pydantic import BaseModel, Field

from repoforge.actions.schema import ActionArguments
from repoforge.actions.view_code import ViewCodeArgs
from repoforge.message_history.assembly import limit_messages
from repoforge.message_history.message_history import MessageHistoryGenerator
from repoforge.node import Node
from repoforge.workspace import Workspace


class NodeMessage(BaseModel):
    user_message: str | None = None
    assistant_message: str | None = None
    actions: list[ActionArguments] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    state_content: str | None = None


class CompactMessageHistoryGenerator(MessageHistoryGenerator):
    message_cache: bool = False

    async def generate_messages(self, node: Node, workspace: Workspace) -> list[dict]:
        messages = []
        required_users = set()
        tool_idx = 0
        for group in await self.get_node_messages(node):
            if group.user_message:
                required_users.add(len(messages))
                messages.append({"role": "user", "content": group.user_message})
            content = [group.assistant_message] if group.assistant_message else []
            calls, results = [], []
            for action, observation in zip(group.actions, group.observations, strict=True):
                tool_idx += 1
                call_id = f"tool_{tool_idx}"
                exclude = None if self.thoughts_in_action else {"thoughts"}
                if not self.thoughts_in_action and action.thoughts:
                    content.append(action.thoughts)
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": action.name, "arguments": action.model_dump_json(exclude=exclude),
                }})
                results.append({"role": "tool", "tool_call_id": call_id, "content": observation})
            if calls or content:
                message = {"role": "assistant"}
                if content:
                    message["content"] = "\n".join(content)
                if calls:
                    message["tool_calls"] = calls
                messages.append(message)
                messages.extend(results)
            if group.state_content:
                messages.append({"role": "user", "content": group.state_content})
        return limit_messages(messages, self.max_tokens, required_users)

    async def get_node_messages(self, node: Node) -> list[NodeMessage]:
        groups = []
        for previous in node.get_trajectory():
            group = NodeMessage(
                user_message=previous.user_message or (
                    previous.feedback_data.feedback if previous.feedback_data else None
                ),
                assistant_message=previous.assistant_message,
            )
            for step in previous.action_steps:
                if step.observation is None:
                    continue
                group.actions.append(step.action)
                observation = step.observation
                # Current code is a separate state block, never a fabricated tool call.
                text = observation.message or observation.summary or "No output found."
                if self.include_file_context and observation.summary and not isinstance(step.action, ViewCodeArgs):
                    text = observation.summary
                group.observations.append(text)
            if group.user_message or group.assistant_message or group.actions:
                groups.append(group)

        context = node.file_context
        state = []
        if context and self.include_file_context:
            code = context.create_prompt(
                show_span_ids=False, show_line_numbers=True, exclude_comments=False,
                show_outcommented_code=True, outcomment_code_comment="... rest of the code",
            )
            if code:
                state.append("Current code context:\n" + code)
        if context and self.include_git_patch:
            patch = context.generate_git_patch()
            if patch:
                state.append("Current workspace diff:\n```diff\n" + patch + "\n```")
        if context and context.test_files:
            state.append("Saved test results (historical; freshness is not established):\n"
                         + context.get_test_failure_details() + "\n" + context.get_test_summary())
        if state:
            groups.append(NodeMessage(state_content="<workspace_state>\n" + "\n\n".join(state)
                                      + "\n</workspace_state>"))
        return groups
