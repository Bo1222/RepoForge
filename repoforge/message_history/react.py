import logging

from repoforge.completion.schema import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
)
from repoforge.message_history.message_history import MessageHistoryGenerator
from repoforge.message_history.assembly import limit_messages
from repoforge.node import Node
from repoforge.utils.tokenizer import count_tokens
from repoforge.workspace import Workspace

logger = logging.getLogger(__name__)


class ReactMessageHistoryGenerator(MessageHistoryGenerator):
    async def generate_messages(self, node: Node, workspace: Workspace) -> list[AllMessageValues]:
        previous_nodes = node.get_trajectory()

        messages = []

        for previous_node in previous_nodes:
            if previous_node.user_message:
                messages.append(ChatCompletionUserMessage(role="user", content=previous_node.user_message))

            if previous_node.assistant_message:
                messages.append(
                    ChatCompletionAssistantMessage(role="assistant", content=previous_node.assistant_message)
                )

            if not previous_node.action_steps:
                continue

            assistant_content = ""

            if previous_node.thoughts:
                assistant_content += f"Thought: {previous_node.thoughts.text}\n"

            user_content = "Observations:\n "

            for action_step in previous_node.action_steps:
                if action_step.observation is None:
                    continue
                from repoforge.actions.think import ThinkArgs

                if isinstance(action_step.action, ThinkArgs):
                    assistant_content += f"Thought: {action_step.action.thought}\n"
                    continue
                elif hasattr(action_step.action, "thoughts") and action_step.action.thoughts:
                    assistant_content += f"Thought: {action_step.action.thoughts}\n"
                else:
                    assistant_content += "\n"

                assistant_content += f"Action: {action_step.action.name}\n"
                assistant_content += action_step.action.format_args_for_llm()

                if action_step.observation:
                    user_content += f"{action_step.observation.message}\n"

            if assistant_content:
                messages.append(ChatCompletionAssistantMessage(role="assistant", content=assistant_content))
                messages.append(ChatCompletionUserMessage(role="user", content=user_content))

        tokens = count_tokens("".join([m["content"] for m in messages if m.get("content")]))
        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")
        return limit_messages(messages, self.max_tokens)
