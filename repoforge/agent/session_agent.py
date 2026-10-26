"""Tool lifecycle events supplement (and never replace) existing action snapshots."""

import asyncio
import time
from repoforge.agent.agent import ActionAgent
from repoforge.events import BaseEvent
from repoforge.context_data import current_action_step


class SessionAgent(ActionAgent):
    async def _execute_action_step(self, node, action_step):
        started = time.monotonic()
        data = {
            "node_id": node.node_id,
            "step": current_action_step.get(),
            "name": action_step.action.name,
            "arguments": action_step.action.model_dump(),
        }
        await self._emit_event(BaseEvent(scope="tool", event_type="running", data=data))
        try:
            observation = await super()._execute_action_step(node, action_step)
            await self._emit_event(
                BaseEvent(
                    scope="tool",
                    event_type="failed" if (observation.properties or {}).get("fail_reason") else "completed",
                    data={
                        **data,
                        "elapsed": time.monotonic() - started,
                        "observation": observation.model_dump(mode="json"),
                    },
                )
            )
            return observation
        except BaseException as exc:
            await self._emit_event(
                BaseEvent(
                    scope="tool",
                    event_type="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                    data={**data, "elapsed": time.monotonic() - started, "error": str(exc)},
                )
            )
            raise
