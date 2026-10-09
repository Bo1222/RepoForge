import asyncio
import os

from dotenv import load_dotenv
from moatless.actions import Respond
from moatless.agent import ActionAgent
from moatless.completion import ToolCallCompletionModel
from moatless.flow import AgenticLoop

load_dotenv()


async def main():
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY in .env before running this demo.")

    agent = ActionAgent(
        completion_model=ToolCallCompletionModel(
            model="openai/deepseek-flash",
            model_base_url="https://api.deepseek.com",
            model_api_key=api_key,
        ),
        system_prompt="You are a helpful assistant. Respond briefly.",
        actions=[Respond()],
    )

    flow = AgenticLoop.create(
        message="用一句话介绍这个项目。",
        agent=agent,
        project_id="demo",
        trajectory_id="demo-001",
        max_iterations=3,
    )

    node = await flow.run()

    print("执行到的 node:", node.node_id)
    print("agent 回复:", node.action_steps[-1].observation.message)


asyncio.run(main())
