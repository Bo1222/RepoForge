import asyncio
import argparse
import os

from dotenv import load_dotenv
from moatless.actions import Respond
from moatless.agent import ActionAgent
from moatless.completion import ToolCallCompletionModel
from moatless.flow import AgenticLoop, LangGraphFlow

load_dotenv()


async def main():
    parser = argparse.ArgumentParser(description="Respond-only demo; offline replay: python -m scripts.migration_smoke")
    parser.add_argument("--flow", choices=["graph", "loop"], default="graph")
    parser.add_argument("--checkpoint-path", help="Local SQLite database for resumable graph execution")
    args = parser.parse_args()
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

    implementation = LangGraphFlow if args.flow == "graph" else AgenticLoop
    options = {"checkpoint_path": args.checkpoint_path} if args.flow == "graph" else {}
    flow = implementation.create(
        message="用一句话介绍这个项目。",
        agent=agent,
        project_id="demo",
        trajectory_id="demo-001",
        max_iterations=3,
        **options,
    )

    node = await flow.run()

    print("执行到的 node:", node.node_id)
    print("agent 回复:", node.action_steps[-1].observation.message)


if __name__ == "__main__":
    asyncio.run(main())
