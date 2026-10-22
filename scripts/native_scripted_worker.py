"""Native worker contract fixture: actual tools/runtime, fixed SDK responses.

This program is deliberately separate from benchmark generation. Its artifacts
are workflow evidence, never five-instance or official resolved results.
"""

import argparse
import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock

from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.actions.verified_finish import VerifiedFinishArgs
from moatless.actions.view_code import CodeSpan, ViewCodeArgs
from moatless.evaluation.experiment import read_json, verify_answer, write_json


async def run(args):
    from litellm.types.utils import ModelResponse
    import litellm
    import moatless.evaluation.generation as worker
    from moatless.index.code_index import CodeIndex
    import scripts.t4_cli_probe

    scripts.t4_cli_probe.model_metadata = lambda: {"id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash", "source": "fixture"}
    CodeIndex.from_persist_dir = lambda *a, **kw: MagicMock(spec=CodeIndex)
    os.environ["DEEPSEEK_API_KEY"] = "native-scripted-fixture-key"
    actions = [ViewCodeArgs(files=[CodeSpan(file_path="calc.py", span_ids=["add"]),
                                   CodeSpan(file_path="test_calc.py", span_ids=["test_add"])]),
               RunTestsArgs(test_files=["test_calc.py"]),
               StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b"),
               RunTestsArgs(test_files=["test_calc.py"]),
               VerifiedFinishArgs(finish_reason="Fixture test passed", test_verification="Real public fixture test failed before and passed after")]
    calls = []

    async def sdk(**kwargs):
        index = len(calls)
        if index >= len(actions):
            raise RuntimeError("Unexpected additional fixture SDK call")
        calls.append(kwargs)
        action = actions[index]
        return ModelResponse(model="deepseek-flash", usage={"prompt_tokens": 100, "completion_tokens": 10},
            choices=[{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": f"fixture_{index}", "type": "function",
                    "function": {"name": action.name, "arguments": action.model_dump_json()}}]}}])

    litellm.acompletion = sdk
    bundle = worker.load_bundle(args.bundle)
    await worker.run(bundle, args.output, args.index)
    answer = verify_answer(args.output, bundle["job"])
    trajectory = read_json(args.output / "trajectory.json")
    observations = [step["observation"] for node in trajectory["nodes"] for step in node.get("action_steps", [])
                    if step["action"]["action_args_class"] == "moatless.actions.run_tests.RunTestsArgs"]
    codes = [obs["properties"]["return_code"] for obs in observations]
    generation = read_json(args.output / "generation.json")
    passed = len(calls) == 5 and codes == [1, 0] and generation["generation_status"] == "completed"
    write_json(args.output / "contract.json", {"passed": passed, "provider_calls": 0,
        "sdk_fixture_calls": len(calls), "test_return_codes": codes, "answer_hash": answer["answer_hash"],
        "manifest_hash": bundle["job"]["manifest_hash"], "execution_backend": "native",
        "variant": bundle["job"]["variant"], "synthetic_workflow": True, "official_harness": "not_run"})
    write_json(args.output / "model-contexts.json", [call["messages"] for call in calls])
    if not passed:
        raise RuntimeError("Native fixed-response worker contract failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
