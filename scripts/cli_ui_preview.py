"""Capture the actual Textual UI with scripted model responses and real local tools.

No provider API calls or changes to the target repository; screenshots are written to --output.
Run: uv run python scripts/cli_ui_preview.py --output docs/design/cli-ui-options/implemented
"""

import argparse
import base64
import asyncio
import json
import os
import re
from pathlib import Path
import tempfile
from unittest.mock import patch

from moatless.actions.finish import FinishArgs
from moatless.actions.local_coding import LocalTestArgs
from moatless.actions.read_file import ReadFileArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.cli.app import CodingApp, TaskInput
from moatless.cli.service import SessionService
from moatless.cli.worktree import git
from litellm.types.utils import ModelResponse


def save_preview(app, name, output):
    """Self-contained Fira Code SVG with explicit Chinese monospace fallback."""
    svg = app.export_screenshot(title="CLI")
    font = Path(__file__).resolve().parents[1] / "docs/design/cli-ui-options/fonts/FiraCode.ttf"
    encoded = base64.b64encode(font.read_bytes()).decode("ascii")
    css = f"""
    @font-face {{
        font-family: 'Fira Code';
        src: url('data:font/ttf;base64,{encoded}') format('truetype');
        font-style: normal;
        font-weight: 300 700;
    }}
    """
    svg = re.sub(r"    @font-face \{.*?    \}\n", "", svg, flags=re.DOTALL)
    svg = svg.replace("<style>", "<style>" + css, 1)
    svg = svg.replace(
        "font-family: Fira Code, monospace;",
        "font-family: 'Fira Code', 'Noto Sans Mono CJK SC', monospace;\n        font-variant-ligatures: none;",
    )
    (output / name).write_text(svg)


async def capture(output):
    # Render the color design even when the invoking automation sets NO_COLOR.
    os.environ.pop("NO_COLOR", None)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moatless-ui-") as temporary:
        root = Path(temporary)
        repo = root / "demo"
        repo.mkdir()
        (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (repo / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n")
        (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
        for args in [
            ("init", "-q"),
            ("config", "user.email", "preview@example.invalid"),
            ("config", "user.name", "Preview"),
            ("add", "."),
            ("commit", "-qm", "baseline"),
        ]:
            git(repo, *args)

        calls = []
        actions = [
            ReadFileArgs(file_path="calc.py"),
            LocalTestArgs(command="python -m pytest -q"),
            StringReplaceArgs(path="calc.py", old_str="return a - b", new_str="return a + b"),
            LocalTestArgs(command="python -m pytest -q"),
            FinishArgs(finish_reason="已修复 add，测试通过。修改保存在独立 worktree，输入 /diff 查看补丁。"),
        ]

        async def complete(**kwargs):
            calls.append(kwargs["model"])
            message = {"role": "assistant", "content": "当前配置的 API 模型是 deepseek/deepseek-chat。"}
            if len(calls) >= 2:
                action = actions[len(calls) - 2]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tool_{i}",
                            "type": "function",
                            "function": {
                                "name": action.name,
                                "arguments": action.model_dump_json(),
                            },
                        }
                        for i in [len(calls) - 2]
                    ],
                }
            return ModelResponse(
                model=kwargs["model"],
                choices=[
                    {
                        "message": message,
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        service = SessionService(root / "sessions")
        app = CodingApp(service, None, config={"model": "deepseek/deepseek-chat"})
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "preview-no-network"}), patch("litellm.acompletion", complete):
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause()
                save_preview(app, "D-dialogue-empty.svg", output)
                editor = app.query_one("#input", TaskInput)
                editor.load_text("/")
                await pilot.pause()
                save_preview(app, "D-command-menu.svg", output)
                editor.load_text("")
                await pilot.pause()
                await app.submit("你当前配置的模型是什么？")
                await app.workers.wait_for_complete()
                await pilot.pause()
                save_preview(app, "D-chat-question.svg", output)
                assert app.session.info["mode"] == "chat"
                await app.submit("/repo " + str(repo))
                await app.submit("修复 calc.py 的 add 函数，先确认测试失败，修改后再运行测试。")
                await app.workers.wait_for_complete()
                await pilot.pause()
                save_preview(app, "D-dialogue-tools.svg", output)
                await app.submit("/architecture")
                await pilot.pause()
                save_preview(app, "D-architecture.svg", output)
                await pilot.press("escape")
                await app.submit("/diff")
                await pilot.pause()
                save_preview(app, "D-diff.svg", output)
                await pilot.press("escape")
                evidence = {
                    "model_responses": "scripted; no API calls or provider charges",
                    "ui": "actual Textual screen exports; no startup page or top status bar",
                    "modes": ["chat", "coding"],
                    "sdk_requests": len(calls),
                    "plain_text": "raw SDK text response parsed through Completion, Agent, Flow, UI",
                    "font": "embedded Fira Code; Noto Sans Mono CJK SC fallback; ligatures disabled",
                    "local_tools": "real temporary Git worktree and pytest",
                    "test_results": [record["test_status"] for record in service.tests(app.session)],
                    "screens": sorted(p.name for p in output.glob("*.svg")),
                    "usage": service.usage(app.session),
                }
                assert evidence["test_results"] == ["failed", "passed"]
                (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
                await service.close()
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(capture(parser.parse_args().output.resolve()))
