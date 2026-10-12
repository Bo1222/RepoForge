"""moatless chat / run / resume command line entry point."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import signal

from dotenv import load_dotenv
from rich.console import Console
from rich.text import Text

from moatless.cli.service import SessionService, error_message, session_mode
from moatless.cli import worktree


class CLIParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed, extra = self.parse_known_args(args, namespace)
        # argparse does not intermix an optional positional after an ID and flags.
        if len(extra) == 1 and not extra[0].startswith("-") and getattr(parsed, "task", None) is None:
            parsed.task = extra.pop()
        if extra:
            self.error("unrecognized arguments: " + " ".join(extra))
        return parsed


def parser():
    root = CLIParser(
        prog="moatless", description="Local chat and coding agent using LangGraph and isolated Git worktrees"
    )
    root.add_argument("--session-dir", help="Session storage (default MOATLESS_CLI_DIR or ~/.moatless/cli)")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("chat", "run", "resume"):
        p = sub.add_parser(name)
        if name == "resume":
            p.add_argument("session_id")
            p.add_argument("--headless", action="store_true", help="Resume without terminal UI")
        else:
            p.add_argument("--repo", help="Bind a clean Git repository; omit for ordinary chat")
            p.add_argument("--model")
            p.add_argument("--config", help="JSON configuration")
            p.add_argument("--max-iterations", type=int)
            p.add_argument("--max-cost", type=float)
            p.add_argument("--instance-id", help="SWE-bench instance ID for predictions export")
        p.add_argument("task", nargs="?", help="Natural-language task (follow-up for resume)")
        p.add_argument("--task-file", help="UTF-8 multiline task file; '-' reads stdin")
        p.add_argument("--export", help="Export completed session to a new directory")
    return root


def read_task(args):
    if args.task and args.task_file:
        raise ValueError("Use a task argument or --task-file, not both")
    if args.task_file == "-":
        import sys

        return sys.stdin.read()
    if args.task_file:
        return Path(args.task_file).expanduser().read_text(encoding="utf-8")
    return args.task


async def batch(service, session, message, export, console):
    def display(identity, event):
        if event.get("scope") == "tool":
            details = event["data"]
            console.print(Text(f"{details['name']} · {event['event_type']}"))
            if event["event_type"] in {"completed", "failed"}:
                output = (details.get("observation") or {}).get("message", details.get("error", ""))
                console.print(
                    Text(output[:2000] + ("\n…full output saved in session logs" if len(output) > 2000 else ""))
                )

    service.listeners.append(display)
    task = service.start(session, message)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        console.print("Task cancelled; checkpoint saved. Unknown effects require reconciliation.")
        return 130
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        await service.close()
    if export:
        console.print(f"Exported: {service.export(session, export)}")
    coding = session_mode(session.info) == "coding"
    context = f"Workspace: {session.info['worktree']}" if coding else "Mode: ordinary chat"
    console.print(Text(f"Session: {session.id}\n{context}\nStatus: {session.info['status']}"))
    if coding:
        tests = service.tests(session)
        console.print("Last recorded test command: " + (tests[-1]["test_status"] if tests else "unknown (not run)"))
    for node in session.flow.root.get_all_nodes():
        for step in node.action_steps:
            if step.action.name == "Finish":
                console.print(Text(step.action.finish_reason))
    if coding:
        console.print("Official Harness: not run")
    status = session.info["status"]
    return 0 if status in {"terminal", "completed", "finished"} else 2


def main(argv=None):
    args = parser().parse_args(argv)
    load_dotenv()
    console = Console()
    config = {}
    try:
        task = read_task(args)
        service = SessionService(args.session_dir)
        if args.command == "resume":
            session = service.load(args.session_id)
            config, repo = session.info["config"], session.info["source"]
        else:
            config = json.loads(Path(args.config).expanduser().read_text()) if args.config else {}
            for field in ("model", "max_iterations", "max_cost", "instance_id"):
                value = getattr(args, field)
                if value is not None:
                    config[field] = value
            repo = worktree.validate(args.repo)[0] if args.repo is not None else None
            if args.command == "run" and not task:
                raise ValueError("moatless run requires a task or --task-file")
            session = asyncio.run(service.create(repo, task, config)) if task else None
        log_path = service.directory / "cli.log"
        logging.basicConfig(filename=log_path, level=logging.WARNING, force=True)
        # LiteLLM installs its own terminal handler; send diagnostics to the
        # session log so background SDK output does not corrupt the TUI.
        sdk_logger = logging.getLogger("LiteLLM")
        sdk_logger.handlers.clear()
        sdk_logger.propagate = True
        if args.command == "run" or args.command == "resume" and args.headless:
            return asyncio.run(
                batch(service, session, task if args.command == "resume" else None, args.export, console)
            )
        from moatless.cli.app import CodingApp

        app = CodingApp(service, repo, config, session, autostart=bool(session))
        if args.command == "resume" and task:
            # Continue the persisted history once the UI is mounted.
            app.autostart = False
            app.call_after_refresh(app.launch, task)
        app.run()
        if args.export and app.session:
            console.print(f"Exported: {service.export(app.session, args.export)}")
        return 0
    except Exception as exc:
        console.print(Text(f"{type(exc).__name__}: {error_message(exc, config)}"), style="red")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
