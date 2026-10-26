#!/usr/bin/env python3
"""Export a readable node-by-node execution flow for a small run.

Offline only.  Reads trajectory.json, policy-decisions.json, grading.json and
the per-node tool logs; writes Markdown and JSON.
"""

import argparse
import json
import re
from pathlib import Path


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def action_name(action):
    return (action or {}).get("action_args_class", "?").rsplit(".", 1)[-1].replace("Args", "")


def short(value, limit=260):
    text = str(value or "").replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def args_summary(action):
    action = action or {}
    name = action_name(action)
    if name == "StringReplace":
        return f"path={action.get('path')} old={short(action.get('old_str'), 70)!r} new={short(action.get('new_str'), 90)!r}"
    if name == "RunTests":
        return f"files={action.get('test_files')} source={action.get('test_source', 'workspace')}"
    if name in {"ViewCode", "ReadFile"}:
        return f"files={short(action.get('files') or action.get('file_path'), 180)}"
    if name in {"FindFunction", "FindClass"}:
        return f"name={action.get('function_name') or action.get('class_name')} pattern={action.get('file_pattern')}"
    if name in {"FindCodeSnippet", "SemanticSearch"}:
        return f"query={short(action.get('code_snippet') or action.get('query'), 120)} pattern={action.get('file_pattern')}"
    if name in {"Finish", "VerifiedFinish"}:
        return f"reason={short(action.get('finish_reason'), 160)}"
    return short({key: value for key, value in action.items()
                  if key not in {"action_args_class", "thoughts"}}, 180)


def node_path(trajectory, selected):
    nodes = {node["node_id"]: node for node in trajectory.get("nodes") or []}
    path, current = [], selected
    while current is not None and current in nodes:
        path.append(nodes[current])
        current = nodes[current].get("parent_id")
    path.reverse()
    return path


def test_command(directory, node_id):
    logs = sorted(Path(directory, "tool-logs").rglob(f"node_{node_id}/*.log"))
    if not logs:
        return ""
    text = logs[0].read_text(errors="replace")
    first = text.splitlines()[0] if text else ""
    return first.replace("Command: ", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    out = args.out or output / "trajectory-flow.md"
    jobs = read(output / "jobs.json", []) or []
    lines = ["# Trajectory execution flow", "", f"Output: `{output}`", ""]
    data = []
    for job in jobs:
        directory = output / "runs" / job["job_id"]
        generation = read(directory / "generation.json", {}) or {}
        grading = read(directory / "grading.json", {}) or {}
        decisions = read(directory / "policy-decisions.json", []) or []
        trajectory = read(directory / "trajectory.json", {}) or {}
        decision_by_id = {entry.get("decision_id"): entry for entry in decisions}
        path = node_path(trajectory, generation.get("selected_node_id"))
        lines += [
            f"## {job['instance_id']}",
            "",
            f"- job: `{job['job_id']}`",
            f"- resolved: `{grading.get('local_resolved')}`; grading: `{grading.get('local_grading')}`; "
            f"finish: `{generation.get('finish_reason')}`; "
            f"steps: `{(generation.get('policy_metrics') or {}).get('actual_tool_steps')}`",
            "",
            "| # | node | origin | action / args | observation | test | policy |",
            "|---:|---:|---|---|---|---|---|",
        ]
        step_index = 0
        for node in path:
            for step in node.get("action_steps") or []:
                step_index += 1
                action = step.get("action") or {}
                observation = step.get("observation") or {}
                properties = observation.get("properties") or {}
                name = action_name(action)
                fail_reason = properties.get("fail_reason")
                message = short(observation.get("message") or observation.get("summary"), 200)
                test = properties.get("test_run") or {}
                test_text = ""
                if test:
                    command = test_command(directory, node["node_id"])
                    test_text = (f"{test.get('test_source')}:{test.get('targets')} "
                                 f"{test.get('status')} rc={test.get('return_code')}")
                    if command:
                        test_text += f" `{command}`"
                decision_id = step.get("policy_decision_id")
                decision_entry = decision_by_id.get(decision_id)
                policy_text = ""
                if decision_entry:
                    decision = decision_entry.get("decision") or {}
                    state = decision_entry.get("state") or {}
                    policy_text = (f"{decision.get('kind')}/{decision.get('reason')}"
                                   f" target={decision.get('target')} "
                                   f"verif={state.get('verification_state')}")
                observation_text = f"fail={fail_reason}; {message}" if fail_reason else message
                lines.append(
                    f"| {step_index} | {node['node_id']} | {step.get('origin', 'model')} | "
                    f"**{name}** {short(args_summary(action), 200)} | {short(observation_text, 260)} | "
                    f"{short(test_text, 260)} | {short(policy_text, 220)} |")
        failures = {}
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            value = (grading.get("tests_status") or {}).get(key) or {}
            if isinstance(value, dict) and value.get("failure"):
                failures[key] = value["failure"]
        lines += ["", f"- grading failures: `{json.dumps(failures, ensure_ascii=False)}`", ""]
        data.append({"instance_id": job["instance_id"], "job_id": job["job_id"],
                     "generation": generation, "grading": grading,
                     "decisions": decisions, "steps": len(path)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    (out.with_suffix(".json")).write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    main()
