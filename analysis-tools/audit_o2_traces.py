#!/usr/bin/env python3
"""Offline audit of O2 trajectories: locate the first deterministic gap.

No model calls, no writes to the candidate.  The audit follows the selected
node path, lists failed tool actions and test evidence, and flags policy
gaps that a contract/verification fix could address.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from moatless.utils.file import is_test


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def action_name(action):
    return (action or {}).get("action_args_class", "?").rsplit(".", 1)[-1].replace("Args", "")


def path_nodes(trajectory, selected):
    nodes = {node["node_id"]: node for node in trajectory.get("nodes") or []}
    path, current = [], selected
    while current is not None and current in nodes:
        path.append(nodes[current])
        current = nodes[current].get("parent_id")
    path.reverse()
    return path


def patch_files(text):
    return sorted(set(re.findall(r"^--- a/(\S+)", text or "", re.M)))


def grading_failures(grade):
    status = grade.get("tests_status") or {}
    out = {}
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = status.get(key) or {}
        failures = value.get("failure") if isinstance(value, dict) else []
        if failures:
            out[key] = list(failures)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("analysis-tools/artifacts/o2_trace_audit.json"))
    args = parser.parse_args()
    output = args.output.resolve()
    jobs = read(output / "jobs.json", []) or []
    report = []
    for job in jobs:
        directory = output / "runs" / job["job_id"]
        generation = read(directory / "generation.json", {}) or {}
        grading = read(directory / "grading.json", {}) or {}
        decisions = read(directory / "policy-decisions.json", []) or []
        trajectory = read(directory / "trajectory.json", {}) or {}
        selected = generation.get("selected_node_id")
        path = path_nodes(trajectory, selected)
        failed, tests, repeats = [], [], Counter()
        previous = None
        for node in path:
            for step in node.get("action_steps") or []:
                action = step.get("action") or {}
                name = action_name(action)
                observation = step.get("observation") or {}
                properties = observation.get("properties") or {}
                reason = properties.get("fail_reason")
                if reason:
                    failed.append({"node": node["node_id"], "action": name,
                                   "reason": reason,
                                   "message": (observation.get("message") or "")[:240]})
                if name == "RunTests" and properties.get("test_run"):
                    run = properties["test_run"]
                    tests.append({"node": node["node_id"], "source": run.get("test_source"),
                                  "targets": run.get("targets"), "status": run.get("status"),
                                  "origins": run.get("test_origins"),
                                  "return_code": run.get("return_code")})
                signature = json.dumps(action, sort_keys=True, default=str)
                if previous and previous[1] and previous[0] == signature:
                    repeats[name] += 1
                previous = (signature, reason)
        patch = (directory / "patch.diff").read_text() if (directory / "patch.diff").exists() else ""
        files = patch_files(patch)
        product = [name for name in files if not is_test(name)]
        test_files = [name for name in files if is_test(name)]
        last = decisions[-1] if decisions else {}
        state = last.get("state") or {}
        decision = last.get("decision") or {}
        changed_tokens = set()
        for name in product:
            changed_tokens.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", Path(name).stem.lower()))
        targets = [entry.get("decision", {}).get("target") for entry in decisions
                   if (entry.get("decision") or {}).get("kind") == "verify"]
        unrelated = []
        for target in {target for target in targets if target}:
            tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", Path(target).stem.lower()))
            if changed_tokens and not (tokens & changed_tokens):
                unrelated.append(target)
        row = {
            "instance_id": job["instance_id"],
            "resolved": grading.get("local_resolved"),
            "grading": grading.get("local_grading"),
            "finish_reason": generation.get("finish_reason"),
            "generation_status": generation.get("generation_status"),
            "final_policy": {"kind": decision.get("kind"), "reason": decision.get("reason"),
                             "verification_state": state.get("verification_state"),
                             "introduced_failures": state.get("introduced_failures") or [],
                             "target": state.get("target")},
            "product_files": product,
            "test_files": test_files,
            "failed_actions": failed,
            "test_runs": tests,
            "repeated_actions": dict(repeats),
            "verify_targets": sorted({target for target in targets if target}),
            "unrelated_verify_targets": unrelated,
            "grading_failures": grading_failures(grading),
            "flags": [],
        }
        if grading.get("local_grading") == "environment_error":
            row["flags"].append("environment_error")
        if product and row["final_policy"]["verification_state"] not in {None, "PUBLIC_VERIFIED"}:
            row["flags"].append("stopped_without_public_verification")
        if product and not tests:
            row["flags"].append("no_test_run_with_product_patch")
        if unrelated:
            row["flags"].append("unrelated_verify_target")
        if repeats:
            row["flags"].append("repeated_action_after_failure")
        if test_files:
            row["flags"].append("submitted_test_files")
        if row["grading_failures"].get("PASS_TO_PASS"):
            row["flags"].append("p2p_regression")
        if row["grading_failures"].get("FAIL_TO_PASS"):
            row["flags"].append("f2p_still_failing")
        report.append(row)

    print(json.dumps(report, ensure_ascii=False, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
