"""Deterministic T7 labels, descriptive statistics and paired analysis.

``label`` writes per-job evidence-bound labels (also consumed by aggregate).
``analyze`` writes facts, stats, paired transitions and a recommendation for
dev; for holdout it writes only descriptive statistics and never a new
recommendation or lock.
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
from collections import Counter

from repoforge.evaluation.experiment import (
    T7_DEV_VARIANTS, T7_HOLDOUT_VARIANTS, content_hash, load_manifest, read_json,
    sha, verify_answer, write_json,
)
from repoforge.evaluation.t7 import (
    PROTOCOL, SELECTION_RULE, FALSIFICATION_RULE, SMALL_SAMPLE_STATEMENT,
    select_winner, validate_lock,
)

SEED = 20261018
BOOTSTRAP = 10000
ACTION_NAMES = {"StringReplace", "CreateFile", "InsertLine", "AppendString", "TextEditor"}
READ_NAMES = {"ViewCode", "ReadFile", "ReadObservation", "GrepTool", "FindFunction",
              "FindClass", "FindCodeSnippet", "SemanticSearch", "Bash"}


def root_of(path):
    for candidate in [Path(path).resolve(), *Path(path).resolve().parents]:
        if (candidate / "repoforge/evaluation/experiment.py").exists():
            return candidate
    raise RuntimeError("Cannot locate the repository root")


def _trajectory(directory):
    path = directory / "trajectory.json"
    return read_json(path) if path.exists() else {}


def _patch_files(directory):
    from unidiff import PatchSet
    path = directory / "patch.diff"
    if not path.exists() or not path.read_text().strip():
        return [], []
    from repoforge.utils.file import is_test
    product, tests = [], []
    for file in PatchSet(path.read_text()):
        (tests if is_test(file.path) else product).append(file.path)
    return product, tests


def _steps(trajectory):
    rows = []
    for node in trajectory.get("nodes") or []:
        for index, step in enumerate(node.get("action_steps") or []):
            observation = step.get("observation") or {}
            properties = observation.get("properties") or {}
            if not isinstance(properties, dict) or properties.get("execution_status") == "skipped":
                continue
            action = step.get("action") or {}
            context = properties.get("execution_context") or {}
            rows.append({"node_id": node.get("node_id"), "index": index, "action": action,
                         "properties": properties, "context": context,
                         "message": observation.get("message") or "",
                         "name": action.get("action_args_class", "").rsplit(".", 1)[-1].replace("Args", "")})
    return rows


def _final_workspace(steps):
    for row in reversed(steps):
        value = (row["context"] or {}).get("workspace_after")
        if value:
            return value
    return None


def _progress(steps):
    """Deterministic no-gain proxy over executed actions.

    Progress markers: a changed workspace after an edit, new view/search/test
    evidence, or a new test result.  The proxy is reported as a diagnostic and
    is not presented as a probability of success.
    """
    seen, no_gain, streak, max_streak, no_gain_steps = set(), 0, 0, 0, 0
    for row in steps:
        props, name, context = row["properties"], row["name"], row["context"]
        before = (context or {}).get("workspace_before")
        after = (context or {}).get("workspace_after")
        signature = content_hash({"action": {k: v for k, v in row["action"].items()
                                             if k not in {"thoughts", "action_args_class"}},
                                  "before": before, "environment": (context or {}).get("environment")})
        progress = False
        if name in ACTION_NAMES and after and after != before:
            marker = ("edit", after)
            progress = marker not in seen
            seen.add(marker)
        view = props.get("code_view")
        if view and not props.get("fail_reason"):
            marker = ("view", signature, content_hash(view))
            progress |= marker not in seen and bool(view.get("files"))
            seen.add(marker)
        search = props.get("search_hits")
        if isinstance(search, dict):
            marker = ("search", signature, content_hash(search.get("files", [])))
            progress |= marker not in seen and bool(search.get("files"))
            seen.add(marker)
        record = props.get("test_run") or props.get("test_execution") or props.get("return_code")
        if record is not None:
            marker = ("test", signature, content_hash(record))
            progress |= marker not in seen
            seen.add(marker)
        if progress:
            streak = 0
        else:
            streak += 1
            no_gain += 1
            no_gain_steps += 1
        max_streak = max(max_streak, streak)
    return {"actual_steps": len(steps), "no_gain_steps": no_gain_steps,
            "no_gain_rate": no_gain_steps / len(steps) if steps else None,
            "max_no_gain_streak": max_streak}


def _tests(directory, final_workspace):
    path = directory / "agent-tests.json"
    records = read_json(path) if path.exists() else []
    effective = [row for row in records if not row.get("superseded")
                 and row.get("effective_status") == "passed"]
    bound = [row for row in effective if not final_workspace
             or row.get("workspace_after") == final_workspace]
    requests = read_json(directory / "test-requests.json") if (
        directory / "test-requests.json").exists() else []
    return {"records": len(records), "effective_passed": len(effective),
            "bound_to_final": len(bound), "base_public_requests": sum(
                row.get("arguments", {}).get("test_source") == "base_public" for row in requests),
            "effective_verification": bool(bound)}


def facts_for(manifest, job):
    directory = Path(manifest["_output"]) / "runs" / job["job_id"]
    generation = read_json(directory / "generation.json") if (
        directory / "generation.json").exists() else {}
    grading = read_json(directory / "grading.json") if (
        directory / "grading.json").exists() else {}
    usage = read_json(directory / "usage.json") if (directory / "usage.json").exists() else {}
    trajectory = _trajectory(directory)
    steps = _steps(trajectory)
    final_workspace = _final_workspace(steps)
    product, tests = _patch_files(directory)
    progress = _progress(steps)
    tests_info = _tests(directory, final_workspace)
    decisions = read_json(directory / "policy-decisions.json") if (
        directory / "policy-decisions.json").exists() else []
    stop_reasons = [row.get("decision", {}).get("reason") for row in decisions
                    if row.get("decision", {}).get("kind") == "stop"]
    return {
        "job_id": job["job_id"], "job_phase": job["phase"], "variant": job["variant"],
        "repeat": job["repeat"], "instance_id": job["instance_id"],
        "generation_status": generation.get("generation_status"),
        "finish_reason": generation.get("finish_reason"),
        "error_kind": generation.get("error_kind"), "error_type": generation.get("error_type"),
        "local_grading": grading.get("local_grading"), "local_resolved": grading.get("local_resolved"),
        "patch_applicable": grading.get("patch_applicable"),
        "original_failure_reproduced": grading.get("original_baseline", {}).get(
            "original_failure_reproduced"),
        "tests_status": grading.get("tests_status", {}),
        "product_files": product, "test_files": tests,
        "has_product_patch": bool(product), "final_workspace": final_workspace,
        "tests": tests_info, "progress": progress,
        "policy_decisions": len(decisions), "policy_stop_reasons": stop_reasons,
        "generation_seconds": generation.get("generation_seconds"),
        "usage": usage,
        "reference_cost": usage.get("reference_cost"),
        "cost_complete": usage.get("cost_complete"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_tokens", 0),
        "policy_metrics": generation.get("policy_metrics", {}),
        "evidence": {name: str((directory / name).relative_to(manifest["_output"]))
                     for name in ("generation.json", "grading.json", "usage.json",
                                  "trajectory.json", "patch.diff")
                     if (directory / name).exists()},
    }


def _ftp_failures(facts):
    status = facts["tests_status"] or {}
    ftp = [name for name, value in (status.get("FAIL_TO_PASS") or {}).items()
           if value in {"FAILED", "ERROR"}]
    ptp = [name for name, value in (status.get("PASS_TO_PASS") or {}).items()
           if value in {"FAILED", "ERROR"}]
    return ftp, ptp


def label_facts(facts, registered=None):
    """Deterministic §11.4 mapping; one primary label with trace evidence."""
    registered = registered or {}
    labels = []
    evidence = [f"runs/{facts['job_id']}/{name}" for name in facts["evidence"]]
    grading = facts["local_grading"]
    if grading == "environment_error":
        labels.append(("environment", f"independent grading ended {grading}", evidence))
    elif facts["generation_status"] and facts["generation_status"] != "completed":
        labels.append(("model_protocol",
                       f"generation ended {facts['generation_status']} "
                       f"({facts.get('error_kind')}/{facts.get('error_type')})", evidence))
    elif facts["finish_reason"] == "max_iterations":
        labels.append(("budget", "generation stopped at the 40-round soft budget", evidence))
    elif any(reason in {"budget_reserve", "unknown_cost"} for reason in facts["policy_stop_reasons"]):
        labels.append(("budget", "action policy stopped on budget reserve/unknown cost", evidence))
    elif facts["local_resolved"] is not True and not facts["has_product_patch"]:
        labels.append(("context_omission", "no non-test product file was changed", evidence))
    elif (facts["local_resolved"] is not True and facts["test_files"]
          and facts["tests"]["base_public_requests"] == 0):
        labels.append(("test_timing",
                       "patch modifies test files and no base_public verification was requested",
                       evidence))
    elif facts["local_resolved"] is not True and (
            facts["progress"]["max_no_gain_streak"] >= 2
            or facts["progress"]["no_gain_rate"] is not None
            and facts["progress"]["no_gain_rate"] >= 0.5):
        labels.append(("invalid_repeat",
                       f"no-gain actions {facts['progress']['no_gain_steps']}/"
                       f"{facts['progress']['actual_steps']}, longest streak "
                       f"{facts['progress']['max_no_gain_streak']}", evidence))
    elif facts["local_resolved"] is not True and facts["finish_reason"] == "terminal" \
            and not facts["tests"]["effective_verification"]:
        labels.append(("unverified_finish",
                       "terminal finish without an effective passing test bound to the final workspace",
                       evidence))
    elif facts["local_resolved"] is not True:
        ftp, ptp = _ftp_failures(facts)
        if facts["patch_applicable"] is False:
            labels.append(("edit_error", "candidate patch was not applicable", evidence))
        elif ptp:
            labels.append(("edit_error", f"PASS_TO_PASS regressions: {ptp[:5]}", evidence))
        elif ftp:
            labels.append(("edit_error", f"FAIL_TO_PASS still failing: {ftp[:5]}", evidence))
        elif facts.get("original_failure_reproduced") is False:
            labels.append(("environment", "original FAIL_TO_PASS failures were not reproduced",
                           evidence))
        else:
            labels.append(("unknown", "no deterministic failure rule matched", evidence))
    if not labels and facts["local_resolved"] is None:
        labels.append(("environment", "execution ended without a business verdict", evidence))
    return [{"category": category, "primary": True, "facts": text, "evidence": paths[:12]}
            for category, text, paths in labels]


def label(output, phase):
    output = Path(output)
    manifest = load_manifest(output)
    manifest["_output"] = str(output)
    root = root_of(__file__)
    registered = {}
    for name in ("anomaly-jobs.json", "env-blocked-jobs.json"):
        path = output / name
        if path.exists():
            for identity, entry in read_json(path).items():
                if entry.get("phase") == phase:
                    registered[identity] = entry
    destination = output / "analysis/t7" if phase == "dev" else output / "analysis/t7/holdout"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in read_json(output / "jobs.json"):
        if job["phase"] != phase:
            continue
        directory = output / "runs" / job["job_id"]
        if not (directory / "answer.json").exists():
            continue
        answer = verify_answer(directory, job)
        facts = facts_for(manifest, job)
        labels = label_facts(facts, registered.get(job["instance_id"]))
        write_json(directory / "labels.json", {
            "answer_hash": answer["answer_hash"], "job_id": job["job_id"],
            "instance_id": job["instance_id"], "variant": job["variant"],
            "resolved": facts["local_resolved"], "labels": labels})
        rows.append({**facts, "answer_hash": answer["answer_hash"], "labels": labels})
    write_json(destination / "facts.json", {"manifest_hash": manifest["manifest_hash"],
                                            "phase": phase, "rows": rows})
    with (destination / "facts.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (destination / "labels.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps({key: row[key] for key in (
                "job_id", "instance_id", "variant", "resolved", "labels",
                "generation_status", "finish_reason", "local_grading")}, ensure_ascii=False) + "\n")
    return {"phase": phase, "labeled": len(rows), "output": str(destination)}


def _percentile(values, quantile):
    values = sorted(value for value in values if isinstance(value, (int, float)))
    if not values:
        return None
    position = (len(values) - 1) * quantile / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _distribution(values):
    return {"n": len(values), "p50": _percentile(values, 50), "p95": _percentile(values, 95),
            "mean": sum(values) / len(values) if values else None}


def _variant_stats(rows, phase):
    result = {}
    for variant in (T7_DEV_VARIANTS if phase == "dev" else T7_HOLDOUT_VARIANTS):
        subset = [row for row in rows if row["variant"] == variant]
        if not subset:
            continue
        attempted = [row for row in subset if row["generation_status"] in {"completed", "failed"}]
        graded = [row for row in subset if row["local_grading"] in {"completed", "patch_error"}]
        resolved = sum(row["local_resolved"] is True for row in subset)
        costs = [row["reference_cost"] for row in subset if row["reference_cost"] is not None]
        complete = all(row["cost_complete"] is True for row in subset) and len(costs) == len(subset)
        walls = [row["generation_seconds"] for row in attempted
                 if isinstance(row["generation_seconds"], (int, float))]
        verifications = [row["tests"]["effective_verification"] for row in attempted]
        progress = [row["progress"]["no_gain_rate"] for row in attempted
                    if row["progress"]["no_gain_rate"] is not None]
        result[variant] = {
            "selected": len(subset), "attempted": len(attempted), "generated": len(attempted),
            "graded": len(graded), "resolved": resolved,
            "resolved_over_selected": resolved / len(subset) if subset else None,
            "known_reference_cost": sum(costs) if costs else None,
            "cost_complete": complete,
            "cost_per_resolved": (sum(costs) / resolved if resolved and complete else None),
            "effective_verification_rate": (sum(verifications) / len(verifications)
                                            if verifications else None),
            "no_progress_action_rate": (sum(progress) / len(progress) if progress else None),
            "finish_reasons": dict(Counter(row["finish_reason"] for row in subset if row["finish_reason"])),
            "budget_stops": sum(row["finish_reason"] == "max_iterations" for row in subset),
            "policy_stops": sum(row["finish_reason"] == "policy_stop" for row in subset),
            "policy_interventions": sum(row["policy_metrics"].get("interventions", 0) for row in subset),
            "skipped_proposals": sum(row["policy_metrics"].get("skipped_proposals", 0) for row in subset),
            "prompt_tokens": sum(row["prompt_tokens"] for row in subset),
            "completion_tokens": sum(row["completion_tokens"] for row in subset),
            "cache_read_tokens": sum(row["cache_read_tokens"] for row in subset),
            "cost_distribution": _distribution(costs),
            "wall_seconds_distribution": _distribution(walls),
            "failure_labels": dict(Counter(
                label["category"] for row in subset if row["local_resolved"] is not True
                for label in row["labels"] if label.get("primary"))),
        }
    return result


def _pairs(rows, phase, lock=None):
    by_instance = {}
    for row in rows:
        by_instance.setdefault(row["instance_id"], {})[row["variant"]] = row["local_resolved"]
    if phase == "dev":
        arms = [variant for variant in T7_DEV_VARIANTS if variant != "a0"]
        baseline = "a0"
    else:
        baseline = lock["control"]
        arms = [lock["winner"]] if lock["winner"] != baseline else []
    output = {}
    for variant in arms:
        transitions = {"success_to_failure": [], "failure_to_success": [],
                       "both_resolved": [], "both_unresolved": [], "not_comparable": []}
        differences = []
        for identity in sorted(by_instance):
            pair = by_instance[identity]
            a, b = pair.get(baseline), pair.get(variant)
            if not isinstance(a, bool) or not isinstance(b, bool):
                transitions["not_comparable"].append(identity)
                continue
            differences.append(int(b) - int(a))
            transitions[("both_resolved" if a and b else "both_unresolved" if not a and not b
                         else "success_to_failure" if a and not b else "failure_to_success")].append(identity)
        bootstrap = []
        if differences:
            rng = random.Random(SEED)
            for _ in range(BOOTSTRAP):
                sample = [rng.choice(differences) for _ in differences]
                bootstrap.append(sum(sample) / len(sample))
        output[variant] = {
            "baseline": baseline, "variant": variant, "paired": len(differences),
            "baseline_resolved": sum(int(by_instance[i][baseline]) for i in by_instance
                                     if isinstance(by_instance[i].get(baseline), bool)
                                     and isinstance(by_instance[i].get(variant), bool)),
            "variant_resolved": sum(int(by_instance[i][variant]) for i in by_instance
                                    if isinstance(by_instance[i].get(baseline), bool)
                                    and isinstance(by_instance[i].get(variant), bool)),
            "difference": sum(differences) / len(differences) if differences else None,
            "bootstrap": {"seed": SEED, "resamples": BOOTSTRAP,
                          "ci95": [_percentile(bootstrap, 2.5), _percentile(bootstrap, 97.5)]
                          if bootstrap else None,
                          "descriptive_only": True},
            "transitions": transitions,
        }
    return output


def _table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(name, "")) for name in columns) + " |")
    return "\n".join(lines)


def _report(manifest, phase, stats, paired, lock=None):
    variants = stats["variants"]
    columns = ["variant", "selected", "graded", "resolved", "resolved_over_selected",
               "known_reference_cost", "cost_complete", "effective_verification_rate",
               "no_progress_action_rate", "budget_stops", "policy_stops"]
    lines = [f"# T7 {phase} report", "",
             f"- manifest: `{manifest['manifest_hash']}`",
             f"- protocol: `{PROTOCOL}`",
             "- official Harness: `not_run`; `official_resolved=null`",
             "- provider bill: not available; costs are frozen-rate project estimates", "",
             "## Variant statistics", "", _table([{"variant": key, **value}
                                                   for key, value in variants.items()], columns),
             "", "## Paired transitions", ""]
    for variant, value in paired.items():
        transitions = value["transitions"]
        lines.append(f"- `{value['baseline']}` vs `{variant}`: "
                     f"+{len(transitions['failure_to_success'])} / "
                     f"-{len(transitions['success_to_failure'])}; "
                     f"difference={value['difference']}; "
                     f"bootstrap 95% CI={value['bootstrap']['ci95']} (descriptive only)")
        if transitions["success_to_failure"]:
            lines.append("  - success -> failure: " + ", ".join(transitions["success_to_failure"]))
        if transitions["failure_to_success"]:
            lines.append("  - failure -> success: " + ", ".join(transitions["failure_to_success"]))
    lines += ["", "## Failure labels", ""]
    for variant, value in variants.items():
        lines.append(f"- `{variant}`: {value['failure_labels']}")
    lines += ["", "## Limitations", "",
              SMALL_SAMPLE_STATEMENT, "",
              "- Diagnostics (no-progress rate, verification rate) are deterministic proxies "
              "computed from public traces; they are not probabilities of task resolution.",
              "- Environment-blocked and anomaly registrations stay in the denominator and are "
              "never counted as resolved.",
              "- Official Harness acceptance remains independent and not_run.", ""]
    if phase == "holdout":
        lines += ["## Holdout integrity", "",
                  f"- lock hash: `{content_hash(lock)}`",
                  f"- locked arms: `{lock['winner']}` and `{lock['control']}`",
                  "- No holdout result was used to choose or revise the lock.", ""]
    return "\n".join(lines)


def analyze(output, phase):
    output = Path(output)
    manifest = load_manifest(output)
    manifest["_output"] = str(output)
    destination = output / "analysis/t7" if phase == "dev" else output / "analysis/t7/holdout"
    facts_path = destination / "facts.json"
    if not facts_path.exists():
        label(output, phase)
    rows = read_json(facts_path)["rows"]
    lock = validate_lock(output, manifest) if phase == "holdout" else None
    stats = {"manifest_hash": manifest["manifest_hash"], "phase": phase,
             "protocol": PROTOCOL, "rows": len(rows),
             "variants": _variant_stats(rows, phase),
             "official_harness": "not_run", "official_resolved": None,
             "analysis_plan_sha256": manifest["t7"]["analysis_plan_sha256"],
             "small_sample_statement": manifest["t7"]["small_sample_statement"]}
    write_json(destination / "stats.json", stats)
    paired = _pairs(rows, phase, lock)
    paired_report = {"manifest_hash": manifest["manifest_hash"], "phase": phase,
                     "seed": SEED, "resamples": BOOTSTRAP, "pairs": paired,
                     "descriptive_only": True,
                     "note": "Paired bootstrap of per-instance resolution differences; not a "
                             "significance test or formal falsification threshold."}
    write_json(destination / "paired.json", paired_report)
    if phase == "dev":
        selection = select_winner(stats)
        improved = [variant for variant in ("a1", "a2", "a3", "a4")
                    if variant not in selection["falsified_variants"]]
        conclusion = ("no policy variant exceeded a0 on dev local_resolved; "
                      "a0 is the degraded single-arm fallback" if not improved else
                      "policy variants exceeding a0: " + ", ".join(improved))
        recommendation = {
            "manifest_hash": manifest["manifest_hash"], "phase": "dev",
            "analysis_plan_sha256": manifest["t7"]["analysis_plan_sha256"],
            "selection_rule": SELECTION_RULE, "falsification_rule": FALSIFICATION_RULE,
            "winner": selection["winner"], "control": "a0",
            "degraded_fallback": selection["degraded_fallback"],
            "falsified_variants": selection["falsified_variants"],
            "ranking": selection["ranking"], "falsification": conclusion,
            "small_sample": SMALL_SAMPLE_STATEMENT,
            "holdout_state": "locked", "official_harness": "not_run",
        }
        write_json(destination / "recommendation.json", recommendation)
    (destination / "report.md").write_text(_report(manifest, phase, stats, paired, lock))
    return {"phase": phase, "stats": str(destination / "stats.json"),
            "paired": str(destination / "paired.json"),
            "recommendation": str(destination / "recommendation.json") if phase == "dev" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["label", "analyze"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["dev", "holdout"], required=True)
    args = parser.parse_args()
    if args.command == "label":
        print(label(args.output, args.phase))
    else:
        print(analyze(args.output, args.phase))


if __name__ == "__main__":
    main()
