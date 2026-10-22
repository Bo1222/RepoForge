#!/usr/bin/env python3
"""Compare official SWE-bench Harness results with frozen T7 local results.

Inputs are the artifacts produced by export_official.py (id-manifest.json,
hash-manifest.json) and either:
  * a directory containing official per-instance report.json files
    (logs/run_evaluation/<run_id>/<model>/<instance_id>/report.json), or
  * an official aggregate report.json with resolved_ids/unresolved_ids, or
  * a JSONL file with instance_id, model_name_or_path, resolved.

The tool is read-only for the experiment.  Official results never change the
lock, winner, selected task IDs or local_resolved values.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = REPO_ROOT / "analysis-tools/artifacts"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def resolve(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("resolved", "is_resolved", "success"):
            if key in value:
                return resolve(value[key])
    if isinstance(value, str):
        return value.lower() in {"true", "resolved", "pass", "passed"}
    return None


def parse_official(path):
    """Return {(model, instance_id): {'resolved': bool|None, 'status': str}}."""
    path = Path(path)
    results = {}
    if not path.exists():
        return results, {"source": str(path), "format": "missing"}
    if path.is_file() and path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            model = row.get("model_name_or_path") or row.get("model") or ""
            instance = row.get("instance_id")
            if instance:
                results[(model, instance)] = {
                    "resolved": resolve(row.get("resolved", row)),
                    "status": row.get("status", "jsonl")}
        return results, {"source": str(path), "format": "jsonl"}

    report_files = []
    if path.is_dir():
        report_files = sorted(path.rglob("report.json"))
        # Prefer per-instance reports; an aggregate report has resolved_ids.
        candidates = []
        for report_file in report_files:
            data = read_json(report_file, {})
            if isinstance(data, dict) and "resolved_ids" in data:
                candidates.append((report_file, data, "aggregate"))
            elif isinstance(data, dict) and len(data) == 1 and isinstance(
                    next(iter(data.values())), dict):
                candidates.append((report_file, data, "instance"))
    else:
        data = read_json(path, {})
        candidates = [(path, data, "single")]

    for report_file, data, kind in candidates:
        if kind == "aggregate" or "resolved_ids" in data:
            model = report_file.parent.name if kind == "aggregate" else ""
            resolved = set(data.get("resolved_ids") or [])
            unresolved = set(data.get("unresolved_ids") or [])
            errors = set(data.get("error_ids") or [])
            empty = set(data.get("empty_patch_ids") or [])
            for instance in resolved:
                results[(model, instance)] = {"resolved": True, "status": "resolved"}
            for instance in unresolved:
                results[(model, instance)] = {"resolved": False, "status": "unresolved"}
            for instance in errors:
                results[(model, instance)] = {"resolved": None, "status": "error"}
            for instance in empty:
                results[(model, instance)] = {"resolved": False, "status": "empty_patch"}
            continue
        # Per-instance report: {instance_id: {"resolved": bool, ...}}
        if len(data) == 1:
            instance, row = next(iter(data.items()))
            model = ""
            parts = report_file.parts
            if len(parts) >= 3 and parts[-3] not in {"logs", "run_evaluation"}:
                model = parts[-3]
            results[(model, instance)] = {
                "resolved": resolve(row), "status": "official_report"}
        else:
            for instance, row in data.items():
                if isinstance(row, dict) and ("resolved" in row or "tests_status" in row):
                    results[("", instance)] = {
                        "resolved": resolve(row), "status": "official_report"}
    return results, {"source": str(path), "format": "directory" if path.is_dir() else "json"}


def pick(results, model, instance):
    if (model, instance) in results:
        return results[(model, instance)]
    matches = [value for (candidate_model, candidate_instance), value in results.items()
               if candidate_instance == instance and candidate_model in {"", model}
               or candidate_instance == instance and candidate_model.endswith(model)]
    if len(matches) == 1:
        return matches[0]
    return None


def build(candidate, artifacts, official, phase_filter):
    artifacts = Path(artifacts).resolve()
    hash_manifest = read_json(artifacts / "hash-manifest.json")
    id_manifest = read_json(artifacts / "id-manifest.json")
    if not hash_manifest or not id_manifest:
        raise SystemExit("Run export_official.py first")
    results, source = parse_official(official) if official else ({}, {"format": "not_provided"})
    rows = []
    for row in hash_manifest["rows"]:
        phase = row["phase"]
        if phase_filter != "all" and phase != phase_filter:
            continue
        model = f"moatless-t7-{phase}-{row['variant']}"
        official_row = pick(results, model, row["instance_id"])
        official_resolved = official_row.get("resolved") if official_row else None
        official_status = official_row.get("status") if official_row else "not_run"
        local = row.get("local_resolved")
        if not row.get("patch_present"):
            comparison = "no_patch"
        elif official_row is None:
            comparison = "official_not_run"
        elif isinstance(local, bool) and isinstance(official_resolved, bool):
            if local and official_resolved:
                comparison = "both_resolved"
            elif local and not official_resolved:
                comparison = "local_only"
            elif not local and official_resolved:
                comparison = "official_only"
            else:
                comparison = "both_unresolved"
        elif isinstance(official_resolved, bool):
            comparison = "official_only_local_null" if official_resolved else "both_unresolved_local_null"
        else:
            comparison = "official_error" if official_status != "not_run" else "official_not_run"
        rows.append({**row, "model_name_or_path": model,
                     "official_resolved": official_resolved,
                     "official_status": official_status,
                     "comparison": comparison})
    expected = {(row["model_name_or_path"], row["instance_id"]) for row in rows}
    unmatched = sorted({f"{model}:{instance}" for (model, instance) in results
                        if (model, instance) not in expected and
                        not any(candidate_instance == instance
                                for _, candidate_instance in expected)})
    by_arm = {}
    for phase in ("dev", "holdout"):
        for arm in ("a0", "a1", "a2", "a3", "a4"):
            subset = [row for row in rows if row["phase"] == phase and row["variant"] == arm]
            if not subset:
                continue
            by_arm[f"{phase}-{arm}"] = {
                "rows": len(subset),
                "patches": sum(row["patch_present"] for row in subset),
                "local_resolved": sum(row.get("local_resolved") is True for row in subset),
                "official_resolved": sum(row.get("official_resolved") is True for row in subset),
                "official_not_run": sum(row["comparison"] == "official_not_run" for row in subset),
                "local_only": sum(row["comparison"] == "local_only" for row in subset),
                "official_only": sum(row["comparison"] == "official_only" for row in subset),
                "both_resolved": sum(row["comparison"] == "both_resolved" for row in subset),
                "both_unresolved": sum(row["comparison"] == "both_unresolved" for row in subset),
                "empty_or_missing": sum(not row["patch_present"] or
                                        row.get("official_status") == "empty_patch"
                                        for row in subset),
            }
    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "candidate": str(candidate),
        "manifest_hash": hash_manifest["manifest_hash"],
        "lock_hash": hash_manifest.get("lock_hash"),
        "official_source": source,
        "official_harness": "completed" if results else "not_run",
        "note": "Official results are reported separately and never modify the frozen lock, task IDs or local_resolved.",
        "by_arm": by_arm,
        "unmatched_official_ids": unmatched,
        "rows": rows,
    }
    out = artifacts / "report"
    write_json(out / "official-comparison.json", report)
    with (out / "official-rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = ["# Official SWE-bench Harness comparison", "",
             f"- candidate: `{candidate}`",
             f"- manifest: `{hash_manifest['manifest_hash']}`",
             f"- lock: `{hash_manifest.get('lock_hash')}`",
             f"- official source: `{source.get('source')}` ({source.get('format')})",
             f"- official Harness: `{report['official_harness']}`", "",
             "| arm | rows | patches | local resolved | official resolved | local only | official only | both | official not_run |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, value in sorted(by_arm.items()):
        lines.append(f"| {arm} | {value['rows']} | {value['patches']} | "
                     f"{value['local_resolved']} | {value['official_resolved']} | "
                     f"{value['local_only']} | {value['official_only']} | "
                     f"{value['both_resolved']} | {value['official_not_run']} |")
    lines += ["", "Official results never change the pre-registered lock or task IDs.",
              "Environment-blocked instances with no patch stay in the denominator and "
              "are reported as missing/empty, not as resolved.", ""]
    (out / "official-report.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--official", type=Path, default=None,
                        help="Official report directory or JSON/JSONL file; omit for not_run")
    parser.add_argument("--phase", choices=["dev", "holdout", "all"], default="all")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    artifacts = (args.artifacts or DEFAULT_ARTIFACTS / candidate.name).resolve()
    report = build(candidate, artifacts, args.official, args.phase)
    print(json.dumps({"official_harness": report["official_harness"],
                      "by_arm": report["by_arm"],
                      "unmatched": report["unmatched_official_ids"],
                      "report": str(artifacts / "report/official-report.md")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
