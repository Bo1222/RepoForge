"""Compare actual native Loop/Graph fixture results, preserving raw differences."""

import argparse
from pathlib import Path
import uuid

from moatless.evaluation.experiment import content_hash, read_json, sha, write_json
from moatless.message_history.policy import TokenEstimator


def audit(output, manifest):
    output = Path(output).resolve()
    auditor_hash = sha(Path(__file__))
    if auditor_hash != manifest["source"]["files"]["scripts/native_migration_audit.py"]:
        raise ValueError("Migration auditor differs from the frozen experiment source")
    summaries = sorted(output.glob("contracts-native/*/summary.json"), key=lambda path: path.stat().st_mtime_ns)
    if not summaries or not read_json(summaries[-1]).get("passed"):
        raise ValueError("Latest actual native contracts must pass before migration audit")
    contracts = summaries[-1].parent
    destination = output / "migration-native" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    metadata, comparisons = [], []

    def normalize(value, environments, path=""):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                location = path + "/" + key
                if key in {"start_time", "end_time"} or (key == "duration" and "/test_execution/" in location) or location == "/sdk_seconds":
                    metadata.append({"path": location, "kind": "timing", "value": item})
                    continue
                result[key] = normalize(item, environments, location)
            return result
        if isinstance(value, list):
            return [normalize(item, environments, path + "/" + str(index)) for index, item in enumerate(value)]
        if isinstance(value, str):
            for token in environments:
                value = value.replace(token, "NATIVE_RUN_ENVIRONMENT_IDENTITY")
        return value

    for left, right in (("loop_legacy", "graph_legacy"), ("loop_managed", "a0")):
        raw = {name: read_json(contracts / name / "results/trajectory.json") for name in (left, right)}
        identities = {name: read_json(contracts / name / "results/runtime-ready.json") for name in (left, right)}
        stable = ("execution_backend", "source_id", "dependencies", "python_packages", "index_digest")
        assert {key: identities[left][key] for key in stable} == {key: identities[right][key] for key in stable}
        business, contexts = {}, {}
        for name in (left, right):
            data = raw[name]
            environments = set()
            for node in data["nodes"]:
                for step in node.get("action_steps", []):
                    record = (step.get("observation") or {}).get("properties", {}).get("test_run", {})
                    if record:
                        assert record["environment_before"] == record["environment_after"]
                        environments.add(record["environment_before"])
            assert len(environments) <= 1
            graph = data.get("graph_state")
            if graph:
                assert graph["current_node_id"] == data["nodes"][-1]["node_id"]
                assert graph.get("finish_reason") == "terminal" and not graph.get("error_kind")
            for field in ("trajectory_id", "finish_reason", "graph_state"):
                if field in data:
                    metadata.append({"variant": name, "path": "/" + field, "kind": "control", "value": data.pop(field)})
            assert read_json(contracts / name / "results/generation.json")["generation_status"] == "completed"
            for node in data["nodes"]:
                context = node.get("context_build", {})
                for index, record in enumerate(context.get("prepared_requests", [])):
                    estimator = TokenEstimator(record["request"]["model"])
                    assert record["request_hash"] == content_hash(record["request"])
                    assert record["estimated_input_tokens"] == estimator.count(record["request"])
                    metadata.append({"variant": name, "path": f"/nodes/{node['node_id']}/context_build/{index}",
                                     "kind": "derived_request_metadata", "request_hash": record["request_hash"],
                                     "estimated_input_tokens": record["estimated_input_tokens"]})
                    request = normalize(record["request"], environments)
                    record["request"] = request
                    record["request_hash"] = content_hash(request)
                    record["estimated_input_tokens"] = estimator.count(request)
                    context["estimated_input_tokens"] = record["estimated_input_tokens"]
            business[name] = normalize(data, environments)
            contexts[name] = normalize(read_json(contracts / name / "results/model-contexts.json"), environments)
            metadata.append({"variant": name, "kind": "native_workspace_identity", "identity": identities[name],
                             "environment_versions": sorted(environments)})
            write_json(destination / f"{name}-normalized.json", business[name])
        checks = {"actions_observations_context_equal": business[left] == business[right],
                  "prepared_model_contexts_equal": contexts[left] == contexts[right],
                  "complete_patch_equal": (contracts / left / "results/patch.diff").read_bytes() == (contracts / right / "results/patch.diff").read_bytes(),
                  "usage_and_calls_equal": normalize(read_json(contracts / left / "results/usage.json"), set()) == normalize(read_json(contracts / right / "results/usage.json"), set())}
        comparisons.append({"pair": [left, right], "checks": checks, "passed": all(checks.values())})
    write_json(destination / "metadata.json", metadata)
    summary = {"manifest_hash": manifest["manifest_hash"], "execution_backend": "native", "comparisons": comparisons,
               "auditor_source_sha256": auditor_hash,
               "passed": all(row["passed"] for row in comparisons), "provider_calls": 0, "synthetic_workflow": True,
               "normalization_scope": "timings, native environment identities and independently recomputed request hashes/token estimates; raw files untouched",
               "evidence": {str(path.relative_to(output)): sha(path) for path in (*contracts.rglob("*"), *destination.rglob("*")) if path.is_file()}}
    write_json(destination / "summary.json", summary)
    if not summary["passed"]:
        raise ValueError("Native Loop/Graph migration comparison failed; raw and normalized evidence retained")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.output, read_json(args.output / "manifest.json"))["comparisons"])


if __name__ == "__main__":
    main()
