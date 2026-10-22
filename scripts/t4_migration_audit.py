"""Save offline historical and same-source migration comparisons without rewriting M0."""

import argparse
import asyncio
import copy
from pathlib import Path

from moatless.evaluation.experiment import content_hash, read_json, sha, write_json
from moatless.flow import AgenticLoop, LangGraphFlow
from scripts.migration_smoke import replay


async def audit(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Use a new audit directory; prior evidence cannot be overwritten")
    output.mkdir(parents=True)
    historical = root / "experiments/baselines/M0"
    summary = {"provider_calls": 0, "official_harness": "not_run", "comparisons": [],
               "historical_manifest_sha256": sha(historical / "manifest.json")}
    for coding in (False, True):
        for managed in (False, True):
            name = ("coding" if coding else "respond") + ("-managed" if managed else "-legacy")
            loop, loop_tests = await replay(AgenticLoop, coding=coding, managed=managed)
            graph, graph_tests = await replay(LangGraphFlow, coding=coding, managed=managed)
            write_json(output / f"{name}-loop.json", loop)
            write_json(output / f"{name}-graph.json", graph)
            write_json(output / f"{name}-pytest.json", {"loop": loop_tests, "graph": graph_tests})
            left, right = copy.deepcopy(loop), copy.deepcopy(graph)
            metadata = []
            if len(left["events"]) != len(right["events"]):
                raise AssertionError("Migration changed event count")
            for index, (a, b) in enumerate(zip(left["events"], right["events"])):
                for field in ("project_id", "trajectory_id"):
                    if a.get(field) != b.get(field):
                        expected = "migration" if field == "project_id" else ("coding" if coding else "respond")
                        if a.get(field) not in (None, expected) or b.get(field) != expected:
                            raise AssertionError("Unexplained event identity change")
                        metadata.append({"event": index, "field": field, "loop": a.get(field), "graph": b.get(field)})
                    a.pop(field, None)
                    b.pop(field, None)
            if left != right:
                raise AssertionError(f"Business replay differs: {name}")
            comparison = {"name": name, "same_source_business_equal": True, "metadata_differences": metadata,
                          "model_calls_each": loop["model_calls"], "normalized_business_hash": content_hash(left)}
            if not managed:
                fixture = historical / ("coding.json" if coding else "respond.json")
                comparison.update(historical_fixture_sha256=sha(fixture), historical_fixture_equal=loop == read_json(fixture))
                if not comparison["historical_fixture_equal"]:
                    raise AssertionError("Legacy replay differs from frozen M0 fixture")
            summary["comparisons"].append(comparison)
    summary["evidence"] = {path.name: sha(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(audit(Path(__file__).resolve().parents[1], args.output))
    print(f"{len(result['comparisons'])} comparisons passed; provider calls=0; official Harness=not_run")


if __name__ == "__main__":
    main()
