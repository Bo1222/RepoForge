"""Public-only native environment pilot; no model generation or grading."""

import argparse
import asyncio
from pathlib import Path

from moatless.evaluation.experiment import read_json, write_json
from moatless.evaluation.experiment_inputs import export_public_dataset
from moatless.evaluation.native_provision import NativeProvisioner
from moatless.evaluation.native_execution import NativeExecutor, NativeMount


async def namespace_probe(output, receipt):
    prefix, repo = Path(receipt["prefix"]), Path(receipt["repo"])
    NativeProvisioner.verify(receipt)
    runner = NativeExecutor([NativeMount(prefix, str(prefix)), NativeMount(repo, "/testbed", True),
                             NativeMount(repo, str(repo), True)],
        environment={"PATH": f"{prefix / 'bin'}:/usr/bin:/bin", "PYTHONPATH": "/testbed/src:/testbed"})
    import_code = ("import json,os,sys,pytest; "
                   "assert os.path.samefile(pytest.__file__, '/testbed/src/pytest/__init__.py'); "
                   "print(json.dumps({'python':sys.version, 'source':pytest.__file__}))")
    rows = []
    for name, argv, timeout in (
        ("namespace-import", [str(prefix / "bin/python"), "-c", import_code], 30),
        ("public-tests", [str(prefix / "bin/python"), "-m", "pytest", "-rA", "testing/test_skipping.py"], 120),
    ):
        result = await runner.run(argv, timeout=timeout)
        (output / f"{name}.log").write_text(result.output)
        rows.append({"command": argv, "return_code": result.return_code,
                     "seconds": result.seconds, "timed_out": result.timed_out, "log": f"{name}.log"})
        write_json(output / "namespace-result.json", {"checks": rows, "provider_calls": 0,
            "official_harness": "not_run", "execution_backend": "native",
            "passed": len(rows) == 2 and all(row["return_code"] == 0 for row in rows)})
        if result.return_code:
            raise RuntimeError(f"Native {name} failed; inspect its preserved command/exit code/log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-id", default="pytest-dev__pytest-7490")
    parser.add_argument("--verify-namespace", action="store_true", help="Check actual imports and public pytest tests inside Bubblewrap")
    args = parser.parse_args()
    manifest = {"dataset": "princeton-nlp/SWE-bench_Verified",
                "smoke_ids": [args.instance_id], "split": {"dev": [args.instance_id]}}
    inventory = export_public_dataset(args.output, manifest, "smoke")
    recipe = read_json(args.output / "host-install-recipes.json").get(args.instance_id, {})
    receipt = asyncio.run(NativeProvisioner(args.output / "environment").prepare(
        inventory[args.instance_id], environment_setup_commit=recipe.get("environment_setup_commit")))
    write_json(args.output / "environment-result.json", receipt)
    if args.verify_namespace:
        if receipt["public"]["repo"] != "pytest-dev/pytest":
            raise ValueError("The initial public-test namespace probe is specific to the pytest first instance")
        asyncio.run(namespace_probe(args.output, receipt))
    print(f"Prepared {args.instance_id}; provider calls=0; official_harness=not_run")


if __name__ == "__main__":
    main()
