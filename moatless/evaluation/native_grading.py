"""Independent local-compatible grading after complete answers are frozen."""

import asyncio
from pathlib import Path
import shutil
import time
import uuid

from moatless.evaluation.experiment import content_hash, read_json, sha, verify_answer, write_json
from moatless.evaluation.native_execution import NativeExecutor, NativeMount
from moatless.evaluation.native_provision import NativeProvisioner


class NativeGrader:
    def __init__(self, output, manifest):
        self.output = Path(output).resolve()
        self.manifest = manifest
        self.revision = sha(Path(__file__))

    @staticmethod
    def scm_version(prepared):
        """Preserve the actual prepared package version after future refs are cut."""
        environment = prepared["environment"]
        name = {"pytest-dev/pytest": "pytest", "astropy/astropy": "astropy",
                "scikit-learn/scikit-learn": "scikit-learn", "pydata/xarray": "xarray",
                "matplotlib/matplotlib": "matplotlib"}.get(prepared["public"]["repo"])
        if not name:
            return None
        dependencies = read_json(Path(environment["prefix"]).parent / "dependencies.json")
        return next((package["version"] for package in dependencies if package["name"] == name), None)

    def _private_instance(self, job):
        """This method is reached only after verify_answer; never return gold patch."""
        from datasets import load_dataset
        dataset = load_dataset(self.manifest["dataset"], split="test",
                               cache_dir=str(self.output / "host-dataset-cache"))
        receipt = read_json(self.output / "dataset-receipt.json")
        if receipt["fingerprint"] != dataset._fingerprint:
            raise ValueError("Dataset fingerprint changed before independent grading")
        allowed = ("instance_id", "repo", "version", "base_commit", "environment_setup_commit",
                   "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")
        row = next(row for row in dataset if row["instance_id"] == job["instance_id"])
        return {key: row[key] for key in allowed if key in row}

    @staticmethod
    def eval_script(spec, prefix):
        """Preserve official commands and retain the actual test command's status."""
        from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
        commands = []
        for command in spec.eval_script_list:
            if command.startswith("source /opt/miniconda3/bin/activate") or command == "conda activate testbed":
                commands.append(": native prefix is already on PATH")
            else:
                if START_TEST_OUTPUT in command:
                    commands.append("set +e")
                if END_TEST_OUTPUT in command:
                    commands.append("native_test_exit=$?")
                    commands.append("set -e")
                commands.append(command)
        commands.append('exit "${native_test_exit:-125}"')
        return "#!/bin/bash\nset -euxo pipefail\n" + "\n".join(commands) + "\n"

    def _runner(self, directory, prepared):
        environment = prepared["environment"]
        NativeProvisioner.verify(environment)
        repo, prefix = directory / "repo", Path(environment["prefix"])
        private_prefix = directory / "runtime"
        shutil.copytree(environment["repo"], repo, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        # Never let official evaluation installation mutate a generator's prefix.
        shutil.copytree(prefix, private_prefix, symlinks=True)
        write_json(directory / "editable-relocation.json",
                   NativeProvisioner.relocate_editable(private_prefix, environment["repo"], "/testbed"))
        (directory / "input").mkdir()
        mounts = [NativeMount(repo, "/testbed", True),
                  NativeMount(repo, environment["repo"], True),
                  NativeMount(private_prefix, str(prefix), True),
                  NativeMount(directory / "input", "/input"), NativeMount(directory, "/out", True)]
        wheel_profile = self.manifest.get("native", {}).get("build_wheels")
        if wheel_profile:
            wheels = Path(wheel_profile["path"])
            if sha(wheels / "manifest.json") != wheel_profile["manifest_sha256"]:
                raise ValueError("Frozen offline build wheel manifest changed")
            for name, digest in wheel_profile["manifest"]["files"].items():
                if sha(wheels / name) != digest:
                    raise ValueError("Frozen offline build dependency changed")
            mounts.append(NativeMount(wheels, "/build-wheels"))
        env = {"PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
            "PYTHONPATH": "/testbed/src:/testbed", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        from moatless.evaluation.native_public_tools import runtime_tools, extend_environment
        tool_mounts, tool_environment = runtime_tools(self.manifest.get("native", {}).get("public_tools"), prepared["public"]["repo"])
        mounts.extend(tool_mounts)
        extend_environment(env, tool_environment)
        if tool_mounts and environment.get("public_tools_digest") != self.manifest["native"]["public_tools"]["digest"]:
            raise ValueError("Prepared environment belongs to another public tool profile")
        version = self.scm_version(prepared)
        if wheel_profile:
            env["PIP_FIND_LINKS"] = "/build-wheels"
        if version:
            env["SETUPTOOLS_SCM_PRETEND_VERSION"] = version
        runner = NativeExecutor(mounts, environment=env, network=False)
        return runner

    async def _case(self, directory, prepared, instance, spec, patch):
        directory.mkdir(parents=True, exist_ok=False)
        runner = self._runner(directory, prepared)
        inputs = directory / "input"
        (inputs / "candidate.patch").write_text(patch)
        script = self.eval_script(spec, prepared["environment"]["prefix"])
        (inputs / "eval.sh").write_text(script)
        started = time.monotonic()
        checks = []
        try:
            patch_commands = ["git", "apply", "--check", "/input/candidate.patch"] if patch.strip() else ["/bin/true"]
            check = await runner.run(patch_commands, timeout=30)
            (directory / "patch-check.log").write_text(check.output)
            checks.append({"command": patch_commands, "return_code": check.return_code})
            if check.return_code:
                result = {"status": "patch_error", "patch_applicable": False, "local_resolved": False,
                          "checks": checks, "official_harness": "not_run"}
            else:
                if patch.strip():
                    apply_command = ["git", "apply", "/input/candidate.patch"]
                    applied = await runner.run(apply_command, timeout=30)
                    checks.append({"command": apply_command, "return_code": applied.return_code})
                    (directory / "patch-apply.log").write_text(applied.output)
                    if applied.return_code:
                        raise RuntimeError("Patch changed between independent check and application")
                command = ["/bin/bash", "/input/eval.sh"]
                execution = await runner.run(command, timeout=600)
                (directory / "test-output.txt").write_text(execution.output)
                checks.append({"command": command, "return_code": execution.return_code,
                               "timed_out": execution.timed_out})
                from swebench.harness.grading import get_eval_report, get_logs_eval
                statuses, found = get_logs_eval(spec, str(directory / "test-output.txt"))
                prediction = {"instance_id": instance["instance_id"], "model_name_or_path": "native-local",
                              "model_patch": patch}
                report = get_eval_report(spec, prediction, str(directory / "test-output.txt"), True)
                write_json(directory / "parsed-tests.json", statuses)
                write_json(directory / "compatible-report.json", report)
                complete = found and bool(statuses) and execution.return_code in {0, 1}
                result = {"status": "completed" if complete else "environment_error", "patch_applicable": True,
                    "local_resolved": report[instance["instance_id"]]["resolved"] if complete else None,
                    "tests_status": report[instance["instance_id"]].get("tests_status", {}),
                    "test_statuses": statuses, "checks": checks, "official_harness": "not_run"}
            result.update(seconds=time.monotonic() - started, environment_identity=prepared["environment"]["recipe_identity"],
                          grader_source_sha256=self.revision, scm_version=self.scm_version(prepared),
                          evaluation_protocol=self.manifest["evaluation_protocol"], execution_backend="native",
                          grading_backend="native", official_resolved=None)
            write_json(directory / "case.json", result)
            return result
        finally:
            # These are our disposable independent copies, never generation or
            # user workspaces. Logs, scripts and parsed results remain intact.
            for name in ("repo", "runtime"):
                path = directory / name
                if path.exists():
                    shutil.rmtree(path)

    def baseline(self, job):
        directory = self.output / "runs" / job["job_id"]
        answer = verify_answer(directory, job)  # Must precede any private read.
        instance = self._private_instance(job)
        from swebench.harness.test_spec.test_spec import make_test_spec
        spec = make_test_spec(instance)
        prepared = read_json(self.output / "provision-native.json")[job["instance_id"]]
        identity = content_hash([self.manifest["manifest_hash"], instance, prepared["environment"]["recipe_identity"], self.revision])
        destination = self.output / "grading-native" / "baselines" / identity
        if (destination / "case.json").exists():
            baseline = read_json(destination / "case.json")
        else:
            baseline = asyncio.run(self._case(destination, prepared, instance, spec, ""))
        ftp = spec.FAIL_TO_PASS
        observed = [name for name in ftp if baseline.get("test_statuses", {}).get(name) in {"FAILED", "ERROR"}]
        result = {"answer_hash": answer["answer_hash"], **baseline,
                  "original_failure_reproduced": bool(ftp) and len(observed) == len(ftp),
                  "observed_ftp_failures": observed, "FAIL_TO_PASS": ftp, "PASS_TO_PASS": spec.PASS_TO_PASS,
                  "evidence": {str(p.relative_to(self.output)): sha(p) for p in destination.rglob("*") if p.is_file()}}
        write_json(directory / "original-baseline.json", result)
        return result

    def diagnose(self, instance_id):
        """Baseline-only diagnostic for pre-admission scans.

        Reuses the baseline case runner with an empty patch and never touches
        runs/, frozen answers or any generation state; the caller classifies the
        returned case (see t7.scan_holdout)."""
        instance = self._private_instance({"instance_id": instance_id})
        from swebench.harness.test_spec.test_spec import make_test_spec
        spec = make_test_spec(instance)
        prepared = read_json(self.output / "provision-native.json")[instance_id]
        identity = content_hash([self.manifest["manifest_hash"], instance,
                                 prepared["environment"]["recipe_identity"], self.revision])
        destination = self.output / "holdout-scan" / identity
        if (destination / "case.json").exists():
            case = read_json(destination / "case.json")
        else:
            case = asyncio.run(self._case(destination, prepared, instance, spec, ""))
        ftp = spec.FAIL_TO_PASS
        observed = [name for name in ftp if case.get("test_statuses", {}).get(name) in {"FAILED", "ERROR"}]
        return {"instance_id": instance_id, "case": case, "observed_ftp_failures": observed,
                "FAIL_TO_PASS": ftp, "PASS_TO_PASS": spec.PASS_TO_PASS,
                "original_failure_reproduced": bool(ftp) and len(observed) == len(ftp),
                "evidence": {str(p.relative_to(self.output)): sha(p) for p in destination.rglob("*") if p.is_file()}}

    def grade(self, job):
        directory = self.output / "runs" / job["job_id"]
        answer = verify_answer(directory, job)
        baseline = self.baseline(job)
        instance = self._private_instance(job)
        from swebench.harness.test_spec.test_spec import make_test_spec
        spec = make_test_spec(instance)
        prepared = read_json(self.output / "provision-native.json")[job["instance_id"]]
        destination = self.output / "grading-native" / "candidates" / content_hash([answer["answer_hash"], self.revision])
        if (destination / "case.json").exists():
            candidate = read_json(destination / "case.json")
        else:
            candidate = asyncio.run(self._case(destination, prepared, instance, spec, (directory / "patch.diff").read_text()))
        applicable = candidate["patch_applicable"]
        result = {"answer_hash": answer["answer_hash"], "local_grading": candidate["status"],
            "grader_source_sha256": self.revision,
            "local_resolved": candidate["local_resolved"] if baseline["original_failure_reproduced"] else None,
            "official_resolved": None, "official_harness": "not_run", "grading_backend": "native",
            "evaluation_protocol": self.manifest["evaluation_protocol"], "patch_applicable": applicable,
            "original_baseline": baseline, "tests_status": candidate.get("tests_status", {}),
            "evidence": {str(p.relative_to(self.output)): sha(p) for p in destination.rglob("*") if p.is_file()}}
        if (directory / "grading.json").exists():
            write_json(directory / "grading-attempts" / f"{uuid.uuid4().hex}.json", read_json(directory / "grading.json"))
        write_json(directory / "grading.json", result)
        return result
