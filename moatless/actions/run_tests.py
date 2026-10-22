import logging
from typing import Literal

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
    RewardScaleEntry,
)
from moatless.artifacts.artifact import ArtifactChange
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.testing.schema import TestFile
from moatless.testing.schema import TestResult, TestStatus
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class RunTestsArgs(ActionArguments):
    """
    Run the specified unit tests in the codebase.

    Note: This action requires explicit paths to test files.
    - Only full test file paths are supported (e.g., "tests/auth/test_login.py")
    - Directory paths, file patterns (e.g., "tests/*.py"), and individual test methods are NOT supported
    - Non-existent test files will be reported as errors
    """

    test_files: list[str] = Field(
        ..., description="The list of explicit test file paths to run (must be file paths, not directories or patterns)"
    )
    test_source: Literal["workspace", "base_public"] = Field(
        default="workspace", description="Use current tests, or unchanged public repository tests with only the product patch."
    )

    product_source: Literal["current", "base"] = Field(default="current", description="Product version: current patch or unmodified base for a separate comparison.")
    check_id: str | None = Field(default=None, description="Registered acceptance check ID, when applicable.")

    model_config = ConfigDict(title="RunTests")

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        if self.test_source == "workspace":
            data.pop("test_source", None)  # legacy action arguments remain readable and stable
        if self.product_source == "current":
            data.pop("product_source", None)
        if self.check_id is None:
            data.pop("check_id", None)
        return data

    def model_dump_json(self, **kwargs):
        exclude = kwargs.get("exclude") or set()
        defaults = set()
        if self.test_source == "workspace":
            defaults.add("test_source")
        if self.product_source == "current":
            defaults.add("product_source")
        if self.check_id is None:
            defaults.add("check_id")
        kwargs["exclude"] = {**exclude, **dict.fromkeys(defaults, True)} if isinstance(exclude, dict) else set(exclude) | defaults
        return super().model_dump_json(**kwargs)

    @property
    def log_name(self):
        return f"RunTests({', '.join(self.test_files)})"

    def to_prompt(self):
        return "Running tests for the following files:\n" + "\n".join(f"* {file}" for file in self.test_files)

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Run the tests for our user authentication module",
                action=RunTestsArgs(
                    thoughts="Running authentication tests to verify the login functionality works as expected",
                    test_files=["tests/auth/test_authentication.py", "tests/auth/test_login.py"],
                ),
            ),
            FewShotExample.create(
                user_input="After fixing the data validation bug, I need to make sure the validation tests pass",
                action=RunTestsArgs(
                    thoughts="Running validation tests to confirm that the bug fix resolved the issues with input validation",
                    test_files=["tests/validation/test_input_validator.py"],
                ),
            ),
        ]


class RunTests(Action):
    """
    Runs specified test files in the codebase.

    Important requirements:
    - Only accepts explicit, complete file paths to test files
    - Does not support directories, file patterns, or individual test methods
    - Will validate that test files exist before attempting to run them
    - Will report errors for non-existent files or if directories are specified
    """

    args_schema = RunTestsArgs

    max_output_tokens: int = Field(
        2000,
        description="The maximum number of tokens in the test result output message",
    )
    timeout: float = Field(
        120,
        description="Per-invocation wall-clock timeout in seconds; the job continues with a tool_timeout observation",
    )

    cleanup_grace: float = Field(default=10, gt=0)

    async def initialize(self, workspace: Workspace):
        await super().initialize(workspace)
        if not workspace.runtime:
            raise ValueError("Runtime is not available for RunTests action")

    async def execute(
        self,
        args: RunTestsArgs,
        file_context: FileContext | None = None,
    ) -> Observation:
        """Run one explicit recipe. No implicit source switching or baseline run."""
        if file_context is None:
            raise ValueError("File context must be provided")
        from moatless.actions.test_targets import discover_test_targets, is_verifiable_test_module, is_related_test
        from moatless.message_history.policy import content_hash
        from moatless.repository_context import file_content
        from moatless.verification_inputs import capture_recipe, verification_patch, verification_input, declared_inputs
        from moatless.runtime.deadline import invoke
        from datetime import datetime, timezone
        import time
        from subprocess import SubprocessError

        targets = list(dict.fromkeys(args.test_files))
        invalid = [path for path in targets if "::" in path or not is_verifiable_test_module(
            _TestSource(self._repository, file_context, args.test_source), path)]
        if not targets or invalid:
            return Observation.create(message="Specify existing Python test modules; invalid targets: " + str(invalid),
                properties={"fail_reason": "no_test_files", "execution_status": "invalid_args",
                            "test_results": [], "candidates": discover_test_targets(self._repository, [])[:5]})
        product = file_context.generate_git_patch(ignore_tests=True)
        full = file_context.generate_git_patch()
        try:
            declared = declared_inputs(self._workspace, targets)
            recipe = capture_recipe(self._workspace, file_context, targets, args.test_source, args.product_source)
        except (OSError, ValueError, SubprocessError) as exc:
            return Observation.create("Verification inputs unavailable: " + str(exc)[:500],
                properties={"fail_reason": "environment_error", "execution_status": "environment_error", "complete": False})
        origins = {}
        for path in targets:
            file = file_context._files.get(path)
            base = file.get_base_content() if file else self._repository.get_file_content(path)
            current = file_content(self._workspace, file_context, path)
            origins[path] = "new" if not base else ("modified" if current != base else "original")
            if args.test_source == "base_public" and not base:
                return Observation.create(message="base_public requires original public tests.",
                    properties={"fail_reason": "no_test_files", "execution_status": "invalid_args"})
        # The runtime starts from base. A baseline with a workspace repro retains
        # exactly the same test changes, but excludes the product changes.
        if args.product_source == "base":
            patch = "" if args.test_source == "base_public" else verification_patch(file_context, declared)
        else:
            patch = ("\n".join(f.patch for f in file_context.files if f.patch and not verification_input(f.file_path, declared))
                     if args.test_source == "base_public" else full)
        tested_product = "" if args.product_source == "base" else product
        props = {**recipe, "product_patch_hash": content_hash(tested_product), "input_identity": content_hash(recipe),
                 "tested_patch_hash": content_hash(patch), "check_id": args.check_id, "test_origins": origins,
                 "introduced_failures": None, "public_baseline": {"status": "unknown", "reason": "baseline_not_run"},
                 "test_files": targets, "runner": type(self._workspace.runtime).__qualname__,
                 "product_files": [f.file_path for f in file_context.files if f.patch and not verification_input(f.file_path, declared)]}
        props["related_targets"] = {path: is_related_test(_TestSource(self._repository, file_context, args.test_source),
                                                       path, props["product_files"]) for path in targets} if props["product_files"] else {}
        runtime = self._workspace.runtime
        # Clear invocation buffers before starting, never reuse previous results.
        for attr in ("last_test_execution", "last_test_results"):
            if hasattr(runtime, attr):
                setattr(runtime, attr, [])
        timed_out = False
        started_at, started = datetime.now(timezone.utc).isoformat(), time.monotonic()
        try:
            kwargs = {"test_files": targets, "patch": patch}
            if getattr(runtime, "supports_managed_deadline", False) is True:
                kwargs["managed_deadline"] = True
            results, timed_out = await invoke(runtime.run_tests(**kwargs), timeout=self.timeout, cleanup_grace=self.cleanup_grace)
            if timed_out:
                results = list(getattr(runtime, "last_test_results", []) or [])
        except (OSError, RuntimeError) as exc:
            return Observation.create(message=f"Test environment error: {type(exc).__name__}: {str(exc)[:800]}",
                properties={**props, "fail_reason": "environment_error", "execution_status": "environment_error",
                            "complete": False, "result_statuses": [], "result_counts": {}})
        executions = list(getattr(runtime, "last_test_execution", []) or [])
        statuses = [result.status.value.lower() for result in results]
        codes = [entry.get("return_code") for entry in executions]
        code = next((value for value in codes if value != 0), 0) if codes else None
        timed_out = timed_out or any(r.timed_out for r in results)
        complete = not timed_out and set(targets) <= {r.file_path for r in results}
        counts = {value: statuses.count(value) for value in set(statuses)}
        binding_error = None
        try:
            after_recipe = capture_recipe(self._workspace, file_context, targets, args.test_source, args.product_source)
        except (OSError, ValueError, SubprocessError) as exc:
            after_recipe, binding_error = None, str(exc)[:500]
        props.update(result_statuses=statuses, result_counts=counts, result_count=len(results),
            test_results=[r.model_dump(mode="json") for r in results], test_execution=executions,
            return_code=code, timed_out=timed_out, complete=complete,
            patch_stale=(recipe != after_recipe),
            execution_status="timeout" if timed_out else "ok")
        props.update(started_at=started_at, duration=time.monotonic() - started,
                     commands=[entry.get("command") for entry in executions], partial=not complete,
                     uncompleted_targets=sorted(set(targets) - {r.file_path for r in results if not r.timed_out}))
        props["input_identity_after"] = content_hash(after_recipe) if after_recipe is not None else None
        if binding_error:
            props.update(fail_reason="environment_error", execution_status="environment_error", complete=False,
                         input_binding_error=binding_error)
        props["continuation_safe"] = getattr(runtime, "last_test_cleanup_confirmed", None) if timed_out else True
        if timed_out:
            props["fail_reason"] = "tool_timeout"
        if any(value in {"new", "modified"} for value in origins.values()) and args.test_source == "workspace":
            props["requires_base_public"] = True
        files = self.create_test_files(targets, results)
        file_context.add_test_files(files)
        summary = "\n" + TestFile.get_test_summary(files)
        details = TestFile.get_test_failure_details(files)
        message = ("Test invocation timed out; results are partial. " if timed_out else "") + ("\n" + details if details else "") + summary
        artifacts = [ArtifactChange(artifact_id=file.file_path, artifact_type="test", change_type="updated",
            properties={"passed": sum(r.status == TestStatus.PASSED for r in file.test_results),
                        "failed": sum(r.status == TestStatus.FAILED for r in file.test_results),
                        "errors": sum(r.status == TestStatus.ERROR for r in file.test_results),
                        "skipped": sum(r.status == TestStatus.SKIPPED for r in file.test_results),
                        "total": len(file.test_results)}, actor="assistant") for file in files]
        return Observation(message=message, summary=summary, properties=props, artifact_changes=artifacts)

    def create_test_files(self, test_file_paths: list[str], test_results: list[TestResult]) -> list[TestFile]:
        results_by_file: dict[str | None, list[TestResult]] = {}
        for result in test_results:
            results_by_file.setdefault(result.file_path, []).append(result)

        test_files = []

        # Create TestFile objects for all groups of results with a valid file_path
        for file_path, file_test_results in results_by_file.items():
            if file_path is None:
                # Handle results with no file path separately
                num_results_no_path = len(file_test_results)
                if num_results_no_path > 0:
                    logger.warning(
                        f"Received {num_results_no_path} test results with no associated file path. These cannot be added to the TestFile list."
                    )
                continue

            # Create TestFile for results with a valid file_path
            test_file = TestFile(file_path=file_path, test_results=file_test_results)

            if file_test_results:
                # Log counts for this group
                logger.info(
                    f"Test results for {file_path}: "
                    f"{sum(1 for r in file_test_results if r.status == TestStatus.PASSED)} passed, "
                    f"{sum(1 for r in file_test_results if r.status == TestStatus.FAILED)} failed, "
                    f"{sum(1 for r in file_test_results if r.status == TestStatus.ERROR)} errors, "
                    f"{sum(1 for r in file_test_results if r.status == TestStatus.SKIPPED)} skipped"
                )

                # Log a warning if these results were not for an explicitly requested file
                if file_path not in test_file_paths:
                    logger.warning(f"Received unexpected test results for {file_path}")

            test_files.append(test_file)

        # Ensure TestFile objects exist for requested paths even if they had no results
        processed_paths = {tf.file_path for tf in test_files}
        for test_file_path in test_file_paths:
            if test_file_path not in processed_paths:
                logger.warning(f"No test results found for requested file {test_file_path}")
                test_file = TestFile(file_path=test_file_path, test_results=[])
                test_files.append(test_file)

        return test_files

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> list[str]:
        criteria = [
            "Test Result Evaluation: Analyze test results in conjunction with the proposed code changes.",
            "Test Failures Categorization: Differentiate between minor, foreseeable, and unforeseeable failures.",
            " * Minor, Easily Fixable Failures: Lightly penalize or treat as neutral.",
            " * Foreseeable Failures: Penalize appropriately based on the complexity of the fix.",
            " * Unforeseeable Failures: Penalize very lightly or reward for providing new insights.",
            "Impact of Failures: Consider the overall impact of test failures on the solution's viability.",
            "Iterative Improvement: Encourage fixing minor issues in subsequent iterations.",
            "Explanation Requirement: In your explanation, describe any test failures, their likely causes, and suggest potential next steps.",
        ]
        return criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> list[RewardScaleEntry]:
        return [
            RewardScaleEntry(
                min_value=90,
                max_value=100,
                description="All tests pass successfully, confirming the solution's correctness.",
            ),
            RewardScaleEntry(
                min_value=75,
                max_value=89,
                description="Most tests pass, with minor, easily fixable failures.",
            ),
            RewardScaleEntry(
                min_value=50,
                max_value=74,
                description="Tests have some failures, but they are minor or unforeseeable, and the agent shows understanding in interpreting results.",
            ),
            RewardScaleEntry(
                min_value=25,
                max_value=49,
                description="Tests have noticeable failures; some may have been foreseeable, but the agent can address them with effort.",
            ),
            RewardScaleEntry(
                min_value=0,
                max_value=24,
                description="Tests have significant failures; the agent's interpretation is minimal or incorrect.",
            ),
            RewardScaleEntry(
                min_value=-49,
                max_value=-1,
                description="Tests fail significantly; the agent misinterprets results or shows lack of progress, foreseeable failures are not addressed.",
            ),
            RewardScaleEntry(
                min_value=-100,
                max_value=-50,
                description="The action is counterproductive, demonstrating misunderstanding or causing setbacks, test failures are severe and could have been anticipated.",
            ),
        ]


class _TestSource:
    def __init__(self, repository, context, source):
        self.repository, self.context, self.source = repository, context, source

    def get_file_content(self, path):
        file = self.context._files.get(path)
        if file:
            return file.get_base_content() if self.source == "base_public" else file.content
        return self.repository.get_file_content(path)
