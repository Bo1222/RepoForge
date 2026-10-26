import logging
from pathlib import Path
from typing import List, Optional

from opentelemetry import trace

from repoforge.actions.schema import Observation
from repoforge.actions.action import Action
from repoforge.file_context import FileContext
from repoforge.actions.schema import RewardScaleEntry

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


class CodeModificationMixin:
    """
    A mixin that provides common functionality for actions that modify code files.
    This includes path normalization, file validation, test running, and observation handling.
    """

    def normalize_path(self, file_path: str) -> str:
        """Normalize file path by removing /repo and leading /"""
        if file_path.startswith("/repo"):
            file_path = file_path[5:]
        if file_path.startswith("/"):
            file_path = file_path[1:]
        return file_path

    def validate_file_access(
        self, file_path: str, file_context: FileContext, focus_text: str | None = None
    ) -> tuple[Optional[Path], Optional[Observation]]:
        """
        Validate file access and return either a valid Path object or an error Observation.

        If the file exists but has not been viewed yet, load it into the context
        automatically.  For very large files, load only the span containing
        ``focus_text`` so the edit does not spend the whole context budget on an
        unrelated file.
        """
        path = Path(file_path)

        if not file_context.file_exists(str(path)):
            from difflib import get_close_matches
            from repoforge.repository_overlay import repository_paths
            nearest = get_close_matches(str(path), repository_paths(file_context.repository), n=5, cutoff=0.2)
            return None, Observation(
                message=f"File {path} not found. Nearest paths: {nearest}",
                properties={"fail_reason": "file_not_found", "nearest_paths": nearest},
            )  # type: ignore

        if not file_context.has_file(str(path)):
            try:
                content = file_context.repository.get_file_content(str(path))
            except (OSError, ValueError):
                content = None
            loaded = False
            if content:
                if focus_text and len(content) > 50_000:
                    index = content.find(focus_text)
                    if index >= 0:
                        start = content.count("\n", 0, index) + 1
                        end = start + focus_text.count("\n")
                        file_context.add_line_span_to_context(
                            str(path), max(1, start - 20), end + 20
                        )
                        loaded = True
                if not loaded and len(content) <= 200_000:
                    file_context.add_file(str(path), show_all_spans=True)
                    loaded = True
            if not loaded:
                return None, Observation(
                    message=f"You have not yet viewed the file {path}. Use ViewCode to view the parts of the file that you want to modify.",
                    properties={"fail_reason": "file_not_in_context"},
                )  # type: ignore

        return path, None

    def validate_python_syntax(
        self, file_path: str, old_content: str, new_content: str
    ) -> Optional[Observation]:
        """Reject a Python edit that turns a valid module into invalid syntax.

        If the file was already invalid, the edit is allowed so the agent can
        repair it incrementally.
        """
        if not str(file_path).endswith(".py"):
            return None
        try:
            compile(old_content, str(file_path), "exec", dont_inherit=True)
        except SyntaxError:
            return None
        try:
            compile(new_content, str(file_path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            line = exc.lineno or 1
            lines = new_content.splitlines()
            start = max(0, line - 3)
            snippet = "\n".join(lines[start:line + 2])
            return Observation.create(
                message=(
                    f"The edit was not applied: it introduces a Python syntax error at "
                    f"line {line}, column {exc.offset or 1}: {exc.msg}.\n"
                    f"```\n{snippet}\n```\n"
                    "Fix the replacement so the file parses before retrying."
                ),
                properties={
                    "fail_reason": "syntax_error",
                    "syntax_error": {"line": line, "column": exc.offset, "message": exc.msg},
                },
            )
        return None

    def edit_token(self, context, path):
        from repoforge.edit_guard import EditConflict, source_version
        try:
            return source_version(context.repository, str(path), context._files.get(str(path))), None
        except EditConflict as exc:
            return None, Observation.create(str(exc), properties={"fail_reason": "edit_conflict"})

    def commit_candidate(self, file, content, token):
        from repoforge.edit_guard import EditConflict
        try:
            file.apply_changes(content, expected_source_version=token)
        except EditConflict as exc:
            return Observation.create(str(exc), properties={"fail_reason": "edit_conflict", "execution_status": "invalid_args"})
        return None

    async def validate_candidate(self, path, old_content, candidate):
        from repoforge.runtime.deadline import invoke
        from repoforge.message_history.policy import content_hash
        from repoforge.execution_memory import environment_version
        from repoforge.runtime.runtime import RuntimeEnvironment
        if not str(path).endswith(".py"):
            return None, {"status": "not_applicable", "path": str(path)}
        runtime = self._workspace.runtime if self._workspace else None
        environment_before = environment_version(self._workspace) if self._workspace else None
        check = {"status": "unavailable", "reason": "target_interpreter_not_configured"}
        if isinstance(runtime, RuntimeEnvironment):
            try:
                result, timeout = await invoke(runtime.compile_python(candidate, str(path)), timeout=10, cleanup_grace=2)
                check = result if not timeout else {"status": "unavailable", "reason": "compiler_timeout"}
            except (OSError, ValueError) as exc:
                check = {"status": "unavailable", "reason": type(exc).__name__}
        check = {**check, "path": str(path), "content_hash": content_hash(candidate),
                 "environment": environment_version(self._workspace) if self._workspace else None}
        if check["environment"] != environment_before:
            check.update(status="unavailable", reason="compiler_environment_changed")
        if check["status"] == "invalid":
            old, timeout = await invoke(runtime.compile_python(old_content, str(path)), timeout=10, cleanup_grace=2)
            if not timeout and old.get("status") == "invalid":
                return None, check  # Allow incremental repair; finalization still requires valid compilation.
            return Observation.create("Candidate not applied: target Python compilation failed. " + str(check.get("message")),
                properties={"fail_reason": "syntax_error", "syntax_error": check, "syntax_checks": [check]}), check
        if check["status"] == "unavailable":
            # Preserve useful host diagnostics without claiming target compatibility.
            diagnostic = self.validate_python_syntax(str(path), old_content, candidate)
            if diagnostic:
                diagnostic.properties["syntax_checks"] = [check]
                diagnostic.properties["syntax_checked"] = False
                return diagnostic, check
        return None, check

    def format_snippet_with_lines(self, snippet: str, start_line: int) -> str:
        """Format a code snippet with line numbers"""
        return "\n".join(f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet.split("\n")))

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        criteria = Action.get_evaluation_criteria(trajectory_length)
        criteria.extend(
            [
                "Instruction Clarity: Ensure that instructions and pseudocode are clear and actionable.",
                "Instruction Compliance: The git diff must *exactly* implement the provided pseudo_code. Identify any discrepancies, omissions, or additions. If discrepancies exist, you should lower the reward accordingly.",
                "Code Modification Accuracy and Quality: Check for correct identification of code spans, accuracy of changes, syntax errors, logical flaws, unintended modifications, and unintended side effects.",
                "Python-Specific Features Utilization: Assess whether the agent has appropriately utilized Python-specific features that enhance the solution.",
                "Common Git Diff Issues and Unintended Changes: Check for issues such as incorrect line numbers, unintended additions or deletions, formatting errors, changes to unrelated parts of the code, and heavily penalize unintended changes.",
                "Addressing Test Failures: Verify if the agent is properly addressing test failures from previous `RunTests` actions.",
            ]
        )
        return criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return Action.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "The code change is optimal, with a perfect Git diff exactly matching the pseudo code, and requires no further changes.",
                ),
                (
                    75,
                    89,
                    "The code change significantly advances the solution, with an accurate Git diff exactly matching the pseudo code,.",
                ),
                (
                    50,
                    74,
                    "The code change is mostly correct but has minor issues or opportunities for optimization; the Git diff exactly matching the pseudo code,.",
                ),
                (
                    25,
                    49,
                    "The code change is acceptable but has noticeable issues or is less effective than possible alternatives;",
                ),
                (
                    0,
                    24,
                    "The code change has minimal impact or introduces minor negative consequences",
                ),
                (
                    -49,
                    -1,
                    "The code change is inappropriate, unhelpful, or introduces new issues; the action did not result in any successful code changes. The Git diff does not match the pseud code and instructions, contains significant inaccuracies or shows no changes. Penalize attempts to modify non-existent code elements (hallucinations) based on severity.",
                ),
                (
                    -100,
                    -50,
                    "The code change is counterproductive, causing significant setbacks or demonstrating persistent repetition without learning. The Git diff is severely flawed or indicates that no effective changes were made. Heavily penalize severe hallucinations or continuous attempts to modify non-existent code elements.",
                ),
            ]
        )
