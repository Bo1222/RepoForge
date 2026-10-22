import logging
import os
import re
from fnmatch import fnmatch
from typing import ClassVar, Optional

from pydantic import Field, model_validator

from moatless.actions.schema import ActionArguments
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.index.hybrid import fuse_hits

logger = logging.getLogger(__name__)


class FindCodeSnippetArgs(SearchBaseArgs):
    """Use this when you know the exact code you want to find.
         It will run the command: grep -n -r "code_snippet" "file_pattern"

    Perfect for:
    - Finding specific constant definitions: code_snippet="MAX_RETRIES = 3"
    - Finding decorator usage: code_snippet="@retry(max_attempts=3)"
    - Finding specific imports: code_snippet="from datetime import datetime"
    - Finding configuration patterns: code_snippet="DEBUG = os.getenv('DEBUG', False)"

    Note: You must know the exact code snippet. Use SemanticSearch if you only know
    what the code does but not its exact implementation.
    """

    code_snippet: str = Field(..., description="The exact code snippet to find.")
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    model_config = {"title": "FindCodeSnippet"}

    @model_validator(mode="after")
    def validate_snippet(self):
        if not self.code_snippet.strip():
            raise ValueError("code_snippet cannot be empty")
        return self

    def to_prompt(self):
        prompt = f"Searching for code snippet: {self.code_snippet}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

    def short_summary(self) -> str:
        param_str = f"code_snippet={self.code_snippet}"
        if self.file_pattern:
            param_str += f", file_pattern={self.file_pattern}"
        return f"{self.name}({param_str})"

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="I need to understand how the User class is structured in our authentication system. Let me find its definition.",
                action=FindCodeSnippetArgs(
                    thoughts="To find the User class definition, I'll search for the exact class declaration line 'class User(BaseModel):'",
                    code_snippet="class User(BaseModel):",
                ),
            ),
            FewShotExample.create(
                user_input="The system seems to use a default timeout value. I should check where DEFAULT_TIMEOUT is defined in the configuration.",
                action=FindCodeSnippetArgs(
                    thoughts="To find the timeout configuration, I'll search for the exact variable declaration 'DEFAULT_TIMEOUT =' in config files",
                    code_snippet="DEFAULT_TIMEOUT =",
                    file_pattern="**/config/*.py",
                ),
            ),
            FewShotExample.create(
                user_input="To understand how request processing works, I need to examine the _handlers dictionary in the processor service.",
                action=FindCodeSnippetArgs(
                    thoughts="To find the handlers mapping, I'll search for the exact dictionary declaration '_handlers =' in the processor service",
                    code_snippet="_handlers =",
                    file_pattern="services/processor.py",
                ),
            ),
        ]


class FindCodeSnippet(SearchBaseAction):
    args_schema: ClassVar[type[ActionArguments]] = FindCodeSnippetArgs

    max_hits: int = Field(
        10,
        description="The maximum number of search results to return. Default is 10.",
    )

    def _fallback_queries(self, args):
        from moatless.actions.query_terms import query_terms
        return query_terms(args.code_snippet)

    async def _search_for_context(self, args: FindCodeSnippetArgs) -> tuple[FileContext, bool]:
        logger.info(f"{self.name}: {args.code_snippet} (file_pattern: {args.file_pattern})")

        matches = await self.search_repository.find_exact_matches(
            search_text=args.code_snippet, file_pattern=args.file_pattern
        )
        logger.info(f"Found {len(matches)} matches")

        if args.file_pattern:
            # Normalize the pattern to handle both **/*.py and *.py cases
            pattern = args.file_pattern.replace("**/", "*")
            directory = args.file_pattern.rstrip("/")
            is_directory = self._repository.is_directory(directory)
            matches = [
                (file_path, line_num)
                for file_path, line_num in matches
                if (is_directory and file_path.startswith(directory + "/"))
                or fnmatch(os.path.basename(file_path), pattern) or fnmatch(file_path, args.file_pattern)
            ]
            logger.info(f"Filtered {len(matches)} matches to {len(matches)} matches")

        search_result_context = FileContext(repo=self.search_repository)
        if matches:
            for file_path, start_line in matches[: self.max_hits]:
                num_lines = len(args.code_snippet.splitlines())
                end_line = start_line + num_lines - 1

                search_result_context.add_line_span_to_context(file_path, start_line, end_line, add_extra=False)
            self._last_search_attempts = {"initial": {"hits": len(matches)}, "fallback_used": False}
            return search_result_context, False

        # Exact grep missed.  Try identifier-level grep first, then semantic
        # retrieval.  This is the common failure mode in long trajectories:
        # the model knows a symbol but not the exact surrounding text.
        tokens = list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", args.code_snippet)))[:3]
        recovered = []
        for token in tokens:
            token_matches = await self.search_repository.find_exact_matches(
                search_text=token, file_pattern=args.file_pattern
            )
            recovered.extend(token_matches)
        recovered = list(dict.fromkeys(recovered))
        if recovered:
            for file_path, start_line in recovered[: self.max_hits]:
                search_result_context.add_line_span_to_context(
                    file_path, start_line, start_line, add_extra=False
                )
            self._last_search_attempts = {"initial": {"hits": 0}, "fallback_used": True,
                                          "fallback": "identifier_grep", "hits": len(recovered)}
            return search_result_context, True

        fallback, details = await self._fallback_context(args)
        if fallback is not None and not fallback.is_empty():
            self._last_search_attempts = {"initial": {"hits": 0}, "fallback_used": True, **details}
            return fallback, True

        self._last_search_attempts = {"initial": {"hits": 0}, "fallback_used": False, "hits": 0}
        return search_result_context, False
