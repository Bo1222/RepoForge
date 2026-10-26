import logging
from abc import ABC
from typing import ClassVar, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from repoforge.actions.action import Action, CompletionModelMixin
from repoforge.actions.schema import ActionArguments, Observation, RewardScaleEntry
from repoforge.completion.base import CompletionRetryError
from repoforge.completion.schema import (
    ChatCompletionUserMessage,
    ResponseSchema,
)
from repoforge.completion.stats import CompletionInvocation
from repoforge.file_context import FileContext
from repoforge.index.code_index import CodeIndex
from repoforge.index.hybrid import fuse_hits
from repoforge.index.lexical import bm25_search
from repoforge.index.types import SearchCodeResponse
from repoforge.workspace import Workspace

logger = logging.getLogger(__name__)

IDENTIFY_SYSTEM_PROMPT = """You are an autonomous AI assistant tasked with identifying relevant code in a codebase. Your goal is to select key code sections from the search results that are most relevant to the search request.

The previous messages will contain:
1. A search request from an AI assistant
2. Search results containing various code sections with their line numbers

# Your Task:

1. Understand the Search Request:
   * Analyze the previous search request to understand what code elements are being looked for
   * Identify key elements such as functions, variables, classes, or patterns that are relevant

2. Evaluate Search Results:
   * Examine each code section in the search results for alignment with the search request
   * Assess the relevance and importance of each code section
   * Consider the complete context of code sections

3. Respond with the Identify Action:
   * Select and respond with the code sections that best match the search request
   * Provide your analysis in the thoughts field
   * List the relevant file paths with start and end line numbers in the identified_spans field
"""


class SearchBaseArgs(ActionArguments, ABC):
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific files or directories.",
    )

    @field_validator("file_pattern")
    @classmethod
    def validate_file_pattern(cls, v):
        if v:
            if "," in v:
                raise ValueError("File pattern cannot contain commas")
        return v


class IdentifiedSpans(BaseModel):
    file_path: str = Field(description="The file path where the relevant code is found.")
    start_line: int = Field(description="Starting line number of the relevant code section.")
    end_line: int = Field(description="Ending line number of the relevant code section.")


class Identify(ResponseSchema):
    """Identify if the provided search result is relevant to the reported issue."""

    thoughts: Optional[str] = Field(
        None,
        description="Your thoughts and analysis on the search results and how they relate to the reported issue.",
    )

    identified_spans: Optional[list[IdentifiedSpans]] = Field(
        default=None,
        description="Files and code sections in the search results identified as relevant to the reported issue.",
    )


class SearchBaseAction(Action, CompletionModelMixin, ABC):
    args_schema: ClassVar[type[ActionArguments]] = SearchBaseArgs
    _last_search_attempts: dict = PrivateAttr(default_factory=dict)
    _active_context: object = PrivateAttr(default=None)

    @property
    def search_repository(self):
        from repoforge.repository_overlay import CurrentRepository
        return CurrentRepository(self._repository, self._active_context) if self._active_context else self._repository

    max_search_tokens: int = Field(
        2000,
        description="The maximum number of tokens allowed in the search results.",
    )
    max_identify_tokens: int = Field(
        8000,
        description="The maximum number of tokens allowed in the identified code sections.",
    )
    max_identify_prompt_tokens: int = Field(
        16000,
        description="The maximum number of tokens allowed in the identify prompt.",
    )
    max_hits: int = Field(
        10,
        description="The maximum number of search hits to display.",
    )
    add_extra_context: bool = Field(
        False,
        description="Whether to add extra context, like imports, to the search results.",
    )
    use_identifier: bool = Field(
        False,
        description="Whether to use an LLM to identify the relevant code sections.",
    )

    async def initialize(self, workspace: Workspace):
        if not workspace.code_index:
            raise ValueError("Code index is not set")

        self._workspace = workspace

    @property
    def code_index(self) -> CodeIndex:
        if not self._workspace:
            raise ValueError("Workspace is not set")

        if not self._workspace.code_index:
            raise ValueError("Code index is not set")

        return self._workspace.code_index

    def _initialize_completion_model(self):
        """Override mixin method to customize initialization"""
        if self.completion_model:

            async def validate_identified_code(
                structured_outputs: list[ResponseSchema],
                text_response: Optional[str],
            ) -> tuple[list[ResponseSchema], Optional[str]]:
                view_context = FileContext(repo=self._repository)

                if not structured_outputs:
                    return structured_outputs, text_response

                for identified_code in structured_outputs:
                    if isinstance(identified_code, Identify) and identified_code.identified_spans:
                        for identified_spans in identified_code.identified_spans:
                            view_context.add_line_span_to_context(
                                identified_spans.file_path,
                                identified_spans.start_line,
                                identified_spans.end_line,
                                add_extra=True,
                            )

                tokens = view_context.context_size()
                if tokens > self.max_identify_tokens:
                    raise CompletionRetryError(
                        f"The identified code sections are too large ({tokens} tokens). Maximum allowed is {self.max_identify_tokens} tokens. "
                        f"Please identify a smaller subset of the most relevant code sections."
                    )

                return structured_outputs, text_response

            self.completion_model.initialize(
                Identify, IDENTIFY_SYSTEM_PROMPT, post_validation_fn=validate_identified_code
            )

    async def execute(self, args: SearchBaseArgs, file_context: FileContext | None = None) -> Observation:
        if file_context is None:
            raise ValueError("File context must be provided to execute the search action.")

        if self.completion_model and not self.completion_model.initialized:
            self.completion_model.initialize(Identify, IDENTIFY_SYSTEM_PROMPT)

        properties = {"search_hits": [], "search_tokens": 0}

        self._last_search_attempts = {}
        self._active_context = file_context
        search_result_context, alternative_suggestion = await self._search_for_context(args)
        # Index recall may use base code. Reconstruct every rendered hit against
        # the same current overlay used by execution-memory file versions.
        current = FileContext(repo=self.search_repository)
        for hit in search_result_context.files:
            source = self.search_repository.get_file(hit.file_path)
            if source is None:
                continue
            for span in hit.spans[:3]:
                if span.span_id and source.module and source.module.find_span_by_id(span.span_id):
                    current.add_span_to_context(hit.file_path, span.span_id, add_extra=False)
                elif span.start_line and self.search_repository.get_file_content(hit.file_path) == hit.content:
                    current.add_line_span_to_context(hit.file_path, span.start_line, span.end_line, add_extra=False)
        # Include exact symbol definitions added by the current shadow patch.
        symbol = getattr(args, "function_name", None) or getattr(args, "class_name", None)
        if symbol:
            import ast
            for path, file in file_context._files.items():
                if not file.patch or not path.endswith(".py"):
                    continue
                try:
                    tree = ast.parse(file.content)
                except SyntaxError:
                    continue
                for item in ast.walk(tree):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == symbol:
                        current.add_line_span_to_context(path, item.lineno, item.end_lineno, add_extra=False)
        search_result_context = current
        properties["search_attempts"] = self._last_search_attempts
        properties["search_hits"] = search_result_context.model_dump(exclude_none=True)
        properties["search_tokens"] = search_result_context.context_size()

        if search_result_context.is_empty():
            properties["fail_reason"] = "no_search_hits"
            from difflib import get_close_matches
            from repoforge.repository_overlay import repository_paths
            query = args.file_pattern or getattr(args, "function_name", None) or getattr(args, "class_name", None) or ""
            properties["nearest_paths"] = get_close_matches(query, repository_paths(self._repository), n=5, cutoff=0.1)
            return Observation.create(message="No search results found. Nearest paths: " + str(properties["nearest_paths"]) +
                ". Try a symbol name or error substring.", properties=properties)

        if not self.use_identifier:
            response_str = ""
            matches = 0
            for file in search_result_context.files:
                response_str += f"\nFile: {file.file_path}\n"
                for span in file.spans:
                    line_content = None

                    if span.start_line:
                        start_line = span.start_line
                        end_line = span.end_line
                    elif file.module:
                        block_span = file.module.find_span_by_id(span.span_id)
                        if block_span:
                            start_line = block_span.start_line
                            end_line = block_span.end_line
                            code_block = block_span.initiating_block
                            if code_block:
                                line_content = code_block.content.strip()

                    if start_line and not line_content:
                        line_content = file.content.split("\n")[start_line - 1].strip()

                    if line_content:
                        response_str += f"Line: {start_line} - {end_line} : {line_content}\n"
                        matches += 1
                    elif start_line:
                        response_str += f"Line: {start_line} - {end_line}\n"
                        matches += 1

            if matches > 0:
                response_str = f"\nFound {matches} matches in {len(search_result_context.files)} files.\n{response_str}"
            else:
                response_str = f"\nFound no matches in {len(search_result_context.files)} files."

            return Observation.create(message=response_str, properties=properties)

        context_size = search_result_context.context_size()
        properties["search_tokens"] = context_size
        properties["search_hits"] = search_result_context.model_dump(exclude_none=True)

        completion = None

        if search_result_context.span_count() == 1 and context_size > self.max_identify_tokens:
            logger.warning(f"{self.name}: Conext is too large ({context_size} tokens).")
            properties["fail_reason"] = "search_too_large"
            return Observation.create(
                message="Search too large. Found a single code section that is too large to view. Please refine the search query.",
                properties=properties,
            )
        elif (
            context_size > self.max_search_tokens and search_result_context.span_count() > 1
        ) or search_result_context.span_count() > self.max_hits:
            logger.info(
                f"{self.name}: Search too large. {properties['search_tokens']} tokens and {search_result_context.span_count()} hits, will ask for clarification."
            )
            view_context, completion = await self._identify_code(args, search_result_context)
        else:
            view_context = search_result_context

        span_count = search_result_context.span_count()
        search_result_str = f"Found {span_count} code sections."

        new_span_ids = file_context.add_file_context(view_context)

        if view_context.is_empty():
            search_result_str += "\n\nNone of the search results was relevant to the task."
            summary = "Didn't find any relevant code sections in the search results."
            message = search_result_str
        else:
            viewed_str = "that has already been viewed" if not new_span_ids else ""

            context_summary = view_context.create_summary()
            if alternative_suggestion:
                summary = f"Did not find an exact match but found the following alternative suggestions {viewed_str}:\n{context_summary}"
            else:
                summary = f"Found the following relevant code spans {viewed_str}:\n{context_summary}"

            message = "Found the following relevant code:\n"
            message += view_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
            )

        properties["new_span_ids"] = new_span_ids

        logger.info(
            f"{self.name}: Found {span_count} code sections in search results. Viewed {view_context.span_count()} code sections."
        )

        return Observation.create(
            message=message,
            summary=summary,
            properties=properties,
            execution_completion=completion,
        )

    def _fallback_queries(self, args: SearchBaseArgs) -> list[str]:
        """Action-specific terms for the last-resort retrieval chain."""
        return []

    async def _fallback_files(self, args: SearchBaseArgs) -> list[tuple[str, str]]:
        """A capped (path, content) set for the lexical fallback."""
        files = getattr(self.search_repository, "files", None)
        if isinstance(files, dict):
            return [(str(path), content) for path, content in files.items()
                    if content and len(content) <= 500_000][:2000]
        try:
            index = self.code_index
            paths = await index.matching_files(args.file_pattern or "**/*.py")
        except (AttributeError, ValueError, OSError):
            return []
        result = []
        for path in paths[:2000]:
            try:
                content = self.search_repository.get_file_content(path)
            except (OSError, ValueError):
                continue
            if content:
                result.append((path, content))
        return result

    async def _fallback_context(self, args: SearchBaseArgs):
        """Deterministic chain: identifier grep -> BM25 -> semantic."""
        queries = [query for query in self._fallback_queries(args) if query]
        if not queries:
            return None, {}
        context = FileContext(repo=self.search_repository)
        for query in queries[:6]:
            try:
                matches = await self.search_repository.find_exact_matches(
                    search_text=query, file_pattern=args.file_pattern)
            except (AttributeError, NotImplementedError, OSError, ValueError):
                matches = []
            for path, line in matches[: self.max_hits]:
                try:
                    context.add_line_span_to_context(path, line, line, add_extra=False)
                except (OSError, ValueError, KeyError):
                    continue
        if not context.is_empty():
            return context, {"fallback": "identifier_grep", "queries": queries[:6]}
        try:
            lexical = bm25_search(await self._fallback_files(args), " ".join(queries),
                                  limit=self.max_hits)
        except (OSError, ValueError):
            lexical = []
        for path, line in lexical:
            try:
                context.add_line_span_to_context(path, line, line, add_extra=False)
            except (OSError, ValueError, KeyError):
                continue
        if not context.is_empty():
            return context, {"fallback": "bm25", "queries": queries[:6]}
        try:
            index = self.code_index
        except (AttributeError, ValueError):
            index = None
        if index is not None:
            try:
                response = await index.semantic_search(
                    query=" ".join(queries), file_pattern=args.file_pattern,
                    max_results=self.max_hits, max_spans_per_file=2)
            except Exception:
                response = None
            if response and response.hits:
                for hit in fuse_hits(" ".join(queries), response.hits):
                    for span in hit.spans:
                        try:
                            context.add_span_to_context(hit.file_path, span.span_id,
                                                        add_extra=False)
                        except (OSError, ValueError, KeyError):
                            continue
                if not context.is_empty():
                    return context, {"fallback": "semantic", "queries": queries[:6]}
        return None, {}

    async def _search_for_context(self, args: SearchBaseArgs) -> tuple[FileContext, bool]:
        alternative_suggestion = False
        search_result = await self._search(args)
        initial_message = search_result.message
        initial_hits = len(search_result.hits)
        if not search_result.hits:
            search_result = await self._search_for_alternative_suggestion(args)
            alternative_suggestion = True
            logger.info(
                f"{self.name}: No relevant search results found. Will use alternative suggestion with {len(search_result.hits)} hits."
            )
        if not search_result.hits:
            fallback_context, fallback_meta = await self._fallback_context(args)
            if fallback_context is not None and not fallback_context.is_empty():
                self._last_search_attempts = {
                    "initial": {"message": initial_message, "hits": initial_hits},
                    "fallback_used": True,
                    **fallback_meta,
                }
                return fallback_context, True
        # These diagnostics survive even when fallback produces useful code.
        self._last_search_attempts = {
            "initial": {"message": initial_message, "hits": initial_hits},
            "fallback_used": alternative_suggestion,
            "final": {"message": search_result.message, "hits": len(search_result.hits)},
        }

        span_count = 0
        search_result_context = FileContext(repo=self.search_repository)
        for hit in search_result.hits:
            span_count += len(hit.spans)
            for span in hit.spans:
                search_result_context.add_span_to_context(hit.file_path, span.span_id, add_extra=self.add_extra_context)

        return search_result_context, alternative_suggestion

    async def _search(self, args: SearchBaseArgs) -> SearchCodeResponse:
        raise NotImplementedError("Subclasses must implement this method.")

    async def _search_for_alternative_suggestion(self, args: SearchBaseArgs) -> SearchCodeResponse:
        return SearchCodeResponse()

    async def _identify_code(
        self, args: SearchBaseArgs, search_result_ctx: FileContext
    ) -> tuple[FileContext, CompletionInvocation]:
        search_result_str = search_result_ctx.create_prompt(
            show_span_ids=True,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="...",
            max_tokens=self.max_identify_prompt_tokens,
        )

        if not self.completion_model:
            raise ValueError("Completion model is not initialized")

        content = "Search request:"
        content += f"\n{args.to_prompt()}"

        content += "\n\nIdentify the relevant code sections in the search results to use them. "
        content += f"\n\n<search_results>\n{search_result_str}\n</search_result>\n"
        identify_message = ChatCompletionUserMessage(role="user", content=content)  # type: ignore

        messages = [identify_message]
        completion_response = await self.completion_model.create_completion(messages=messages)  # type: ignore

        view_context = FileContext(repo=self.search_repository)
        if completion_response.structured_outputs:
            for identified_code in completion_response.structured_outputs:
                if isinstance(identified_code, Identify) and identified_code.identified_spans:
                    for identified_spans in identified_code.identified_spans:
                        view_context.add_line_span_to_context(
                            identified_spans.file_path,
                            identified_spans.start_line,
                            identified_spans.end_line,
                            add_extra=True,
                        )

        return view_context, completion_response.completion_invocation

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> list[str]:
        evaluation_criteria = super().get_evaluation_criteria(trajectory_length)
        evaluation_criteria.extend(
            [
                "Query Relevance: Evaluate if the search query or parameters are well-defined and likely to find relevant code.",
                "Search Scope Appropriateness: Check if the file patterns and class/function names narrow down the search effectively.",
                "Relevance of Search Results: Assess whether the search results are directly related to the problem and useful for making progress.",
                "Size of Search Results: Ensure that the code context provided is appropriately sized—not too large to overwhelm nor too small to be unhelpful.",
            ]
        )

        return evaluation_criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> list[RewardScaleEntry]:
        if trajectory_length <= 3:
            return cls.generate_reward_scale_entries(
                [
                    (
                        90,
                        100,
                        "The search action is excellent, with well-defined parameters yielding only highly relevant results.",
                    ),
                    (
                        75,
                        89,
                        "The search action is good, with reasonable parameters yielding relevant results.",
                    ),
                    (
                        25,
                        74,
                        "The search action have issues with parameters or yields few or no relevant results.",
                    ),
                    (
                        0,
                        24,
                        "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks.",
                    ),
                ]
            )
        else:
            return cls.generate_reward_scale_entries(
                [
                    (
                        90,
                        100,
                        "The search action significantly advances the solution, providing highly relevant and appropriately sized search results.",
                    ),
                    (
                        75,
                        89,
                        "The search action contributes positively towards solving the problem, with relevant results and minor issues.",
                    ),
                    (
                        50,
                        74,
                        "The search action is acceptable but may have issues with relevance or provides search results that are too large or too small.",
                    ),
                    (
                        25,
                        49,
                        "The search action provides results that are not helpful due to relevance or size issues.",
                    ),
                    (
                        0,
                        24,
                        "The search action has minimal impact, providing few relevant results.",
                    ),
                    (
                        -50,
                        -1,
                        "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks.",
                    ),
                ]
            )
