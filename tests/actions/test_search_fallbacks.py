import pytest
from unittest.mock import AsyncMock, MagicMock

from moatless.actions.find_class import FindClass, FindClassArgs
from moatless.actions.find_function import FindFunction, FindFunctionArgs
from moatless.actions.query_terms import query_terms
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.index.lexical import bm25_search
from moatless.index.types import SearchCodeResponse
from moatless.repository.repository import InMemRepository
from moatless.workspace import Workspace


class FallbackRepository(InMemRepository):
    async def find_exact_matches(self, search_text, file_pattern=None):
        if search_text == "target_symbol":
            return [("module.py", 1)]
        return []


def test_query_terms_prefers_identifiers_and_keeps_order():
    terms = query_terms("Call `process_payment` and handle HTTPError in billing/charge.py")
    assert "process_payment" in terms
    assert "HTTPError" in terms
    assert "billing/charge.py" in terms
    assert len(terms) == len(set(terms))


def test_bm25_ranks_the_file_containing_the_identifier():
    hits = bm25_search([
        ("a.py", "def process_payment():\n    return 1\n"),
        ("b.py", "def unrelated():\n    return 2\n"),
    ], "process_payment", limit=5)
    assert hits and hits[0][0] == "a.py"


@pytest.mark.asyncio
async def test_find_function_falls_back_to_identifier_grep():
    repo = FallbackRepository()
    repo.save_file("module.py", "def target_symbol():\n    return 42\n")
    workspace = Workspace(repository=repo)
    workspace._code_index = MagicMock(spec=CodeIndex)
    workspace._code_index.find_function = AsyncMock(return_value=SearchCodeResponse())
    workspace._code_index.matching_files = AsyncMock(return_value=["module.py"])
    action = FindFunction()
    await action.initialize(workspace)

    observation = await action.execute(
        FindFunctionArgs(function_name="target_symbol"), FileContext(repo=repo))

    assert "module.py" in observation.message
    assert observation.properties["search_attempts"]["fallback"] == "identifier_grep"


@pytest.mark.asyncio
async def test_find_class_uses_the_same_fallback_chain():
    repo = FallbackRepository()
    repo.save_file("module.py", "class TargetSymbol:\n    pass\n")
    workspace = Workspace(repository=repo)
    workspace._code_index = MagicMock(spec=CodeIndex)
    workspace._code_index.find_class = AsyncMock(return_value=SearchCodeResponse())
    workspace._code_index.matching_files = AsyncMock(return_value=["module.py"])
    action = FindClass()
    await action.initialize(workspace)

    observation = await action.execute(
        FindClassArgs(class_name="TargetSymbol"), FileContext(repo=repo))

    assert "module.py" in observation.message
    assert observation.properties["search_attempts"]["fallback_used"] is True
