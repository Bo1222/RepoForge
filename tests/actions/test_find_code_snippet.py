import pytest

from unittest.mock import MagicMock

from repoforge.actions.find_code_snippet import FindCodeSnippet, FindCodeSnippetArgs
from repoforge.file_context import FileContext
from repoforge.index.code_index import CodeIndex
from repoforge.repository.repository import InMemRepository
from repoforge.workspace import Workspace


class GrepRepository(InMemRepository):
    async def find_exact_matches(self, search_text, file_pattern=None):
        if search_text == "target_symbol":
            return [("module.py", 1)]
        return []


@pytest.mark.asyncio
async def test_exact_miss_falls_back_to_identifier_grep():
    repo = GrepRepository()
    repo.save_file("module.py", "def target_symbol():\n    return 42\n")

    action = FindCodeSnippet(max_hits=5)
    workspace = Workspace(repository=repo)
    workspace._code_index = MagicMock(spec=CodeIndex)
    await action.initialize(workspace)
    context = FileContext(repo=repo)

    observation = await action.execute(
        FindCodeSnippetArgs(code_snippet="value = target_symbol()", file_pattern="**/*.py"),
        context,
    )

    assert "module.py" in observation.message
    assert observation.properties["search_attempts"]["fallback_used"] is True
    assert observation.properties["search_attempts"]["fallback"] == "identifier_grep"
