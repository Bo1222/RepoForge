import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest
import pytest_asyncio
from repoforge.actions.schema import Observation
from repoforge.actions.string_replace import StringReplace, StringReplaceArgs
from repoforge.file_context import FileContext, ContextFile
from repoforge.repository.repository import InMemRepository
from repoforge.workspace import Workspace


def dedent(text):
    """Remove common leading whitespace from a multi-line string."""
    return textwrap.dedent(text).strip()


@pytest_asyncio.fixture
async def repository():
    repo = InMemRepository()
    # Add a file with specific content to test string replacement
    repo.save_file(
        "existing.py",
        """def hello_world():
    message = "Hello World"
    return message

if __name__ == "__main__":
    print(hello_world())
""",
    )
    return repo


@pytest_asyncio.fixture
async def workspace(repository):
    workspace = Mock(spec=Workspace)
    workspace.repository = repository
    return workspace


@pytest_asyncio.fixture
async def file_context(repository):
    context = FileContext(repo=repository, shadow_mode=False)
    return context


@pytest.mark.asyncio
async def test_string_replace_basic(file_context, repository, workspace):
    """Test basic string replacement with StringReplace."""
    # Set up string replace action
    string_replace_action = StringReplace()
    string_replace_action.workspace = workspace

    # Double-check the file content to use the exact string
    content = repository.get_file_content("existing.py")
    print(f"Original content:\n{content}")

    # Make sure the file is in context with show_all_spans=True and mark as viewed
    file_context.add_file("existing.py", show_all_spans=True)
    context_file = file_context.get_file("existing.py")
    context_file.was_viewed = True

    # Find the exact string we want to replace
    lines = content.splitlines()
    message_line = None
    for line in lines:
        if 'message = "Hello World"' in line:
            message_line = line
            break

    # Replace the message
    replace_args = StringReplaceArgs(
        path="existing.py", old_str=message_line, new_str=message_line.replace("Hello World", "Hello Universe")
    )

    # Execute the string replace action
    replace_observation = await string_replace_action.execute(replace_args, file_context=file_context)

    # Verify the observation
    assert isinstance(replace_observation, Observation)
    assert "The file existing.py has been edited" in replace_observation.message
    assert "Review the changes" in replace_observation.summary
    assert "cat -n" in replace_observation.message

    # Verify content change
    modified_content = repository.get_file_content("existing.py")
    assert 'message = "Hello Universe"' in modified_content
    assert "def hello_world():" in modified_content  # Ensure other parts are preserved


@pytest.mark.asyncio
async def test_string_replace_multiline(file_context, repository, workspace):
    """Test replacing multiple lines at once."""
    # Set up string replace action
    string_replace_action = StringReplace()
    string_replace_action.workspace = workspace

    # Make sure the file is in context with show_all_spans=True and mark as viewed
    file_context.add_file("existing.py", show_all_spans=True)
    context_file = file_context.get_file("existing.py")
    context_file.was_viewed = True

    # Verify initial content
    content = repository.get_file_content("existing.py")
    print(f"Original content:\n{content}")

    # Replace the entire function
    original_func = """def hello_world():
    message = "Hello World"
    return message"""

    new_func = """def hello_universe():
    message = "Hello Universe"
    greeting = f"{message}!"
    return greeting"""

    replace_args = StringReplaceArgs(path="existing.py", old_str=original_func, new_str=new_func)

    # Execute string replace action
    replace_observation = await string_replace_action.execute(replace_args, file_context=file_context)

    # Verify the observation
    assert isinstance(replace_observation, Observation)
    assert "The file existing.py has been edited" in replace_observation.message

    # Verify content change
    modified_content = repository.get_file_content("existing.py")
    assert "def hello_universe():" in modified_content
    assert 'greeting = f"{message}!"' in modified_content


@pytest.mark.asyncio
async def test_string_replace_non_existent(file_context, repository, workspace):
    """Test replacement with non-existent string."""
    # Set up string replace action
    string_replace_action = StringReplace()
    string_replace_action.workspace = workspace

    # Make sure the file is in context with show_all_spans=True and mark as viewed
    file_context.add_file("existing.py", show_all_spans=True)
    context_file = file_context.get_file("existing.py")
    context_file.was_viewed = True

    replace_args = StringReplaceArgs(
        path="existing.py", old_str="This string does not exist in the file", new_str="New content"
    )

    # Execute string replace action
    replace_observation = await string_replace_action.execute(replace_args, file_context=file_context)

    # Verify error observation
    assert "String 'This string does not exist in the file' not found" in replace_observation.message
    assert replace_observation.properties.get("fail_reason") == "string_not_found"

    # Content should remain unchanged
    content = repository.get_file_content("existing.py")
    assert "New content" not in content


@pytest.mark.asyncio
async def test_string_replace_non_existent_file(file_context, repository, workspace):
    """Test replacement in a non-existent file."""
    # Set up string replace action
    string_replace_action = StringReplace()
    string_replace_action.workspace = workspace

    # Don't add the file to context - it doesn't exist

    replace_args = StringReplaceArgs(path="nonexistent.py", old_str="Any string", new_str="New content")

    # Execute string replace action
    replace_observation = await string_replace_action.execute(replace_args, file_context=file_context)

    # Verify error observation indicates file not found
    assert replace_observation.properties.get("fail_reason") == "file_not_found"


@pytest.mark.asyncio
async def test_string_replace_auto_loads_existing_file(file_context, repository, workspace):
    """A file that exists but was never viewed should be loaded for the edit."""
    action = StringReplace()
    action.workspace = workspace

    args = StringReplaceArgs(
        path="existing.py",
        old_str='message = "Hello World"',
        new_str='message = "Hello Auto Load"',
    )
    observation = await action.execute(args, file_context=file_context)

    assert observation.properties.get("fail_reason") is None
    assert "has been edited" in observation.message
    assert 'message = "Hello Auto Load"' in repository.get_file_content("existing.py")


@pytest.mark.asyncio
async def test_string_replace_rejects_noop(file_context, repository, workspace):
    """An identical old_str/new_str should not count as a successful edit."""
    action = StringReplace()
    action.workspace = workspace
    file_context.add_file("existing.py", show_all_spans=True)

    before = repository.get_file_content("existing.py")
    args = StringReplaceArgs(
        path="existing.py",
        old_str='message = "Hello World"',
        new_str='message = "Hello World"',
    )
    observation = await action.execute(args, file_context=file_context)

    assert observation.properties.get("fail_reason") == "no_changes"
    assert repository.get_file_content("existing.py") == before


@pytest.mark.asyncio
async def test_string_replace_rejects_new_syntax_error(file_context, repository, workspace):
    """A Python edit that makes a previously valid file invalid must not commit."""
    action = StringReplace()
    action.workspace = workspace
    file_context.add_file("existing.py", show_all_spans=True)

    before = repository.get_file_content("existing.py")
    args = StringReplaceArgs(
        path="existing.py",
        old_str='message = "Hello World"',
        new_str="message = (",
    )
    observation = await action.execute(args, file_context=file_context)

    assert observation.properties.get("fail_reason") == "syntax_error"
    assert observation.properties.get("syntax_error", {}).get("line")
    assert repository.get_file_content("existing.py") == before
