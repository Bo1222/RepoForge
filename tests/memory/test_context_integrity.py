"""Regressions for lost interactions, patches and budgets in long tasks."""

import pytest

from moatless.actions.finish import FinishArgs
from moatless.actions.schema import Observation
from moatless.actions.view_code import CodeSpan, ViewCodeArgs
from moatless.file_context import FileContext
from moatless.message_history.compact import CompactMessageHistoryGenerator
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.message_history.react_compact import ReactCompactMessageHistoryGenerator
from moatless.message_history.react import ReactMessageHistoryGenerator
from moatless.message_history.summary import SummaryMessageHistoryGenerator
from moatless.utils.tokenizer import count_tokens
from moatless.node import ActionStep, Node
from moatless.repository.repository import InMemRepository
from moatless.workspace import Workspace


def trajectory():
    repo = InMemRepository(files={"a.py": "def a():\n    return 1\n", "b.py": "def b():\n    return 2\n"})
    root = Node(node_id=0, user_message="Fix issue", file_context=FileContext(repo=repo))
    child = root.create_child()
    child.action_steps = [
        ActionStep(action=FinishArgs(finish_reason="A"), observation=Observation(message="A " * 150)),
        ActionStep(action=FinishArgs(finish_reason="B"), observation=Observation(message="B " * 150)),
    ]
    return repo, root, child


@pytest.mark.asyncio
async def test_budget_keeps_tool_groups_and_all_user_requirements():
    repo, root, child = trajectory()
    child.user_message = "Also preserve the API"
    for budget in (124, 250, 500, 1000):
        messages = await MessageHistoryGenerator(max_tokens=budget).generate_messages(child, Workspace(repository=repo))
        calls = [c["id"] for m in messages for c in m.get("tool_calls", [])]
        results = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
        assert calls == results
        assert [m["content"][0]["text"] for m in messages if m["role"] == "user"] == [
            root.user_message, child.user_message
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("generator", [CompactMessageHistoryGenerator, ReactCompactMessageHistoryGenerator])
async def test_compact_retains_task_and_every_completed_action(generator):
    repo, root, child = trajectory()
    memory = generator()
    groups = await memory.get_node_messages(child)
    assert [a.finish_reason for g in groups for a in g.actions] == ["A", "B"]
    messages = await memory.generate_messages(child, Workspace(repository=repo))
    assert messages[0]["role"] == "user"
    assert root.user_message in str(messages[0])
    if generator is CompactMessageHistoryGenerator:
        assert len([m for m in messages if m["role"] == "tool"]) == 2


@pytest.mark.asyncio
async def test_compact_handles_multifile_view():
    repo, root, child = trajectory()
    child.file_context.add_file("a.py", show_all_spans=True)
    child.file_context.add_file("b.py", show_all_spans=True)
    child.action_steps = [ActionStep(
        action=ViewCodeArgs(files=[CodeSpan(file_path="a.py"), CodeSpan(file_path="b.py")]),
        observation=Observation(message="Both files read"),
    )]
    messages = await CompactMessageHistoryGenerator().generate_messages(child, Workspace(repository=repo))
    assert "a.py" in str(messages) and "b.py" in str(messages)


def test_cleanup_and_clone_preserve_edits_and_budget():
    repo, _, _ = trajectory()
    context = FileContext(repo=repo, max_tokens=1234)
    context.add_file("a.py", show_all_spans=True).apply_changes("def a():\n    return 3\n")
    patch = context.generate_git_patch()
    context.remove_files(["a.py"])
    assert context.generate_git_patch() == patch
    assert "a.py" not in context.create_prompt()
    cloned = context.clone()
    restored = FileContext.from_dict(context.model_dump(), repo=repo)
    for copy in (cloned, restored):
        assert copy.model_dump()["max_tokens"] == 1234
        assert copy.generate_git_patch() == patch
        assert "a.py" not in copy.create_prompt()
    context.add_file("a.py", show_all_spans=True)
    assert "return 3" in context.create_prompt()


def test_clone_budget_without_cleanup():
    repo, _, _ = trajectory()
    original = FileContext(repo=repo, max_tokens=1234)
    assert original.clone().model_dump()["max_tokens"] == 1234


def test_cleanup_unmodified_file_retains_content_and_is_reversible():
    repo, _, _ = trajectory()
    context = FileContext(repo=repo)
    context.add_file("a.py", show_all_spans=True)
    context.remove_files(["a.py"])
    assert context._files["a.py"].content == repo.get_file_content("a.py")
    assert not context.has_file("a.py") and context.file_paths == []
    restored = FileContext.from_dict(context.model_dump(), repo=repo)
    restored.add_file("a.py")
    assert restored.has_file("a.py")


@pytest.mark.asyncio
async def test_pending_steps_do_not_fabricate_completed_tool_calls():
    repo, _, child = trajectory()
    child.action_steps[1].observation = None
    messages = await MessageHistoryGenerator().generate_messages(child, Workspace(repository=repo))
    assert len([m for m in messages if m["role"] == "tool"]) == 1
    assert "B" not in str([m.get("tool_calls", []) for m in messages])


@pytest.mark.asyncio
@pytest.mark.parametrize("generator", [
    MessageHistoryGenerator, CompactMessageHistoryGenerator, ReactCompactMessageHistoryGenerator,
    ReactMessageHistoryGenerator, SummaryMessageHistoryGenerator,
])
async def test_all_formats_preserve_requirements_and_enforce_budget(generator):
    repo, root, child = trajectory()
    child.user_message = "Additional API constraint"
    messages = await generator(max_tokens=130).generate_messages(child, Workspace(repository=repo))
    assert root.user_message in str(messages) and child.user_message in str(messages)
    assert sum(count_tokens(str(m)) for m in messages) <= 130
    with pytest.raises(RuntimeError, match="context_budget_exceeded"):
        await generator(max_tokens=0).generate_messages(child, Workspace(repository=repo))


@pytest.mark.asyncio
async def test_reconstructed_state_never_claims_a_new_tool_execution():
    repo, _, child = trajectory()
    child.file_context.add_file("a.py", show_all_spans=True).apply_changes("def a():\n    return 3\n")
    messages = await CompactMessageHistoryGenerator().generate_messages(child, Workspace(repository=repo))
    calls = [c for m in messages for c in m.get("tool_calls", [])]
    assert len(calls) == 2
    assert all(c["function"]["name"] == "Finish" for c in calls)
    assert "<workspace_state>" in messages[-1]["content"]
    assert "return 3" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_duplicate_user_messages_keep_chronological_positions():
    repo, root, child = trajectory()
    child.user_message = root.user_message
    messages = await MessageHistoryGenerator(max_tokens=90).generate_messages(child, Workspace(repository=repo))
    assert len([m for m in messages if m["role"] == "user"]) == 2
