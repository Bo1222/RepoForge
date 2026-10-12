"""Current versions and scoped instructions are derived from task-local facts."""

import os

import pytest

from moatless.actions.bash import BashArgs
from moatless.actions.finish import Finish
from moatless.actions.local_coding import LocalBash, LocalReadFile, LocalSearch
from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.actions.grep_tool import GrepToolArgs
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.agent.agent import ActionAgent
from moatless.cli.environment import SessionEnvironment
from moatless.file_context import FileContext
from moatless.message_history.managed import ManagedMessageHistoryGenerator
from moatless.message_history.policy import ContextBudgetExceeded, ContextPolicy
from moatless.node import ActionStep, Node
from moatless.repository.file import FileRepository
from moatless.repository_context import refresh_context, workspace_version
from moatless.workspace import Workspace
from scripts.migration_smoke import ScriptedCompletion


async def setup(tmp_path, load=False, actions=None, shadow=False, runtime=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = FileRepository(repo_path=str(tmp_path), shadow_mode=shadow)
    workspace = Workspace(repository=repo, environment=SessionEnvironment(str(tmp_path)), runtime=runtime)
    memory = ManagedMessageHistoryGenerator(policy=ContextPolicy(context_window=32768, load_repo_instructions=load))
    agent = ActionAgent(completion_model=ScriptedCompletion([], max_tokens=512),
                        actions=actions or [LocalReadFile(), LocalBash(), LocalSearch(), Finish()],
                        system_prompt="Fix the issue", memory=memory, shadow_mode=shadow)
    await agent.initialize(workspace)
    memory.bind_model(agent.completion_model)
    root = Node(node_id=0, user_message="Fix empty input", file_context=FileContext(repo=repo))
    return agent, root, workspace


async def execute(agent, parent, *args):
    node = parent.create_child()
    node.file_context = parent.file_context.clone() if parent.file_context else None
    node.action_steps = [ActionStep(action=arg) for arg in args]
    for index in range(len(args)):
        await agent.execute_action_step(node, index)
    return node


@pytest.mark.asyncio
async def test_same_size_bash_edit_invalidates_old_read_without_mutating_raw_output(tmp_path):
    (tmp_path / "calc.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path)
    read = await execute(agent, root, ReadFileArgs(file_path="calc.py"))
    original = read.action_steps[0].observation.model_dump()
    stat = (tmp_path / "calc.py").stat()
    changed = await execute(agent, read, BashArgs(command="printf 'value = 2\\n' > calc.py"))
    os.utime(tmp_path / "calc.py", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    messages = await agent.memory.generate_messages(changed, workspace)
    stale = next(message["content"] for message in messages if message.get("tool_call_id") == f"node_{read.node_id}_step_0")
    assert "Stale code observation" in stale and "value = 1" not in stale
    assert "value = 2" in str(messages)
    assert read.action_steps[0].observation.model_dump() == original
    assert f"{read.node_id}:0" in changed.context_build["stale_sources"]
    assert "current_code:calc.py" in changed.context_build["included_sources"]


@pytest.mark.asyncio
async def test_absolute_path_inside_repository_uses_canonical_version_key(tmp_path):
    (tmp_path / "calc.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path)
    node = await execute(agent, root, ReadFileArgs(file_path=str(tmp_path / "calc.py")))
    assert list(node.observation.properties["code_view"]["files"]) == ["calc.py"]
    messages = await agent.memory.generate_messages(node, workspace)
    assert "value = 1" in str(messages)
    assert not node.context_build["stale_sources"]


@pytest.mark.asyncio
async def test_batch_read_refreshes_context_cache_after_external_disk_edit(tmp_path):
    (tmp_path / "calc.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path, actions=[ReadFile()])
    first = await execute(agent, root, ReadFileArgs(file_path="calc.py"))
    (tmp_path / "calc.py").write_text("value = 2\n")
    second = await execute(agent, first, ReadFileArgs(file_path="calc.py"))
    assert "value = 2" in second.observation.message
    assert "value = 1" not in second.observation.message
    assert first.observation.properties["code_view"]["files"] != second.observation.properties["code_view"]["files"]


@pytest.mark.asyncio
async def test_versions_detect_new_deleted_files_and_same_version_read_dedup(tmp_path):
    (tmp_path / "calc.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path)
    first = await execute(agent, root, ReadFileArgs(file_path="calc.py"))
    second = await execute(agent, first, ReadFileArgs(file_path="calc.py"))
    messages = await agent.memory.generate_messages(second, workspace)
    assert f"{first.node_id}:0" in second.context_build["deduplicated_sources"]
    assert "Repeated read" in str(messages)
    before = workspace_version(workspace, second.file_context)
    (tmp_path / "new.py").write_text("created\n")
    assert workspace_version(workspace, second.file_context) != before
    (tmp_path / "calc.py").unlink()
    messages = await agent.memory.generate_messages(second, workspace)
    assert second.repository_context.files["calc.py"]["content_hash"] == "deleted"
    assert "current_code:calc.py" not in second.context_build["included_sources"]
    assert f"{second.node_id}:0" in second.context_build["stale_sources"]


@pytest.mark.asyncio
async def test_search_hits_bound_to_workspace_and_repeated_search_dedup(tmp_path):
    (tmp_path / "calc.py").write_text("MATCH = 1\n")
    agent, root, workspace = await setup(tmp_path)
    first = await execute(agent, root, GrepToolArgs(pattern="MATCH"))
    second = await execute(agent, first, GrepToolArgs(pattern="MATCH"))
    await agent.memory.generate_messages(second, workspace)
    assert f"{first.node_id}:0" in second.context_build["deduplicated_sources"]
    (tmp_path / "new.py").write_text("MATCH = 2\n")
    await agent.memory.generate_messages(second, workspace)
    assert f"{second.node_id}:0" in second.context_build["stale_sources"]


@pytest.mark.asyncio
async def test_instructions_opt_in_scoped_order_refresh_deletion_and_roundtrip(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "AGENTS.md").write_text("Root guidance")
    (tmp_path / "pkg" / "AGENTS.md").write_text("Package guidance")
    (tmp_path / "other" / "AGENTS.md").write_text("Unrelated guidance")
    (tmp_path / "pkg" / "code.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path)
    assert not refresh_context(root, workspace, False)
    assert not root.instruction_updates
    agent.memory.policy.load_repo_instructions = True
    first = await agent.memory.generate_messages(root, workspace)
    assert "Root guidance" in str(first) and "Package guidance" not in str(first)
    node = await execute(agent, root, ReadFileArgs(file_path="pkg/code.py"))
    messages = await agent.memory.generate_messages(node, workspace)
    sources = node.context_build["instruction_sources"]
    assert [source["path"] for source in sources] == ["AGENTS.md", "pkg/AGENTS.md"]
    assert sources[1]["scope"] == "pkg"
    assert "Unrelated guidance" not in str(messages)
    (tmp_path / "pkg" / "AGENTS.md").write_text("Updated package guidance")
    await agent.memory.generate_messages(node, workspace)
    versions = [item for item in node.instruction_updates if item.path == "pkg/AGENTS.md"]
    assert len(versions) == 2 and versions[0].content == "Package guidance"
    (tmp_path / "pkg" / "AGENTS.md").unlink()
    messages = await agent.memory.generate_messages(node, workspace)
    assert "Updated package guidance" not in str(messages)
    assert node.instruction_updates[-1].content is None
    (tmp_path / "pkg" / "AGENTS.md").write_text("Restored rule")
    await agent.memory.generate_messages(node, workspace)
    # A previously loaded rule that becomes an escaping symlink cannot remain
    # in the active instruction projection merely because its history exists.
    (tmp_path / "pkg" / "AGENTS.md").unlink()
    (tmp_path / "pkg" / "AGENTS.md").symlink_to(tmp_path.parent / "personal.md")
    messages = await agent.memory.generate_messages(node, workspace)
    assert "Restored rule" not in str(messages)
    restored = Node.reconstruct(root.model_dump(), repo=workspace.repository)
    saved = restored.get_node_by_id(node.node_id)
    assert saved.repository_context == node.repository_context
    assert saved.instruction_updates == node.instruction_updates


@pytest.mark.asyncio
async def test_instruction_symlink_outside_repo_never_loaded_and_large_rules_budgeted(tmp_path):
    external = tmp_path / "personal.md"
    external.write_text("Personal instructions should not be read")
    repo = tmp_path / "repo"
    agent, root, workspace = await setup(repo, load=True)
    (repo / "AGENTS.md").symlink_to(external)
    messages = await agent.memory.generate_messages(root, workspace)
    assert "Personal instructions" not in str(messages)
    assert root.repository_context.errors == ["AGENTS.md: ValueError"]
    (repo / "AGENTS.md").unlink()
    (repo / "AGENTS.md").write_text("Large guidance " * 30000)
    with pytest.raises(ContextBudgetExceeded):
        await agent.memory.generate_messages(root, workspace)


@pytest.mark.asyncio
async def test_shadow_versions_include_preserved_patch_and_branch_isolation(tmp_path):
    (tmp_path / "calc.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path, shadow=True, actions=[ReadFile()])
    read = await execute(agent, root, ReadFileArgs(file_path="calc.py"))
    left, right = read.create_child(), read.create_child()
    left.file_context = read.file_context.clone()
    right.file_context = read.file_context.clone()
    left.file_context.get_file("calc.py").apply_changes("value = 2\n")
    await agent.memory.generate_messages(left, workspace)
    await agent.memory.generate_messages(right, workspace)
    assert "value = 2" in str(await agent.memory.generate_messages(left, workspace))
    assert left.repository_context.files != right.repository_context.files
    assert (tmp_path / "calc.py").read_text() == "value = 1\n"


@pytest.mark.asyncio
async def test_batch_disk_read_after_patch_and_bash_change_preserves_edit_base(tmp_path):
    (tmp_path / "calc.py").write_text("value = 1\n")
    agent, root, workspace = await setup(tmp_path, actions=[ReadFile(), StringReplace(), LocalBash()], shadow=False)
    viewed = await execute(agent, root, ReadFileArgs(file_path="calc.py"))
    edited = await execute(agent, viewed, StringReplaceArgs(path="calc.py", old_str="value = 1", new_str="value = 2"))
    changed = await execute(agent, edited, BashArgs(command="printf 'value = 3\\n' > calc.py"))
    read = await execute(agent, changed, ReadFileArgs(file_path="calc.py"))
    assert "value = 3" in read.observation.message
    assert read.file_context.get_file("calc.py").get_base_content() == "value = 1\n"
    assert "+value = 3" in read.file_context.generate_git_patch()
    assert "+value = 2" not in read.file_context.generate_git_patch()


def test_legacy_nodes_do_not_gain_empty_optional_snapshot_fields():
    node = Node(node_id=0, user_message="old")
    dump = node.model_dump()
    assert not {"repository_context", "instruction_updates", "completion_status"}.intersection(dump)
