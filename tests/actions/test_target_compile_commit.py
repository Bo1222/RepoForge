"""Target compiler facts and optimistic commit checks across write tools."""

import asyncio
from pathlib import Path
import sys

import pytest

from tests.memory.test_repository_context import setup, execute
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.actions.create_file import CreateFile, CreateFileArgs
from moatless.actions.insert_line import InsertLine, InsertLinesArgs
from moatless.actions.append_string import AppendString, AppendStringArgs
from moatless.actions.claude_text_editor import ClaudeEditTool, EditActionArguments
from moatless.runtime.runtime import NoEnvironment
from moatless.file_context import FileContext
from moatless.repository.file import FileRepository


def write_cases():
    return [
        (StringReplace(), StringReplaceArgs(path="calc.py", old_str="value = 1", new_str="value = 2")),
        (AppendString(), AppendStringArgs(path="calc.py", new_str="\nother = 2\n")),
        (InsertLine(), InsertLinesArgs(path="calc.py", insert_line=1, new_str="other = 2\n")),
        (CreateFile(), CreateFileArgs(path="new.py", file_text="value = 2\n")),
        (ClaudeEditTool(), EditActionArguments(command="insert", path="calc.py", insert_line=1, new_str="other = 2\n")),
    ]


@pytest.mark.asyncio
async def test_real_target_interpreter_and_nonexecuting_compile(tmp_path):
    runtime = NoEnvironment()
    runtime.python_executable = "/usr/bin/python3"
    side_effect = tmp_path / "executed"
    check = await runtime.compile_python(f"open({str(side_effect)!r},'w').write('bad')\n", "calc.py")
    assert check["status"] == "valid" and check["interpreter"]["executable"] == "/usr/bin/python3"
    assert check["interpreter"]["executable"] != sys.executable
    assert not side_effect.exists()
    invalid = await runtime.compile_python("return True\n", "calc.py")
    assert invalid["status"] == "invalid" and invalid["line"] == 1


@pytest.mark.asyncio
async def test_unavailable_target_does_not_claim_host_compatibility(tmp_path):
    (tmp_path / "calc.py").write_text("value = 1\n")
    agent, root, _ = await setup(tmp_path, actions=[StringReplace()], runtime=NoEnvironment())
    node = await execute(agent, root, StringReplaceArgs(path="calc.py", old_str="value = 1", new_str="value = 2"))
    assert node.observation.properties["syntax_checked"] is False
    assert node.observation.properties["syntax_checks"][0]["status"] == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("action,args", write_cases(), ids=["replace", "append", "insert", "create", "text_editor"])
async def test_source_mutated_during_compilation_is_never_overwritten(tmp_path, action, args):
    (tmp_path / "calc.py").write_text("value = 1\n")
    runtime = NoEnvironment()
    runtime.python_executable = "/usr/bin/python3"
    compile_target = runtime.compile_python

    async def concurrent_writer(source, path):
        check = await compile_target(source, path)
        (tmp_path / args.path).write_text("external = 99\n")
        return check

    runtime.compile_python = concurrent_writer
    agent, root, _ = await setup(tmp_path, actions=[action], runtime=runtime)
    root.file_context.add_file("calc.py", show_all_spans=True)
    node = await execute(agent, root, args)
    assert node.observation.properties["fail_reason"] == "edit_conflict"
    assert (tmp_path / args.path).read_text() == "external = 99\n"
    assert not node.file_context.generate_git_patch().strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("action,args", write_cases(), ids=["replace", "append", "insert", "create", "text_editor"])
async def test_target_rejection_reaches_all_write_tools_before_commit(tmp_path, action, args):
    (tmp_path / "calc.py").write_text("value = 1\n")
    runtime = NoEnvironment()

    async def target(source, path):
        # Deliberately stricter target than the host, to prove tools obey the
        # adapter result rather than a host compile. The real adapter is tested above.
        return {
            "status": "valid" if source.strip() in {"", "value = 1"} else "invalid",
            "interpreter": {"executable": "target"},
            "message": "target grammar rejects candidate",
        }

    runtime.compile_python = target
    agent, root, _ = await setup(tmp_path, actions=[action], runtime=runtime)
    root.file_context.add_file("calc.py", show_all_spans=True)
    node = await execute(agent, root, args)
    assert node.observation.properties["fail_reason"] == "syntax_error"
    assert (tmp_path / "calc.py").read_text() == "value = 1\n"
    assert not (tmp_path / "new.py").exists()
    assert not node.file_context.generate_git_patch().strip()


def test_failed_atomic_write_does_not_publish_context_patch(tmp_path, monkeypatch):
    (tmp_path / "calc.py").write_text("value = 1\n")
    repo = FileRepository(repo_path=str(tmp_path))
    context = FileContext(repo=repo)
    file = context.add_file("calc.py")

    def fail(*args):
        raise OSError("filesystem full")

    monkeypatch.setattr("moatless.repository.file.os.replace", fail)
    with pytest.raises(OSError, match="full"):
        file.apply_changes("value = 2\n")
    assert not file.patch and not file.was_edited
    assert file.content == "value = 1\n" and (tmp_path / "calc.py").read_text() == "value = 1\n"
    assert list(tmp_path.iterdir()) == [tmp_path / "calc.py"]


@pytest.mark.asyncio
async def test_real_older_target_rejects_host_valid_match_syntax(tmp_path):
    target = Path("/mnt/data2/home/geqb/tmp-t7-smoke-rebuild/pytest-online-full/bin/python3.9")
    if not target.exists():
        pytest.skip("Optional public Python 3.9 task prefix is unavailable")
    source = "match 1:\n    case 1:\n        value = 2\n"
    compile(source, "calc.py", "exec")  # valid in the framework's Python 3.10
    (tmp_path / "calc.py").write_text("value = 1\n")
    runtime = NoEnvironment()
    runtime.python_executable = str(target)
    agent, root, _ = await setup(tmp_path, actions=[StringReplace()], runtime=runtime)
    node = await execute(agent, root, StringReplaceArgs(path="calc.py", old_str="value = 1", new_str=source))
    check = node.observation.properties["syntax_error"]
    assert check["status"] == "invalid" and check["interpreter"]["version"].startswith("3.9.")
    assert (tmp_path / "calc.py").read_text() == "value = 1\n"


@pytest.mark.asyncio
async def test_large_candidate_uses_stdin_instead_of_shell_argument_limit():
    runtime = NoEnvironment()
    runtime.python_executable = "/usr/bin/python3"
    check = await runtime.compile_python("#" + "large source " * 30000 + "\nvalue = 1\n", "large.py")
    assert check["status"] == "valid"


@pytest.mark.asyncio
async def test_switching_target_interpreter_invalidates_compilation_and_test_receipts(tmp_path):
    from tests.policy.test_acceptance_repair import fixture, plan, edit
    from moatless.actions.register_acceptance_plan import RegisterAcceptancePlanArgs
    from moatless.actions.run_tests import RunTestsArgs
    from moatless.actions.check_syntax import CheckSyntaxArgs
    from moatless.policy.repair import assess

    agent, root, workspace, runtime = await fixture(tmp_path)
    node = await execute(
        agent,
        root,
        RegisterAcceptancePlanArgs(plan=plan()),
        edit(),
        RunTestsArgs(test_files=["test_calc.py"], test_source="base_public"),
        RunTestsArgs(test_files=["test_calc.py"], test_source="base_public", product_source="base"),
    )
    assert assess(node, workspace).state == "ready"
    runtime.python_executable = "/usr/bin/python3"
    pending = assess(node, workspace)
    assert pending.state != "ready" and pending.syntax_pending == ["calc.py"]
    node = await execute(agent, node, CheckSyntaxArgs(paths=["calc.py"]))
    assert node.observation.properties["syntax_checks"][0]["interpreter"]["executable"] == "/usr/bin/python3"


@pytest.mark.asyncio
async def test_batch_runtime_compiles_stdin_in_actual_task_shell_without_execution(tmp_path, monkeypatch):
    from moatless.runtime.local import SweBenchLocalEnvironment
    from tests.runtime.test_generation_runtime import PUBLIC

    monkeypatch.setenv("SKIP_INSTALL", "true")
    monkeypatch.setenv("SKIP_CONDA_ACTIVATE", "true")
    runtime = SweBenchLocalEnvironment(tmp_path, PUBLIC, None, generation_only=True)
    await runtime._wait_for_install()
    source = "open('must-not-exist','w').write('unsafe')\n" + "#" + "large source " * 30000 + "\n"
    check = await runtime.compile_python(source, "path with 'quotes.py")
    assert check["status"] == "valid" and check["interpreter"]["executable"]
    assert not (tmp_path / "must-not-exist").exists()
    invalid = await runtime.compile_python("return True\n", "bad.py")
    assert invalid["status"] == "invalid"
