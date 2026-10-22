"""Regression cases found while auditing frozen T4/T5 trajectories."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from moatless.actions.append_string import AppendString, AppendStringArgs
from moatless.actions.find_code_snippet import FindCodeSnippet, FindCodeSnippetArgs
from moatless.actions.insert_line import InsertLine, InsertLinesArgs
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.actions.finish import Finish
from moatless.actions.view_code import ViewCode, ViewCodeArgs, CodeSpan
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.repository.repository import InMemRepository
from moatless.testing.schema import TestResult as Result, TestStatus as Status
from moatless.workspace import Workspace
from tests.memory.test_execution_memory import ResultsRuntime
from tests.memory.test_repository_context import setup, execute


@pytest.mark.asyncio
async def test_view_current_lines_after_patch_and_preserve_multiple_ranges():
    base = 'def target():\n    return 1\n\ndef other():\n    return 9\n'
    repo = InMemRepository({'demo.py': base}, shadow_mode=True)
    context = FileContext(repo=repo)
    context.add_file('demo.py').apply_changes('# inserted\n' * 5 + base)
    action = ViewCode()
    await action.initialize(Workspace(repository=repo))
    args = ViewCodeArgs(files=[CodeSpan(file_path='demo.py', start_line=6, end_line=7),
                               CodeSpan(file_path='demo.py', start_line=9, end_line=10)])
    observation = await action.execute(args, context)
    assert 'def target()' in observation.message and 'def other()' in observation.message
    assert observation.properties['files']['demo.py']['requested_ranges'] == [[6, 7], [9, 10]]
    single = await action.execute(ViewCodeArgs(files=[args.files[0]]), context)
    assert 'def target()' in single.message and 'def other()' not in single.message
    full = await action.execute(ViewCodeArgs(files=[CodeSpan(file_path='demo.py')]), context)
    assert 'def target()' in full.message and 'def other()' in full.message


@pytest.mark.asyncio
async def test_auxiliary_selection_also_uses_the_edited_version():
    from moatless.actions.identify_mixin import Identify, IdentifiedSpans
    from scripts.migration_smoke import ScriptedCompletion, response
    base = 'def target():\n    return 1\n\ndef other():\n    return 9\n'
    repo = InMemRepository({'demo.py': base}, shadow_mode=True)
    context = FileContext(repo=repo)
    context.add_file('demo.py').apply_changes('# inserted\n' * 5 + base)
    model = ScriptedCompletion([response(Identify(identified_spans=[
        IdentifiedSpans(file_path='demo.py', start_line=6, end_line=7)]))])
    action = ViewCode(max_tokens=1, completion_model=model)
    await action.initialize(Workspace(repository=repo))
    observed = await action.execute(ViewCodeArgs(files=[CodeSpan(file_path='demo.py')]), context)
    assert 'def target()' in observed.message and 'def other()' not in observed.message
    assert observed.properties['files']['demo.py']['display_ranges'][0] == [6, 7]
    assert model._calls == 1


@pytest.mark.asyncio
async def test_current_snippet_follows_requested_view_instead_of_first_200_lines(tmp_path):
    (tmp_path / 'code.py').write_text('# padding\n' * 400 + 'def target():\n    return 42\n')
    agent, root, workspace = await setup(tmp_path, actions=[ViewCode(), Finish()], shadow=True)
    node = await execute(agent, root, ViewCodeArgs(files=[CodeSpan(file_path='code.py', start_line=401, end_line=402)]))
    messages = await agent.memory.generate_messages(node, workspace)
    assert not any('<current_code ' in (m.get('content') or '') for m in messages)
    assert 'def target()' in str(messages) and '1: # padding' not in str(messages)


class DirectoryRepo(InMemRepository):
    def is_directory(self, path):
        return path.rstrip('/') == 'tests'

    async def find_exact_matches(self, search_text, file_pattern=None):
        return [('tests/demo.py', 2)]


@pytest.mark.asyncio
@pytest.mark.parametrize('pattern', ['tests', 'tests/', 'tests/*.py', 'tests/demo.py'])
async def test_directory_and_glob_do_not_delete_real_matches(pattern):
    repo = DirectoryRepo({'tests/demo.py': 'def demo():\n    return "filterable"\n'})
    action = FindCodeSnippet()
    action._workspace = Workspace(repository=repo)
    context, _ = await action._search_for_context(FindCodeSnippetArgs(code_snippet='filterable', file_pattern=pattern))
    assert not context.is_empty()


@pytest.mark.asyncio
async def test_test_category_reaches_vector_search_with_test_file_filter():
    index = object.__new__(CodeIndex)
    index.matching_files = AsyncMock(return_value=['tests/demo.py'])
    index._vector_search = AsyncMock(return_value=[])
    result = await index.semantic_search('demo', file_pattern='tests/*.py', category='test')
    index._vector_search.assert_awaited_once()
    assert result.hits == []


@pytest.mark.asyncio
@pytest.mark.parametrize('statuses,code,expected', [
    ([Status.PASSED, Status.SKIPPED], 0, 'passed'),
    ([Status.SKIPPED], 0, 'unknown'),
    ([], 0, 'unknown'),
    ([Status.PASSED], 2, 'error'),
    ([Status.PASSED], 1, 'unknown'),
    ([Status.ERROR], 2, 'error'),
])
async def test_exit_status_and_skips_are_classified_without_false_success(tmp_path, statuses, code, expected):
    (tmp_path / 'test_code.py').write_text('def test_code(): pass\n')
    runtime = ResultsRuntime([Result(status=s, name=str(i), file_path='test_code.py') for i, s in enumerate(statuses)])
    runtime.return_code = code
    agent, root, _ = await setup(tmp_path, actions=[RunTests(), Finish()], runtime=runtime)
    node = await execute(agent, root, RunTestsArgs(test_files=['test_code.py']))
    assert node.observation.properties['test_run']['status'] == expected


@pytest.mark.asyncio
async def test_base_public_excludes_test_changes_and_records_origin(tmp_path):
    (tmp_path / 'code.py').write_text('value = 1\n')
    (tmp_path / 'test_code.py').write_text('def test_code(): pass\n')
    runtime = ResultsRuntime([Result(status=Status.PASSED, name='test_code', file_path='test_code.py')])
    runtime.run_tests = AsyncMock(wraps=runtime.run_tests)
    agent, root, _ = await setup(tmp_path, actions=[RunTests()], runtime=runtime, shadow=True)
    root.file_context.add_file('code.py').apply_changes('value = 2\n')
    root.file_context.add_file('test_code.py').apply_changes('def test_code(): assert True\n')
    node = await execute(agent, root, RunTestsArgs(test_files=['test_code.py'], test_source='base_public'))
    patch = runtime.run_tests.call_args.kwargs['patch']
    assert 'code.py' in patch and 'test_code.py' not in patch
    assert node.observation.properties['test_run']['test_source'] == 'base_public'
    assert node.observation.properties['test_run']['test_origins'] == {'test_code.py': 'modified'}


@pytest.mark.asyncio
async def test_run_tests_timeout_returns_observation_instead_of_killing_job(tmp_path):
    (tmp_path / 'test_code.py').write_text('def test_code(): pass\n')
    runtime = ResultsRuntime([])
    runtime.run_tests = AsyncMock(side_effect=asyncio.TimeoutError)
    agent, root, _ = await setup(tmp_path, actions=[RunTests(timeout=1), Finish()], runtime=runtime)
    node = await execute(agent, root, RunTestsArgs(test_files=['test_code.py']))
    assert node.observation.properties.get('fail_reason') == 'tool_timeout'
    assert node.observation.properties.get('timed_out') is True


@pytest.mark.asyncio
async def test_modified_tests_require_base_public(tmp_path):
    (tmp_path / 'code.py').write_text('value = 1\n')
    (tmp_path / 'test_code.py').write_text('def test_code(): pass\n')
    runtime = ResultsRuntime([Result(status=Status.PASSED, name='test_code', file_path='test_code.py')])
    agent, root, _ = await setup(tmp_path, actions=[RunTests()], runtime=runtime, shadow=True)
    root.file_context.add_file('code.py').apply_changes('value = 2\n')
    root.file_context.add_file('test_code.py').apply_changes('def test_code(): assert True\n')
    node = await execute(agent, root, RunTestsArgs(test_files=['test_code.py']))
    assert node.observation.properties.get('requires_base_public') is True
    assert node.observation.properties.get('test_source') == 'workspace'


@pytest.mark.asyncio
async def test_run_tests_timeout_returns_completed_partial_results(tmp_path):
    (tmp_path / 'test_code.py').write_text('def test_code(): pass\n')
    runtime = ResultsRuntime([Result(status=Status.PASSED, name='test_code', file_path='test_code.py')])
    async def timeout_with_partial(**kwargs):
        runtime.last_test_results = [Result(status=Status.PASSED, name='test_code', file_path='test_code.py')]
        runtime.last_test_execution = [{'command': 'pytest', 'target': 'test_code.py', 'return_code': 0}]
        raise asyncio.TimeoutError()
    runtime.run_tests = timeout_with_partial
    agent, root, _ = await setup(tmp_path, actions=[RunTests(timeout=1), Finish()], runtime=runtime)
    node = await execute(agent, root, RunTestsArgs(test_files=['test_code.py']))
    props = node.observation.properties
    assert props['fail_reason'] == 'tool_timeout'
    assert len(props['test_results']) == 1
    assert props['test_execution'] == runtime.last_test_execution
    assert props['timed_out'] and not props['complete']


@pytest.mark.asyncio
async def test_insert_lines_autoloads_and_rejects_syntax_error(tmp_path):
    (tmp_path / 'code.py').write_text('def value():\n    return 1\n')
    agent, root, _ = await setup(tmp_path, actions=[InsertLine(), Finish()])
    node = await execute(agent, root, InsertLinesArgs(path='code.py', insert_line=1, new_str='# header'))
    assert node.observation.properties.get('fail_reason') is None
    assert '# header' in (tmp_path / 'code.py').read_text()
    bad = await execute(agent, node, InsertLinesArgs(path='code.py', insert_line=1, new_str='def broken(:'))
    assert bad.observation.properties.get('fail_reason') == 'syntax_error'
    assert 'def broken(' not in (tmp_path / 'code.py').read_text()


@pytest.mark.asyncio
async def test_append_string_rejects_syntax_error(tmp_path):
    (tmp_path / 'code.py').write_text('value = 1\n')
    agent, root, _ = await setup(tmp_path, actions=[AppendString(), Finish()])
    node = await execute(agent, root, AppendStringArgs(path='code.py', new_str='def broken(:'))
    assert node.observation.properties.get('fail_reason') == 'syntax_error'
    assert (tmp_path / 'code.py').read_text() == 'value = 1\n'
