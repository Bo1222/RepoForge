"""Deterministic policy boundaries and real ordered-flow integration."""
import asyncio
import pytest

from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.run_tests import RunTests
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.actions.verified_finish import VerifiedFinishArgs
from moatless.flow.langgraph_flow import LangGraphFlow
from moatless.flow.loop import AgenticLoop
from moatless.node import ActionStep
from moatless.policy import ActionPolicy, ActionPolicyConfig, BudgetSnapshot, PolicyState
from scripts.migration_smoke import response
from tests.memory.test_execution_memory import ResultsRuntime
from tests.memory.test_repository_context import setup
from moatless.testing.schema import TestResult as Result, TestStatus as Status


def state(**changes):
    values = dict(node_id=1, step_index=0, patch_hash='full', product_patch_hash='product',
                  has_product_patch=True, signature='action', correction_key='check')
    values.update(changes)
    return PolicyState(**values)


def decide(variant, features, action=None, **budget):
    return ActionPolicy(ActionPolicyConfig(variant=variant)).decide(features,
        ActionStep(action=action or FinishArgs(finish_reason='done')),
        BudgetSnapshot(remaining_rounds=budget.pop('remaining_rounds', 20), **budget))


def test_finish_correction_is_bounded_and_disabled_is_passthrough():
    assert decide('off', state()).kind == 'allow'
    assert decide('a1', state()).reason == 'need_public_test_target'
    assert decide('a1', state(corrected=True)).kind == 'stop'
    assert decide('a1', state(target='test_code.py')).kind == 'verify'
    assert decide('a1', state(public_test_passed=True)).kind == 'allow'


def test_repeat_and_stagnation_ablation_and_budget_reserve():
    assert decide('a2', state(repeat_count=2)).kind == 'replan'
    assert decide('a3', state(no_progress=4)).kind == 'replan'
    assert decide('a4', state(no_progress=4, repeat_count=2, public_test_passed=True)).kind == 'allow'
    assert decide('a2', state(repeat_count=2, replan_attempts=1)).kind == 'stop'
    assert decide('a3', state(target='test_code.py'), remaining_rounds=3).reason == 'budget_reserve'
    assert decide('a3', state(target='test_code.py'), cost_complete=False).kind == 'stop'


def test_feedback_and_cost_control_have_distinct_edit_thresholds():
    read = ReadFileArgs(file_path='code.py')
    assert decide('a1', state(edits_since_test=2), read).kind == 'allow'
    assert decide('a2', state(edits_since_test=2, target='test_code.py'), read).kind == 'verify'
    assert decide('a3', state(edits_since_test=2, changed_product_files=1), read).kind == 'allow'
    assert decide('a3', state(edits_since_test=1, changed_product_files=2, target='test_code.py'), read).kind == 'verify'


@pytest.mark.asyncio
async def test_ordered_edit_then_finish_inserts_public_test_and_keeps_proposals(tmp_path):
    (tmp_path / 'code.py').write_text('value = 1\n')
    (tmp_path / 'test_code.py').write_text('def test_code(): pass\n')
    runtime = ResultsRuntime([Result(status=Status.PASSED, name='test_code', file_path='test_code.py')])
    agent, root, workspace = await setup(tmp_path, actions=[StringReplace(), ReadFile(), RunTests(), Finish()],
                                         runtime=runtime, shadow=True)
    root.file_context.add_file('code.py', show_all_spans=True)
    root.file_context.add_file('test_code.py', show_all_spans=True)
    agent.completion_model._responses = [
        response(StringReplaceArgs(path='code.py', old_str='value = 1', new_str='value = 2'),
                 FinishArgs(finish_reason='premature')),
        response(FinishArgs(finish_reason='checked')),
    ]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=10, action_policy={'variant': 'a1'})
    last = await asyncio.wait_for(flow.run(workspace=workspace), timeout=10)
    edited = root.children[0]
    assert edited.action_steps[1].observation.properties['execution_status'] == 'skipped'
    assert edited.action_steps[2].origin == 'policy'
    assert edited.action_steps[2].observation.properties['test_source'] == 'base_public'
    assert last.is_terminal() and last.completion_status['status'] == 'verified'
    assert agent.completion_model._calls == 2
    assert flow.graph_state['policy_state']['corrections']
    assert 'Action scheduled by policy' in str(agent.completion_model._messages[-1])


@pytest.mark.asyncio
async def test_failed_verification_stops_without_finish_interception_loop(tmp_path):
    (tmp_path / 'code.py').write_text('value = 1\n')
    (tmp_path / 'test_code.py').write_text('def test_code(): pass\n')
    runtime = ResultsRuntime([Result(status=Status.FAILED, name='test_code', file_path='test_code.py')])
    agent, root, workspace = await setup(tmp_path, actions=[StringReplace(), RunTests(), Finish()], runtime=runtime, shadow=True)
    root.file_context.add_file('code.py').apply_changes('value = 2\n')
    root.file_context.add_file('test_code.py')
    agent.completion_model._responses = [response(FinishArgs(finish_reason='done'))] * 3
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=10, action_policy={'variant': 'a1'})
    last = await asyncio.wait_for(flow.run(workspace=workspace), timeout=10)
    assert flow.is_finished() == 'policy_stop' and last.completion_status['status'] == 'unverified'
    assert agent.completion_model._calls == 2


def test_loop_cannot_silently_ignore_enabled_policy():
    with pytest.raises(ValueError, match='LangGraphFlow'):
        AgenticLoop(action_policy={'variant': 'a1'})


@pytest.mark.asyncio
@pytest.mark.parametrize('variant,proposals,executed,reason', [
    ('a2', 3, 2, 'policy_stop'), ('a2', 4, 2, 'policy_stop'), ('a4', 3, 3, 'policy_stop')])
async def test_actual_noop_edits_are_suppressed_and_ablation_is_real(tmp_path, variant, proposals, executed, reason):
    (tmp_path / 'code.py').write_text('value = 1\n')
    agent, root, workspace = await setup(tmp_path, actions=[StringReplace(), Finish()], shadow=True)
    root.file_context.add_file('code.py', show_all_spans=True)
    noop = StringReplaceArgs(path='code.py', old_str='value = 1', new_str='value = 1')
    agent.completion_model._responses = [response(noop)] * proposals + [response(FinishArgs(finish_reason='no change'))]
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=10, action_policy={'variant': variant})
    await asyncio.wait_for(flow.run(workspace=workspace), timeout=10)
    attempts = [s for n in flow.root.get_all_nodes() for s in n.action_steps
                if s.observation and s.observation.properties.get('fail_reason') == 'no_changes']
    assert len(attempts) == executed
    assert flow.is_finished() == reason


class SimulatedCrash(BaseException):
    pass


class RecoverableRuntime(ResultsRuntime):
    def __init__(self, fault):
        super().__init__([Result(status=Status.PASSED, name='test_code', file_path='test_code.py')])
        self.calls = 0
        self.fault = fault

    def recovery_state(self):
        return {'kind': 'policy-test-runtime', 'version': 1}

    async def run_tests(self, patch=None, test_files=None):
        if self.fault == 'before_effect':
            self.fault = None
            raise SimulatedCrash()
        self.calls += 1
        if self.fault == 'after_effect':
            self.fault = None
            raise SimulatedCrash()
        self.last_test_execution = [{"target": path, "command": "fixture-test", "return_code": 0} for path in test_files]
        return self.results


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['policy_committed', 'test_committed', 'before_effect', 'after_effect'])
async def test_policy_verification_crash_windows_do_not_replay_effects(tmp_path, fault):
    from moatless.flow.durable import RecoveryConflict
    directory = tmp_path / 'repo'
    directory.mkdir()
    (directory / 'code.py').write_text('value = 1\n')
    (directory / 'test_code.py').write_text('def test_code(): pass\n')
    runtime = RecoverableRuntime(fault)
    agent, root, workspace = await setup(directory, actions=[RunTests(), Finish()], runtime=runtime, shadow=True)
    root.file_context.add_file('code.py').apply_changes('value = 2\n')
    root.file_context.add_file('test_code.py')
    agent.completion_model._responses = [response(FinishArgs(finish_reason='done'))] * 3
    flow = LangGraphFlow.create(root=root, agent=agent, max_iterations=10, action_policy={'variant': 'a1'},
        checkpoint_path=str(tmp_path / 'state.sqlite'), trajectory_id='crash-' + fault)
    fired = False

    async def crash(current):
        nonlocal fired
        state = current.graph_state
        node = current.get_node_by_id(state['current_node_id'])
        cursor = state['action_cursor']
        policy_step = cursor < len(node.action_steps) and node.action_steps[cursor].origin == 'policy'
        if not fired and policy_step and ((fault == 'policy_committed' and state['phase'] == 'execute')
                or (fault == 'test_committed' and state['phase'] == 'record')):
            fired = True
            raise SimulatedCrash()

    flow._on_snapshot = crash
    with pytest.raises(SimulatedCrash):
        await asyncio.wait_for(flow.run(workspace=workspace), timeout=10)
    calls = runtime.calls
    flow._on_snapshot = None
    if fault in {'before_effect', 'after_effect'}:
        with pytest.raises(RecoveryConflict, match='Unconfirmed'):
            await flow.run(workspace=workspace)
        assert runtime.calls == calls
    else:
        last = await asyncio.wait_for(flow.run(workspace=workspace), timeout=10)
        assert runtime.calls == 1 and last.is_terminal()
        assert last.completion_status['status'] == 'verified'
        assert agent.completion_model._calls == 2


def test_no_progress_requires_verification_before_stop():
    read = ReadFileArgs(file_path='code.py')
    # No target yet: ask for a public test target instead of stopping.
    decision = decide('a3', state(no_progress=4), read)
    assert decision.kind == 'replan' and decision.reason == 'need_public_test_target'
    # Target exists: verify before accepting the no-progress stop.
    decision = decide('a3', state(no_progress=4, target='test_code.py'), read)
    assert decision.kind == 'verify' and decision.target == 'test_code.py'
    # A failed verification is repair evidence, not a stop.
    decision = decide('a3', state(no_progress=4, target='test_code.py',
                                  verification_state='TESTED_FAIL', current_test_failed=True,
                                  introduced_failures=['test_code.py::test_reg']), read)
    assert decision.kind == 'replan' and decision.reason == 'p2p_regression'
    # Bounded: after one replan the policy may stop.
    assert decide('a3', state(no_progress=4, target='test_code.py', replan_attempts=1), read).kind == 'stop'


def test_failed_verification_requests_repair_not_repeat():
    decision = decide('a3', state(current_test_failed=True, target='test_code.py',
                                  verification_state='TESTED_FAIL',
                                  introduced_failures=['test_code.py::test_reg']))
    assert decision.kind == 'replan' and decision.reason == 'p2p_regression'
    decision = decide('a3', state(current_test_failed=True, target='test_code.py',
                                  verification_state='TESTED_FAIL', failure_kind='assertion_failure'))
    assert decision.kind == 'replan' and decision.reason == 'assertion_failure'
    # A corrected verification is bounded: a terminal proposal stops.
    assert decide('a3', state(current_test_failed=True, target='test_code.py',
                              verification_state='TESTED_FAIL', corrected=True)).kind == 'stop'


def test_target_found_without_target_does_not_stop():
    read = ReadFileArgs(file_path='code.py')
    decision = decide('a3', state(no_progress=4, verification_state='TARGET_FOUND'), read)
    assert decision.kind == 'replan' and decision.reason == 'need_public_test_target'


def test_o2_verified_finish_is_terminal():
    finish = VerifiedFinishArgs(finish_reason="done", test_verification="public tests pass")
    assert decide('a5', state(has_product_patch=False), action=finish).reason == 'no_product_patch'
    decision = decide('a5', state(target='test_code.py', public_test_passed=False), action=finish)
    assert decision.kind == 'verify' and decision.target == 'test_code.py'
    assert decide('a5', state(public_test_passed=True), action=finish).kind == 'allow'


def test_a5_uses_the_same_verification_first_contract():
    read = ReadFileArgs(file_path='code.py')
    assert decide('a5', state(no_progress=4, target='test_code.py'), read).kind == 'verify'
    assert decide('a5', state(no_progress=4, target='test_code.py', replan_attempts=1), read).kind == 'stop'


def test_exact_repeated_action_after_failure_is_replanned():
    read = ReadFileArgs(file_path='code.py')
    decision = decide('a3', state(last_action_signature='action',
                                  failure_signature='sig', failure_kind='assertion_failure'), read)
    assert decision.kind == 'replan' and decision.reason == 'repeated_assertion_failure'
    decision = decide('a3', state(last_action_signature='action',
                                  failure_signature='sig', failure_kind='assertion_failure',
                                  corrected=True), read)
    assert decision.kind == 'stop'
    # A1 keeps the historical behavior; the hard repeat rule is O2/a3 behavior.
    assert decide('a1', state(last_action_signature='action',
                              failure_signature='sig', failure_kind='assertion_failure'), read).kind == 'allow'
