"""Save task-local implementation evidence; never run a model or benchmark."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / 'analysis-tools/research/a6-six-groups'
DOC = ROOT / 'docs/2026-10-22_A6_六组缺口修复验收.md'
BASE = json.loads((ART/'baseline.json').read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

runs = [
    ('final-acceptance-current.log',124,'MOATLESS_TEST_SELECTOR_TICK=1 timeout 300 .venv/bin/python -m pytest -p tests.selector_tick -q -o log_cli=false --tb=short tests/policy tests/actions/test_target_compile_commit.py tests/runtime/test_deadline_contract.py'),
    ('final-compatibility-tests-round2.log',245,'MOATLESS_TEST_SELECTOR_TICK=1 timeout 220 .venv/bin/python -m pytest -p tests.selector_tick -q -o log_cli=false --tb=short tests/flow tests/runtime/test_generation_runtime.py tests/runtime/test_native_tool_isolation.py tests/agent/test_public_steps.py tests/memory tests/actions/test_run_tests.py tests/actions/test_t6_tool_contracts.py tests/actions/test_string_replace.py tests/actions/test_insert_line.py tests/actions/test_create_file.py tests/actions/test_search_fallbacks.py tests/actions/test_find_code_snippet.py tests/evaluation/test_verification.py'),
    ('final-protocol-tests.log',64,'timeout 120 .venv/bin/python -m pytest -p tests.selector_tick -q -o log_cli=false --tb=short tests/evaluation/test_t4_t5_experiment.py tests/evaluation/test_t6_experiment.py tests/evaluation/test_t7_experiment.py tests/evaluation/test_native_experiment.py tests/evaluation/test_native_configuration_audit.py tests/evaluation/test_generation_worker.py'),
    ('final-followup-tests.log',61,'MOATLESS_TEST_SELECTOR_TICK=1 timeout 120 .venv/bin/python -m pytest -p tests.selector_tick -q -o log_cli=false --tb=short tests/runtime/test_generation_runtime.py tests/runtime/test_native_tool_isolation.py tests/flow/test_langgraph_flow.py::test_m0_m1_frozen_business_replay tests/flow/test_loop_contract.py::test_frozen_replay tests/actions/test_run_tests.py tests/actions/test_t6_tool_contracts.py tests/actions/test_string_replace.py tests/memory/test_execution_memory.py'),
    ('final-compiler-recovery-tests.log',51,'MOATLESS_TEST_SELECTOR_TICK=1 timeout 120 .venv/bin/python -m pytest -p tests.selector_tick -q -o log_cli=false --tb=short tests/actions/test_target_compile_commit.py tests/runtime/test_deadline_contract.py tests/runtime/test_generation_runtime.py tests/policy/test_six_group_flow.py tests/policy/test_acceptance_repair.py::test_actual_graph_durable_resume_never_replays_completed_effects tests/evaluation/test_t4_t5_experiment.py::test_same_source_loop_graph_fixed_replay tests/memory/test_execution_memory.py'),
    ('final-replan-counter-tests.log',8,'timeout 70 .venv/bin/python -m pytest -p tests.selector_tick -q -o log_cli=false --tb=short tests/policy/test_six_group_contracts.py::test_repair_revisit_target_change_and_new_evidence_replan_are_durable tests/policy/test_six_group_contracts.py::test_new_repro_can_reprove_unchanged_old_failure_with_real_new_bundle_baseline tests/policy/test_six_group_contracts.py::test_added_product_file_or_changed_exception_cannot_reset_same_behavior_repair_cap tests/policy/test_six_group_flow.py'),
    ('final-batch-compiler-test.log',1,'timeout 20 .venv/bin/python -m pytest -p tests.selector_tick -q -o log_cli=false --tb=short tests/actions/test_target_compile_commit.py::test_batch_runtime_compiles_stdin_in_actual_task_shell_without_execution'),
]
validation=[]
for name,count,command in runs:
    log=ART/name
    text=log.read_text()
    assert re.search(rf'\b{count} passed\b',text),name
    assert not re.search(r'^FAILED |\b[1-9]\d* failed\b',text,re.M),name
    validation.append({'command':command,'cwd':str(ROOT),'exit_code':0,'passed':count,'log':str(log.relative_to(ROOT)),'sha256':sha(log)})
assert (ART/'final-lint.log').read_text().strip() == 'All checks passed!'

with tarfile.open(ART/'baseline.tar.gz') as archive:
    old={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
changed={};parts=[]
paths=set(BASE['files']) | {str(p.relative_to(ROOT)) for p in ROOT.glob('moatless/**/*.py')} | {str(p.relative_to(ROOT)) for p in ROOT.glob('tests/**/*.py')}
for name in sorted(paths):
    path=ROOT/name
    if not path.is_file() or BASE['files'].get(name)==sha(path): continue
    changed[name]={'before':BASE['files'].get(name),'after':sha(path)}
    if name.endswith('.py'):
        parts.extend(difflib.unified_diff(old.get(name,b'').decode().splitlines(True),path.read_text().splitlines(True),fromfile='baseline/'+name,tofile='final/'+name))
(ART/'changes.diff').write_text(''.join(parts))
(ART/'changed-files.json').write_text(json.dumps(changed,indent=2,ensure_ascii=False)+'\n')

# The accepted plan and previous report remain evidence, not rewritten claims.
for name in ('docs/2026-10-21_批量Agent_验证驱动修复优化方案.md','docs/2026-10-22_A6_验收驱动修复_实施说明.md'):
    assert sha(ROOT/name)==BASE['files'][name],name
broken=[]
for target in re.findall(r'\]\(([^)]+)\)',DOC.read_text()):
    if '://' in target or target.startswith('#'): continue
    if not (DOC.parent/target.split('#')[0]).exists():broken.append(target)
assert not broken,broken

groups=[
    ('G1','历史相关失败不会因版本、目标或当前 import 改变而消失',['§4.1','§4.5','§5'],[3,10],['moatless/policy/obligations.py','moatless/policy/repair.py'],['test_old_related_failure_survives_edit_and_unrelated_green_until_actual_rerun','test_removing_import_from_unregistered_failed_test_cannot_erase_its_original_relation','test_new_repro_can_reprove_unchanged_old_failure_with_real_new_bundle_baseline']),
    ('G2','一个执行 deadline 与独立清理宽限，取消/恢复可信',['§3.3'],[8],['moatless/runtime/deadline.py','moatless/runtime/local.py','moatless/agent/agent.py','moatless/flow/langgraph_flow.py'],['test_execution_and_cleanup_have_independent_bounded_budgets','test_real_test_process_killed_reaped_and_patch_restored','test_cleanup_failure_is_recovery_error_but_cancellation_still_propagates','test_cleanup_exceeding_grace_leaves_pending_operation_and_resume_cannot_replay']),
    ('G3','按时间顺序计数，有真实相关新证据的 replan 和最后一次修复',['§5'],[10],['moatless/policy/replanning.py','moatless/actions/replan_repair.py','moatless/flow/langgraph_flow.py'],['test_repair_revisit_target_change_and_new_evidence_replan_are_durable','test_added_product_file_or_changed_exception_cannot_reset_same_behavior_repair_cap','test_real_replan_gate_survives_durable_resume_and_one_final_failed_trial_exits','test_model_proposed_tests_share_the_global_verification_budget']),
    ('G4','真实同配方基准差异，无法比较时 unknown/null',['§4.3','§4.4'],[4,6],['moatless/verification_inputs.py','moatless/execution_memory.py'],['test_public_baseline_comparison_reports_regressions_preexisting_and_fixed_by_full_id','test_untrustworthy_comparison_is_unknown_never_empty_regression_set','test_exit_code_and_parsed_counts_must_agree_for_comparison']),
    ('G5','目标解释器编译、源版本提交检查和原子保存',['§3.2'],[1,2],['moatless/edit_guard.py','moatless/runtime/python_check.py','moatless/actions/code_modification_mixin.py','moatless/file_context.py','moatless/repository/file.py'],['test_source_mutated_during_compilation_is_never_overwritten','test_target_rejection_reaches_all_write_tools_before_commit','test_unavailable_target_does_not_claim_host_compatibility','test_real_older_target_rejects_host_valid_match_syntax','test_failed_atomic_write_does_not_publish_context_patch','test_batch_runtime_compiles_stdin_in_actual_task_shell_without_execution']),
    ('G6','完整输入配方、辅助输入、runner/环境及恢复绑定',['§4.3','§4.4'],[3,4,6,13],['moatless/verification_inputs.py','moatless/actions/run_tests.py','moatless/execution_memory.py','moatless/flow/langgraph_flow.py'],['test_workspace_auxiliary_edits_invalidate_receipts_and_require_same_baseline_bundle','test_cache_outputs_preserve_receipt_but_data_add_delete_runner_and_env_stale_it','test_read_cache_cannot_hide_changed_auxiliary_disk_input','test_internal_symlink_input_is_bound_and_external_input_is_unavailable','test_completed_checkpoint_rechecks_compiler_and_runner_inputs_before_reusing_verified']),
]
test_source='\n'.join(p.read_text() for p in (ROOT/'tests').glob('**/*.py'))
requirements=[]
for key,problem,clauses,cases,files,tests in groups:
    assert all((ROOT/name).is_file() for name in files)
    assert all('def '+name+'(' in test_source for name in tests),(key,tests)
    requirements.append({'id':key,'problem':problem,'plan_clauses':clauses,'acceptance_scenarios':cases,'implementation':files,'tests':tests,'status':'complete_offline'})

result={
 'objective':'完整完成六组 plan 缺口及最低离线行为验收',
 'baseline_commit':BASE['commit'],'comparison':'开始本次任务的工作区快照 → 最终工作区；保留用户原有修改',
 'baseline_snapshot':{'path':'analysis-tools/research/a6-six-groups/baseline.tar.gz','sha256':sha(ART/'baseline.tar.gz')},
 'python':sys.version,'requirements':requirements,'validation_runs':validation,
 'coverage_note':'各套件互有重叠，不合计总通过数。主验收 124 项之后的同用例失败族别名调整由独立 8 项计数/真实 Graph 恢复复查覆盖，新增真实 batch 编译由独立 1 项覆盖。',
 'plan_minimum_acceptance':{'items':list(range(1,15)),'mapping':'docs/2026-10-22_A6_六组缺口修复验收.md#3-plan-最低验收清单','status':'covered_offline'},
 'delivery_documents':{str(p.relative_to(ROOT)):sha(p) for p in (DOC,ROOT/'CHANGELOG.md')},
 'changed_python_files':changed,'diff':{'path':'analysis-tools/research/a6-six-groups/changes.diff','sha256':sha(ART/'changes.diff')},
 'lint':{'log':'analysis-tools/research/a6-six-groups/final-lint.log','exit_code':0,'scope':'本次主要新增/修改模块；未宣称仓库历史全部 lint 无问题'},
 'compileall':{'command':'.venv/bin/python -m compileall -q moatless tests/policy tests/runtime tests/actions/test_target_compile_commit.py','exit_code':0},
 'preserved_nonpassing_runs':{'core-round1.log':124,'final-repair-tests.log':124,'final-input-contract-tests.log':124,'final-failure-obligations-tests.log':124,'new-contracts-round1.log':1,'protocol-round1.log':1,'compatibility-round1.log':1,'final-tools-context-tests.log':1,'compatibility-fixes-round2.log':1},
 'model_api_calls':0,'new_benchmark_runs':0,'official_harness':'not_run','resolved':None,
 'remaining_required_work':[],
 'limits':['验证范围限于公开任务和选定测试，不能由此证明 hidden tests 通过。','完整输入范围保守；特殊辅助路径由 runtime 明确声明，外部可变依赖仍需 runtime 环境身份。','旧期望冲突只能支持 partially_verified；原始失败及证据仍保留。','没有提交或推送，也未部署。']}
(ART/'completion-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'groups':len(requirements),'minimum_scenarios':14,'validation_runs':len(validation),'changed_files':len(changed),'broken_doc_links':broken,'remaining_required_work':[]},ensure_ascii=False))
