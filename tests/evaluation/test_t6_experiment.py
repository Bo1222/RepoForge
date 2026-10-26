"""T6 selection and provenance cannot silently inherit T4's experiment scope."""
from copy import deepcopy

import pytest

from repoforge.evaluation.experiment import SMOKE_IDS, PROFILES, make_jobs
from repoforge.evaluation.t6 import PROTOCOL, non_query_ast, query_compatibility


def manifest():
    return {'manifest_hash': 'frozen', 'profiles': deepcopy(PROFILES), 'smoke_ids': list(SMOKE_IDS),
            'split': {'dev': ['dev-' + str(i) for i in range(30)]}, 't6': {'protocol': PROTOCOL}}


def test_t6_selects_only_five_a0_batch_jobs_without_cli_or_dev():
    jobs = make_jobs(manifest())
    assert len(jobs) == 5
    assert {j['instance_id'] for j in jobs} == set(SMOKE_IDS)
    assert all(j['variant'] == 'a0' and j['phase'] == 'smoke'
               and j['entrypoint'] == 'batch' and j['repeat'] == 0 for j in jobs)
    assert len({j['job_id'] for j in jobs}) == 5


def test_original_t4_job_matrix_remains_unchanged():
    old = manifest()
    del old['t6']
    jobs = make_jobs(old)
    assert len([j for j in jobs if j['phase'] == 'smoke']) == 15
    assert len([j for j in jobs if j['phase'] == 'cli_smoke']) == 5
    assert len([j for j in jobs if j['phase'] == 'dev']) == 90


def test_t6_rejects_changed_smoke_selection_or_protocol():
    changed = manifest()
    changed['smoke_ids'][0] = 'replacement'
    with pytest.raises(ValueError, match='original five'):
        make_jobs(changed)
    changed = manifest()
    changed['t6']['protocol'] = 'old-t4'
    with pytest.raises(ValueError):
        make_jobs(changed)


def test_index_reuse_proof_excludes_only_query_method():
    base = 'class CodeIndex:\n def semantic_search(self): return 1\n def ingest(self): return 2\n'
    assert non_query_ast(base) == non_query_ast(base.replace('return 1', 'return 99'))
    assert non_query_ast(base) != non_query_ast(base.replace('return 2', 'return 99'))


def test_disabled_policy_serialization_preserves_legacy_arguments():
    from repoforge.actions.run_tests import RunTestsArgs
    args = RunTestsArgs(test_files=['test_a.py'])
    assert 'test_source' not in args.model_dump() and 'test_source' not in args.model_dump_json()
    original = RunTestsArgs(test_files=['test_a.py'], test_source='base_public')
    assert original.model_dump()['test_source'] == 'base_public'


def test_query_only_source_change_can_reuse_actual_archived_index_producer():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    reference = root / 'experiments/results/T4_T5_bge_a0_40iter_1usd_20261016T074244Z'
    if not (reference / 'source.tar').exists():
        pytest.skip('optional local historical archive')
    proof = query_compatibility(root, reference)
    assert proof['original_sha256'] != proof['current_sha256']
    assert proof['non_query_ast_sha256']
