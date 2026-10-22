"""Full builder path fixtures only; not evidence of real research acceptance."""
import pytest

from moatless.evaluation.experiment import write_json, sha
from moatless.evaluation.native_gate import build_native_gate, validate_native_gate
from moatless.evaluation.native_gate_scoped import build_scoped_native_gate
from tests.evaluation.test_native_configuration_audit import config_fixture
from tests.evaluation.test_native_experiment import frozen_answer


def full_fixture(tmp_path, monkeypatch):
    output, manifest, jobs = config_fixture(tmp_path)
    for name in ['contracts', 'migration', 'isolation', 'recovery', 'cli-boundaries']:
        summary = {'passed': True, 'manifest_hash': manifest['manifest_hash'], 'evidence': {}}
        if name == 'recovery':
            summary['cases'] = [{'case': mode + '-' + fault, 'passed': True, 'first_exit_code': 88}
                                for mode in ['shadow', 'disk']
                                for fault in ['after_generation', 'before_tool', 'after_effect', 'after_commit']]
        write_json(output / (name + '-native') / 'fixture/summary.json', summary)
    for job in jobs:
        existing = output / 'runs' / job['job_id']
        loaded = (existing / 'loaded-config.json').read_bytes()
        (existing / 'loaded-config.json').unlink()
        existing.rmdir()
        directory, answer = frozen_answer(output, job)
        (directory / 'loaded-config.json').write_bytes(loaded)
        # Different patches must not overwrite a basename from another job.
        # Frozen helper uses empty patches; give the baseline/candidate unique logs.
        baseline_log = directory / 'baseline.txt'
        baseline_log.write_text('baseline fixture ' + job['job_id'])
        relative = str(baseline_log.relative_to(output))
        baseline = {'original_failure_reproduced': True, 'evidence': {relative: sha(baseline_log)}}
        write_json(directory / 'original-baseline.json', baseline)
        write_json(directory / 'grading.json', {'answer_hash': answer['answer_hash'],
            'grading_backend': 'native', 'evaluation_protocol': manifest['evaluation_protocol'],
            'grader_source_sha256': manifest['source']['files']['moatless/evaluation/native_grading.py'],
            'local_grading': 'completed', 'local_resolved': False, 'original_baseline': baseline,
            'official_harness': 'not_run', 'official_resolved': None})
        write_json(directory / 'own-tests.json', {'answer_hash': answer['answer_hash'],
            'hidden_tests': False, 'status': 'not_run'})
    write_json(output / 'provision-native.json', {identity: {
        'source_sha256': manifest['source']['sha256'], 'environment': {'fixture_only': True}}
        for identity in manifest['smoke_ids']})
    monkeypatch.setattr('moatless.evaluation.native_provision.NativeProvisioner.verify', lambda _: None)
    return output, manifest, jobs


def test_builder_scopes_answer_files_to_each_job(tmp_path, monkeypatch):
    output, manifest, jobs = full_fixture(tmp_path, monkeypatch)
    gate = build_scoped_native_gate(output, manifest)
    assert 'patch.diff' not in gate['evidence']
    for job in jobs:
        path = 'runs/' + job['job_id'] + '/patch.diff'
        assert gate['evidence'][path] == sha(output / path)
    validate_native_gate(output, manifest)
    job = jobs[0]
    (output / 'runs' / job['job_id'] / 'patch.diff').write_text('changed fixture')
    with pytest.raises(ValueError):
        validate_native_gate(output, manifest)


def test_original_builder_reproduces_unscoped_answer_path_failure(tmp_path, monkeypatch):
    output, manifest, _ = full_fixture(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match='patch.diff'):
        build_native_gate(output, manifest)


def test_cli_gate_uses_job_scoped_builder(tmp_path, monkeypatch):
    import sys
    from scripts.t4_t5_experiment import main

    output, manifest, _ = full_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["t4_t5_experiment", "gate", "--output", str(output)])
    main()
    gate = validate_native_gate(output, manifest)
    assert gate["gate_builder"] == "scoped-answer-evidence-v1"
    assert "patch.diff" not in gate["evidence"]


def test_gate_rejects_startup_infrastructure_error_even_with_complete_grading(tmp_path, monkeypatch):
    from moatless.evaluation.experiment import freeze_answer, read_json

    def freeze_startup_error(directory, job):
        generation = read_json(directory / "generation.json")
        generation.update(generation_status="failed", error_type="IsADirectoryError", selected_node_id=0)
        write_json(directory / "generation.json", generation)
        return freeze_answer(directory, job)

    monkeypatch.setattr("tests.evaluation.test_native_experiment.freeze_answer", freeze_startup_error)
    output, manifest, _ = full_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Generation infrastructure failure"):
        build_scoped_native_gate(output, manifest)
    assert not (output / "native-gate.json").exists()
