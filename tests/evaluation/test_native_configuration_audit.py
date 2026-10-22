"""Configuration audit fixtures: schema aliases, forbidden drift, immutable raw logs."""
from pathlib import Path

from moatless.evaluation.experiment import prepare, read_json, sha, write_json
from moatless.message_history.policy import ContextPolicy
from scripts.native_configuration_audit import audit_configuration

ROOT = Path(__file__).resolve().parents[2]
CLI_TOOLS = {'LocalReadFile': 'ReadFile', 'LocalSearch': 'GrepTool',
             'LocalStringReplace': 'StringReplace', 'LocalCreateFile': 'CreateFile',
             'LocalBash': 'Bash', 'LocalRunTests': 'RunTests', 'Finish': 'Finish',
             'ReadObservation': 'ReadObservation'}


def config_fixture(tmp_path):
    output = tmp_path / 'native'
    manifest, jobs = prepare(ROOT, output, execution_backend='native')
    config = manifest['cli_config']
    selected = [job for job in jobs if job['phase'] in {'smoke', 'cli_smoke'}]
    for job in selected:
        managed = job['entrypoint'] == 'cli' or manifest['profiles'][job['variant']]['managed']
        graph = job['entrypoint'] == 'cli' or manifest['profiles'][job['variant']]['flow'] == 'graph'
        tools = list(CLI_TOOLS) if job['entrypoint'] == 'cli' else [
            action['action_class'].rsplit('.', 1)[-1] for action in manifest['agent_configs'][job['variant']]['actions']]
        if managed and job['entrypoint'] != 'cli':
            tools.append('ReadObservation')
        schemas = [CLI_TOOLS[name] for name in tools] if job['entrypoint'] == 'cli' else [
            'Finish' if name == 'VerifiedFinish' else name for name in tools]
        loaded = {'manifest_hash': manifest['manifest_hash'], 'job_id': job['job_id'],
                  'flow_type': 'LangGraphFlow' if graph else 'AgenticLoop',
                  'model': config['model'], 'model_base_url': config['base_url'],
                  'key_env_matches': True, 'model_parameters': config['params'],
                  'memory_type': 'ManagedMessageHistoryGenerator' if managed else 'MessageHistoryGenerator',
                  'tool_names': tools, 'schema_names': schemas,
                  'tool_isolation': 'native-offline-v1', 'index_settings': manifest['embedding_settings'],
                  'policy': ContextPolicy(**config['context_policy']).model_dump(mode='json')}
        loaded.update({key: config[key] for key in
                       ('max_tokens', 'temperature', 'max_iterations', 'max_cost', 'model_timeout')})
        write_json(output / 'runs' / job['job_id'] / 'loaded-config.json', loaded)
    return output, manifest, selected


def test_real_cli_aliases_accepted_without_rewriting_raw_configuration(tmp_path):
    output, manifest, _ = config_fixture(tmp_path)
    before = {str(path): sha(path) for path in output.glob('runs/*/loaded-config.json')}
    result = audit_configuration(output, manifest)
    assert result['passed'] and len(result['rows']) == 20
    assert all(sha(Path(name)) == expected for name, expected in before.items())
    assert result['implementation_schema_bindings'][next(job['job_id'] for job in
        read_json(output / 'jobs.json') if job['entrypoint'] == 'cli')]['LocalSearch'] == 'GrepTool'
    artifact = output / 'audit-source/native_configuration_audit.py'
    assert result['evidence'][str(artifact.relative_to(output))] == sha(artifact)


def test_matching_schema_lists_cannot_hide_an_extra_or_missing_tool(tmp_path):
    output, manifest, jobs = config_fixture(tmp_path)
    job = next(job for job in jobs if job['entrypoint'] == 'cli')
    path = output / 'runs' / job['job_id'] / 'loaded-config.json'
    raw = read_json(path)
    raw['tool_names'].append('UnconfiguredTool')
    raw['schema_names'].append('UnconfiguredTool')
    write_json(path, raw)
    result = audit_configuration(output, manifest)
    row = next(row for row in result['rows'] if row['job_id'] == job['job_id'])
    assert row['checks']['tool_schema']  # Old mutual-consistency check passes.
    assert not row['checks']['frozen_tool_bindings'] and not result['passed']
