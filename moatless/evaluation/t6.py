"""T6-only A0 smoke protocol. Historical T4's 15+5 contract is unchanged."""
import ast
from collections import Counter
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tarfile

from moatless.evaluation.experiment import (
    SMOKE_IDS, agent_config, content_hash, freeze_source, load_manifest, make_jobs, now,
    read_json, sha, verify_answer, write_json,
)

PROTOCOL = "t6-a0-smoke-v1"


def non_query_ast(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CodeIndex":
            node.body = [child for child in node.body if not (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "semantic_search")]
    return content_hash(ast.dump(tree, include_attributes=False))


def query_compatibility(root, reference):
    name = "moatless/index/code_index.py"
    with tarfile.open(Path(reference) / "source.tar") as archive:
        original = archive.extractfile(name).read().decode()
    old = read_json(Path(reference) / "manifest.json")
    import hashlib
    if hashlib.sha256(original.encode()).hexdigest() != old["source"]["files"][name]:
        raise ValueError("Reference query source differs from its archived identity")
    current = (Path(root) / name).read_text()
    if non_query_ast(original) != non_query_ast(current):
        raise ValueError("Index producer changed; query-only reuse is not justified")
    return {"path": name, "original_sha256": old["source"]["files"][name],
            "current_sha256": sha(Path(root) / name), "non_query_ast_sha256": non_query_ast(current),
            "reference_manifest": old["manifest_hash"]}


def prepare(root, output, reference):
    root, output, reference = Path(root).resolve(), Path(output).resolve(), Path(reference).resolve()
    old = load_manifest(reference)
    if output.exists():
        raise ValueError("Use a new T6 candidate; identities are immutable")
    compatibility = query_compatibility(root, reference)
    output.mkdir(parents=True)
    manifest = deepcopy(old)
    manifest.pop("manifest_hash")
    manifest["created"] = now()
    manifest["source"] = freeze_source(root, output)
    manifest["baseline_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    manifest["diff"] = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'moatless', 'scripts'], cwd=root, text=True)
    manifest["environment"]["python"] = sys.version
    manifest["t6"] = {"protocol": PROTOCOL, "historical_manifest": old["manifest_hash"],
                      "baseline_revision": "tools-fixed-action-policy-off", "real_runs": 5,
                      "total_cost_limit": None, "holdout": "locked"}
    manifest["smoke_ids"] = list(SMOKE_IDS)
    manifest["profiles"]["a0"]["action_policy"] = {"version": 1, "variant": "off"}
    manifest["agent_configs"] = {v: agent_config(root, v, manifest["model_config"]) for v in manifest["profiles"]}
    manifest["native"]["index_store"]["query_only_compatibility"] = compatibility
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "jobs.json", make_jobs(manifest))
    # Reuse verified immutable public environments, not old answers or grades.
    from moatless.evaluation.native_provision import NativeProvisioner
    receipts = read_json(reference / "provision-native.json")
    prepared = {}
    for name in SMOKE_IDS:
        item = deepcopy(receipts[name])
        NativeProvisioner.verify(item["environment"])
        item["source_sha256"] = manifest["source"]["sha256"]
        prepared[name] = item
    write_json(output / "provision-native.json", prepared)
    for name in ("dataset-receipt.json", "host-install-recipes.json"):
        write_json(output / name, read_json(reference / name))
    write_json(output / "public/instances.json", {name: row["public"] for name, row in prepared.items()})
    write_json(output / "query-compatibility.json", compatibility)
    return manifest


def audit(reference, destination):
    reference, destination = Path(reference), Path(destination)
    labels = [__import__('json').loads(line) for line in (reference / 'analysis/labels.jsonl').read_text().splitlines()]
    counts = Counter('true' if row['resolved_adjudicated'] is True else 'false'
                     if row['resolved_adjudicated'] is False else 'null' for row in labels)
    report = {"timestamp": now(), "historical_manifest": load_manifest(reference)["manifest_hash"],
              "evidence": {"labels": {"path": str(reference / 'analysis/labels.jsonl'),
                                      "sha256": sha(reference / 'analysis/labels.jsonl')}},
              "scope": {"effective_repeats": [0], "selected": 30, "historical_planned": 90,
                        "excluded_repeats": {"1": "not_run", "2": "not_run"}},
              "formal_resolved": sum(row['local_resolved_formal'] is True for row in labels),
              "diagnostic_adjudication": dict(counts), "total": len(labels),
              "unknown_instances": [row['instance_id'] for row in labels if row['resolved_adjudicated'] is None],
              "corrections": ["17+11+2=30; never convert a formal null silently",
                "Adding or modifying tests alone does not establish weakened verification",
                "Retrieval interface failures exist; vector-ranking causality remains unproven",
                "+2/30 is exploratory, not a significance or falsification threshold"],
              "historical_artifacts_modified": False, "provider_calls": 0}
    write_json(destination, report)
    from moatless.evaluation.experiment import aggregate
    corrected = aggregate(reference, 'dev', repeats=[0])
    corrected['effective_scope'] = {'phase': 'dev', 'repeats': [0], 'historical_planned': 90,
                                  'excluded_repeats': [1, 2], 'scope_source': 'user 2026-10-17 instruction'}
    write_json(destination.parent / 't5-dev-r0-report.json', corrected)
    return report


def expected_jobs(output, manifest):
    jobs = read_json(Path(output) / "jobs.json")
    if manifest.get("t6", {}).get("protocol") != PROTOCOL or jobs != make_jobs(manifest) or len(jobs) != 5:
        raise ValueError("T6 smoke requires exactly five frozen batch A0 jobs")
    return jobs


def configuration(output, manifest):
    from moatless.message_history.policy import ContextPolicy
    from moatless.evaluation.configuration_bindings import tool_bindings
    from moatless.policy import ActionPolicyConfig
    output = Path(output)
    rows, evidence = [], {}
    for job in expected_jobs(output, manifest):
        path = output / 'runs' / job['job_id'] / 'loaded-config.json'
        loaded = read_json(path)
        config = manifest['cli_config']
        tools = tool_bindings(manifest, job)
        checks = {
            'identity': loaded.get('manifest_hash') == manifest['manifest_hash'] and loaded.get('job_id') == job['job_id'],
            'flow': loaded.get('flow_type') == 'LangGraphFlow',
            'memory': loaded.get('memory_type') == 'ManagedMessageHistoryGenerator',
            'context': loaded.get('policy') == ContextPolicy(**config['context_policy']).model_dump(mode='json'),
            'action_policy': loaded.get('action_policy') == ActionPolicyConfig().model_dump(mode='json'),
            'model': loaded.get('model') == config['model'] and loaded.get('model_base_url') == config['base_url'],
            'params': loaded.get('model_parameters') == config['params'] and loaded.get('key_env_matches') is True,
            'limits': all(loaded.get(k) == config[k] for k in (
                'max_tokens', 'temperature', 'max_iterations', 'max_cost', 'model_timeout')),
            'tools': set(loaded.get('tool_names', [])) == set(tools) and set(loaded.get('schema_names', [])) == set(tools.values()),
            'index': loaded.get('index_settings') == manifest['embedding_settings'],
            'isolation': loaded.get('tool_isolation') == 'native-offline-v1',
        }
        rows.append({'job_id': job['job_id'], 'checks': checks, 'passed': all(checks.values())})
        evidence[str(path.relative_to(output))] = sha(path)
    report = {'manifest_hash': manifest['manifest_hash'], 'protocol': PROTOCOL,
              'passed': all(row['passed'] for row in rows), 'rows': rows, 'evidence': evidence}
    write_json(output / 'configuration-t6.json', report)
    return report


def build_gate(output, manifest):
    """Bind five answers plus fresh component evidence, without weakening T4."""
    from moatless.evaluation.generation import validate_generation_health
    output = Path(output).resolve()
    preflight = read_json(output / 'preflight-t6.json')
    offline = read_json(output / 'offline-check.json')
    if (not preflight.get('passed') or preflight.get('manifest_hash') != manifest['manifest_hash']
            or not offline.get('passed') or offline.get('source_digest') != manifest['source']['digest']):
        raise ValueError('T6 requires source-bound preflight and offline verification')
    config = configuration(output, manifest)
    if not config['passed']:
        raise ValueError('T6 actual A0 configuration failed')
    evidence = dict(config['evidence'])
    evidence['configuration-t6.json'] = sha(output / 'configuration-t6.json')
    for component in ('contracts', 'isolation', 'recovery'):
        files = sorted(output.glob(component + '-native/*/summary.json'), key=lambda p: p.stat().st_mtime_ns)
        if not files:
            raise ValueError('Missing T6 component: ' + component)
        path = files[-1]
        result = read_json(path)
        if not result.get('passed') or result.get('manifest_hash') != manifest['manifest_hash']:
            raise ValueError('Latest component failed or has another identity: ' + component)
        evidence[str(path.relative_to(output))] = sha(path)
        evidence.update(result.get('evidence', {}))
    rows = []
    for job in expected_jobs(output, manifest):
        directory = output / 'runs' / job['job_id']
        answer = verify_answer(directory, job)
        validate_generation_health(directory)
        grade, own = read_json(directory / 'grading.json'), read_json(directory / 'own-tests.json')
        if (grade.get('answer_hash') != answer['answer_hash'] or grade.get('grading_backend') != 'native'
                or grade.get('grader_source_sha256') != manifest['source']['files']['moatless/evaluation/native_grading.py']
                or grade.get('evaluation_protocol') != manifest['evaluation_protocol']
                or grade.get('local_grading') not in {'completed', 'patch_error'}
                or not isinstance(grade.get('local_resolved'), bool)
                or not grade.get('original_baseline', {}).get('original_failure_reproduced')
                or grade.get('official_harness') != 'not_run' or grade.get('official_resolved') is not None):
            raise ValueError('T6 independent grading incomplete: ' + job['instance_id'])
        if (own.get('answer_hash') != answer['answer_hash'] or own.get('hidden_tests') is not False
                or own.get('status') not in {'executed', 'not_run'}):
            raise ValueError('T6 public own-test evidence incomplete')
        for name in ('answer.json', 'grading.json', 'own-tests.json', 'original-baseline.json', 'usage.json'):
            path = directory / name
            evidence[str(path.relative_to(output))] = sha(path)
        evidence.update({str((directory / name).relative_to(output)): digest for name, digest in answer['evidence'].items()})
        evidence.update(grade.get('evidence', {}))
        evidence.update(grade.get('original_baseline', {}).get('evidence', {}))
        evidence.update(own.get('evidence', {}))
        rows.append({'instance_id': job['instance_id'], 'job_id': job['job_id'],
                     'local_resolved': grade['local_resolved'], 'usage': read_json(directory / 'usage.json')})
    for name in ('manifest.json', 'jobs.json', 'provision-native.json', 'query-compatibility.json',
                 'offline-check.json', 'preflight-t6.json'):
        evidence[name] = sha(output / name)
    for name, digest in evidence.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or sha(path) != digest:
            raise ValueError('T6 evidence changed or escaped output: ' + name)
    gate = {'protocol': PROTOCOL, 'manifest_hash': manifest['manifest_hash'], 'timestamp': now(),
            'passed': True, 'selected': 5, 'resolved': sum(row['local_resolved'] for row in rows),
            'rows': rows, 'evidence': evidence, 'official_harness': 'not_run', 'official_resolved': None,
            'holdout_state': 'locked', 'algorithm_effectiveness': 'not_evaluated'}
    if (output / 't6-gate.json').exists():
        previous = read_json(output / 't6-gate.json')
        if previous.get('manifest_hash') != manifest['manifest_hash'] or previous.get('evidence') != evidence:
            raise ValueError('Accepted T6 gate evidence changed; create a new candidate')
        return previous
    write_json(output / 't6-gate.json', gate)
    return gate
