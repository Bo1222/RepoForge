"""A0-only T6 runner: offline checks, immutable preparation, five jobs and report."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
from filelock import FileLock

from repoforge.evaluation.experiment import (
    aggregate, append_json, freeze_source, load_manifest, now, read_json, sha,
    validate_execution_source, write_json,
)
from repoforge.evaluation import t6


SUITES = ["tests/actions/test_t6_tool_contracts.py", "tests/policy", "tests/memory",
          "tests/flow", "tests/agent/test_public_steps.py", "tests/cli/test_managed_sessions.py",
          "tests/evaluation/test_t6_experiment.py", "tests/evaluation/test_native_experiment.py",
          "tests/evaluation/test_native_configuration_audit.py", "tests/evaluation/test_t4_t5_experiment.py"]


def check(root, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-v", "-o", "log_cli=false", "-o", "faulthandler_timeout=60", *SUITES]
    with (destination / "pytest.log").open("w") as stream:
        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    with tempfile.TemporaryDirectory() as tmp:
        source = freeze_source(root, Path(tmp))
    report = {'timestamp': now(), 'command': command, 'cwd': str(root), 'return_code': result.returncode,
              'passed': result.returncode == 0, 'source_digest': source['digest'],
              'source_files': source['files'], 'test_files': {str(p.relative_to(root)): sha(p)
                  for p in (root / 'tests').rglob('*.py')},
              'log': str(destination / 'pytest.log'), 'log_sha256': sha(destination / 'pytest.log'),
              'provider_calls': 0}
    write_json(destination / 'offline-check.json', report)
    return report


def validate_offline(root, report, manifest=None):
    if not report.get('passed') or report.get('return_code') != 0 or report.get('provider_calls') != 0:
        raise ValueError('T6 requires successful offline verification')
    for path, expected in {**report['source_files'], **report['test_files']}.items():
        if sha(root / path) != expected:
            raise ValueError('Source/test changed after offline verification: ' + path)
    if manifest and report['source_digest'] != manifest['source']['digest']:
        raise ValueError('Offline evidence belongs to a different frozen source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'audit', 'prepare', 'preflight', 'run', 'report', 'gate', 'status'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path,
        default=Path('experiments/results/T4_T5_bge_a0_40iter_1usd_20261016T074244Z'))
    parser.add_argument('--offline-report', type=Path,
        default=Path('experiments/results/T6_implementation_20261017/check/offline-check.json'))
    parser.add_argument('--job-id')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')
    output = args.output.resolve()
    if args.command == 'check':
        report = check(root, output)
        print({'passed': report['passed'], 'return_code': report['return_code'], 'report': str(output / 'offline-check.json')})
        raise SystemExit(report['return_code'])
    if args.command == 'audit':
        result = t6.audit(args.reference, output / 't5-audit.json')
        print({'total': result['total'], 'diagnostic': result['diagnostic_adjudication']})
        return
    if args.command == 'prepare':
        offline = read_json(args.offline_report)
        validate_offline(root, offline)
        manifest = t6.prepare(root, output, args.reference)
        validate_offline(root, offline, manifest)
        write_json(output / 'offline-check.json', offline)
        print({'manifest_hash': manifest['manifest_hash'], 'jobs': 5, 'protocol': t6.PROTOCOL})
        return
    manifest = load_manifest(output)
    jobs = t6.expected_jobs(output, manifest)
    if args.command == 'status':
        print({'protocol': t6.PROTOCOL, 'selected': 5, 'jobs': [
            {'instance_id': j['instance_id'], 'job_id': j['job_id'],
             'generated': (output / 'runs' / j['job_id'] / 'answer.json').exists(),
             'graded': (output / 'runs' / j['job_id'] / 'grading.json').exists()} for j in jobs]})
        return
    if args.command == 'report':
        report = aggregate(output, 'smoke')
        write_json(output / 'report-smoke.json', report)
        print(report['groups'])
        return
    validate_execution_source(manifest, root)
    validate_offline(root, read_json(output / 'offline-check.json'), manifest)
    from repoforge.evaluation.experiment_native import (
        preflight, probe_contracts, probe_isolation, probe_recovery, run_job, grade_job,
        validate_agent_tests, release_job_workspace, shared_index, verify_index_receipt,
    )
    with FileLock(str(output / 'scheduler.lock'), timeout=0):
        if args.command == 'preflight':
            result = preflight(root)
            if result['blockers']:
                raise ValueError(str(result['blockers']))
            from repoforge.evaluation.bge_identity import validate_identity
            validate_identity(root, manifest)
            prepared = read_json(output / 'provision-native.json')
            for job in jobs:
                item = prepared[job['instance_id']]
                index = shared_index(output, manifest, item)[2]
                if index is None:
                    raise ValueError('T6 forbids rebuilding an index; compatible existing index missing')
                verify_index_receipt(index, manifest, item['public'],
                                     setup_patch_sha256=item['environment'].get('setup_patch_sha256'))
            probe_contracts(output, manifest)
            probe_isolation(output, manifest)
            probe_recovery(output, manifest)
            write_json(output / 'preflight-t6.json', {'passed': True, 'manifest_hash': manifest['manifest_hash'],
                'provider_calls': 0, 'timestamp': now(), 'checks': result})
            print({'preflight': 'passed', 'provider_calls': 0})
            return
        if args.command == 'gate':
            gate = t6.build_gate(output, manifest)
            print({'passed': gate['passed'], 'resolved': gate['resolved'], 'selected': 5})
            return
        if not read_json(output / 'preflight-t6.json').get('passed'):
            raise ValueError('T6 preflight is required before real generation')
        selected = [j for j in jobs if not args.job_id or j['job_id'] == args.job_id]
        if not selected or args.resume and len(selected) != 1:
            raise ValueError('Explicit resume requires one frozen job')
        try:
            for job in selected:
                directory = output / 'runs' / job['job_id']
                operations = [('generate', lambda: run_job(output, manifest, job, resume=args.resume)),
                    ('own-tests', lambda: validate_agent_tests(output, manifest, job)),
                    ('grade', lambda: grade_job(output, manifest, job)),
                    ('release', lambda: release_job_workspace(output, manifest, job))]
                for stage, operation in operations:
                    if stage == 'own-tests' and (directory / 'own-tests.json').exists():
                        continue
                    if stage == 'grade' and (directory / 'grading.json').exists():
                        continue
                    print({'instance_id': job['instance_id'], 'stage': stage, 'status': 'started'}, flush=True)
                    append_json(output / 'operations.jsonl', {'timestamp': now(), 'job_id': job['job_id'],
                        'stage': stage, 'status': 'started'})
                    operation()
                    append_json(output / 'operations.jsonl', {'timestamp': now(), 'job_id': job['job_id'],
                        'stage': stage, 'status': 'completed'})
        finally:
            write_json(output / 'report-smoke.json', aggregate(output, 'smoke'))
        gate = t6.build_gate(output, manifest)
        print({'passed': gate['passed'], 'resolved': gate['resolved'], 'selected': 5})


if __name__ == '__main__':
    main()
