"""Read-only native configuration audit and real-evidence gate entrypoint."""
import argparse
from pathlib import Path
import shutil

from filelock import FileLock

from repoforge.evaluation.experiment import load_manifest, sha, validate_execution_source, write_json
from repoforge.evaluation import native_gate
from repoforge.evaluation.configuration_bindings import tool_bindings  # noqa: F401


def audit_configuration(output, manifest):
    output = Path(output).resolve()
    summary = native_gate.configuration_evidence(output, manifest)
    artifact = output / 'audit-source' / Path(__file__).name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, artifact)
    summary.update(auditor='implementation-schema-bindings-v1',
                   generation_archive_unchanged=True, raw_loaded_config_unchanged=True)
    summary['evidence'][str(artifact.relative_to(output))] = sha(artifact)
    write_json(output / 'configuration-native.json', summary)
    return summary


def build_gate(output, manifest):
    return native_gate.build_native_gate(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.output)
    if manifest.get('execution_backend') != 'native':
        raise ValueError('Native configuration audit requires a native manifest')
    validate_execution_source(manifest, Path(__file__).resolve().parents[1])
    with FileLock(str(args.output / 'scheduler.lock'), timeout=0):
        print(build_gate(args.output, manifest)['local_research_acceptance'])


if __name__ == '__main__':
    main()
