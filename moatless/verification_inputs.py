"""Public execution recipes, including test support files and runner configuration.

Hashes are task-local evidence, never an index/answer lookup. Output/cache paths
are deliberately excluded; unclassified public data is conservatively included.
"""

import hashlib
import os
from pathlib import Path
import subprocess

from moatless.message_history.policy import content_hash
from moatless.utils.file import is_test


CONFIG_NAMES = {
    "pytest.ini",
    ".pytest.ini",
    "tox.ini",
    "setup.cfg",
    "pyproject.toml",
    "conftest.py",
    "manage.py",
    "runtests.py",
    "test-requirements.txt",
}
CACHE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".coverage",
    "htmlcov",
    "node_modules",
}
OUTPUT_NAMES = {"coverage.xml", "junit.xml", "report.xml", ".coverage", "pytestdebug.log"}


def ignored_output(path):
    pure = Path(path)
    if CACHE_PARTS.intersection(pure.parts):
        return True
    if verification_input(path):
        return False  # A fixture named report.xml or *.log is still input.
    return pure.name in OUTPUT_NAMES or pure.suffix in {".pyc", ".pyo", ".log"}


def verification_input(path, declared=()):
    """Configs/data may affect validation even outside a tests directory."""
    pure = Path(path)
    return (
        path in declared
        or is_test(path)
        or pure.name in CONFIG_NAMES
        or any(part in {"fixtures", "testdata", "test_support", "testsupport", "helpers"} for part in pure.parts)
        or pure.stem in {"helpers", "testutils", "test_helpers"}
    )


def public_paths(repository):
    memory = getattr(repository, "files", None)
    if isinstance(memory, dict):
        return sorted(path for path in memory if not CACHE_PARTS.intersection(Path(path).parts))
    root = getattr(repository, "repo_path", None)
    if not root:
        return []
    result = subprocess.run(["git", "ls-files", "-z", "--cached"], cwd=root, capture_output=True, timeout=5)
    if result.returncode == 0:
        tracked = {p for p in result.stdout.decode().split("\0") if p and not CACHE_PARTS.intersection(Path(p).parts)}
        others = subprocess.run(
            ["git", "ls-files", "-z", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=5,
        )
        return sorted(tracked | {p for p in others.stdout.decode().split("\0") if p and not ignored_output(p)})
    return sorted(
        str(path.relative_to(root))
        for path in Path(root).rglob("*")
        if path.is_file() and not ignored_output(str(path.relative_to(root)))
    )


def content_bytes(repository, context, path, *, base=False):
    file = context._files.get(path) if context else None
    if file and (file.patch or file.was_edited or file._is_new) and (base or repository.shadow_mode):
        if base and file._is_new:
            return None
        content = file.get_base_content() if base else file.content
        return content.encode() if content is not None else None
    root = getattr(repository, "repo_path", None)
    if root:
        target = Path(root) / path
        if not target.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError(f"Unbound verification symlink/path: {path}")
        if target.is_symlink():
            return (
                b"symlink\0"
                + os.readlink(target).encode()
                + b"\0"
                + (target.read_bytes() if target.is_file() else b"missing")
            )
        return target.read_bytes() if target.is_file() else None
    content = repository.get_file_content(path)
    return content.encode() if isinstance(content, str) else content


def runner_configuration(runtime, targets):
    from moatless.runtime.runtime import RuntimeEnvironment

    if not isinstance(runtime, RuntimeEnvironment):
        return None
    adapter = getattr(runtime, "verification_configuration", None)
    if callable(adapter):
        configuration = adapter(targets)
        return content_hash(configuration) if isinstance(configuration, dict) and configuration else None
    # Older adapters have no trusted configuration contract. Keep unknown,
    # rather than presenting a class name as a complete runner identity.
    return None


def declared_inputs(workspace, targets):
    from moatless.runtime.runtime import RuntimeEnvironment

    if not isinstance(workspace.runtime, RuntimeEnvironment):
        return []
    paths = workspace.runtime.verification_input_paths(targets)
    if not isinstance(paths, list) or any(
        not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts for path in paths
    ):
        raise ValueError("Verification input paths must be explicit public repository-relative files")
    return sorted(set(paths))


def capture_recipe(workspace, context, targets, test_source="workspace", product_source="current"):
    from moatless.execution_memory import environment_version

    repository = workspace.repository
    declared = declared_inputs(workspace, targets)
    paths = (
        set(public_paths(repository))
        | set(declared)
        | {p for p, f in context._files.items() if f.patch or not ignored_output(p)}
    )
    base_files, products, tests = {}, {}, {}
    for path in sorted(paths):
        if CACHE_PARTS.intersection(Path(path).parts):
            continue
        original = content_bytes(repository, context, path, base=True)
        original_hash = hashlib.sha256(original).hexdigest() if original is not None else "missing"
        base_files[path] = original_hash
        auxiliary = verification_input(path, declared) or path in targets
        use_base = test_source == "base_public" if auxiliary else product_source == "base"
        value = content_bytes(repository, context, path, base=use_base)
        if value is not None:
            (tests if auxiliary else products)[path] = hashlib.sha256(value).hexdigest()
    identity = {"commit": getattr(repository, "initial_commit", None) or None, "tree": content_hash(base_files)}
    # A new workspace repro is absent from the base; its absence must not
    # change the immutable base identity when the context learns that file.
    identity["tree"] = content_hash({p: h for p, h in base_files.items() if h != "missing"})
    return {
        "recipe_version": 3,
        "base_identity": identity,
        "verification_inputs": tests,
        "test_bundle_hash": content_hash(tests),
        "product_inputs_hash": content_hash(products),
        "runner_fingerprint": runner_configuration(workspace.runtime, targets),
        "environment": environment_version(workspace),
        "targets": sorted(targets),
        "test_source": test_source,
        "product_source": product_source,
    }


def verification_patch(context, declared=()):
    """Retain the same workspace test/config/data bundle on base product code."""
    return "\n".join(
        file.patch for file in context.files if file.patch and verification_input(file.file_path, declared)
    )


def test_identity(result):
    name = result.get("name")
    path = result.get("file_path")
    if not name or not path:
        return None
    # Preserve class and parametrization supplied by the runner. Never reduce
    # a node id to its final function name.
    method = result.get("method")
    return f"{path}::{name}" + (f"::{method}" if method and method not in name else "")


def receipt_complete(record) -> bool:
    """One validity predicate for an execution receipt.

    Verification snapshots, acceptance assessment and public-baseline
    comparison must agree on what counts as a complete, bound execution.
    Callers add outcome-specific checks (return code, parsed counts).
    """
    return (
        record.get("recipe_version") == 3
        and record.get("complete") is True
        and not record.get("timed_out")
        and not record.get("patch_stale")
        and bool(record.get("input_identity"))
        and record.get("input_identity") == record.get("input_identity_after")
        and record.get("environment_before") == record.get("environment_after")
    )


def compare_runs(candidate, baseline):
    """Compare executions only when the actual public recipes are compatible."""
    unknown = {
        "status": "unknown",
        "introduced_failures": None,
        "preexisting_failures": None,
        "baseline_source": baseline.get("source_id") if baseline else None,
    }
    if not baseline:
        return {**unknown, "reason": "baseline_not_run"}
    for record in (candidate, baseline):
        if (
            not receipt_complete(record)
            or not record.get("runner_fingerprint")
            or record.get("return_code") not in {0, 1}
            or record.get("counts", {}).get("error")
        ):
            return {**unknown, "reason": "incomplete_or_unbound_execution"}
    if candidate.get("product_source") != "current" or baseline.get("product_source") != "base":
        return {**unknown, "reason": "not_a_product_baseline_pair"}
    for key in ("base_identity", "test_bundle_hash", "runner_fingerprint", "environment_after", "test_source"):
        if candidate.get(key) != baseline.get(key):
            return {**unknown, "reason": "different_" + key}
    if set(candidate.get("targets", [])) != set(baseline.get("targets", [])):
        return {**unknown, "reason": "different_targets"}
    rows = []
    for record in (candidate, baseline):
        outcomes = {}
        for item in record.get("results", []):
            key = test_identity(item)
            status = str(item.get("status", "")).lower()
            if key is None or key in outcomes or status not in {"passed", "failed"}:
                return {**unknown, "reason": "missing_ambiguous_or_unexecuted_test_identity"}
            outcomes[key] = status
        observed = {status: list(outcomes.values()).count(status) for status in set(outcomes.values())}
        if (
            any(record.get("counts", {}).get(status, 0) != observed.get(status, 0) for status in {"passed", "failed"})
            or (record["return_code"] == 0 and observed.get("failed", 0))
            or (record["return_code"] == 1 and not observed.get("failed", 0))
        ):
            return {**unknown, "reason": "exit_or_count_conflicts_with_parsed_results"}
        rows.append(outcomes)
    actual, original = rows
    if not actual or actual.keys() != original.keys():
        return {**unknown, "reason": "different_collected_tests"}
    introduced = sorted(key for key in actual if actual[key] == "failed" and original[key] == "passed")
    preexisting = sorted(key for key in actual if actual[key] == original[key] == "failed")
    fixed = sorted(key for key in actual if actual[key] == "passed" and original[key] == "failed")
    return {
        "status": "known",
        "introduced_failures": introduced,
        "preexisting_failures": preexisting,
        "fixed_tests": fixed,
        "baseline_source": baseline.get("source_id"),
        "candidate_source": candidate.get("source_id"),
    }
