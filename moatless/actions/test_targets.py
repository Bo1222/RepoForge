"""Helpers for choosing test modules that can actually verify a patch.

The policy used to treat any path accepted by ``is_test`` as a verification
target.  That includes fixtures, docs, ``__init__.py`` and product modules
whose path merely contains a ``test`` component.  This module implements the
stricter predicate used by the policy and by RunTests contract checks.
"""

import ast
from pathlib import Path


_SUPPORT_PARTS = {"fixtures", "fixture", "docs", "doc", "roots", "vendor", "third_party"}


def is_verifiable_test_module(repository, path: str) -> bool:
    """Return True only for a Python module that defines real tests."""
    path = str(path)
    if not path.endswith(".py"):
        return False
    parts = Path(path).parts
    if not parts:
        return False
    if parts[-1] in {"__init__.py", "conftest.py", "setup.py"}:
        return False
    if any(part.lower() in _SUPPORT_PARTS for part in parts[:-1]):
        return False
    try:
        content = repository.get_file_content(path)
    except (OSError, ValueError):
        return False
    if not isinstance(content, str) or not content:
        return False
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            return True
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = getattr(base, "id", None) or getattr(base, "attr", None)
                if name == "TestCase":
                    return True
    return False


class _ContentSource:
    """Minimal repository adapter for an already-resolved file body."""

    def __init__(self, content):
        self._content = content

    def get_file_content(self, path):
        return self._content


def is_existing_test_module(workspace, context, path, test_source="workspace") -> bool:
    """Whether a plan check target is an executable test module right now.

    ``base_public`` resolves the unchanged base body; ``workspace`` resolves
    the current overlay/disk body.  A target that does not exist yet is not
    executable and must not be treated as an immutable acceptance obligation.
    """
    from moatless.repository_context import file_content, relative_path
    try:
        path = relative_path(workspace, path)
    except (OSError, ValueError):
        return False
    if test_source == "base_public":
        file = context._files.get(path) if context else None
        content = file.get_base_content() if file else workspace.repository.get_file_content(path)
    else:
        content = file_content(workspace, context, path)
    if not isinstance(content, str) or not content:
        return False
    return is_verifiable_test_module(_ContentSource(content), path)


def _tokens(value: str) -> set[str]:
    import re
    return {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", value)}


def is_related_test(repository, path: str, changed_files) -> bool:
    """Whether a test module has a name or import relation to a changed file."""
    changed = [str(item) for item in changed_files]
    if not changed:
        return True
    changed_tokens = set()
    for item in changed:
        changed_tokens |= _tokens(Path(item).stem)
    if changed_tokens & _tokens(Path(path).stem):
        return True
    try:
        content = repository.get_file_content(path) or ""
    except (OSError, ValueError):
        return False
    return any(Path(item).stem.lower() in content.lower()
               for item in changed if Path(item).stem)


def discover_test_targets(repository, changed_files, candidates=(), require_relation: bool = False) -> list[str]:
    """Rank existing test modules by relation to the changed product files.

    Only candidates already visible in the repository/context are considered;
    this avoids scanning a huge repository on every policy decision.  The
    ranking is intentionally explainable: name overlap, import text, and
    test-directory proximity.

    ``require_relation`` drops tests with no name/import relation to the
    changed product files.  Without it, an arbitrary test module can win by
    path order and become a false verification target.
    """
    changed = [str(path) for path in changed_files]
    changed_tokens = set()
    for path in changed:
        changed_tokens |= _tokens(Path(path).stem)
    candidate_paths = set(str(path) for path in candidates)
    from moatless.repository_overlay import repository_paths
    candidate_paths.update(repository_paths(repository))
    scored = []
    for path in candidate_paths:
        if not is_verifiable_test_module(repository, path):
            continue
        try:
            content = repository.get_file_content(path) or ""
        except (OSError, ValueError):
            continue
        relevance = 0.0
        name_tokens = _tokens(Path(path).stem)
        relevance += 4.0 * len(changed_tokens & name_tokens)
        for module in changed:
            stem = Path(module).stem
            if stem and stem.lower() in content.lower():
                relevance += 3.0
        if require_relation and relevance <= 0:
            continue
        score = relevance
        if any(part in Path(path).parts for part in ("tests", "test")):
            score += 0.5
        scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [path for _, path in scored]
