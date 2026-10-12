import pytest
from moatless.cli.worktree import git
from moatless.completion.base import CompletionResponse
from moatless.completion.tool_call import ToolCallCompletionModel


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # Same-length edits within one second must not reuse timestamp-based pyc
    # files in these deliberately tiny repositories.
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "tests@example.invalid")
    git(path, "config", "user.name", "Tests")
    (path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (path / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n")
    (path / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(path, "add", ".")
    git(path, "commit", "-qm", "base")
    return path


@pytest.fixture
def scripted(monkeypatch):
    outputs = []
    calls = []

    async def complete(self, messages, **kwargs):
        calls.append(messages)
        item = outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, CompletionResponse) else CompletionResponse(structured_outputs=item)

    monkeypatch.setattr(ToolCallCompletionModel, "create_completion", complete)
    return outputs, calls


@pytest.fixture(autouse=True)
def preserve_evidence(tmp_path, request):
    """Opt-in complete snapshots/logs for acceptance, without burdening normal tests."""
    yield
    import os
    from pathlib import Path
    from moatless.cli.service import SessionService

    output = os.environ.get("MOATLESS_CLI_EVIDENCE")
    if not output:
        return
    test_directory = Path(output) / request.node.name.replace("[", "-").replace("]", "")
    for session_file in tmp_path.glob("**/sessions/*/session.json"):
        service = SessionService(session_file.parent.parent)
        session = service.load(session_file.parent.name)
        test_directory.mkdir(parents=True, exist_ok=True)
        destination = test_directory / session.id
        if not destination.exists():
            service.export(session, str(destination))
    for name in ("model-calls", "error", "id", "process-group"):
        for path in tmp_path.glob(f"**/{name}"):
            test_directory.mkdir(parents=True, exist_ok=True)
            (test_directory / (path.parent.name + "-" + name + ".txt")).write_text(path.read_text())
