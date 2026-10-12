"""Git workspace operations. Never stage or commit in the user's repository."""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


def git(path: str | Path, *args: str, env=None, input=None) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        input=input,
        capture_output=True,
        env=env,
        timeout=30,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout.decode(errors="replace")


def clean(path: str | Path):
    if git(path, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Source repository must be clean (including untracked files); no automatic stash or commit")


def validate(source: str) -> tuple[str, str]:
    requested = Path(source).expanduser().resolve()
    if not requested.is_dir():
        raise ValueError(f"目标仓库目录不存在或不是目录：{requested}。请通过 --repo 指定实际 Git 仓库路径。")
    try:
        source = git(requested, "rev-parse", "--show-toplevel").strip()
    except ValueError as exc:
        raise ValueError(
            f"目标目录不是可用的 Git 工作目录：{requested}。"
            "请通过 --repo 指定实际 Git 仓库路径；省略 --repo 可进入普通聊天。"
            f"\nGit 诊断：{exc}"
        ) from exc
    clean(source)
    try:
        base = git(source, "rev-parse", "HEAD").strip()
    except ValueError as exc:
        raise ValueError(f"目标仓库没有可用的 HEAD 提交：{source}。创建会话需要已有提交且干净的仓库。") from exc
    return source, base


def create(source: str, target: Path) -> tuple[str, str]:
    source, base = validate(source)
    if target.resolve().is_relative_to(Path(source)):
        raise ValueError("Store CLI sessions outside the source repository")
    git(source, "worktree", "add", "--detach", str(target), base)
    return source, base


def diff(worktree: str, base: str) -> str:
    # Temporary index includes new/deleted/binary files without changing the real index.
    with tempfile.TemporaryDirectory() as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
        git(worktree, "read-tree", base, env=env)
        git(worktree, "add", "-A", "--", ".", env=env)
        return git(worktree, "diff", "--cached", "--binary", "--no-ext-diff", base, "--", env=env)


def patch_hash(patch: str) -> str:
    return hashlib.sha256(patch.encode()).hexdigest()


def apply(source: str, patch: str):
    clean(source)
    if not patch.strip():
        raise ValueError("No patch to apply")
    git(source, "apply", "--check", "-", input=patch.encode())
    git(source, "apply", "-", input=patch.encode())
