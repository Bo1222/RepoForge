"""Versioned repository facts and scoped instructions for one task lineage."""

import hashlib
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field

from moatless.message_history.policy import content_hash


class InstructionContext(BaseModel):
    version: Literal[1] = 1
    path: str
    scope: str
    content: str | None
    content_hash: str
    source_node_id: int


class RepositoryContext(BaseModel):
    version: Literal[1] = 1
    workspace_version: str
    files: dict[str, dict] = Field(default_factory=dict)
    instructions: list[dict] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def relative_path(workspace, path: str) -> str:
    repo = workspace.repository
    location = getattr(repo, "repo_path", None)
    if location:
        base = Path(location).resolve()
        target = (base / path).resolve()
        if not target.is_relative_to(base) or ".git" in target.relative_to(base).parts:
            raise ValueError("Repository context path must stay inside the repository and outside .git")
        return target.relative_to(base).as_posix()
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or ".git" in pure.parts:
        raise ValueError("Invalid repository context path")
    return pure.as_posix()


def file_content(workspace, context, path: str) -> str | None:
    path = relative_path(workspace, path)
    repo = workspace.repository
    file = context._files.get(path) if context else None
    if file is not None and file.patch and repo.shadow_mode:
        return file.content
    location = getattr(repo, "repo_path", None)
    if location:
        target = Path(location) / path
        return target.read_text(errors="replace") if target.is_file() else None
    return repo.get_file_content(path)


def file_version(workspace, context, path: str) -> str:
    content = file_content(workspace, context, path)
    return "deleted" if content is None else hashlib.sha256(content.encode()).hexdigest()


def workspace_version(workspace, context) -> str:
    # Hash actual bytes, not mtimes or mutable runtime counters. Never follow
    # repository symlinks into personal directories while collecting facts.
    try:
        repo = workspace.repository
    except ValueError:
        return content_hash({"repository": None})
    location = getattr(repo, "repo_path", None)
    if location:
        base = Path(location).resolve()
        result = subprocess.run(["git", "-C", str(base), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                capture_output=True, timeout=10)
        paths = [base / os.fsdecode(p) for p in result.stdout.split(b"\0") if p] if result.returncode == 0 else list(base.rglob("*"))
        contents = {}
        for path in sorted(set(paths)):
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.relative_to(base).parts):
                continue
            if path.is_symlink():
                contents[str(path.relative_to(base))] = content_hash({"symlink": os.readlink(path)})
            elif path.is_file() and path.resolve().is_relative_to(base):
                contents[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif not path.exists():
                contents[str(path.relative_to(base))] = "deleted"
    else:
        contents = getattr(repo, "files", {})
    return content_hash({"files": contents, "shadow_patch":
                         context.generate_git_patch() if context and repo.shadow_mode else None})


def action_paths(args) -> list[str]:
    paths = []
    for name in ("file_path", "path"):
        value = getattr(args, name, None)
        if isinstance(value, str):
            paths.append(value)
    for item in getattr(args, "files", []) or []:
        path = getattr(item, "file_path", item if isinstance(item, str) else None)
        if isinstance(path, str):
            paths.append(path)
    paths.extend(getattr(args, "test_files", []) or [])
    return sorted(set(path.split("::")[0] for path in paths))


def refresh_context(node, workspace, load_instructions: bool, requested_paths: list[str] = ()):
    try:
        workspace.repository
    except ValueError:
        node.repository_context = RepositoryContext(workspace_version=workspace_version(workspace, node.file_context))
        return []
    lineage = node.get_trajectory()
    paths = set(requested_paths)
    for previous in lineage:
        for step in previous.action_steps:
            if step.observation is not None:
                paths.update(action_paths(step.action))
        if previous.repository_context:
            paths.update(previous.repository_context.files)
        if previous.task_state:
            paths.update(previous.task_state.related_files)
    if node.file_context:
        paths.update(node.file_context.file_paths)
    files, errors = {}, []
    for path in sorted(paths):
        try:
            path = relative_path(workspace, path)
            files[path] = {"content_hash": file_version(workspace, node.file_context, path)}
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {type(exc).__name__}")

    current = {}
    unavailable_rules = set()
    for previous in lineage:
        for instruction in previous.instruction_updates:
            current[instruction.path] = instruction
    if load_instructions:
        rule_paths = {"AGENTS.md"}
        for path in files:
            parent = PurePosixPath(path).parent
            for directory in [*reversed(parent.parents), parent]:
                rule_paths.add(str(directory / "AGENTS.md"))
        for path in sorted(rule_paths, key=lambda p: (len(PurePosixPath(p).parts), p)):
            try:
                content = file_content(workspace, node.file_context, path)
                digest = content_hash(content)
                old = current.get(path)
                if old is not None and old.content_hash == digest:
                    continue
                if old is None and content is None:
                    continue
                instruction = InstructionContext(path=path, scope=str(PurePosixPath(path).parent),
                                                 content=content, content_hash=digest, source_node_id=node.node_id)
                node.instruction_updates.append(instruction)
                current[path] = instruction
            except (OSError, ValueError) as exc:
                errors.append(f"{path}: {type(exc).__name__}")
                unavailable_rules.add(path)
    active = sorted((item for item in current.values() if item.content is not None and item.path in rule_paths and item.path not in unavailable_rules),
                    key=lambda item: (len(PurePosixPath(item.path).parts), item.path)) if load_instructions else []
    node.repository_context = RepositoryContext(
        workspace_version=workspace_version(workspace, node.file_context), files=files,
        instructions=[item.model_dump(exclude={"content"}) for item in active], errors=errors,
    )
    return active


def invalidate_read_cache(workspace, context, paths: list[str]) -> None:
    if not context:
        return
    for path in paths:
        path = relative_path(workspace, path)
        file = context._files.get(path)
        if file and not workspace.repository.shadow_mode:
            # Keep the original edit base/patch, but read actual disk content.
            current = file_content(workspace, None, path)
            if file.patch and current is not None:
                # A Bash/external edit must not leave an older patch associated
                # with the now-current disk view. Keep its original base.
                file.patch = file.generate_patch(file.get_base_content(), current)
            file._cached_content = current
            file._cached_module = None
        elif file and not file.patch:
            # Preserve patched bases; refresh only unedited code before a read.
            file._cached_content = None
            file._cached_base_content = None
            file._cached_module = None
