"""One current-content view for search recall, rendering and file versions."""

from fnmatch import fnmatch
from pathlib import Path
import subprocess

from moatless.repository.file import CodeFile


def repository_paths(repository) -> list[str]:
    files = getattr(repository, "files", None)
    if isinstance(files, dict):
        return sorted(files)
    root = getattr(repository, "repo_dir", None) or getattr(repository, "repo_path", None)
    if not root:
        return []
    try:
        result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                cwd=root, capture_output=True, check=True, timeout=5)
        return sorted(set(result.stdout.decode().split("\0")) - {""})
    except (OSError, subprocess.SubprocessError):
        return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*.py")
                      if not {".git", ".venv", "__pycache__"}.intersection(p.relative_to(root).parts))


class CurrentRepository:
    def __init__(self, repository, context):
        self.base, self.context = repository, context

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def files(self):
        return {path: self.get_file_content(path) for path in
                sorted(set(repository_paths(self.base)) | set(self.context._files))}

    def get_file_content(self, path):
        file = self.context._files.get(path)
        if file is not None and self.base.shadow_mode:
            return file.content
        try:
            return self.base.get_file_content(path)
        except (OSError, UnicodeError, ValueError):
            return None

    def get_file(self, path):
        content = self.get_file_content(path)
        return CodeFile.from_content(path, content) if content is not None else None

    def file_exists(self, path):
        return self.get_file_content(path) is not None

    async def find_exact_matches(self, search_text, file_pattern=None):
        touched = {path for path, file in self.context._files.items() if file.patch}
        finder = getattr(self.base, "find_exact_matches", None)
        if callable(finder):
            base_hits = await finder(search_text=search_text, file_pattern=file_pattern)
            hits = [(path, line) for path, line in base_hits if path not in touched]
            paths = touched
        else:
            hits = []
            paths = set(repository_paths(self.base)) | set(self.context._files)
        for path in sorted(paths):
            content = self.get_file_content(path)
            if not content or (file_pattern and not (fnmatch(path, file_pattern) or
                    fnmatch(path, file_pattern.removeprefix("**/")))):
                continue
            lines = content.splitlines()
            for index in range(len(lines)):
                if "\n".join(lines[index:index + len(search_text.splitlines())]).find(search_text) >= 0:
                    hits.append((path, index + 1))
        return sorted(set(hits))
