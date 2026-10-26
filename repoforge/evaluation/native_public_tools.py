"""Frozen public media/TeX tools, installed without changing host system packages."""

import os
from pathlib import Path

from repoforge.evaluation.experiment import content_hash, read_json, sha
from repoforge.evaluation.native_execution import NativeMount


def tree_identity(directory):
    root = Path(directory).resolve()
    files, links = {}, {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        if path.is_symlink():
            links[name] = os.readlink(path)
        elif path.is_file():
            files[name] = {"sha256": sha(path), "mode": path.stat().st_mode & 0o777}
    return {"files": files, "symlinks": links}


def freeze_profile(path):
    path = Path(path).resolve()
    profile = read_json(path)
    for entry in profile["roots"].values():
        if tree_identity(entry["path"]) != entry["identity"]:
            raise ValueError("Public native tool tree changed")
    return {"path": str(path), "sha256": sha(path), "profile": profile,
            "digest": content_hash(profile)}


def verify_profile(frozen):
    if freeze_profile(frozen["path"]) != frozen:
        raise ValueError("Frozen public native tools changed")
    return frozen["profile"]


def runtime_tools(frozen, repo):
    if repo != "matplotlib/matplotlib":
        return [], {}
    if not frozen:
        raise ValueError("Matplotlib native preparation requires frozen public tools")
    profile = verify_profile(frozen)
    roots = [Path(profile["roots"][name]["path"]) for name in ("media", "texlive")]
    mounts = [NativeMount(path, str(path)) for path in roots]
    env = {"PATH": f"{roots[0] / 'bin'}:{roots[1] / 'bin/x86_64-linux'}",
           "TEXMFHOME": "/tmp/texmf-home", "TEXMFCONFIG": "/tmp/texmf-config",
           "TEXMFVAR": "/tmp/texmf-var", "MPLCONFIGDIR": "/tmp/matplotlib"}
    return mounts, env


def extend_environment(environment, extra):
    """Keep the target Python first while making public executables available."""
    if extra:
        first, _, rest = environment["PATH"].partition(":")
        environment.update({key: value for key, value in extra.items() if key != "PATH"})
        environment["PATH"] = f"{first}:{extra['PATH']}:{rest}"


def matplotlib_pre_install(specs, frozen, repo):
    """Replace this exact Docker-only recipe with verified private equivalents."""
    import shlex
    profile = verify_profile(frozen)
    commands = specs.get("pre_install", [])
    if len(commands) != 7 or not commands[0].startswith("apt-get -y update") or not any(
            'QHULL_BUILD_DIR="/testbed/build"' == command for command in commands):
        raise ValueError("Unrecognized Matplotlib system preparation recipe")
    sources = Path(profile["roots"]["sources"]["path"])
    # Public source archives were checked against hashes in exact-base setupext.py.
    return [f"mkdir -p {shlex.quote(str(Path(repo) / 'build'))}",
            f"cp -a {shlex.quote(str(sources))}/. {shlex.quote(str(Path(repo) / 'build'))}/",
            "command -v convert ffmpeg gs latex pdflatex xelatex lualatex dvipng"]
