"""Public tools must stay immutable, offline and outside host system prefixes."""
from pathlib import Path

import pytest

from repoforge.evaluation.experiment import write_json
from repoforge.evaluation.native_public_tools import (
    extend_environment, freeze_profile, matplotlib_pre_install, runtime_tools, tree_identity,
)


def profile_at(tmp_path):
    roots = {}
    for name in ("media", "texlive", "sources"):
        path = tmp_path / name
        path.mkdir()
        (path / "public-file").write_text(name)
        roots[name] = {"path": str(path), "identity": tree_identity(path)}
    path = tmp_path / "tools.json"
    write_json(path, {"roots": roots})
    return freeze_profile(path)


def test_tool_mounts_are_readonly_and_target_python_keeps_precedence(tmp_path):
    frozen = profile_at(tmp_path)
    mounts, extra = runtime_tools(frozen, "matplotlib/matplotlib")
    assert len(mounts) == 2 and all(not mount.writable for mount in mounts)
    assert all(mount.target == str(mount.source) for mount in mounts)
    env = {"PATH": "/private/target/bin:/usr/bin:/bin"}
    extend_environment(env, extra)
    assert env["PATH"].split(":")[0] == "/private/target/bin"
    assert env["TEXMFVAR"].startswith("/tmp/")
    assert runtime_tools(frozen, "django/django") == ([], {})
    with pytest.raises(ValueError, match="requires frozen"):
        runtime_tools(None, "matplotlib/matplotlib")


@pytest.mark.parametrize("tamper", ["content", "extra-file", "link"])
def test_tool_mutation_is_rejected_before_mounting(tmp_path, tamper):
    frozen = profile_at(tmp_path)
    if tamper == "content":
        (tmp_path / "media/public-file").write_text("changed")
    elif tamper == "extra-file":
        (tmp_path / "media/new-file").write_text("extra")
    else:
        (tmp_path / "media/link").symlink_to("public-file")
    with pytest.raises(ValueError, match="changed"):
        runtime_tools(frozen, "matplotlib/matplotlib")


def test_matplotlib_docker_recipe_uses_verified_private_sources_and_no_apt(tmp_path):
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    frozen = profile_at(tmp_path)
    for version in ("3.6", "3.7"):
        specs = MAP_REPO_VERSION_TO_SPECS["matplotlib/matplotlib"][version]
        commands = matplotlib_pre_install(specs, frozen, Path("/private/public-repo"))
        assert not any("apt-get" in command or "wget" in command or "/testbed" in command for command in commands)
        assert "/private/public-repo/build" in "\n".join(commands)
        assert "latex" in commands[-1] and "ffmpeg" in commands[-1]
    with pytest.raises(ValueError, match="Unrecognized"):
        matplotlib_pre_install({"pre_install": ["apt-get install something-new"]}, frozen, Path("/repo"))
