from types import SimpleNamespace
import subprocess

import pytest

from moatless.evaluation.native_provision import NativeProvisioner
from moatless.evaluation.native_grading import NativeGrader


def test_private_editable_routes_match_namespace_source_and_preserve_other_packages(tmp_path):
    site = tmp_path / "lib/python3.6/site-packages"
    site.mkdir(parents=True)
    link = site / "project.egg-link"
    pth = site / "easy-install.pth"
    link.write_text("/original/source\n.\n")
    pth.write_text("/other/dependency\n/original/source\nimport sys\n")
    changes = NativeProvisioner.relocate_editable(tmp_path, "/original/source", "/testbed")
    assert len(changes) == 2
    assert link.read_text() == "/testbed\n.\n"
    assert pth.read_text() == "/other/dependency\n/testbed\nimport sys\n"
    # CLI worktrees use another source path; explicit routing remains coherent.
    NativeProvisioner.relocate_editable(tmp_path, "/testbed", "/out/session/worktree")
    assert link.read_text().splitlines()[0] == "/out/session/worktree"
    assert NativeProvisioner.relocate_editable(tmp_path, "/testbed", "/out/session/worktree") == []


def test_editable_symlink_cannot_modify_the_shared_prepared_prefix(tmp_path, monkeypatch):
    shared = tmp_path / "prepared-link"
    shared.write_text("/original/source\n.\n")
    private = tmp_path / "private/lib/python3.6/site-packages"
    private.mkdir(parents=True)
    (private / "project.egg-link").symlink_to(shared)
    link = private / "project.egg-link"
    original_read = type(link).read_text
    def read(path, *args, **kwargs):
        assert path != link, "Unsafe metadata must be rejected before reading its target"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(type(link), "read_text", read)
    with pytest.raises(ValueError, match="escapes"):
        NativeProvisioner.relocate_editable(tmp_path / "private", "/original/source", "/testbed")
    assert shared.read_text() == "/original/source\n.\n"


def test_pretest_install_failure_cannot_be_hidden_by_a_passing_test(tmp_path):
    from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
    spec = SimpleNamespace(eval_script_list=["/bin/bash -c 'exit 7'", f"echo '{START_TEST_OUTPUT}'",
        "true", f"echo '{END_TEST_OUTPUT}'", "true"])
    script = tmp_path / "eval.sh"
    script.write_text(NativeGrader.eval_script(spec, tmp_path))
    result = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True)
    assert result.returncode == 7
    assert START_TEST_OUTPUT not in result.stdout


def test_test_failure_still_publishes_end_marker_and_keeps_its_status(tmp_path):
    from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
    spec = SimpleNamespace(eval_script_list=["true", f"echo '{START_TEST_OUTPUT}'",
        "false", f"echo '{END_TEST_OUTPUT}'", "true"])
    script = tmp_path / "eval.sh"
    script.write_text(NativeGrader.eval_script(spec, tmp_path))
    result = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True)
    assert result.returncode == 1
    assert END_TEST_OUTPUT in result.stdout
