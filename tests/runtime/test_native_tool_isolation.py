import subprocess

import pytest

from repoforge.evaluation.generation import copy_setup_artifacts
from repoforge.runtime.native_tools import tool_process


def test_native_tool_environment_is_offline_and_drops_credentials(monkeypatch):
    monkeypatch.setattr("repoforge.runtime.native_tools.shutil.which", lambda *args, **kwargs: "/usr/bin/bwrap")
    original = {"GENERATION_TOOL_ISOLATION": "native-offline-v1", "PATH": "/usr/bin:/bin",
                "DEEPSEEK_API_KEY": "offline-deepseek", "JINA_API_KEY": "offline-jina",
                "https_proxy": "http://127.0.0.1:17890", "PUBLIC_SETTING": "kept"}
    argv, env = tool_process(["python", "test.py"], original)
    assert "--unshare-net" in argv and "--unshare-pid" in argv
    assert argv[argv.index("--proc") + 1] == "/proc"
    assert env["PUBLIC_SETTING"] == "kept"
    assert not {"DEEPSEEK_API_KEY", "JINA_API_KEY", "https_proxy"} & env.keys()
    assert original["JINA_API_KEY"] == "offline-jina"


def test_non_native_tools_keep_existing_environment():
    args = ["bash", "-c", "true"]
    env = {"PATH": "/bin", "PUBLIC_SETTING": "kept"}
    assert tool_process(args, env) == (args, env)


def test_native_tools_refuse_missing_bubblewrap(monkeypatch):
    monkeypatch.setattr("repoforge.runtime.native_tools.shutil.which", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="no network fallback"):
        tool_process(["python"], {"GENERATION_TOOL_ISOLATION": "native-offline-v1"})


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)


def test_worktree_copies_only_public_ignored_setup_artifacts(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    git(source, "init", "-q")
    (source / ".gitignore").write_text("build/\n__pycache__/\n")
    (source / "tracked.py").write_text("original\n")
    git(source, "add", ".")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    (source / "build").mkdir()
    (source / "build/generated.so").write_bytes(b"compiled fixture")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/stale.pyc").write_bytes(b"stale")
    copied = copy_setup_artifacts(source, target)
    assert [row["path"] for row in copied] == ["build/generated.so"]
    assert (target / "build/generated.so").read_bytes() == b"compiled fixture"
    assert not (target / "__pycache__").exists() and not (target / "tracked.py").exists()


def test_worktree_setup_artifact_symlink_is_rejected(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    git(source, "init", "-q")
    (source / ".gitignore").write_text("generated\n")
    (source / "generated").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes public source"):
        copy_setup_artifacts(source, target)


def test_private_runtime_copy_cannot_modify_prepared_environment(tmp_path, monkeypatch):
    import repoforge.evaluation.experiment_native as native
    from repoforge.evaluation.experiment import content_hash, write_json
    from repoforge.evaluation.experiment_inputs import index_settings_for
    prefix, repo = tmp_path / "prepared-prefix", tmp_path / "prepared-repo"
    prefix.mkdir()
    (prefix / "dependency").write_text("original dependency\n")
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "calc.py").write_text("public fixture\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    base = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    public = {"instance_id": "pytest-dev__pytest-7490", "repo": "pytest-dev/pytest",
              "version": "6.0", "base_commit": base, "problem_statement": "public fixture"}
    output = tmp_path / "experiment"
    job = {"job_id": "job", "variant": "a0", "instance_id": public["instance_id"], "entrypoint": "batch"}
    manifest = {"manifest_hash": "manifest", "source": {"sha256": "source", "files": {}},
                "cli_config": {"api_key_env": "DEEPSEEK_API_KEY"}, "agent_configs": {"a0": {}},
                "profiles": {"a0": {}}, "prices": {}, "embedding_settings": {"embed_model": "jina-embeddings-v3"}}
    write_json(output / "provision-native.json", {public["instance_id"]: {
        "public": public, "source_sha256": "source", "environment": {"repo": str(repo), "prefix": str(prefix)}}})
    write_json(output / "indexes" / public["instance_id"] / "index-receipt.json", {
        "instance": public, "settings": index_settings_for(manifest).model_dump(mode="json"),
        "files": {}, "digest": content_hash({})})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline")
    calls = []
    monkeypatch.setattr(native, "executor_for", lambda *args, **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr(native, "worker_command", lambda *args, **kwargs: ["offline"])
    def worker(*args, **kwargs):
        # Model the healthy artifacts emitted by the real worker, rather than
        # bypassing the host's generation/usage acceptance boundary.
        directory = output / "runs" / job["job_id"]
        write_json(directory / "usage.json", {"cost_complete": True, "reference_cost": 0})
        write_json(directory / "generation.json", {"generation_status": "completed", "selected_node_id": 1})
        return 0

    monkeypatch.setattr(native, "run_worker", worker)
    monkeypatch.setattr(native, "verify_answer", lambda *args, **kwargs: {"verified": True})
    assert native.run_job(output, manifest, job) == {"verified": True}
    private = output / "runtime-prefixes/job/dependency"
    assert private.stat().st_ino != (prefix / "dependency").stat().st_ino
    private.write_text("changed privately\n")
    assert (prefix / "dependency").read_text() == "original dependency\n"
    assert calls[-1]["runtime_writable"] is True
    (output / "runtime-prefixes/job").rename(output / "missing-prefix")
    with pytest.raises(ValueError, match="Original private runtime is missing"):
        native.run_job(output, manifest, job, resume=True)


def test_native_repository_cannot_read_parent_proc_or_escape_symlink(tmp_path, monkeypatch):
    from repoforge.repository.file import FileRepository
    monkeypatch.setenv("GENERATION_EXECUTION_BACKEND", "native")
    repository = FileRepository(repo_path=str(tmp_path))
    (tmp_path / "outside").symlink_to("/proc")
    with pytest.raises(ValueError, match="escapes the public task"):
        repository.get_full_path("../../proc/self/environ")
    with pytest.raises(ValueError, match="escapes the public task"):
        repository.get_full_path("outside/self/environ")
    assert repository.get_full_path("src/module.py") == str(tmp_path / "src/module.py")
