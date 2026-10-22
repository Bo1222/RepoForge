"""Public preparation boundaries and real metadata tampering checks."""

import pytest

from moatless.evaluation.native_provision import NativeProvisioner, native_env_commands


PUBLIC = {"instance_id": "pytest-dev__pytest-7490", "repo": "pytest-dev/pytest", "version": "6.0",
          "base_commit": "7f7a36478abe7dd1fa993b115d22606aa0e35e88", "problem_statement": "Public issue"}


@pytest.mark.asyncio
async def test_hidden_fields_rejected_before_any_preparation_side_effect(tmp_path):
    provisioner = NativeProvisioner(tmp_path / "preparation", conda="/bin/true")
    with pytest.raises(ValueError, match="only public"):
        await provisioner.prepare({**PUBLIC, "test_patch": "private", "FAIL_TO_PASS": ["hidden"]})
    assert not (tmp_path / "preparation" / "instances").exists()


@pytest.mark.asyncio
async def test_missing_runtime_never_becomes_ready(tmp_path):
    with pytest.raises(RuntimeError, match="Conda is required"):
        NativeProvisioner(tmp_path / "preparation", conda=str(tmp_path / "missing"))


@pytest.mark.asyncio
async def test_astropy_compatibility_change_gets_a_new_recipe_before_install(tmp_path, monkeypatch):
    import moatless.evaluation.native_provision as native
    from moatless.evaluation.experiment import read_json

    public = {**PUBLIC, "instance_id": "astropy__astropy-8872", "repo": "astropy/astropy", "version": "3.1"}
    provisioner = NativeProvisioner(tmp_path / "preparation", conda="/bin/true")
    monkeypatch.setattr(native, "native_env_commands", lambda *args: [])

    async def stop_before_effects(*args, **kwargs):
        raise RuntimeError("recipe inspection only")

    monkeypatch.setattr(provisioner, "_run", stop_before_effects)
    with pytest.raises(RuntimeError, match="recipe inspection"):
        await provisioner.prepare(public)
    recipes = list((tmp_path / "preparation/instances").glob("*/recipe.json"))
    assert len(recipes) == 1
    commands = read_json(recipes[0])["install_commands"]
    assert commands.index(native.ASTROPY_COMPATIBILITY_BOOTSTRAP) < commands.index(
        "python -m pip install -e .[test] --verbose")
    monkeypatch.setattr(native, "ASTROPY_COMPATIBILITY_BOOTSTRAP", "python -m pip install pyparsing==2.4.6 setuptools==58.0.4")
    with pytest.raises(RuntimeError, match="recipe inspection"):
        await provisioner.prepare(public)
    assert len(list((tmp_path / "preparation/instances").glob("*/recipe.json"))) == 2
    assert not list((tmp_path / "preparation/instances").glob("*/ready.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("version,legacy", [("0.22", True), ("1.3", False)])
async def test_sklearn_legacy_bootstrap_does_not_downgrade_modern_public_recipe(tmp_path, monkeypatch, version, legacy):
    import moatless.evaluation.native_provision as native
    from moatless.evaluation.experiment import read_json

    public = {**PUBLIC, "repo": "scikit-learn/scikit-learn", "version": version}
    provisioner = NativeProvisioner(tmp_path / "preparation", conda="/bin/true")
    monkeypatch.setattr(native, "native_env_commands", lambda *args: [])

    async def stop_before_effects(*args, **kwargs):
        raise RuntimeError("recipe inspection only")

    monkeypatch.setattr(provisioner, "_run", stop_before_effects)
    with pytest.raises(RuntimeError, match="recipe inspection"):
        await provisioner.prepare(public)
    path = next((tmp_path / "preparation/instances").glob("*/recipe.json"))
    commands = read_json(path)["install_commands"]
    assert (native.SCIKIT_PYTEST_BOOTSTRAP in commands) is legacy
    assert any("Cython==0.29.36" in command for command in commands) is legacy
    assert (native.SCIKIT_MODERN_BOOTSTRAP in commands) is (version == "1.3")


@pytest.mark.parametrize("specs", [
    {"python": "3.9", "packages": "pytest numpy"},
    {"python": "3.9", "packages": "requirements.txt"},
    {"python": "3.9", "packages": "environment.yml"},
    {"python": "3.9", "packages": "environment.yml", "no_use_env": True},
])
def test_every_conda_recipe_uses_private_prefix_and_preserves_dependencies(tmp_path, monkeypatch, specs):
    monkeypatch.setattr("swebench.harness.test_spec.python.get_requirements", lambda _: "attrs==23.1.0")
    monkeypatch.setattr("swebench.harness.test_spec.python.get_environment_yml", lambda *_: "name: testbed\ndependencies:\n- python=3.9\n")
    prefix = tmp_path / "private prefix"
    commands = native_env_commands(PUBLIC, specs, prefix, tmp_path / "conda-base")
    for command in commands:
        if "conda create " in command or "conda env create " in command or "conda env update " in command:
            assert "--prefix " in command
            assert str(prefix) in command
        assert "-n testbed" not in command and "activate testbed" not in command
        assert "$HOME" not in command and "/opt/miniconda3" not in command
    if specs["packages"] == "requirements.txt":
        assert any("attrs==23.1.0" in command for command in commands)


def test_pip_metadata_changes_are_visible_even_when_conda_records_do_not_change(tmp_path):
    metadata = tmp_path / "lib/python3.9/site-packages/example.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: example\nVersion: 1\n")
    conda = NativeProvisioner.metadata(tmp_path)
    before = NativeProvisioner.package_metadata(tmp_path)
    metadata.write_text("Name: example\nVersion: 2\n")
    assert NativeProvisioner.metadata(tmp_path) == conda
    assert NativeProvisioner.package_metadata(tmp_path) != before
def test_python36_bootstrap_precedes_requirements_without_changing_runtime_version(tmp_path, monkeypatch):
    from moatless.evaluation.native_provision import native_env_commands, PYTHON36_BOOTSTRAP
    instance = {"instance_id": "django__django-10914", "repo": "django/django",
                "version": "3.0", "base_commit": "0" * 40, "problem_statement": "public"}
    specs = {"python": "3.6", "packages": "requirements.txt", "install": "python -m pip install -e ."}
    monkeypatch.setattr("swebench.harness.test_spec.python.make_env_script_list_py", lambda *args: [
        "conda create -n testbed python=3.6 -y",
        "conda activate testbed && python -m pip install -r requirements.txt", "conda activate testbed"])
    commands = native_env_commands(instance, specs, tmp_path / "prefix", tmp_path / "conda-root")
    joined = "\n".join(commands)
    assert joined.count(PYTHON36_BOOTSTRAP) == 1
    assert joined.index(PYTHON36_BOOTSTRAP) < joined.index("python -m pip install -r requirements.txt")
    assert "python=3.6" in joined



@pytest.mark.asyncio
async def test_preparation_failure_keeps_raw_exit_and_command_evidence(tmp_path):
    import json
    provisioner = NativeProvisioner(tmp_path / "preparation", conda="/bin/true")
    log = tmp_path / "failure.log"
    argv = ["/bin/bash", "-c", "echo original-failure; exit 3"]
    with pytest.raises(RuntimeError, match="exit 3"):
        await provisioner._run(argv, tmp_path, log)
    records = [json.loads(line) for line in (tmp_path / "native-commands.jsonl").read_text().splitlines()]
    assert records[-1]["exit_code"] == 3 and records[-1]["argv"] == argv
    assert isinstance(records[-1]["process"]["pid"], int)
    assert log.read_text() == "original-failure\n"


@pytest.mark.asyncio
async def test_preparation_log_grows_before_cancel_and_original_process_is_reaped(tmp_path):
    import asyncio
    import json
    provisioner = NativeProvisioner(tmp_path / "preparation", conda="/bin/true")
    log = tmp_path / "active.log"
    task = asyncio.create_task(provisioner._run(["/bin/bash", "-c", "echo active; sleep 10"], tmp_path, log))
    for _ in range(100):
        if log.exists() and log.read_text().strip() == "active":
            break
        await asyncio.sleep(.01)
    assert log.read_text().strip() == "active" and not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    records = [json.loads(line) for line in (tmp_path / "native-commands.jsonl").read_text().splitlines()]
    assert records[-1]["exit_code"] == -9
    assert records[-1]["wall_seconds"] < 5
