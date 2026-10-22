"""Historical public-only source/Conda preparation for native evaluation.

The coordinator owns this module. Neither dataset rows with answers nor grader
directories are inputs to the generated agent's preparation recipe.
"""

import asyncio
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import time

from moatless.evaluation.experiment import append_json, content_hash, now, public_instance, read_json, sha, write_json


PYTHON36_BOOTSTRAP = "python -m pip install --upgrade pip==21.3.1 setuptools==59.6.0 wheel==0.37.1"
SCIKIT_CYTHON_BOOTSTRAP = "conda remove --offline --force --yes --prefix {prefix} cython\npython -m pip install Cython==0.29.36"
SCIKIT_PYTEST_BOOTSTRAP = "python -m pip install pytest==4.6.11"
SCIKIT_MODERN_BOOTSTRAP = "python -m pip install setuptools==58.0.4 pytest==7.4.0"
ASTROPY_COMPATIBILITY_BOOTSTRAP = "python -m pip install pyparsing==2.4.7 setuptools==58.0.4"


def native_env_commands(instance, specs, prefix, conda_root):
    """Use pinned Harness installation recipes; only adapt prefix/path routing."""
    from swebench.harness.test_spec.python import make_env_script_list_py

    prefix = Path(prefix).resolve()
    commands = make_env_script_list_py(instance, specs, "testbed")
    adapted = []
    bootstrapped = False
    for command in commands:
        command = command.replace("/opt/miniconda3", str(Path(conda_root).resolve()))
        command = command.replace("conda create -n testbed", f"conda create --prefix {shlex.quote(str(prefix))}")
        command = command.replace("-c conda-forge -n testbed", f"-c conda-forge --prefix {shlex.quote(str(prefix))}")
        command = command.replace("conda activate testbed", f"conda activate {shlex.quote(str(prefix))}")
        activation = f"conda activate {shlex.quote(str(prefix))}"
        if str(specs["python"]) == "3.6" and not bootstrapped and activation in command:
            # The Python 3.6 defaults-channel pip cannot select current abi3 /
            # manylinux wheels and incorrectly attempts Rust source builds.
            command = command.replace(activation, activation + " && " + PYTHON36_BOOTSTRAP, 1)
            bootstrapped = True
        if "conda env create --file" in command or "conda env update -f" in command:
            command += f" --prefix {shlex.quote(str(prefix))}"
        # Keep public recipe files in the preparation directory, not the user's home.
        command = command.replace("$HOME/requirements.txt", "requirements.txt")
        adapted.append(command)
    return adapted


class NativeProvisioner:
    """Prepare disposable task copies and preserve failures/readiness evidence."""

    def __init__(self, directory, *, conda=None, public_tools=None):
        self.directory = Path(directory).resolve()
        self.conda = Path(conda or shutil.which("conda") or "")
        if not self.conda.is_file():
            raise RuntimeError("Conda is required for historical native task environments")
        self.conda_root = self.conda.resolve().parent.parent
        self.public_tools = public_tools
        self.directory.mkdir(parents=True, exist_ok=True)

    async def _run(self, argv, cwd, log, *, timeout=1800, extra_environment=None):
        env = {"PATH": f"{self.conda_root / 'bin'}:/usr/bin:/bin", "LANG": "C.UTF-8",
               "CONDA_PKGS_DIRS": str(self.directory / "conda-packages"),
               "CONDA_ENVS_PATH": str(self.directory / "named-environments"),
               "PIP_CACHE_DIR": str(self.directory / "pip-cache"),
               "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_TERMINAL_PROMPT": "0", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        from moatless.evaluation.native_public_tools import extend_environment
        extend_environment(env, extra_environment)
        # Retain network transport settings without inheriting model credentials.
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
                     "https_proxy", "http_proxy", "all_proxy", "no_proxy", "SSL_CERT_FILE"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        log = Path(log)
        started, clock = now(), time.monotonic()
        with log.open("wb") as stream:
            process = await asyncio.create_subprocess_exec(*argv, cwd=cwd, env=env,
                stdout=stream, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            try:
                ticks = Path(f"/proc/{process.pid}/stat").read_text().split()[21]
            except FileNotFoundError:
                ticks = None  # A short command may already have been reaped.
            record = {"argv": list(argv), "cwd": str(cwd), "log": str(log), "started": started,
                      "process": {"pid": process.pid, "start_ticks": ticks}}
            append_json(log.parent / "native-commands.jsonl", {**record, "event": "started"})
            waiting = asyncio.create_task(process.wait())
            try:
                await asyncio.wait_for(asyncio.shield(waiting), timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await waiting
                raise
            finally:
                append_json(log.parent / "native-commands.jsonl", {**record, "event": "completed",
                    "ended": now(), "wall_seconds": time.monotonic() - clock,
                    "exit_code": process.returncode, "log_sha256": sha(log)})
        if process.returncode:
            raise RuntimeError(f"Native preparation failed (exit {process.returncode}); see {log}")
        return log.read_text(errors="replace")

    async def prepare(self, instance, *, environment_setup_commit=None):
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, MAP_REPO_TO_INSTALL

        public = public_instance(instance)
        if set(instance) != set(public):
            raise ValueError("Native preparation accepts only public instance fields; installation recipes are separate")
        instance = public
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", instance["repo"]):
            raise ValueError("Invalid public repository name")
        if not re.fullmatch(r"[0-9a-f]{40}", instance["base_commit"]):
            raise ValueError("Native source requires the full base commit")
        if environment_setup_commit and not re.fullmatch(r"[0-9a-f]{40}", environment_setup_commit):
            raise ValueError("Installation recipe requires a full setup commit")
        specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
        identity_parts = [instance, environment_setup_commit, specs, importlib.metadata.version("swebench")]
        if str(specs["python"]) == "3.6":
            identity_parts.append({"python36_bootstrap": PYTHON36_BOOTSTRAP})
        if instance["repo"] == "scikit-learn/scikit-learn" and instance["version"] == "0.22":
            identity_parts.append({"cython_bootstrap": SCIKIT_CYTHON_BOOTSTRAP})
            identity_parts.append({"pytest_bootstrap": SCIKIT_PYTEST_BOOTSTRAP})
        if instance["repo"] == "scikit-learn/scikit-learn" and instance["version"] == "1.3":
            identity_parts.append({"modern_bootstrap": SCIKIT_MODERN_BOOTSTRAP})
        if instance["repo"] == "astropy/astropy":
            identity_parts.append({"compatibility_bootstrap": ASTROPY_COMPATIBILITY_BOOTSTRAP})
        tool_environment = None
        if instance["repo"] == "matplotlib/matplotlib":
            from moatless.evaluation.native_public_tools import runtime_tools
            _, tool_environment = runtime_tools(self.public_tools, instance["repo"])
            identity_parts.append({"public_tools": self.public_tools["digest"], "adaptation": "private-media-tex-sources-v1"})
        identity = content_hash(identity_parts)
        target = self.directory / "instances" / identity
        ready = target / "ready.json"
        if ready.exists():
            receipt = read_json(ready)
            if receipt['public'] != public:
                raise ValueError('Cached preparation belongs to another public task')
            self.verify(receipt)
            return receipt
        if target.exists():
            raise ValueError("Preparation has uncertain/failed prior effects; preserve it and use a new preparation directory")
        target.mkdir(parents=True)
        cached_source = self.verified_cached_source(public)
        recipe = {**instance}
        if environment_setup_commit:
            recipe["environment_setup_commit"] = environment_setup_commit
        prefix, repo = target / "conda", target / "baseline"
        commands = native_env_commands(recipe, specs, prefix, self.conda_root)
        install = [f"source {shlex.quote(str(self.conda_root / 'bin/activate'))}",
                   f"conda activate {shlex.quote(str(prefix))}",
                   f"cd {shlex.quote(str(repo))}"]
        if instance["repo"] in MAP_REPO_TO_INSTALL:
            install.append(MAP_REPO_TO_INSTALL[instance["repo"]])
        if instance["repo"] == "matplotlib/matplotlib":
            from moatless.evaluation.native_public_tools import matplotlib_pre_install
            install.extend(matplotlib_pre_install(specs, self.public_tools, repo))
        else:
            if any("apt-get" in command for command in specs.get("pre_install", [])):
                raise ValueError("Native preparation cannot execute host system package installation")
            install.extend(specs.get("pre_install", []))
        if instance["repo"] == "scikit-learn/scikit-learn" and instance["version"] == "0.22":
            # Pip cannot uninstall Conda's distutils package; overlaying leaves
            # old generic .so modules and mixes Cython ABIs. Remove the package
            # with its owning manager before installing the compatible wheel.
            install.append(SCIKIT_CYTHON_BOOTSTRAP.format(prefix=shlex.quote(str(prefix))))
            # Public conftest.py requires pytest >= 3.3. Conda's unpinned
            # historical solve can select 3.2; freeze a Python-3.6 runner.
            install.append(SCIKIT_PYTEST_BOOTSTRAP)
        if instance["repo"] == "scikit-learn/scikit-learn" and instance["version"] == "1.3":
            # This public commit requires pytest >=5.3.1 and setuptools <60;
            # the legacy 0.22 runner pin must not override its requirements.
            install.append(SCIKIT_MODERN_BOOTSTRAP)
        if instance["repo"] == "astropy/astropy":
            # Historical tests make dependency deprecations fatal, including
            # pyparsing's camelCase APIs and setuptools' distutils shim.
            install.append(ASTROPY_COMPATIBILITY_BOOTSTRAP)
        if instance['repo'] in {'pytest-dev/pytest', 'astropy/astropy', 'matplotlib/matplotlib', 'pydata/xarray'}:
            # Exact-base fetches deliberately have no future tags. Public
            # release version is a declared native installation adaptation.
            install.append('export SETUPTOOLS_SCM_PRETEND_VERSION=' + shlex.quote(instance['version']))
        if specs.get("install"):
            install.append(specs["install"])
        env_script, repo_script = target / "prepare-env.sh", target / "install-repo.sh"
        for path, lines in ((env_script, commands), (repo_script, install)):
            path.write_text("#!/bin/bash\nset -euo pipefail\n" + "\n".join(lines) + "\n")
        clone_script = target / 'clone-public-source.sh'
        remote = f"https://github.com/{instance['repo']}.git"
        source_commands = [['git', 'init', str(repo)],
            ['git', '-c', 'http.version=HTTP/1.1', '-C', str(repo), 'fetch', '--depth=1', remote, instance['base_commit']],
            ['git', '-C', str(repo), 'remote', 'add', 'origin', remote]]
        if cached_source:
            source_commands = [['git', 'clone', '--no-hardlinks', '--local', cached_source['repo'], str(repo)]]
        clone_script.write_text('#!/bin/bash\nset -euo pipefail\n' + '\n'.join(shlex.join(argv) for argv in source_commands) + '\n')
        write_json(target / "recipe.json", {"public": instance, "environment_setup_commit": environment_setup_commit,
            "specs": specs, "swebench": importlib.metadata.version("swebench"), "prefix": str(prefix),
            "commands": commands, "install_commands": install, "source_commands": source_commands,
            "public_source_fetch": 'verified-public-local-clone-v1' if cached_source else 'exact-base-shallow-http1-v1',
            "cached_public_source": cached_source, "provider_calls": 0})
        steps = []
        try:
            for name, argv, cwd in (
                ("clone", ["/bin/bash", str(clone_script)], target),
                ("checkout", ["git", "checkout", "--detach", instance["base_commit"]], repo),
                ("environment", ["/bin/bash", str(env_script)], target),
                ("install", ["/bin/bash", str(repo_script)], repo),
            ):
                log = target / f"{name}.log"
                if name == "install" and tool_environment:
                    await self._run(argv, cwd, log, extra_environment=tool_environment)
                else:
                    await self._run(argv, cwd, log)
                steps.append({"command": argv, "cwd": str(cwd), "exit_code": 0, "log": str(log)})
                write_json(target / "steps.json", steps)
            # Drop refs and unreachable future commits in our disposable copy.
            await self._run(["git", "remote", "remove", "origin"], repo, target / "remove-remote.log")
            refs = await self._run(["git", "for-each-ref", "--format=%(refname)"], repo, target / "refs.log")
            for ref in refs.splitlines():
                await self._run(["git", "update-ref", "-d", ref], repo, target / "drop-refs.log")
            # GitRepository requires a local branch, but newer public commits
            # must stay unreachable. Keep only the exact dataset base reference.
            await self._run(["git", "update-ref", "refs/heads/native-baseline", instance["base_commit"]],
                            repo, target / "base-reference.log")
            await self._run(["git", "reflog", "expire", "--expire=now", "--all"], repo, target / "reflog.log")
            await self._run(["git", "gc", "--prune=now"], repo, target / "gc.log")
            await self._run([str(self.conda), "list", "--prefix", str(prefix), "--json"],
                            target, target / "dependencies.json")
            module = {"pytest-dev/pytest": "pytest", "django/django": "django",
                      "sympy/sympy": "sympy", "scikit-learn/scikit-learn": "sklearn",
                      "astropy/astropy": "astropy", "matplotlib/matplotlib": "matplotlib",
                      "pydata/xarray": "xarray", "pylint-dev/pylint": "pylint",
                      "sphinx-doc/sphinx": "sphinx"}.get(instance["repo"])
            if not module:
                raise ValueError("Native provision requires an explicit source import probe for this repository")
            probe = ("import importlib,json,sys; from pathlib import Path; "
                     f"m=importlib.import_module({module!r}); "
                     f"p=Path(m.__file__).resolve(); assert Path({str(repo)!r}).resolve() in p.parents, str(p); "
                     "print(json.dumps({'python':sys.version, 'executable':sys.executable, 'source':str(p)}))")
            await self._run([str(prefix / "bin/python"), "-c", probe], repo, target / "import-probe.json")
            if instance["repo"] == "astropy/astropy":
                await self._run([str(prefix / "bin/python"), "-m", "pip", "check"],
                                repo, target / "dependency-check.log")
                public_test = repo / "astropy/units/tests/test_quantity.py"
                if public_test.exists():
                    # Use the actual console entrypoint. python -m pytest can
                    # change import order and conceal fatal collection warnings.
                    await self._run([str(prefix / "bin/pytest"), "--collect-only", "-q", str(public_test)],
                                    repo, target / "public-collection.log", timeout=120)
            if instance["repo"] == "scikit-learn/scikit-learn":
                await self._run([str(prefix / "bin/python"), "-c",
                    "import pytest,json; from distutils.version import LooseVersion; "
                    "assert LooseVersion(pytest.__version__) >= LooseVersion('3.3.0'); "
                    "print(json.dumps({'pytest':pytest.__version__}))"], repo, target / "test-runner-probe.json")
                public_test = repo / "sklearn/tree/tests/test_export.py"
                if public_test.exists():
                    await self._run([str(prefix / "bin/python"), "-m", "pytest", "-q", str(public_test)],
                                    repo, target / "public-tests.log", timeout=120)
            await self._run(["ldd", str(prefix / "bin/python")], repo, target / "system-libraries.log")
            # Record setup changes separately. Actual base remains the dataset commit.
            setup = await self._run(["git", "diff", "--binary", instance["base_commit"]], repo, target / "setup.patch")
            receipt = {"version": 1, "public": instance, "recipe_identity": identity,
                "repo": str(repo), "prefix": str(prefix), "python": specs["python"],
                "dependencies": content_hash(read_json(target / "dependencies.json")),
                "conda_metadata": self.metadata(prefix), "package_metadata": self.package_metadata(prefix),
                "setup_patch_sha256": sha(target / "setup.patch"),
                "setup_has_tracked_changes": bool(setup.strip()), "created": now(),
                "host": {"os": platform.platform(), "machine": platform.machine()},
                "evidence": {p.name: sha(p) for p in target.iterdir() if p.is_file()},
                "execution_backend": "native", "official_harness": "not_run", "provider_calls": 0}
            if tool_environment:
                receipt["public_tools_digest"] = self.public_tools["digest"]
            write_json(ready, receipt)
            return receipt
        except BaseException as exc:
            write_json(target / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                "steps_completed": steps, "provider_calls": 0, "official_harness": "not_run"})
            raise

    @staticmethod
    def metadata(prefix):
        return {p.name: sha(p) for p in sorted((Path(prefix) / "conda-meta").glob("*.json"))}

    def verified_cached_source(self, public):
        """Reuse only immutable public source, never another task or its runtime."""
        for cached in sorted((self.directory / "instances").glob("*/ready.json")):
            candidate = read_json(cached)
            if candidate.get("public") != public or candidate.get("setup_has_tracked_changes"):
                continue
            self.verify(candidate)
            return {"repo": candidate["repo"], "receipt": str(cached), "receipt_sha256": sha(cached)}
        return None

    @staticmethod
    def package_metadata(prefix):
        prefix = Path(prefix)
        paths = []
        for pattern in ("lib/python*/site-packages/*.dist-info/METADATA",
                        "lib/python*/site-packages/*.dist-info/direct_url.json",
                        "lib/python*/site-packages/*.egg-info/PKG-INFO"):
            paths.extend(prefix.glob(pattern))
        return {str(p.relative_to(prefix)): sha(p) for p in sorted(paths)}

    @staticmethod
    def relocate_editable(prefix, source_repo, destination_repo):
        """Route a private prefix's editable metadata to its namespace copy.

        Never modify the shared prepared prefix. Versions and dependencies are
        unchanged; this fixes pip's string-path assertion across bind aliases.
        """
        prefix = Path(prefix).resolve()
        original = str(source_repo).rstrip("/")
        destination = str(destination_repo).rstrip("/")
        paths = list(prefix.glob("lib/python*/site-packages/*.egg-link"))
        paths.extend(prefix.glob("lib/python*/site-packages/easy-install.pth"))
        changes = []
        for path in sorted(paths):
            if path.is_symlink() or not path.resolve().is_relative_to(prefix):
                raise ValueError("Editable metadata escapes the private runtime")
            body = path.read_text()
            lines = body.splitlines(keepends=True)
            updated = "".join(destination + line[len(original):]
                if line.rstrip("\r\n").rstrip("/") == original else line for line in lines)
            if updated == body:
                continue
            before = sha(path)
            path.write_text(updated)
            changes.append({"path": str(path.relative_to(prefix)), "before": before, "after": sha(path),
                            "source": original, "destination": destination})
        return changes

    @classmethod
    def verify(cls, receipt):
        repo, prefix = Path(receipt["repo"]), Path(receipt["prefix"])
        if not repo.is_dir() or not prefix.is_dir():
            raise ValueError("Original native repository/environment is missing; no silent reprovisioning")
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if actual != receipt["public"]["base_commit"] or cls.metadata(prefix) != receipt["conda_metadata"]:
            raise ValueError("Native preparation identity changed")
        if "package_metadata" in receipt and cls.package_metadata(prefix) != receipt["package_metadata"]:
            raise ValueError("Native Python package metadata changed")
        for name, expected in receipt["evidence"].items():
            if sha(repo.parent / name) != expected:
                raise ValueError("Native preparation evidence changed")
