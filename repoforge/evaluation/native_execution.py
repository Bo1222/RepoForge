"""Explicit Bubblewrap mounts and process lifecycle for native experiments.

Only the host coordinator constructs these mounts. Generated agents receive the
public bundle and operate inside the namespace; they cannot add host mounts.
"""

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import time


@dataclass(frozen=True)
class NativeMount:
    source: Path
    target: str
    writable: bool = False

    def validate(self):
        target = PurePosixPath(self.target)
        if not target.is_absolute() or ".." in target.parts or str(target) != self.target:
            raise ValueError("Native mount targets must be normalized absolute paths")
        if target in map(PurePosixPath, ("/", "/proc", "/dev", "/tmp", "/run", "/etc", "/home")):
            raise ValueError("Cannot replace a namespace root or private system directory")
        if any(target == system or system in target.parents for system in map(
                PurePosixPath, ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/proc", "/dev", "/run", "/home"))):
            raise ValueError("Additional mounts cannot replace readonly system paths")
        source = Path(self.source)
        if source.is_symlink() or not source.exists():
            raise ValueError("Native mounts require existing, nonsymlink roots")
        if source.resolve() == Path("/"):
            raise ValueError("The host filesystem root cannot be exposed to an agent")


@dataclass(frozen=True)
class NativeCommandResult:
    return_code: int
    output: str
    seconds: float
    timed_out: bool = False


class NativeExecutor:
    """Run one process tree in an allowlisted filesystem without shell fallback."""

    def __init__(self, mounts, *, environment=None, network=True):
        self.mounts = tuple(mounts)
        self._environment = dict(environment or {})
        self.network = network
        targets = []
        for mount in self.mounts:
            mount.validate()
            target = PurePosixPath(mount.target)
            if any(target == prior or target in prior.parents or prior in target.parents for prior in targets):
                raise ValueError("Overlapping native mounts can bypass readonly boundaries")
            targets.append(target)
        if any(not isinstance(k, str) or not isinstance(v, str) or "\0" in k + v
               for k, v in self._environment.items()):
            raise ValueError("Native environment must contain string names and values")

    def command(self, argv, *, cwd="/testbed"):
        if not argv or not all(isinstance(value, str) and "\0" not in value for value in argv):
            raise ValueError("Native commands require an explicit argv")
        executable = shutil.which("bwrap")
        if not executable:
            raise RuntimeError("Bubblewrap is required; refusing unisolated fallback")
        command = [executable, "--die-with-parent", "--new-session", "--unshare-user",
                   "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--proc", "/proc",
                   "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run",
                   "--dir", "/home", "--dir", "/etc"]
        if not self.network:
            command.append("--unshare-net")
        # System programs/libraries and selected public resolver/certificate files.
        # No host /home, /root, /tmp, dataset caches, grader or Docker socket.
        command.extend(["--ro-bind", "/usr", "/usr"])
        for name in ("bin", "sbin", "lib", "lib64"):
            path = Path("/") / name
            if path.is_symlink():
                command.extend(["--symlink", os.readlink(path), str(path)])
            elif path.exists():
                command.extend(["--ro-bind", str(path), str(path)])
        for name in ("ssl", "hosts", "resolv.conf", "nsswitch.conf", "passwd", "group", "ld.so.cache"):
            path = Path("/etc") / name
            if path.exists():
                command.extend(["--ro-bind", str(path), str(path)])
        for mount in self.mounts:
            mount.validate()  # Recheck at execution, not just at construction.
            command.extend(["--bind" if mount.writable else "--ro-bind",
                            str(Path(mount.source).resolve()), mount.target])
        command.extend(["--chdir", cwd, "--", *argv])
        return command

    def environment(self):
        # subprocess.env replaces the parent environment. Credentials must be
        # deliberately supplied by the coordinator and never appear in argv.
        return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "XDG_CACHE_HOME": "/tmp/cache", "TMPDIR": "/tmp",
                "GIT_CONFIG_GLOBAL": "/tmp/gitconfig", "GIT_CONFIG_NOSYSTEM": "1",
                **self._environment}

    async def run(self, argv, *, cwd="/testbed", timeout=120, redact=()):
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *self.command(argv, cwd=cwd), env=self.environment(),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        communication = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            stdout, _ = await asyncio.wait_for(asyncio.shield(communication), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, _ = await communication
            if isinstance(exc, asyncio.CancelledError):
                raise
            timed_out = True
        output = stdout.decode(errors="replace")
        for secret in redact:
            if secret:
                output = output.replace(secret, "[REDACTED]")
        return NativeCommandResult(-1 if timed_out else process.returncode, output,
                                   time.monotonic() - started, timed_out)
