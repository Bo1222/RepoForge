from abc import ABC, abstractmethod

from repoforge.testing.schema import TestResult


class RuntimeEnvironment(ABC):
    supports_managed_deadline = False

    def verification_input_paths(self, targets) -> list[str]:
        """Explicit public support/data inputs outside conventional test paths."""
        return list(getattr(self, "verification_files", []))

    def compilation_configuration(self) -> dict:
        return {"python_executable": getattr(self, "python_executable", None)}
    def verification_configuration(self, targets) -> dict | None:
        """Adapters must expose stable public runner settings, never output buffers."""
        if not getattr(self, "test_command", None) or not getattr(self, "python_executable", None):
            return None
        return {"adapter": f"{type(self).__module__}.{type(self).__name__}",
                "command": getattr(self, "test_command", None),
                "python_executable": getattr(self, "python_executable", None)}

    async def compile_python(self, source: str, path: str) -> dict:
        executable = getattr(self, "python_executable", None)
        if not executable:
            return {"status": "unavailable", "reason": "target_interpreter_not_configured", "path": path}
        import asyncio
        import os
        import signal
        from repoforge.runtime.python_check import arguments, parse_output
        process = await asyncio.create_subprocess_exec(executable, *arguments(path), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        communication = asyncio.create_task(process.communicate(source.encode()))
        try:
            stdout, _ = await asyncio.shield(communication)
            return parse_output(stdout.decode(errors="replace"), process.returncode)
        except asyncio.CancelledError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await communication
            raise

    @abstractmethod
    async def run_tests(self, patch: str | None = None, test_files: list[str] | None = None) -> list[TestResult]:
        pass


class NoEnvironment(RuntimeEnvironment):
    def recovery_state(self) -> dict:
        return {"kind": "no_environment"}

    async def run_tests(self, patch: str | None = None, test_files: list[str] | None = None) -> list[TestResult]:
        return []
