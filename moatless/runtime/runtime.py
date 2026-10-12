from abc import ABC, abstractmethod

from moatless.testing.schema import TestResult


class RuntimeEnvironment(ABC):
    @abstractmethod
    async def run_tests(self, patch: str | None = None, test_files: list[str] | None = None) -> list[TestResult]:
        pass


class NoEnvironment(RuntimeEnvironment):
    def recovery_state(self) -> dict:
        return {"kind": "no_environment"}

    async def run_tests(self, patch: str | None = None, test_files: list[str] | None = None) -> list[TestResult]:
        return []
