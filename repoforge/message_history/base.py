from abc import abstractmethod

from repoforge.completion.schema import AllMessageValues
from repoforge.component import RepoForgeComponent
from repoforge.node import Node
from repoforge.workspace import Workspace


class BaseMemory(RepoForgeComponent):
    @abstractmethod
    async def generate_messages(self, node: Node, workspace: Workspace) -> list[AllMessageValues]:
        pass

    @classmethod
    def get_component_type(cls) -> str:
        return "memory"

    @classmethod
    def _get_package(cls) -> str:
        return "repoforge.message_history"

    @classmethod
    def _get_base_class(cls) -> type:
        return BaseMemory
