import logging
from abc import abstractmethod

from repoforge.component import RepoForgeComponent
from repoforge.node import Node, Selection

logger = logging.getLogger(__name__)


class BaseSelector(RepoForgeComponent):
    @abstractmethod
    async def select(self, expandable_nodes: list[Node]) -> Selection:
        pass

    @classmethod
    def get_component_type(cls) -> str:
        return "selector"

    @classmethod
    def _get_package(cls) -> str:
        return "repoforge.selector"

    @classmethod
    def _get_base_class(cls) -> type:
        return BaseSelector
