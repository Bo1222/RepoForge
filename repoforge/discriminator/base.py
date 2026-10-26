import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from repoforge.component import RepoForgeComponent
from repoforge.node import Node, DiscriminatorResult
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseDiscriminator(RepoForgeComponent, ABC):
    @abstractmethod
    def select(self, nodes: list[Node]) -> DiscriminatorResult:
        raise NotImplementedError

    @classmethod
    def get_component_type(cls) -> str:
        return "discriminator"

    @classmethod
    def _get_package(cls) -> str:
        return "repoforge.discriminator"

    @classmethod
    def _get_base_class(cls) -> type:
        return BaseDiscriminator
