import logging
from abc import abstractmethod

from repoforge.component import RepoForgeComponent
from repoforge.node import FeedbackData, Node

logger = logging.getLogger(__name__)


class BaseFeedbackGenerator(RepoForgeComponent):
    @abstractmethod
    async def generate_feedback(self, node: Node) -> FeedbackData | None:
        """Generate feedback based on the node."""
        pass

    @classmethod
    def get_component_type(cls) -> str:
        return "feedback_generator"

    @classmethod
    def _get_package(cls) -> str:
        return "repoforge.feedback"

    @classmethod
    def _get_base_class(cls) -> type:
        return BaseFeedbackGenerator
