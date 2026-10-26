from pydantic import BaseModel

from repoforge.artifacts.artifact import Artifact


class Task(BaseModel):
    description: str
    completed: bool = False


class PlanArtifact(Artifact):
    goal: str
    tasks: list[Task]
