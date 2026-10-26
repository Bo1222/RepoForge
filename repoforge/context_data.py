import contextvars
import os
from pathlib import Path

# Used to track the current action step, node id, trajectory id, and project id
current_phase = contextvars.ContextVar[str | None]("current_phase", default=None)
current_action_step = contextvars.ContextVar[int | None]("current_action_step", default=None)
current_node_id = contextvars.ContextVar[int | None]("current_node_id", default=None)
current_trajectory_id = contextvars.ContextVar[str | None]("current_trajectory_id", default=None)
current_project_id = contextvars.ContextVar[str | None]("current_project_id", default=None)


def get_repoforge_dir() -> Path:
    """Get the repoforge directory."""
    if not os.getenv("REPOFORGE_DIR"):
        raise ValueError("REPOFORGE_DIR environment variable is not set")
    return Path(os.getenv("REPOFORGE_DIR"))


def get_projects_dir() -> Path:
    """Get the repoforge projects directory."""
    return get_repoforge_dir() / "projects"


def get_project_dir(project_id: str | None = None) -> Path:
    """Get the repoforge project directory."""

    if project_id:
        project_id = project_id
    else:
        project_id = current_project_id.get()

    if project_id:
        project_dir = get_repoforge_dir() / "projects" / project_id / "trajs"
    else:
        project_dir = get_repoforge_dir() / "projects" / "default" / "trajs"

    if not project_dir.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def get_trajectory_dir(project_id: str | None = None, trajectory_id: str | None = None) -> Path:
    """Get the repoforge trajectory directory."""

    trajectory_base_dir = get_project_dir(project_id)
    if not trajectory_id:
        return trajectory_base_dir
    trajectory_dir = trajectory_base_dir / trajectory_id
    if not trajectory_dir.exists():
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    return trajectory_dir
