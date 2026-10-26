from repoforge.repository.file import FileRepository


def __getattr__(name):
    if name == "GitRepository":
        from repoforge.repository.git import GitRepository

        return GitRepository
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
