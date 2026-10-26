"""Read historical serialized class names without rewriting their source files."""


def canonical_classpath(classpath: str) -> str:
    """Map the previous package namespace at deserialization boundaries only."""
    if classpath.startswith("moatless."):
        return "repoforge." + classpath[len("moatless.") :]
    return classpath
