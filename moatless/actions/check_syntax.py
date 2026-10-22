from pydantic import ConfigDict, Field
from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.actions.code_modification_mixin import CodeModificationMixin


class CheckSyntaxArgs(ActionArguments):
    """Compile current Python sources in the task interpreter without executing them."""

    model_config = ConfigDict(title="CheckSyntax")
    paths: list[str] = Field(min_length=1, max_length=20)


class CheckSyntax(Action, CodeModificationMixin):
    args_schema = CheckSyntaxArgs

    async def execute(self, args, file_context=None):
        from moatless.repository_context import file_content

        checks = []
        for path in args.paths:
            content = file_content(self._workspace, file_context, path)
            if content is None:
                return Observation.create("Source not found: " + path, properties={"fail_reason": "file_not_found"})
            _, check = await self.validate_candidate(path, "", content)
            checks.append(check)
        return Observation.create(
            "Target compilation: " + str([{k: v for k, v in check.items() if k != "environment"} for check in checks]),
            properties={"syntax_checks": checks},
        )
