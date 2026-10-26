import logging
from pathlib import Path

from pydantic import ConfigDict, Field

from repoforge.actions.action import Action
from repoforge.actions.code_action_value_mixin import CodeActionValueMixin
from repoforge.actions.code_modification_mixin import CodeModificationMixin
from repoforge.actions.run_python_script import RunPythonScript, RunPythonScriptArgs
from repoforge.actions.schema import ActionArguments, Observation
from repoforge.completion.schema import FewShotExample
from repoforge.file_context import FileContext

logger = logging.getLogger(__name__)


class CreatePythonFileArgs(ActionArguments):
    """
    Create a new Python file with specified content and optionally execute it.
    If execute is True, the file will be run after creation using Python
    """

    path: str = Field(..., description="Path where the new Python file should be created")
    file_text: str = Field(..., description="Complete content to write to the new file")
    execute: bool = Field(
        default=False, 
        description="Whether to execute the Python file after creating it. Set to True to run the script immediately after creation."
    )

    model_config = ConfigDict(title="CreateFile")

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<file_text>
{self.file_text}
</file_text>
<execute>{self.execute}</execute>"""

    @classmethod
    def format_schema_for_llm(cls, thoughts_in_action: bool = False) -> str:
        return cls.format_xml_schema({
            "path": "file/path.py", 
            "file_text": "\ncomplete file content\n",
            "execute": "false"
        })

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Create a new Python file for handling user authentication",
                action=CreatePythonFileArgs(
                    thoughts="Creating a new authentication module with basic user authentication functionality",
                    path="auth/user_auth.py",
                    execute=False,
                    file_text="""import logging
from typing import Optional

logger = logging.getLogger(__name__)

class UserAuth:
    def __init__(self):
        self._users = {}

    def authenticate(self, username: str, password: str) -> bool:
        if username not in self._users:
            logger.warning(f"Authentication failed: User {username} not found")
            return False

        return self._users[username] == password

    def register(self, username: str, password: str) -> bool:
        if username in self._users:
            logger.error(f"Registration failed: User {username} already exists")
            return False

        self._users[username] = password
        logger.info(f"User {username} registered successfully")
        return True""",
                ),
            )
        ]


class CreatePythonFile(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to create a new file with specified content.
    """

    args_schema = CreatePythonFileArgs

    async def execute(
        self,
        args: CreatePythonFileArgs,  # type: ignore[override]
        file_context: FileContext | None = None,
    ) -> Observation:
        if not file_context:
            raise RuntimeError("File context is not set")
        
        if args.path.startswith("/repo"):
            args.path = args.path[5:]
        if args.path.startswith("/"):
            args.path = args.path[1:]

        path = Path(args.path)

        if file_context.file_exists(str(path)):
            return Observation.create(
                message=f"File already exists at: {path}. Cannot overwrite files using create command.",
                properties={"fail_reason": "file_exists"},
            )

        token, conflict = self.edit_token(file_context, path)
        if conflict:
            return conflict
        error, check = await self.validate_candidate(str(path), "", args.file_text)
        if error:
            return error
        context_file = file_context.add_file(str(path), show_all_spans=True)
        conflict = self.commit_candidate(context_file, args.file_text, token)
        if conflict:
            return conflict

        message = f"File created successfully at: {path}"

        # If execute is True, run the script using RunPythonScript
        if args.execute:
            run_script = RunPythonScript()
            await run_script.initialize(self.workspace)
            run_args = RunPythonScriptArgs(thoughts="Executing created Python file", script_path=str(path))
            observation = await run_script.execute(run_args, file_context)
            observation.properties = {**(observation.properties or {}), "syntax_checks": [check],
                                      "syntax_checked": check["status"] == "valid"}
            return observation
            
        return Observation.create(message=message, summary=message, properties={"syntax_checks": [check], "syntax_checked": check["status"] == "valid"})
