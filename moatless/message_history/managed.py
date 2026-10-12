"""Task-oriented context projection. Raw observations stay in Node history."""

from copy import deepcopy

from pydantic import PrivateAttr

from moatless.message_history.assembly import ContextUnit
from moatless.message_history.builder import ContextBuilder, ContextBuildRecord
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.message_history.policy import (
    ContextBudgetExceeded, ContextPolicy, TokenEstimator, content_hash, output_limit, request_envelope,
)
from moatless.node import Node
from moatless.workspace import Workspace
from moatless.execution_memory import READ_ACTIONS, failure_records, test_records
from moatless.repository_context import file_content, refresh_context


class ManagedMessageHistoryGenerator(MessageHistoryGenerator):
    policy: ContextPolicy
    _model: object = PrivateAttr(default=None)

    def bind_model(self, model) -> None:
        if {"messages", "model"}.intersection(model._completion_params or {}):
            raise ValueError("Managed context forbids params overriding the bound model or assembled messages")
        self._model = model

    async def generate_messages(self, node: Node, workspace: Workspace) -> list[dict]:
        if self._model is None:
            raise ValueError("Managed memory must be bound to the request model")
        model = self._model
        builder = ContextBuilder(model, self.policy)
        estimator, budget = builder.estimator, builder.budget
        units: list[ContextUnit] = []
        sources: list[str] = []
        shortened = []
        stale, deduplicated = [], []
        instructions = refresh_context(node, workspace, self.policy.load_repo_instructions)
        repository = node.repository_context
        tests = test_records(node, workspace)
        tests_by_source = {record["source_id"]: record for record in tests}
        latest_reads = {}
        read_keys = {}
        current_views = {}
        for previous in node.get_trajectory():
            for index, step in enumerate(previous.action_steps):
                if step.observation is None:
                    continue
                view = (step.observation.properties or {}).get("code_view")
                if not view:
                    continue
                source = f"{previous.node_id}:{index}"
                key = content_hash({"action": step.action.model_dump(exclude={"thoughts"}),
                                    "files": view["files"],
                                    "workspace": view["workspace_version"] if view["workspace_scoped"] else None})
                read_keys[source] = key
                latest_reads[key] = source
                for path in view["files"]:
                    current_views[path] = (source, view)
        for previous in node.get_trajectory():
            if previous.user_message:
                units.append(ContextUnit([{"role": "user", "content": previous.user_message}], required=True))
                sources.append(f"user:{previous.node_id}")
            completed = [(i, step) for i, step in enumerate(previous.action_steps) if step.observation is not None]
            if not completed:
                if previous.assistant_message:
                    units.append(ContextUnit([{"role": "assistant", "content": previous.assistant_message}]))
                    sources.append(f"assistant:{previous.node_id}")
                continue
            calls, results = [], []
            for index, step in completed:
                source = f"{previous.node_id}:{index}"
                calls.append({"id": f"node_{previous.node_id}_step_{index}", "type": "function", "function": {
                    "name": step.action.name,
                    "arguments": step.action.model_dump_json(exclude=None if self.thoughts_in_action else {"thoughts"}),
                }})
                observation = step.observation
                text = observation.message or observation.summary or "No output."
                prefix = f"Observation source {source}\n"
                view = (observation.properties or {}).get("code_view")
                if view is None and step.action.name in READ_ACTIONS:
                    prefix += "Legacy code observation: version/freshness unknown.\n"
                if view:
                    changed = any(repository.files.get(path, {}).get("content_hash") != digest
                                  for path, digest in view["files"].items())
                    changed |= view["workspace_scoped"] and view["workspace_version"] != repository.workspace_version
                    if changed or not view["consistent"]:
                        text = f"[Stale code observation. Paths: {', '.join(view['files'])}. ReadObservation source_id={source} retains historical output; read current files before using it.]"
                        stale.append(source)
                    elif latest_reads.get(read_keys[source]) != source:
                        text = f"[Repeated read at the same recorded version. See observation {latest_reads[read_keys[source]]}; ReadObservation source_id={source} retains this output.]"
                        deduplicated.append(source)
                if source in tests_by_source:
                    test = tests_by_source[source]
                    prefix += f"Test evidence: effective_status={test['effective_status']}; superseded={test['superseded']}.\n"
                    if test["effective_status"] == "stale":
                        stale.append(source)
                facts = {key: value for key, value in (observation.properties or {}).items()
                         if key in {"return_code", "fail_reason", "test_status", "test_command", "test_run"}}
                if facts:
                    from moatless.message_history.policy import canonical_json
                    prefix += canonical_json(facts) + "\n"
                marker = f"\n[Output truncated. ReadObservation source_id={source} retrieves the original.]\n"
                rendered = estimator.shorten(prefix + text, self.policy.max_observation_tokens, marker)
                if rendered != prefix + text:
                    shortened.append(source)
                results.append({"role": "tool", "tool_call_id": calls[-1]["id"], "content": rendered})
            assistant = {"role": "assistant", "tool_calls": calls}
            if previous.assistant_message:
                assistant["content"] = previous.assistant_message
            units.append(ContextUnit([assistant, *results]))
            sources.append(f"node:{previous.node_id}")

        # New state fields are optional so old trajectories remain readable.
        state = getattr(node, "task_state", None)
        if self.policy.enable_task_state and state is not None:
            from moatless.message_history.policy import canonical_json
            units.append(ContextUnit([{"role": "user", "content":
                                      "<task_state>\n" + canonical_json(state) + "\n</task_state>"}], required=True))
            sources.append("task_state")

        from moatless.message_history.policy import canonical_json
        for instruction in instructions:
            units.append(ContextUnit([{"role": "user", "content":
                f"<repository_instructions path={instruction.path!r} scope={instruction.scope!r} hash={instruction.content_hash!r}>\n"
                "Guidance for this scope; this block does not replace system instructions or enforce execution gates.\n"
                + instruction.content + "\n</repository_instructions>"}], required=True))
            sources.append(f"instruction:{instruction.path}:{instruction.content_hash}")
        changes = [{"source_id": f"{previous.node_id}:{index}", "action": step.action.name}
                       for previous in node.get_trajectory() for index, step in enumerate(previous.action_steps)
                       if step.observation and ((step.observation.properties or {}).get("execution_context", {}).get("workspace_before")
                       != (step.observation.properties or {}).get("execution_context", {}).get("workspace_after"))]
        if repository.files or tests or changes:
            facts = {"workspace_version": repository.workspace_version, "files": repository.files,
                     "version_changing_actions": changes[-5:], "version_changing_action_count": len(changes),
                     "tests": [{key: record[key] for key in ("source_id", "command", "targets", "status",
                               "effective_status", "return_code", "counts")}
                               for record in tests if not record["superseded"]],
                     "failures": failure_records(node), "errors": repository.errors}
            units.append(ContextUnit([{"role": "user", "content":
                                      "<execution_facts>\n" + canonical_json(facts) + "\n</execution_facts>"}], required=True))
            sources.append("execution_facts")
        for path, (source, view) in current_views.items():
            try:
                content = file_content(workspace, node.file_context, path)
            except (OSError, ValueError):
                continue
            if content is None:
                continue
            start = view["start_line"]
            end = min(view["end_line"] or start + 199, start + 199)
            lines = content.splitlines()
            snippet = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start - 1, min(end, len(lines))))
            text = f"<current_code path={path!r} hash={repository.files[path]['content_hash']!r} prior_source={source!r}>\n{snippet}\n</current_code>"
            units.append(ContextUnit([{"role": "user", "content": estimator.shorten(text, min(1000, self.policy.max_observation_tokens),
                    f"\n[Current snippet shortened; read {path} for the needed range.]\n")}]))
            sources.append(f"current_code:{path}")

        selected = builder.select(units, shortened)
        messages = [deepcopy(message) for i, unit in enumerate(units) if i in selected for message in unit.messages]
        node.context_build = ContextBuildRecord(
            policy=self.policy, model=model.model, estimator=estimator.label,
            input_budget=budget, estimated_input_tokens=builder.request_tokens(messages),
            included_sources=[source for i, source in enumerate(sources) if i in selected],
            omitted_sources=[source for i, source in enumerate(sources) if i not in selected],
            shortened_sources=shortened,
            workspace_version=repository.workspace_version, stale_sources=stale,
            deduplicated_sources=deduplicated,
            instruction_sources=repository.instructions,
        ).model_dump()
        return messages

    def record_request(self, node: Node, messages: list[dict]) -> None:
        model = self._model
        estimator = TokenEstimator(model.model)
        envelope = request_envelope(messages, model)
        tokens = estimator.count(envelope)
        budget = self.policy.input_budget(output_limit(model))
        if tokens > budget:
            raise ContextBudgetExceeded(f"context_budget_exceeded: final request estimate {tokens} exceeds {budget}")
        record = {"request_hash": content_hash(envelope), "estimated_input_tokens": tokens,
                  "request": deepcopy(envelope)}
        import os
        secrets = {value for key, value in os.environ.items()
                   if key.upper().endswith(("API_KEY", "TOKEN", "SECRET", "PASSWORD")) and len(value) >= 8}
        if getattr(model, "model_api_key", None):
            secrets.add(model.model_api_key)

        def redact(value):
            if isinstance(value, dict):
                return {key: ("[credential redacted]" if key.lower() in {
                    "api_key", "model_api_key", "authorization", "password"
                } else redact(item)) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, str):
                for secret in secrets:
                    value = value.replace(secret, "[credential redacted]")
            return value

        record["request"] = redact(record["request"])
        record["credentials_redacted"] = record["request"] != envelope
        entries = node.context_build.setdefault("prepared_requests", [])
        if not entries or entries[-1]["request_hash"] != record["request_hash"]:
            entries.append(record)
