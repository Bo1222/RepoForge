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
from moatless.evaluation.verification import snapshot as verification_snapshot
from moatless.execution_memory import READ_ACTIONS, failure_records, test_records
from moatless.repository_context import file_content, refresh_context


class ManagedMessageHistoryGenerator(MessageHistoryGenerator):
    policy: ContextPolicy
    _model: object = PrivateAttr(default=None)
    _optional_groups: list = PrivateAttr(default_factory=list)

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
        product_patch = ""
        if node.file_context is not None:
            product_patch = node.file_context.generate_git_patch(ignore_tests=True)
        verification = verification_snapshot(
            content_hash(product_patch) if product_patch.strip() else None, None,
            [record for record in tests if not record.get("superseded")],
        )
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
                if step.origin == "policy":
                    prefix += f"Action scheduled by policy, not proposed by model; decision={step.policy_decision_id}.\n"
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
                    if test["effective_status"] in {"failed", "error"}:
                        failing = [line for line in (observation.message or "").splitlines()
                                   if "FAILED" in line or "ERROR" in line][:8]
                        if failing:
                            prefix += "Failing test evidence:\n" + "\n".join(failing) + "\n"
                    if test["effective_status"] == "stale":
                        stale.append(source)
                facts = {key: value for key, value in (observation.properties or {}).items()
                         if key in {"return_code", "fail_reason", "test_status", "test_command", "test_run"}}
                if "test_run" in facts:
                    # The full immutable manifest/timing/output stays on the
                    # receipt and is readable by source ID. Project only the
                    # current control facts into the model request.
                    receipt = tests_by_source.get(source, facts["test_run"])
                    fields = {"source_id", "status", "effective_status", "test_source", "product_source", "targets",
                              "return_code", "counts", "input_identity", "input_identity_after", "complete", "timed_out",
                              "product_patch_hash", "tested_patch_hash", "comparison", "public_baseline", "introduced_failures"}
                    facts["test_run"] = {key: value for key, value in receipt.items() if key in fields}
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

        from moatless.policy.repair import registered_plan
        plan = registered_plan(node)
        if plan:
            from moatless.message_history.policy import canonical_json
            units.append(ContextUnit([{"role": "user", "content": "<acceptance_plan>\n" + canonical_json({
                "revision": plan.revision, "criteria": [{"id": item.id, "kind": item.kind,
                "behavior": item.behavior[:300]} for item in plan.criteria],
                "mandatory_checks": [check.id for check in plan.checks]}) + "\n</acceptance_plan>"}], required=True))
            sources.append("acceptance_plan")
        # New state fields are optional so old trajectories remain readable.
        feedback = next((previous.policy_feedback for previous in reversed(node.get_trajectory())
                         if previous.policy_feedback), None)
        if feedback:
            units.append(ContextUnit([{"role": "user", "content": "<action_policy_feedback>\n" +
                feedback + "\n</action_policy_feedback>"}], required=True))
            sources.append("action_policy_feedback")
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
        from moatless.message_history.policy import canonical_json
        units.append(ContextUnit([{"role": "user", "content":
            "<verification>\n" + canonical_json({
                "state": verification.state.value,
                "tests": verification.tests[:5],
                "introduced_failures": verification.introduced_failures[:5],
                "reasons": verification.reasons,
            }) + "\n</verification>"}], required=True))
        sources.append("verification")
        if repository.files or tests or changes:
            facts = {"workspace_version": repository.workspace_version, "files": repository.files,
                     "version_changing_actions": changes[-5:], "version_changing_action_count": len(changes),
                     "tests": [{key: record[key] for key in ("source_id", "command", "targets", "status",
                               "effective_status", "return_code", "counts", "test_source", "product_source",
                               "input_identity", "complete", "timed_out", "check_id")}
                               for record in tests if not record["superseded"]][-8:],
                     "failures": failure_records(node)[-8:], "errors": repository.errors[-5:]}
            units.append(ContextUnit([{"role": "user", "content":
                                      "<execution_facts>\n" + canonical_json(facts) + "\n</execution_facts>"}], required=True))
            sources.append("execution_facts")
        recent_node_sources = set([source for source in sources
                                   if source.startswith("node:")][-self.policy.recent_interactions:])
        seen_current_code = set()
        current_code_tokens = 0
        for path, (source, view) in current_views.items():
            # A fresh read interaction already carries this path's current
            # content; emitting current_code again duplicates 45-49% of it.
            if f"node:{source.split(':')[0]}" in recent_node_sources and source not in stale and source not in deduplicated:
                continue
            content_hash_value = repository.files.get(path, {}).get("content_hash")
            key = (path, content_hash_value)
            if key in seen_current_code:
                continue
            seen_current_code.add(key)
            try:
                content = file_content(workspace, node.file_context, path)
            except (OSError, ValueError):
                continue
            if content is None:
                continue
            lines = content.splitlines()
            ranges = view.get("ranges_by_file", {}).get(path) or [[view["start_line"], view["end_line"]]]
            indices = []
            for start, end in ranges:
                start = max(1, start or 1)
                for index in range(start - 1, min(end or start + 199, len(lines))):
                    if index not in indices:
                        indices.append(index)
                    if len(indices) >= 200:
                        break
                if len(indices) >= 200:
                    break
            snippet = "\n".join(f"{i + 1}: {lines[i]}" for i in indices)
            text = f"<current_code path={path!r} hash={repository.files[path]['content_hash']!r} prior_source={source!r}>\n{snippet}\n</current_code>"
            rendered = estimator.shorten(text, min(1000, self.policy.max_observation_tokens),
                    f"\n[Current snippet shortened; read {path} for the needed range.]\n")
            cost = estimator.count(rendered)
            if current_code_tokens + cost > int(budget * 0.2) and current_code_tokens:
                continue
            current_code_tokens += cost
            units.append(ContextUnit([{"role": "user", "content": rendered}]))
            sources.append(f"current_code:{path}")

        selected = builder.select(units, shortened)
        if any(i not in selected for i in range(len(units))):
            # Reserve space, then build references from the final selection.
            units.append(ContextUnit([{"role": "user", "content": "<omitted_sources>" + " ref" * 128 + "</omitted_sources>"}], required=True))
            sources.append("omitted_sources")
            selected = builder.select(units, shortened)
            omitted = [source for i, source in enumerate(sources) if i not in selected]
            omitted_refs = []
            for source in omitted:
                if source.startswith("node:"):
                    node_id = int(source.split(":")[1])
                    previous = next(n for n in node.get_trajectory() if n.node_id == node_id)
                    omitted_refs.extend(f"{node_id}:{i}" for i, step in enumerate(previous.action_steps) if step.observation)
            catalog = "<omitted_sources>\n" + "\n".join(omitted_refs[:20]) + "\n</omitted_sources>"
            units[-1].messages[0]["content"] = catalog
        # Deduplicate against every selected fresh read, including older units
        # that fit after the recent groups. Ranges come from that same view.
        for path, (source, view) in current_views.items():
            interaction = f"node:{source.split(':')[0]}"
            if source not in stale and source not in deduplicated and source not in shortened:
                if interaction in sources and sources.index(interaction) in selected:
                    duplicate = f"current_code:{path}"
                    if duplicate in sources:
                        selected.discard(sources.index(duplicate))
        self._optional_groups = [{"source": sources[i], "hashes": [content_hash(message) for message in unit.messages]}
                                 for i, unit in enumerate(units) if i in selected and not unit.required]
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
            if node.context_build is not None:
                node.context_build.setdefault("rejected_envelopes", []).append(
                    {"request_hash": content_hash(envelope), "estimated_input_tokens": tokens, "input_budget": budget})
            for group in self._optional_groups:
                matching = [i for i, message in enumerate(messages) if content_hash(message) in group["hashes"]]
                if len(matching) != len(group["hashes"]):
                    continue
                messages[:] = [message for i, message in enumerate(messages) if i not in matching]
                if node.context_build is not None:
                    source = group["source"]
                    included = node.context_build["included_sources"]
                    if source in included:
                        included.remove(source)
                    if source not in node.context_build["omitted_sources"]:
                        node.context_build["omitted_sources"].append(source)
                    refs = []
                    for omitted in node.context_build["omitted_sources"]:
                        if omitted.startswith("node:"):
                            previous = next(n for n in node.get_trajectory() if n.node_id == int(omitted.split(":")[1]))
                            refs.extend(f"{previous.node_id}:{i}" for i, step in enumerate(previous.action_steps) if step.observation)
                    for message in messages:
                        if message.get("role") == "user" and str(message.get("content", "")).startswith("<omitted_sources>"):
                            message["content"] = "<omitted_sources>\n" + "\n".join(refs[:20]) + "\n</omitted_sources>"
                envelope = request_envelope(messages, model)
                tokens = estimator.count(envelope)
                if tokens <= budget:
                    break
        if tokens > budget:
            raise ContextBudgetExceeded(f"context_budget_exceeded: final request estimate {tokens} exceeds {budget}")
        if node.context_build is not None:
            node.context_build["estimated_input_tokens"] = tokens
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
