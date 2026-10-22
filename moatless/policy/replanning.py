"""Durable replan gate and chronological repair attempts, independent of targets."""

from moatless.message_history.policy import content_hash
from moatless.policy.obligations import failure_family
from moatless.utils.file import is_test


def failure_detail(record):
    """Keep the runner command/exit code when no test identity was parsed."""
    command = record.get("commands") or record.get("command")
    return (f"command={command!r} return_code={record.get('return_code')} "
            f"counts={record.get('counts')} targets={record.get('targets')}")


def mentioned_files(record):
    """Existing repository paths named by the runner's own failure text.

    Pytest reports a version/config failure in ``pyproject.toml`` or a
    ``_version.py`` path; that file is legitimate new evidence even though it
    is neither a product file nor the failing test module.
    """
    import re
    text = "\n".join(str(item.get("failure_output") or "") for item in record.get("results", []))
    text += "\n" + str(record.get("commands") or record.get("command") or "")
    paths = set()
    for raw in re.findall(r"(?:/testbed/)?([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|toml|cfg|ini|json|txt))", text):
        path = raw.lstrip("./")
        if path and ".." not in path.split("/"):
            paths.add(path)
    return paths


def update_no_gain(saved, node, index, last_props, *, stagnation_limit, repeat_limit, plan=None,
                   open_gate=True):
    """Track no-gain reads/searches and optionally open a ReplanRepair gate.

    O2 counted no-progress through ``extract_state``.  The a6 assessment path
    only counted test-failure families, so repeated searches/views could run
    until max_iterations.  This keeps the same progress definition (new patch,
    new code view, new test receipt, new plan/replan/syntax evidence) and opens
    one bounded gate for the no_progress family.  A7 calls it with
    ``open_gate=False``: it persists only progress/repeat facts and lets the
    single O2 policy own the gate decision.
    """
    state = saved.setdefault("no_gain", {
        "count": 0, "repeat_count": 0, "last_signature": None,
        "seen_views": [], "seen_tests": [], "last_patch": None, "last_plan": None,
    })
    progress = False
    context = node.file_context
    patch = content_hash(context.generate_git_patch(ignore_tests=True)) if context else ""
    if state.get("last_patch") is not None and patch != state["last_patch"]:
        progress = True
    state["last_patch"] = patch

    view = last_props.get("code_view")
    if view and view.get("consistent") is True:
        fingerprint = content_hash(view)
        if fingerprint not in state["seen_views"]:
            progress = True
            state["seen_views"].append(fingerprint)
            state["seen_views"] = state["seen_views"][-50:]

    run = last_props.get("test_run")
    if run:
        # A new node id is not progress: a repeated identical test run has the
        # same source/status/counts/product/input identity.
        fingerprint = content_hash({key: run.get(key) for key in (
            "test_source", "targets", "status", "counts", "return_code",
            "input_identity", "test_bundle_hash", "product_source", "check_id")})
        if fingerprint not in state["seen_tests"]:
            progress = True
            state["seen_tests"].append(fingerprint)
            state["seen_tests"] = state["seen_tests"][-50:]

    if plan is not None and state.get("last_plan") != plan.revision:
        progress = True
        state["last_plan"] = plan.revision
    if (last_props.get("replan_validation") or {}).get("accepted"):
        progress = True
    if last_props.get("syntax_checks"):
        progress = True

    step = node.action_steps[index] if 0 <= index < len(node.action_steps) else None
    signature = content_hash(step.action.model_dump(exclude={"thoughts"})) if step else None
    if signature is not None and signature == state.get("last_signature"):
        state["repeat_count"] = state.get("repeat_count", 0) + 1
    else:
        state["repeat_count"] = 0
    state["last_signature"] = signature

    if progress:
        state["count"] = 0
        state["repeat_count"] = 0
    else:
        state["count"] = state.get("count", 0) + 1

    gate = (state["count"] >= stagnation_limit
            or state.get("repeat_count", 0) >= repeat_limit)
    if not gate or not open_gate:
        return "", False

    family = "no_progress"
    replans = saved.setdefault("replans", {})
    if family in replans:
        if replans[family].get("status") == "accepted":
            return "", False
        return (f"No progress for {state['count']} actions; stop searching. "
                f"Submit ReplanRepair(family_id='{family}') with evidence read after "
                f"{node.node_id}:{index}."), False

    changed = [file.file_path for file in context.files
               if file.patch and not is_test(file.file_path)] if context else []
    targets = [target for check in (plan.checks if plan else []) for target in check.targets]
    evidence = evidence_rows(node)
    replans[family] = {
        "status": "required",
        "trigger": f"{node.node_id}:{index}",
        "rejections": 0,
        "read_attempts": 0,
        "related_files": sorted(set(changed + targets)),
        "known_evidence": [row["fingerprint"] for row in evidence],
        "no_progress": True,
    }
    return (f"No progress for {state['count']} actions; stop searching. "
            f"Submit ReplanRepair(family_id='{family}') with evidence read after "
            f"{node.node_id}:{index}."), True


def evidence_rows(node):
    rows = []
    for previous in node.get_trajectory():
        for index, step in enumerate(previous.action_steps):
            if step.observation is None:
                continue
            props = step.observation.properties or {}
            if props.get("execution_status") == "skipped":
                continue
            fact = None
            if props.get("code_view") and not props.get("fail_reason") and props["code_view"].get("consistent") is True:
                view = props["code_view"]
                fact = {
                    "kind": "read",
                    "files": view.get("files"),
                    "ranges": view.get("ranges_by_file"),
                    "start": view.get("start_line"),
                    "end": view.get("end_line"),
                }
            elif props.get("test_run"):
                run = props["test_run"]
                fact = {
                    "kind": "test",
                    "inputs": run.get("input_identity"),
                    "status": run.get("status"),
                    "results": run.get("results"),
                    "targets": run.get("targets", []),
                }
            if fact:
                paths = list(fact.get("files") or fact.get("targets") or [])
                rows.append(
                    {"source": f"{previous.node_id}:{index}", "fingerprint": content_hash(fact), "paths": paths}
                )
    return rows


def update_repair_control(node, records, saved, max_repairs, obligations=None):
    from moatless.verification_inputs import test_identity

    families = saved.setdefault("failure_families", {})
    case_families = saved.setdefault("case_families", {})
    replans = saved.setdefault("replans", {})
    seen = saved.setdefault("counted_failure_receipts", [])
    evidence = evidence_rows(node)
    relevant_sources = {item["source_id"] for item in obligations} if obligations is not None else None
    relevant_cases = {item["case"] for item in obligations if item.get("case")} if obligations is not None else set()
    sources = [
        f"{n.node_id}:{i}" for n in node.get_trajectory() for i, step in enumerate(n.action_steps) if step.observation
    ]
    for record in records:
        if (
            record["source_id"] in seen
            or record.get("product_source") != "current"
            or record.get("status") not in {"failed", "error"}
            or record.get("timed_out")
        ):
            continue
        if (
            relevant_sources is not None
            and record["source_id"] not in relevant_sources
            and not any(test_identity(item) in relevant_cases for item in record.get("results", []))
        ):
            continue
        seen.append(record["source_id"])
        failing = [
            item for item in record.get("results", []) if str(item.get("status", "")).lower() in {"failed", "error"}
        ]
        if not failing:
            failing = [{"failure_output": failure_detail(record)}]
        keys = set()
        for item in failing:
            case = test_identity(item)
            # Preserve the first causal family of an unresolved behavior even
            # when a later patch adds product files or changes the exception.
            family = (
                case_families.setdefault(case, failure_family(record, item))
                if case
                else content_hash({"unparsed": record["status"]})
            )
            keys.add(family)
        for family in keys:
            counter = families.setdefault(family, {"last_patch": None, "repair_attempts": 0, "last_source": None})
            counter["related_files"] = sorted(
                set(counter.get("related_files", []))
                | set(record.get("product_files", []) + record.get("targets", []))
                | mentioned_files(record)
            )
            patch = record.get("product_patch_hash")
            # A -> B -> A is two repair trials, not two unique hashes.
            if counter["last_patch"] is not None and patch != counter["last_patch"]:
                counter["repair_attempts"] += 1
            counter.update(last_patch=patch, last_source=record["source_id"])
            if counter["repair_attempts"] >= max_repairs:
                return "repair_exhausted"
            if counter["repair_attempts"] >= 2 and family not in replans:
                cutoff = sources.index(record["source_id"])
                replans[family] = {
                    "status": "required",
                    "trigger": record["source_id"],
                    "rejections": 0,
                    "related_files": counter["related_files"],
                    "known_evidence": [r["fingerprint"] for r in evidence if sources.index(r["source"]) <= cutoff],
                }
    return None


def require_replan(saved, node, family, trigger=None):
    """Open (or reuse) the one evidence gate for an O2 replan decision.

    A7's single controller may decide to replan for no-progress or a failure
    family.  The evidence requirements stay in A6: product writes are blocked
    until ReplanRepair cites new execution/read evidence for this family.
    """
    replans = saved.setdefault("replans", {})
    gate = replans.get(family)
    if gate and gate.get("status") in {"required", "accepted"}:
        return gate
    sources = [
        f"{n.node_id}:{i}" for n in node.get_trajectory() for i, step in enumerate(n.action_steps) if step.observation
    ]
    context = node.file_context
    changed = [file.file_path for file in context.files
               if file.patch and not is_test(file.file_path)] if context else []
    from moatless.policy.repair import registered_plan
    plan = registered_plan(node)
    targets = [target for check in (plan.checks if plan else []) for target in check.targets]
    gate = {
        "status": "required",
        "trigger": trigger or (sources[-1] if sources else None),
        "rejections": 0,
        "read_attempts": 0,
        "related_files": sorted(set(changed + targets)),
        "known_evidence": [row["fingerprint"] for row in evidence_rows(node)],
        "no_progress": family == "no_progress",
    }
    replans[family] = gate
    return gate


def accept_replan(node, proposal, saved):
    family = proposal.get("family_id")
    gate = saved.get("replans", {}).get(family)
    if not gate or gate["status"] != "required":
        return False, "No pending replan for this failure family"
    sources = [
        f"{n.node_id}:{i}" for n in node.get_trajectory() for i, step in enumerate(n.action_steps) if step.observation
    ]
    trigger = sources.index(gate["trigger"]) if gate.get("trigger") in sources else -1
    rows = {r["source"]: r for r in evidence_rows(node)}
    cited = proposal.get("evidence_sources", [])

    def resolve(ref):
        if ref in rows and sources.index(ref) > trigger:
            return ref
        # Models often cite a repository path instead of a source ID.  Map it
        # to the latest post-trigger read/search row that actually contained
        # that path; no new evidence is invented.
        normalized = str(ref).lstrip("./")
        if normalized.startswith("/testbed/"):
            normalized = normalized[len("/testbed/"):]
        matches = [
            source for source, row in rows.items()
            if sources.index(source) > trigger
            and any(path == normalized or path.endswith("/" + normalized)
                    or normalized.endswith("/" + path) for path in row.get("paths", []))
        ]
        return max(matches, key=lambda source: sources.index(source)) if matches else None

    # A proposal may cite background reads together with new evidence.  Only
    # the new, post-trigger rows are required; an older cited row must not
    # invalidate the whole proposal.
    fresh = list(dict.fromkeys(
        resolved for ref in cited for resolved in [resolve(ref)] if resolved))
    if not fresh:
        return False, "Read or execute new evidence after the triggering failure and cite its source IDs"
    if not any(rows[ref]["fingerprint"] not in gate["known_evidence"] for ref in fresh):
        return False, "Repeated output at the same inputs is not new evidence"
    if gate.get("related_files") and not any(
        set(rows[ref]["paths"]) & set(gate["related_files"]) and rows[ref]["fingerprint"] not in gate["known_evidence"]
        for ref in fresh
    ):
        return False, "New evidence must concern this failure's product files or failing test modules"
    gate.update(
        status="accepted", evidence_sources=fresh, hypothesis=proposal["hypothesis"], next_steps=proposal["next_steps"]
    )
    return True, "Evidence accepted; one final repair trial is available"
