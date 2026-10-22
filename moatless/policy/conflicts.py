"""Failing original expectations are explained, never rewritten as success."""

import ast


def validate_conflict(node, workspace, claim, records=None):
    from moatless.policy.repair import registered_plan
    from moatless.execution_memory import test_records
    from moatless.verification_inputs import test_identity, content_bytes

    plan = registered_plan(node)
    criterion = next((item for item in plan.criteria if item.id == claim.get("criterion_id")), None) if plan else None
    if not criterion or criterion.kind != "issue_behavior" or criterion.public_quote != claim.get("public_quote"):
        return "Conflict must cite an exact registered public issue behavior quote"
    record = next(
        (
            r
            for r in (records if records is not None else test_records(node, workspace))
            if r["source_id"] == claim.get("candidate_source")
        ),
        None,
    )
    if not record or record.get("effective_status") != "failed" or record.get("test_source") != "base_public":
        return "Conflict requires a fresh failing original-public execution"
    comparison = record.get("comparison") or {}
    if comparison.get("status") != "known" or claim.get("failing_case") not in (
        comparison.get("introduced_failures") or []
    ):
        return "Conflict requires a comparable original pass -> current failure, with full test identity"
    result = next((item for item in record.get("results", []) if test_identity(item) == claim["failing_case"]), None)
    if not result:
        return "Failing test identity not found"
    source = content_bytes(workspace.repository, node.file_context, result["file_path"], base=True)
    try:
        text = source.decode() if source is not None else ""
        assertions = {
            ast.get_source_segment(text, item) for item in ast.walk(ast.parse(text)) if isinstance(item, ast.Assert)
        }
    except (ValueError, SyntaxError, UnicodeError):
        return "Original assertion cannot be verified"
    if claim.get("original_assertion") not in assertions:
        return "Cite the exact original assert statement; a model explanation alone cannot exempt a failure"
    return None


def explained_conflicts(node, workspace, records, satisfied):
    claims = []
    for previous in node.get_trajectory():
        for step in previous.action_steps:
            claim = (step.observation.properties or {}).get("expectation_conflict") if step.observation else None
            if (
                claim
                and claim.get("criterion_id") in satisfied
                and validate_conflict(node, workspace, claim, records) is None
            ):
                claims.append(claim)
    return claims
