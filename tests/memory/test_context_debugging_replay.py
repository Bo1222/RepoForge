import pytest

from scripts.context_memory_smoke import replay


@pytest.mark.asyncio
async def test_wrong_hypothesis_failed_fix_then_success_across_both_entrypoints(monkeypatch):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    results = await replay()
    for entrypoint in ("batch", "cli"):
        result = results[entrypoint]
        assert result["completion_status"]["status"] == "verified"
        assert result["completion_status"]["official_harness"] == "not_run"
        assert result["provider_calls"] == 0 and result["model_fixture_calls"] == 8
        assert [record["status"] for record in result["tests"]] == ["failed", "passed"]
        assert result["tests"][0]["effective_status"] == "stale"
        assert result["tests"][0]["superseded"]
        state = result["nodes"][-1]["task_state"]
        assert state["hypotheses"][0]["status"] == "rejected"
        assert state["hypotheses"][0]["evidence"]
        assert result["failures"] and "return a + b" in result["patch"]
    assert results["cli"]["reload_verified"]
