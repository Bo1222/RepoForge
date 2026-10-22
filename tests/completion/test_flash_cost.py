from unittest.mock import patch

import pytest

from moatless.completion.stats import Usage


def test_flash_fixed_price_counts_cache_once_and_overrides_unknown_sdk_price():
    with patch("litellm.cost_calculator.completion_cost", return_value=0):
        usage = Usage.from_completion_response({"usage": {
            "prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 400,
        }}, "openai/deepseek-flash")
    assert usage.completion_cost == pytest.approx((600 * .30 + 400 * .006 + 100 * 1.20) / 1_000_000)


@pytest.mark.asyncio
async def test_flash_never_falls_back_to_openai_credential(monkeypatch):
    from unittest.mock import AsyncMock
    from moatless.actions.respond import RespondArgs
    from moatless.completion.tool_call import ToolCallCompletionModel
    from moatless.exceptions import CompletionRuntimeError
    sdk = AsyncMock()
    monkeypatch.setattr("litellm.acompletion", sdk)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-other-provider-key")
    model = ToolCallCompletionModel(model="openai/deepseek-flash", model_base_url="https://api.deepseek.com")
    model.initialize([RespondArgs], "fixed")
    with pytest.raises(CompletionRuntimeError, match="DEEPSEEK_API_KEY"):
        await model.create_completion([{"role": "user", "content": "hello"}])
    assert not sdk.called
