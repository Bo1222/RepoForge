import httpx
import pytest

from scripts.t4_cli_probe import model_metadata


def test_metadata_uses_sdk_transport_and_validates_canonical_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-metadata")
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, request=httpx.Request("GET", url), json={"data": [{
            "id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash", "context_window": 1048576}]})
    monkeypatch.setattr("scripts.t4_cli_probe.httpx.get", get)
    assert model_metadata()["id"] == "deepseek-flash"
    assert calls[0][0] == "https://api.deepseek.com/models"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer offline-metadata"
    assert calls[0][1]["timeout"] == 30
    assert calls[0][1].get("verify", True) is True


def test_changed_model_metadata_requires_new_configuration(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-metadata")
    monkeypatch.setattr("scripts.t4_cli_probe.httpx.get", lambda url, **kwargs: httpx.Response(
        200, request=httpx.Request("GET", url), json={"data": [{
            "id": "deepseek-flash", "name": "different-provider-model", "context_window": 1048576}]}))
    with pytest.raises(ValueError, match="freeze a new configuration"):
        model_metadata()
