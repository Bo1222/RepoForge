import io
import json
from pathlib import Path

import pytest

from moatless.evaluation import bge_identity


def test_health_rejects_changed_revision_and_missing_fields(monkeypatch):
    response = dict(bge_identity.SPEC)

    class Opener:
        def open(self, url, timeout):
            assert url == "http://127.0.0.1:8787/health"
            assert timeout == 5
            return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(bge_identity.urllib.request, "build_opener", lambda *args: Opener())
    assert bge_identity.check_health("http://127.0.0.1:8787/v1/") == response
    response["revision"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        bge_identity.check_health("http://127.0.0.1:8787/v1")
    del response["revision"]
    with pytest.raises(ValueError, match="identity"):
        bge_identity.check_health("http://127.0.0.1:8787/v1")


@pytest.mark.parametrize("key", ["specification", "files", "dependencies"])
def test_resume_rejects_deployment_drift_but_accepts_usage_changes(monkeypatch, key):
    frozen = {"specification": bge_identity.SPEC, "files": {"model": "hash"},
              "dependencies": {"torch": "pinned"}, "health_at_freeze": {"request_count": 0}}
    current = {**frozen, "health_at_freeze": {"request_count": 100}}
    monkeypatch.setattr(bge_identity, "capture_identity", lambda *args: current)
    manifest = {"embedding_service": frozen, "embedding_settings": {"embed_base_url": "loopback"}}
    assert bge_identity.validate_identity(Path("."), manifest) == {"request_count": 100}
    current[key] = {"changed": True}
    with pytest.raises(ValueError, match="changed"):
        bge_identity.validate_identity(Path("."), manifest)


def test_old_local_manifest_cannot_authorize_new_execution():
    with pytest.raises(ValueError, match="new manifest"):
        bge_identity.validate_identity(Path("."), {})
