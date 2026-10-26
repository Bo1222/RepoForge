import json
import sys
from types import SimpleNamespace

import pytest

from scripts.bge_token_audit import count


@pytest.mark.parametrize("length,passed", [(2048, True), (2049, False)])
def test_actual_token_limit_keeps_full_input_and_failure_evidence(tmp_path, monkeypatch, length, passed):
    seen = []

    def tokenize(text, **kwargs):
        seen.append((text, kwargs))
        return {"input_ids": list(range(length))}

    tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenize)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    chunks, report = tmp_path / "chunks.json", tmp_path / "report.json"
    chunks.write_text(json.dumps({"settings": {"provider": "local"},
                                 "nodes": [{"id": "1", "file": "public.py", "text": "full input"}]}))
    if passed:
        count(chunks, tmp_path, report)
    else:
        with pytest.raises(SystemExit) as exc:
            count(chunks, tmp_path, report)
        assert exc.value.code == 2
    result = json.loads(report.read_text())
    assert result["passed"] is passed
    assert result["max_tokens"] == length
    assert result["provider_calls"] == 0
    assert seen == [("full input", {"truncation": False, "add_special_tokens": True})]
    assert bool(result["overlength"]) is (not passed)
