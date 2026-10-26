from llama_index.core.schema import Document

from repoforge.index.code_node import CodeNode
from repoforge.index.epic_split import EpicSplitter
from repoforge.index.settings import IndexSettings


def test_local_content_preservation_keeps_legacy_provider_defaults():
    assert IndexSettings().preserve_chunk_content
    assert not IndexSettings(provider="jina").preserve_chunk_content
    assert not IndexSettings(provider="openai").preserve_chunk_content


def test_unicode_fragments_rejoin_exactly_and_keep_span_ids():
    splitter = EpicSplitter(repo_path="/repo", preserve_content=True, hard_token_limit=30)
    text = ("中文标识 = '🙂'\n" * 30) + "FINAL_SENTINEL"
    node = CodeNode(id_="source", text=text, metadata={"file_path": "public.py", "span_ids": ["function"]})
    parts = splitter._split_preserved_node(node)
    assert len(parts) > 1
    assert "".join(part.text for part in parts) == text
    assert len({part.id_ for part in parts}) == len(parts)
    assert all(splitter._count_tokens(part.text) <= 30 for part in parts)
    assert all(part.metadata["span_ids"] == ["function"] for part in parts)


def test_generated_public_code_and_long_statement_are_not_dropped_or_cut():
    content = "# generated public example\ndef example():\n    value = [" + ",".join(str(i) for i in range(200)) + "]\n    return 'FINAL_SENTINEL'\n"
    document = Document(id_="/repo/public.py", text=content, metadata={"file_path": "public.py"})
    legacy = EpicSplitter(repo_path="/repo", hard_token_limit=80)
    assert legacy.get_nodes_from_documents([document]) == []
    splitter = EpicSplitter(repo_path="/repo", preserve_content=True, hard_token_limit=80)
    parts = splitter.get_nodes_from_documents([document])
    assert parts
    joined = "".join(part.text for part in parts)
    assert "FINAL_SENTINEL" in joined
    assert "199" in joined
    assert all(splitter._count_tokens(part.text) <= 80 for part in parts)


def test_parse_failure_preserves_public_input(monkeypatch):
    splitter = EpicSplitter(repo_path="/repo", preserve_content=True, hard_token_limit=30)
    def fail(*args, **kwargs):
        raise ValueError("fixture parse failure")
    monkeypatch.setattr(type(splitter.parser), "parse", fail)
    content = "unparseable public text\n" * 20
    document = Document(id_="/repo/public.py", text=content, metadata={"file_path": "public.py"})
    parts = splitter.get_nodes_from_documents([document])
    assert "".join(part.text for part in parts) == content
