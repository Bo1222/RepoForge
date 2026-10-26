"""Check exact ingestion inputs with the deployed tokenizer before embedding.

Export runs in the framework environment; count runs in the service environment.
Neither operation makes an embedding or LLM request or modifies source/indexes.
"""

import argparse
import hashlib
import json
from pathlib import Path


def export(repo, settings, output):
    from unittest.mock import patch
    from llama_index.core.schema import MetadataMode
    from repoforge.index.code_index import CodeIndex
    from repoforge.index.settings import IndexSettings
    from repoforge.repository import FileRepository

    captured = []
    repo = repo.resolve()

    def capture(pipeline, *, nodes, **kwargs):
        captured.extend({"id": node.node_id, "file": node.metadata.get("file_path"),
                         "text": node.get_content(metadata_mode=MetadataMode.EMBED)} for node in nodes)
        return nodes

    index = CodeIndex(file_repo=FileRepository(repo_path=str(repo)),
                      settings=IndexSettings(**settings))
    with patch("repoforge.index.code_index.IngestionPipeline.run", capture):
        index.run_ingestion(repo_path=str(repo), num_workers=1)
    output.write_text(json.dumps({"settings": settings, "nodes": captured}, ensure_ascii=False))


def count(chunks, model, output):
    from transformers import AutoTokenizer
    payload = json.loads(chunks.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    rows = []
    for node in payload["nodes"]:
        length = len(tokenizer(node["text"], truncation=False, add_special_tokens=True)["input_ids"])
        rows.append({"id": node["id"], "file": node["file"], "tokens": length,
                     "input_sha256": hashlib.sha256(node["text"].encode()).hexdigest()})
    report = {"settings": payload["settings"], "nodes": rows,
              "chunks_sha256": hashlib.sha256(chunks.read_bytes()).hexdigest(),
              "count": len(rows), "max_tokens": max((r["tokens"] for r in rows), default=0),
              "total_tokens": sum(r["tokens"] for r in rows),
              "overlength": [r for r in rows if r["tokens"] > 2048],
              "provider_calls": 0, "passed": bool(rows) and all(r["tokens"] <= 2048 for r in rows)}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"nodes", "overlength", "settings"}}))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "count"])
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        export(args.repo, json.loads(args.settings.read_text()), args.chunks)
    else:
        count(args.chunks, args.model, args.output)
