"""Audit the exact ingestion inputs with the deployed BGE tokenizer, without inference.

The tokenizer runs in the service environment on the host. Neither its dependencies
nor model files are added to generation workers. No dataset answers are inputs.
"""

import argparse
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from llama_index.core.schema import MetadataMode

from repoforge.evaluation.experiment import content_hash, sha, write_json
from repoforge.index.code_index import CodeIndex
from repoforge.index.settings import IndexSettings
from repoforge.repository import FileRepository


def audit(repo, settings, runtime, output):
    repo, runtime = Path(repo).resolve(), Path(runtime).resolve()
    captured = []

    class Captured(Exception):
        pass

    def capture(_pipeline, *, nodes, **kwargs):
        captured.extend(nodes)
        raise Captured()

    # Observe the production splitter/metadata path immediately before embedding.
    # Raising here prevents both inference and publication of a partial index.
    with patch("repoforge.index.code_index.IngestionPipeline.run", capture):
        try:
            index = CodeIndex(file_repo=FileRepository(repo_path=str(repo)), settings=settings)
            index.run_ingestion(repo_path=str(repo), num_workers=1)
        except Captured:
            pass
        else:
            raise RuntimeError("Ingestion did not reach the audited embedding boundary")
    texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in captured]
    code = """
import json,sys
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=False)
texts=json.load(sys.stdin)
lengths=[]
for start in range(0,len(texts),64):
    lengths.extend(map(len,tokenizer(texts[start:start+64],truncation=False,add_special_tokens=True)['input_ids']))
print(json.dumps(lengths))
"""
    process = subprocess.run([str(runtime / "venv/bin/python"), "-c", code, str(runtime / "model")],
                             input=json.dumps(texts), capture_output=True, text=True, timeout=300)
    if process.returncode:
        raise RuntimeError(f"Offline BGE token audit failed: {process.stderr}")
    lengths = json.loads(process.stdout)
    if len(lengths) != len(captured) or not lengths:
        raise ValueError("Tokenizer did not account for every ingestion node")
    rows = [{"node_id": node.node_id, "file_path": node.metadata.get("file_path"),
             "input_sha256": content_hash(text), "actual_bge_tokens": length}
            for node, text, length in zip(captured, texts, lengths)]
    result = {"settings": settings.model_dump(mode="json"), "repo": str(repo),
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "source_diff_sha256": content_hash(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo, text=True)),
              "model_manifest_sha256": sha(runtime / "model-manifest.json"),
              "service_dependency_lock_sha256": sha(Path(__file__).parent / "local_bge/requirements.lock"),
              "node_count": len(rows), "actual_bge_tokens": sum(lengths), "max_actual_bge_tokens": max(lengths),
              "overlength_count": sum(length > 2048 for length in lengths),
              "passed": max(lengths) <= 2048, "embedding_calls": 0, "llm_calls": 0, "nodes": rows}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("experiments/runtime/local_bge"))
    parser.add_argument("--settings", type=Path)
    args = parser.parse_args()
    settings = IndexSettings(**json.loads(args.settings.read_text())) if args.settings else IndexSettings()
    if settings.embed_provider != "local":
        raise ValueError("This audit applies only to local BGE")
    result = audit(args.repo, settings, args.runtime, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "nodes"}, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
