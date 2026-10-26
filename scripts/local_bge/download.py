"""Download one pinned model weight format plus Sentence Transformers configs."""

import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_url, get_hf_file_metadata

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FILES = [
    "pytorch_model.bin",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--http-ranges", action="store_true")
    args = parser.parse_args()
    target = Path("experiments/runtime/local_bge/model").resolve()
    if not args.verify_only:
        selected = [name for name in FILES if name != "pytorch_model.bin"] if args.http_ranges else FILES
        snapshot_download("BAAI/bge-m3", revision=REVISION, local_dir=target, allow_patterns=selected)
        if args.http_ranges:
            from scripts.local_bge.range_download import download

            url = hf_hub_url("BAAI/bge-m3", "pytorch_model.bin", revision=REVISION)
            metadata = get_hf_file_metadata(url)
            if metadata.commit_hash != REVISION:
                raise RuntimeError("Unexpected official revision")
            download(url, target / "pytorch_model.bin", metadata.size, metadata.etag)
    missing = [name for name in FILES if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing model files: {missing}")
    rows = {}
    for name in FILES:
        path = target / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        rows[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    manifest = {
        "model": "BAAI/bge-m3",
        "revision": REVISION,
        "files": rows,
        "total_bytes": sum(row["bytes"] for row in rows.values()),
    }
    (target.parent / "model-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"revision": REVISION, "model_bytes": manifest["total_bytes"]}))


if __name__ == "__main__":
    main()
