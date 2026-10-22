"""Immutable code indexes shared by repository snapshot and embedding pipeline.

This host-side catalog never builds embeddings. Workers receive only one index,
not the catalog or other repositories. Experiment results retain their own usage.
"""

import ast
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile

from filelock import FileLock

from moatless.evaluation.experiment import content_hash, now, read_json, sha, write_json
from moatless.evaluation.experiment_inputs import embedding_service_identity


CORE_FILES = (
    "vector_index.faiss", "vector_index.json", "docstore.json", "settings.json",
    "inverted_indexes.json", "blocks_by_class_name.json", "blocks_by_function_name.json",
)
EMPTY_PATCH = hashlib.sha256(b"").hexdigest()
PIPELINE_FILES = (
    "moatless/index/code_index.py", "moatless/index/epic_split.py",
    "moatless/index/settings.py", "moatless/index/simple_faiss.py",
    "moatless/index/local.py", "moatless/index/embed_model.py", "moatless/utils/tokenizer.py",
)


def builder_fingerprint(source):
    """Ignore unrelated CLI/grader changes in the same producer module."""
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_index")
    return hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()


def index_profile(manifest, *, builder=None, setup_patch_sha256=EMPTY_PATCH):
    files = manifest["source"]["files"]
    pipeline = {name: files[name] for name in PIPELINE_FILES}
    compatibility = manifest.get("native", {}).get("index_store", {}).get("query_only_compatibility")
    if compatibility:
        name = "moatless/index/code_index.py"
        if (compatibility.get("path") != name or compatibility.get("current_sha256") != files[name]
                or not compatibility.get("non_query_ast_sha256")):
            raise ValueError("Invalid query-only index compatibility identity")
        # The preparation driver verifies both archived ASTs after removing only
        # semantic_search. Embeddings/chunking are unchanged, provenance is not.
        pipeline[name] = compatibility["original_sha256"]
    pipeline.update({name: value for name, value in files.items()
                     if name.startswith("moatless/codeblocks/")})
    identity = embedding_service_identity(manifest)
    if manifest["embedding_settings"]["embed_provider"] != "local" or identity is None:
        raise ValueError("Shared local indexes require a frozen embedding deployment identity")
    builder = builder or manifest["native"]["index_store"]["builder_sha256"]
    return {"version": 1, "settings": manifest["embedding_settings"],
            "embedding_service_identity": identity, "pipeline_files": pipeline,
            "builder_sha256": builder, "setup_patch_sha256": setup_patch_sha256}


def checked_file(directory, relative):
    path = directory / relative
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Index file is missing or escaped its directory: " + relative)
    return path


class RepositoryIndexStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    @staticmethod
    def profile_id(profile):
        return "bge-m3-" + content_hash(profile)[:16]

    def directory(self, public, profile):
        repo, commit = public["repo"], public["base_commit"]
        parts = repo.split("/")
        if (len(parts) != 2 or any(not re.fullmatch(r"[\w.-]+", p) or p in (".", "..") for p in parts)
                or not re.fullmatch(r"[0-9a-f]{40}", commit)):
            raise ValueError("Expected owner/repository and full lowercase Git commit")
        directory = self.root / "repositories" / parts[0] / parts[1] / commit / self.profile_id(profile)
        if not directory.resolve().is_relative_to(self.root):
            raise ValueError("Shared index path escaped its store")
        return directory

    def find(self, public, profile):
        directory = self.directory(public, profile)
        if not directory.exists():
            return None
        index = directory / "index"
        self.verify(index, public=public, profile=profile)
        return index

    def verify(self, index, *, public=None, profile=None):
        index = Path(index)
        receipt = read_json(checked_file(index, "index-receipt.json"))
        payload = {key: value for key, value in receipt.items() if key != "receipt_hash"}
        if receipt.get("format") != "repository-index-v1" or content_hash(payload) != receipt.get("receipt_hash"):
            raise ValueError("Shared index receipt changed")
        snapshot = receipt["repository_snapshot"]
        expected_profile = receipt["profile"] if profile is None else profile
        expected_public = snapshot if public is None else public
        if (snapshot != {k: expected_public[k] for k in ("repo", "base_commit")}
                or receipt["profile"] != expected_profile
                or self.directory(snapshot, expected_profile).resolve() != index.parent.resolve()):
            raise ValueError("Shared index source, pipeline or configuration differs")
        if set(receipt["files"]) != set(CORE_FILES):
            raise ValueError("Shared index omits required data files")
        for name, expected in receipt["files"].items():
            if sha(checked_file(index, name)) != expected:
                raise ValueError("Shared index file changed: " + name)
        if content_hash(receipt["files"]) != receipt["digest"]:
            raise ValueError("Shared index digest changed")
        if read_json(index / "settings.json") != expected_profile["settings"]:
            raise ValueError("Stored embedding settings differ from profile")
        return receipt

    def publish(self, source, public, profile, *, provenance):
        """Verify a completed producer receipt before copying any index data."""
        source = Path(source).resolve()
        receipt = read_json(checked_file(source, "index-receipt.json"))
        if ({k: receipt["instance"][k] for k in ("repo", "base_commit")}
                != {k: public[k] for k in ("repo", "base_commit")}
                or receipt["settings"] != profile["settings"]
                or receipt.get("embedding_service_identity") != profile["embedding_service_identity"]):
            raise ValueError("Producer index source/settings/service differs from requested profile")
        if content_hash(receipt["files"]) != receipt["digest"]:
            raise ValueError("Producer receipt digest changed")
        for name, expected in receipt["files"].items():
            if sha(checked_file(source, name)) != expected:
                raise ValueError("Producer index file changed: " + name)
        core = {name: sha(checked_file(source, name)) for name in CORE_FILES}
        directory = self.directory(public, profile)
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.root / ".publish.lock")):
            existing = self.find(public, profile)
            if existing is not None:
                if self.verify(existing)["files"] != core:
                    raise ValueError("Different index data already exists for this immutable snapshot")
            else:
                directory.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=".publishing-", dir=directory.parent))
                try:
                    index = temporary / "index"
                    index.mkdir()
                    for name in CORE_FILES:
                        shutil.copyfile(checked_file(source, name), index / name)
                        if sha(index / name) != core[name]:
                            raise ValueError("Index changed while copying")
                    metadata = {"format": "repository-index-v1", "created": now(),
                                "repository_snapshot": {k: public[k] for k in ("repo", "base_commit")},
                                "profile": profile, "files": core, "digest": content_hash(core)}
                    metadata["receipt_hash"] = content_hash(metadata)
                    write_json(index / "index-receipt.json", metadata)
                    os.rename(temporary, directory)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            self.verify(directory / "index", public=public, profile=profile)
            record = {"producer": provenance, "source_directory": str(source),
                      "source_receipt": receipt, "source_receipt_sha256": sha(source / "index-receipt.json")}
            write_json(directory / "provenance" / (content_hash(record) + ".json"), record)
            profile_path = self.root / "profiles" / (self.profile_id(profile) + ".json")
            if profile_path.exists() and read_json(profile_path) != profile:
                raise ValueError("Shared index profile collision")
            write_json(profile_path, profile)
            self.refresh_catalog()
        return directory / "index"

    def refresh_catalog(self):
        rows = []
        for path in sorted(self.root.glob("repositories/*/*/*/*/index/index-receipt.json")):
            receipt = self.verify(path.parent)
            mapping = read_json(path.parent / "vector_index.json")
            rows.append({**receipt["repository_snapshot"], "profile_id": self.profile_id(receipt["profile"]),
                         "index": str(path.parent.relative_to(self.root)),
                         "vectors": len(mapping["vector_id_to_text_id"]),
                         "dimensions": receipt["profile"]["settings"]["dimensions"],
                         "data_bytes": sum((path.parent / name).stat().st_size for name in CORE_FILES),
                         "receipt_sha256": sha(path), "digest": receipt["digest"]})
        write_json(self.root / "catalog.json", {"format": "repository-index-catalog-v1", "indexes": rows})
        return rows


def retire_index_data(source, *, reason, replacement=None):
    """Delete only receipted index payloads; keep original logs and receipts."""
    source = Path(source).resolve()
    receipt = read_json(checked_file(source, "index-receipt.json"))
    planned = []
    for name in CORE_FILES:
        path = checked_file(source, name)
        if sha(path) != receipt["files"][name]:
            raise ValueError("Refusing to delete changed index data: " + name)
        planned.append({"file": name, "bytes": path.stat().st_size, "sha256": receipt["files"][name]})
    record = {"reason": reason, "replacement": str(replacement) if replacement else None,
              "source_receipt_sha256": sha(source / "index-receipt.json"), "files": planned,
              "status": "planned", "timestamp": now()}
    write_json(source / "index-retirement.json", record)
    for row in planned:
        (source / row["file"]).unlink()
    record.update(status="completed", deleted_bytes=sum(row["bytes"] for row in planned))
    write_json(source / "index-retirement.json", record)
    return record
