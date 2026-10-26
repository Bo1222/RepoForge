"""Small deterministic BM25 fallback over a capped set of repository files.

This is intentionally not a persistent index: it runs only when exact and
identifier searches have no hits, and the caller caps the file set.  It adds
lexical recall without changing the primary retrieval order.
"""

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_MAX_CHARS = 500_000


def _tokens(text: str) -> list[str]:
    result = []
    for raw in _TOKEN.findall(text or ""):
        result.append(raw.lower())
        for part in _CAMEL.findall(raw):
            if len(part) >= 3:
                result.append(part.lower())
    return result


def _line_for(content: str, terms: list[str]) -> int:
    lowered = content.lower()
    for term in terms:
        index = lowered.find(term.lower())
        if index >= 0:
            return content.count("\n", 0, index) + 1
    return 1


def bm25_search(files, query: str, limit: int = 20, max_files: int = 2000,
                k1: float = 1.5, b: float = 0.75) -> list[tuple[str, int]]:
    """Return ``(file_path, line)`` hits ranked by BM25."""
    query_tokens = list(dict.fromkeys(_tokens(query)))
    if not query_tokens:
        return []
    docs = []
    for path, content in files:
        if len(docs) >= max_files:
            break
        if not content:
            continue
        content = content[:_MAX_CHARS]
        counts = Counter(_tokens(content))
        if counts:
            docs.append((str(path), content, counts))
    if not docs:
        return []
    count = len(docs)
    average_length = sum(sum(item.values()) for _, _, item in docs) / count
    document_frequency = Counter()
    query_set = set(query_tokens)
    for _, _, item in docs:
        for token in set(item) & query_set:
            document_frequency[token] += 1
    scored = []
    for path, content, item in docs:
        length = sum(item.values())
        score = 0.0
        for token in query_tokens:
            frequency = item.get(token, 0)
            if not frequency:
                continue
            idf = math.log(1 + (count - document_frequency[token] + 0.5)
                           / (document_frequency[token] + 0.5))
            score += idf * frequency * (k1 + 1) / (
                frequency + k1 * (1 - b + b * length / average_length))
        if score > 0:
            scored.append((score, path, _line_for(content, query_tokens)))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [(path, line) for _, path, line in scored[:limit]]
