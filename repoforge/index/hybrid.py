"""Small, deterministic rank fusion helpers for query-time retrieval."""

import re

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text or "")}


def rrf(rank_lists, k: int = 60):
    """Reciprocal rank fusion; inputs are ordered lists of hashable ids."""
    scores = {}
    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))]


def fuse_hits(query: str, hits):
    """Fuse semantic order with lexical path/name overlap, returning hits."""
    if len(hits) < 2:
        return hits
    query_tokens = tokens(query)
    lexical = sorted(
        range(len(hits)),
        key=lambda index: (-len(query_tokens & tokens(hits[index].file_path)),
                           -len(query_tokens & tokens(" ".join(span.span_id for span in hits[index].spans))),
                           index),
    )
    semantic = list(range(len(hits)))
    order = rrf([semantic, lexical])
    return [hits[index] for index in order]
