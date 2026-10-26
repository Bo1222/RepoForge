"""Deterministic query-term extraction for search fallbacks.

No model call and no repository scan happens here: the same query always
produces the same ordered terms.  The caller decides which terms to grep.
"""

import re

_BACKTICK = re.compile(r"`([^`]+)`")
_QUALIFIED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_PATH = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
_ERROR = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_CAMEL_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "into", "when",
    "where", "what", "which", "while", "have", "has", "had", "are", "was",
    "were", "not", "but", "you", "your", "use", "using", "can", "could",
    "should", "would", "will", "def", "class", "return", "import", "self",
}


def _parts(value: str) -> list[str]:
    return [part.lower() for part in _CAMEL_PART.findall(value)
            if len(part) >= 3 and part.lower() not in _STOP]


def query_terms(text: str, limit: int = 12) -> list[str]:
    """Return full identifiers first, then their snake/camel sub-tokens."""
    text = text or ""
    ordered: list[str] = []
    seen: set[str] = set()

    def add(value: str):
        value = value.strip()
        if not value or value.lower() in _STOP or value in seen:
            return
        seen.add(value)
        ordered.append(value)

    for match in _BACKTICK.finditer(text):
        add(match.group(1))
    for pattern in (_QUALIFIED, _PATH, _ERROR):
        for match in pattern.finditer(text):
            add(match.group(0))
    for match in _IDENTIFIER.finditer(text):
        token = match.group(0)
        add(token)
        for part in _parts(token):
            add(part)
    return ordered[:limit]
