"""Tokenization shared by the index build and the query build."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MIN_TOKEN_LENGTH = 2

# Function words, plus the conversational verbs a shopper frames a request
# with. None of them describes a product, so they are noise in both halves: an
# index entry for "want" matches everything and discriminates nothing.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
})


def tokens(value: str) -> list[str]:
    """Returns lowercased tokens, minus stopwords and single characters."""
    result: list[str] = []
    for match in TOKEN_RE.findall(value):
        if len(match) < MIN_TOKEN_LENGTH:
            continue
        lowered = match.lower()
        if lowered not in STOPWORDS:
            result.append(lowered)
    return result


def unique_tokens(value: str) -> list[str]:
    """Returns `tokens` deduped, so a repeated word cannot dominate a query."""
    return list(dict.fromkeys(tokens(value)))
