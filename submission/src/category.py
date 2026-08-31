"""Resolving free customer text to catalog category buckets.

The coarse category is the highest-value and least fragile signal in the
problem: filtering to it cuts 50,000 products to a median of 182 at
recall@100 = 0.990, worth 0.30 of recall@10 over using it as query terms
(measurements 3.6). Recovering it by stripping a known template is exact while the
template holds and yields nothing at all when it does not, which is measured at
a total collapse (measurements 3.24).

This module is the path that does not depend on the wording. Bucket keys are
real category names drawn from the catalog, so matching a customer's words
against them is ordinary taxonomy classification, and it degrades to a wider
pool rather than to an empty one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from submission.src import text

# A candidate must account for at least this much of a bucket name's weight
# before it is worth considering. Low enough that naming only the generic
# half ("shoes") still nominates the shoe buckets, high enough that one
# incidental word does not nominate hundreds. Only reached when the customer
# never states a bucket name outright, so it trades precision for a pool
# that is merely wide instead of empty.
MIN_COVERAGE = 0.3

# How many buckets a soft match may union. A median bucket holds 182 products,
# so three keeps the pool comparable to one large bucket while covering the
# confusions that actually occur.
MAX_BUCKETS = 3

# How far below the leader a runner-up may sit and still be included. A decisive
# leader resolves to one bucket, the same hard filter as the exact path.
DECISION_MARGIN = 0.05


@dataclass(frozen=True)
class Match:
    """One candidate bucket for a customer message."""

    key: str
    coverage: float
    matched: int


class Resolver:
    """Maps customer words to bucket keys, best first.

    Scores a bucket by how much of its own name the message accounts for,
    weighted by how rare each word is across the catalog's bucket names. A
    message naming every word of `Shirts T-Shirts` scores 1.0 there; one naming
    only `Shirts` scores whatever share that single word carries.
    """

    def __init__(
        self,
        keys: tuple[str, ...],
        postings: dict[str, tuple[int, ...]],
        key_tokens: tuple[tuple[str, ...], ...],
        weights: dict[str, float],
        totals: tuple[float, ...],
    ) -> None:
        self._keys = keys
        self._postings = postings
        self._key_tokens = key_tokens
        self._weights = weights
        self._totals = totals
        self._known = frozenset(keys)

    def contains(self, key: str | None) -> bool:
        """Returns whether `key` is a bucket key exactly."""
        return bool(key) and key in self._known

    def resolve(self, tokens: list[str]) -> tuple[Match, ...]:
        """Returns the buckets `tokens` best accounts for, strongest first.

        Args:
            tokens: Tokens from the customer message, already lowercased by
              `submission.src.text`.
        """
        present = set(tokens)
        candidates: set[int] = set()
        for token in present:
            candidates.update(self._postings.get(token, ()))

        matches = []
        for index in candidates:
            total = self._totals[index]
            if total <= 0.0:
                continue
            hit = [
                token for token in self._key_tokens[index] if token in present
            ]
            coverage = sum(self._weights[token] for token in hit) / total
            if coverage >= MIN_COVERAGE:
                matches.append(Match(self._keys[index], coverage, len(hit)))

        # A more specific name winning a tie is the safer default: its
        # vocabulary is a superset, so the customer said strictly more.
        matches.sort(key=lambda match: (-match.coverage, -match.matched))
        return tuple(matches)

    def buckets(self, value: str) -> tuple[str, ...]:
        """Returns the buckets to retrieve from, empty when nothing matched.

        A customer who states a category name verbatim has been unambiguous, so
        that alone is retrieved from however it was framed. Only when nobody's
        name is stated outright do near-ties union, and widening the pool then
        is the point: an uncertain read should cost precision, not the session.
        """
        matches = self.resolve(text.tokens(value))
        if not matches:
            return ()
        named = self._named(value, matches)
        if named:
            return (named,)
        floor = matches[0].coverage - DECISION_MARGIN
        near = [match.key for match in matches if match.coverage >= floor]
        return tuple(near[:MAX_BUCKETS])

    def _named(self, value: str, matches: tuple[Match, ...]) -> str:
        """Returns the longest bucket name the message states outright."""
        lowered = value.casefold()
        best = ""
        for match in matches:
            key = match.key
            if match.coverage < 1.0 or len(key) <= len(best):
                continue
            if key.casefold() in lowered:
                best = key
        return best


def build(keys: tuple[str, ...]) -> Resolver:
    """Returns a resolver over the catalog's bucket keys."""
    key_tokens = tuple(
        tuple(dict.fromkeys(text.tokens(key))) for key in keys
    )

    frequency: dict[str, int] = {}
    postings: dict[str, list[int]] = {}
    for index, tokens in enumerate(key_tokens):
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
            postings.setdefault(token, []).append(index)

    count = len(keys) or 1
    weights = {
        token: math.log(1.0 + count / value)
        for token, value in frequency.items()
    }
    totals = tuple(
        sum(weights[token] for token in tokens) for tokens in key_tokens
    )
    return Resolver(
        keys=keys,
        postings={token: tuple(value) for token, value in postings.items()},
        key_tokens=key_tokens,
        weights=weights,
        totals=totals,
    )
