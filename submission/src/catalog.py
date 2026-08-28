"""Catalog loading: category buckets, popularity priors, and the BM25 index."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from submission.src import bm25
from submission.src import category as category_module
from submission.src import phrases as phrases_module
from submission.src import slots as slots_module
from submission.src import text

FALLBACK_CATEGORY = "clothing item"
FALLBACK_POOL_SIZE = 100
EXCLUDED_CATEGORIES = frozenset({
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
})


def category_parts(values: list[str]) -> list[str]:
    """Returns comma-split category segments, minus the top-level buckets."""
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in EXCLUDED_CATEGORIES:
                cleaned.append(part)
    return cleaned


def coarse_category(values: list[str]) -> str:
    """Returns the bucket key: the last two category segments, space-joined.

    The catalog's `categories` field is a breadcrumb from general to specific,
    so its last two segments name the narrowest real department a product sits
    in. Scoring reads the same field, and `submission/src/tests/test_catalog.py`
    asserts the two agree over all 50,000 products.
    """
    cleaned = category_parts(values)
    return " ".join(cleaned[-2:]) if cleaned else FALLBACK_CATEGORY


@dataclass(frozen=True)
class Catalog:
    """The frozen catalog: identifiers, priors, buckets, and the BM25 index."""

    asins: tuple[str, ...]
    ids: frozenset[str]
    prior: tuple[float, ...]
    buckets: dict[str, tuple[int, ...]]
    pad_groups: dict[str, tuple[int, ...]]
    pad_key_of: dict[str, str]
    popular: tuple[int, ...]
    index: bm25.Bm25Index
    phrases: phrases_module.PhraseIndex
    resolver: category_module.Resolver
    taxonomy: slots_module.Taxonomy

    def bucket(self, key: str | None) -> tuple[int, ...]:
        """Returns the pool for one bucket key, popularity-ordered.

        An unknown or missing key yields an empty pool rather than raising.
        """
        return self.buckets.get(key, ()) if key else ()

    def pool(self, keys: tuple[str, ...]) -> tuple[int, ...]:
        """Returns the union of several buckets, popularity-ordered.

        A confident category read yields one key and this is `bucket`. An
        uncertain one yields several, which widens the pool instead of emptying
        it; the ordering still falls back to the prior when nothing matches
        lexically, which is what keeps a soft read from ranking randomly.
        """
        if not keys:
            return ()
        if len(keys) == 1:
            return self.bucket(keys[0])
        merged: list[int] = []
        seen: set[int] = set()
        for key in keys:
            for index in self.buckets.get(key, ()):
                if index not in seen:
                    seen.add(index)
                    merged.append(index)
        merged.sort(key=lambda index: -self.prior[index])
        return tuple(merged)

    def fallback_pool(self, key: str | None) -> tuple[int, ...]:
        """Returns the coarser group the bucket sits inside, popularity-ordered.

        Serves two callers: padding a bucket holding fewer than ten products,
        and recovering when the parsed category is not a known bucket at all.
        The second never fires on the public 200, which is why it is tested.
        """
        if not key:
            return ()
        inside = self.pad_groups.get(self.pad_key_of.get(key, ""))
        if inside:
            return inside
        return self.pad_groups.get(key.rsplit(" ", 1)[-1], ())

    def slate_of(self, indices: Iterable[int]) -> list[str]:
        """Returns the parent ASINs for document indices."""
        return [self.asins[index] for index in indices]


def build(catalog_path: str | Path) -> Catalog:
    """Reads the catalog once and returns everything ranking needs."""
    asins: list[str] = []
    prior: list[float] = []
    bucket_members: dict[str, list[int]] = {}
    pad_members: dict[str, list[int]] = {}
    pad_key_of: dict[str, str] = {}
    builder = bm25.Bm25Builder()
    phrase_builder = phrases_module.PhraseBuilder()
    taxonomy_builder = slots_module.TaxonomyBuilder()

    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asins.append(str(product["parent_asin"]))
            prior.append(_prior(product))
            document = _document_text(product)
            builder.add(text.tokens(document))
            phrase_builder.add(phrases_module.candidates(product, document))
            taxonomy_builder.add(
                product.get("details"), bool(product.get("features"))
            )

            index = len(asins) - 1
            cleaned = category_parts(product.get("categories") or [])
            bucket_key = (
                " ".join(cleaned[-2:]) if cleaned else FALLBACK_CATEGORY
            )
            pad_key = cleaned[-1] if cleaned else FALLBACK_CATEGORY
            bucket_members.setdefault(bucket_key, []).append(index)
            pad_members.setdefault(pad_key, []).append(index)
            pad_key_of.setdefault(bucket_key, pad_key)

    def by_popularity(index: int) -> float:
        return -prior[index]

    return Catalog(
        asins=tuple(asins),
        ids=frozenset(asins),
        prior=tuple(prior),
        buckets={
            key: tuple(sorted(members, key=by_popularity))
            for key, members in bucket_members.items()
        },
        pad_groups={
            key: tuple(sorted(members, key=by_popularity))
            for key, members in pad_members.items()
        },
        pad_key_of=pad_key_of,
        popular=tuple(
            sorted(range(len(asins)), key=by_popularity)[:FALLBACK_POOL_SIZE]
        ),
        index=builder.freeze(),
        phrases=phrase_builder.freeze(),
        resolver=category_module.build(tuple(bucket_members)),
        taxonomy=taxonomy_builder.freeze(),
    )


def _document_text(product: dict) -> str:
    """Returns the indexed text: `title` plus `features`, and nothing else.

    Measured: adding `description` costs 4.5 points of hit@10, and `details` is
    287 sub-keys of shipping metadata that drown the few useful ones.
    """
    parts = [str(product.get("title") or "")]
    features = product.get("features")
    if isinstance(features, list):
        parts.extend(str(item) for item in features if item not in (None, ""))
    elif features not in (None, ""):
        parts.append(str(features))
    return " ".join(parts)


def _prior(product: dict) -> float:
    """Returns the popularity prior, `log1p` of the review count."""
    count = product.get("rating_number")
    if isinstance(count, bool) or not isinstance(count, (int, float)):
        return 0.0
    return math.log1p(float(count)) if count > 0 else 0.0
