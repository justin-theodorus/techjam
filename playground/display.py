"""The product fields the agent throws away, kept for the screen.

`catalog.build` folds `title` into the BM25 index and `rating_number` into a
float prior, then drops both: nothing on the scoring path ever needs to show a
product to anyone. A UI does, so this reads the catalog a second time and keeps
only what a card displays. Measured at 0.2s for all 50,000 records, paid once
at server start.

There is no image URL anywhere in the catalog schema, so a card is text. That
is not a limitation worth apologising for on camera: `rating_number` is the
popularity prior the ranking actually blends, so putting the review count on
the card is what makes `alpha` legible.
"""

from __future__ import annotations

import json
from pathlib import Path

# Long enough to identify a product, short enough not to wrap a card twice.
TITLE_LIMIT = 110

# Features are the field the simulator quotes from, so the drawer shows them
# rather than `description`, which the index deliberately excludes (3.13).
FEATURE_LIMIT = 8


def load(catalog_path: str | Path = "data/catalog.jsonl") -> dict[str, dict]:
    """Returns display fields keyed by `parent_asin`.

    Args:
        catalog_path: The frozen catalog, the same file the agent indexes.
    """
    products: dict[str, dict] = {}
    with open(catalog_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            asin = str(record["parent_asin"])
            products[asin] = {
                "asin": asin,
                "title": _clip(record.get("title")),
                "store": record.get("store") or "",
                "price": record.get("price"),
                "rating": record.get("average_rating"),
                "reviews": record.get("rating_number") or 0,
                "categories": record.get("categories") or [],
                "features": (record.get("features") or [])[:FEATURE_LIMIT],
            }
    return products


def card(products: dict[str, dict], asin: str) -> dict:
    """Returns one product's display fields, or a placeholder for a stranger.

    A slate ASIN always exists in the catalog because the evaluator drops the
    ones that do not, but a hand-typed goal product need not, and a missing
    card should not take the panel down with it.
    """
    found = products.get(asin)
    if found is not None:
        return found
    return {
        "asin": asin, "title": asin, "store": "", "price": None,
        "rating": None, "reviews": 0, "categories": [], "features": [],
    }


def _clip(title: object) -> str:
    """Returns a single-line title, cut at a word boundary where it can be."""
    text = " ".join(str(title or "").split())
    if len(text) <= TITLE_LIMIT:
        return text
    cut = text[:TITLE_LIMIT].rsplit(" ", 1)[0]
    return f"{cut or text[:TITLE_LIMIT]}..."
