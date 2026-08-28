"""Tiny self-contained catalog fixture for the agent tests."""

from __future__ import annotations

import json
from pathlib import Path

SNEAKER_BUCKET = "Footwear Sneakers"
BOOT_BUCKET = "Footwear Boots"
FILLER_BUCKET = "Footwear Tees"
DEEP_BUCKET = "Footwear Sandals"
TUNIC_BUCKET = "Footwear Tunics"
DEEP_SIZE = 30


def _product(
    asin: str, title: str, features: list[str], leaf: str, ratings: int
) -> dict:
    return {
        "parent_asin": asin,
        "title": title,
        "features": features,
        "description": ["ignored prose that must never reach the index"],
        "categories": [
            "Clothing, Shoes & Jewelry", "Women", f"Footwear, {leaf}",
        ],
        "details": {"Department": "womens"},
        "store": "Example",
        "price": None,
        "average_rating": 4.5,
        "rating_number": ratings,
    }


# Sneakers holds three and Boots two, so Boots must be padded. Tees holds
# twelve, enough to fill a slate on its own. Sandals holds thirty, which is
# the only bucket deep enough for a narrow head to explore past the slate.
CATALOG_ROWS = [
    _product("SNEAK_POP", "Popular canvas sneaker",
             ["cotton canvas upper"], "Sneakers", 90_000),
    _product("SNEAK_MID", "Midrange leather sneaker",
             ["leather upper", "rubber sole"], "Sneakers", 500),
    _product("SNEAK_RARE", "Obscure hemp sneaker",
             ["hemp upper", "cork footbed"], "Sneakers", 3),
    _product("BOOT_POP", "Popular winter boot",
             ["wool lining"], "Boots", 40_000),
    _product("BOOT_RARE", "Obscure winter boot",
             ["shearling lining"], "Boots", 7),
] + [
    _product(f"FILLER_{n:02d}", f"Filler tee {n}",
             ["polyester blend"], "Tees", 1000 - n * 10)
    for n in range(12)
] + [
    # Deep enough that a narrow head must explore past the slate rather than
    # below it, with strictly decreasing review counts so prior rank is exactly
    # the index. DEEP_15 carries the rare phrase a later turn should promote.
    _product(f"DEEP_{n:02d}", f"Deep sandal {n}",
             ["hemp footbed marker"] if n == 15 else ["synthetic footbed"],
             "Sandals", (DEEP_SIZE - n) * 100)
    for n in range(DEEP_SIZE)
] + [
    # A term-frequency trap. Under the textbook K1 the repeater outranks the
    # exact match on term frequency alone; it must not.
    _product("TF_REPEATER", "Repeater tunic",
             ["merino " * 10, "merino wool blend"] + ["filler bullet"] * 20,
             "Tunics", 100),
    _product("TF_EXACT", "Exact tunic", ["merino"], "Tunics", 100),
]


def write_catalog(root: Path) -> Path:
    """Writes the fixture catalog into `root` and returns its path."""
    path = root / "catalog.jsonl"
    rows = "".join(json.dumps(row) + "\n" for row in CATALOG_ROWS)
    path.write_text(rows, encoding="utf-8")
    return path
