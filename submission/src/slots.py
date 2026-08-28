"""Typing a constraint string: which attribute of a product it talks about.

The taxonomy is learned from the catalog, not declared. Products describe
themselves through a `details` mapping whose keys name attributes directly
("Fabric Type", "Closure Type", "Occasion"), and the values under those keys
are the vocabulary customers use for them. Reading both gives an attribute
classifier that is grounded in the same data the products are.

Typing is bookkeeping, not filtering. No catalog dimension is populated densely
enough to filter on: `Color` covers 4.9% of products, `Material` 4.1%, `Style`
3.5% (findings 3.19), and a hard filter on a 4%-covered dimension discards 96%
of the pool on no evidence. What types buy is knowing which attributes a session
has already heard about, which is what a targeted override, a probe policy and a
grounded reply all need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The attributes a probe may ask about, ordered by how much of this catalog's
# constraint text they account for.
MATERIAL = "material"
COLOR = "color"
SIZE = "size"
STYLE = "style"
USE_CASE = "use_case"
FEATURE = "feature"
BUDGET = "budget"
BRAND = "brand"
CATEGORY = "category"

# What a constraint is about when nothing more specific fits. Half of this
# catalog's constraint strings are plain product bullets with no attribute word
# in them at all, so this is the honest majority class rather than a failure.
DEFAULT = FEATURE

# Substrings of a `details` key, mapped to the attribute that key describes.
# Checked longest first, so "fabric type" beats "type".
KEY_FAMILIES = (
    (COLOR, ("color", "colour")),
    (MATERIAL, ("material", "fabric", "metal type", "lining", "sole",
                "leather")),
    (SIZE, ("size", "fit type", "length", "width")),
    (STYLE, ("style", "pattern", "shape", "neck", "sleeve", "collar",
             "department", "theme", "cut")),
    (USE_CASE, ("occasion", "sport", "activity", "suggested user",
                "target audience", "age range", "recommended use")),
    (BRAND, ("brand", "manufacturer", "model name")),
    (FEATURE, ("closure", "special feature", "component", "batteries")),
)

# A price is stated, not named, so it has no key to look up.
_BUDGET_RE = re.compile(
    r"(?:\$|budget|under|below|less than|cheaper than|<=)\s*\d"
    r"|\d+\s*(?:dollars|usd|bucks)",
    re.IGNORECASE,
)

# Logistics, not description. `details` is 287 sub-keys dominated by shipping
# metadata (findings 3.11), and counting a parcel's dimensions as a size the
# customer might care about makes every attribute look equally worth asking.
_LOGISTICS = (
    "package", "shipping", "date first", "item model", "part number",
    "best sellers", "discontinued", "batteries", "item weight",
    "product dimensions", "item dimensions", "manufacturer recommended",
    "number of items", "quantity",
)

_KEY_RE = re.compile(r"^\s*([^:]{1,40}?)\s*:\s*(.+)$", re.DOTALL)

# Values longer than this are prose, not attribute vocabulary, and indexing them
# would teach the classifier whole marketing sentences.
MAX_VALUE_LENGTH = 30

# How much of a value's occurrences must agree before it is taught as that
# attribute's vocabulary. A plurality rather than unanimity: catalog rows are
# dirty, and "large" appears 86 times under `Size` against one stray row each
# under `Department` and `Style`, which should not disqualify it.
MIN_AGREEMENT = 0.7


@dataclass(frozen=True)
class Slot:
    """One thing the customer has told us, and what it was about."""

    attribute: str
    value: str
    turn: int


class Taxonomy:
    """Maps a constraint string to the attribute it describes."""

    def __init__(
        self,
        values: dict[str, str],
        declared: dict[str, int],
        documents: int,
    ) -> None:
        self._values = values
        self._declared = declared
        self._documents = documents or 1

    def prevalence(self, attribute: str) -> float:
        """Returns the share of products that say anything about `attribute`.

        Read as the prior probability that a customer shopping in this catalog
        has an opinion about that attribute at all, which is what a question is
        worth before anything has been asked.
        """
        return self._declared.get(attribute, 0) / self._documents

    def classify(self, value: str) -> str:
        """Returns the attribute `value` talks about.

        Tries, in order: a stated price, the key of a `"Key: value"` pair, the
        learned value vocabulary, and finally the majority class.
        """
        if _BUDGET_RE.search(value):
            return BUDGET

        match = _KEY_RE.match(value)
        if match:
            named = _from_key(match.group(1))
            if named:
                return named
            body = match.group(2)
        else:
            body = value

        learned = self._values.get(body.strip().casefold())
        if learned:
            return learned
        return DEFAULT

    def slots(
        self, constraints: tuple[str, ...], turn: int
    ) -> tuple[Slot, ...]:
        """Types a whole constraint list, keeping order."""
        return tuple(
            Slot(self.classify(value), value, turn) for value in constraints
        )


class TaxonomyBuilder:
    """Accumulates attribute vocabulary while the catalog is read once."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._declared: dict[str, int] = {}
        self._documents = 0

    def add(self, details: object, has_features: bool = False) -> None:
        """Records what one product declares about itself."""
        self._documents += 1
        named: set[str] = set()
        if isinstance(details, dict):
            for key, value in details.items():
                attribute = _from_key(str(key))
                if not attribute:
                    continue
                named.add(attribute)
                if attribute in (BRAND, FEATURE):
                    continue
                for word in _candidate_values(value):
                    tally = self._counts.setdefault(word, {})
                    tally[attribute] = tally.get(attribute, 0) + 1

        # A product with bullets can always be described by one of them, and
        # half of this catalog's constraint text is exactly that: a plain
        # bullet naming no attribute (findings 3.4).
        if has_features:
            named.add(FEATURE)
        for attribute in named:
            self._declared[attribute] = self._declared.get(attribute, 0) + 1

    def freeze(self) -> Taxonomy:
        """Returns an immutable taxonomy over the values that agree enough."""
        learned = {}
        for word, tally in self._counts.items():
            total = sum(tally.values())
            attribute = max(tally, key=tally.get)
            if tally[attribute] / total >= MIN_AGREEMENT:
                learned[word] = attribute
        return Taxonomy(learned, dict(self._declared), self._documents)


def _candidate_values(value: object) -> list[str]:
    """Returns the vocabulary entries one `details` value contributes."""
    if not isinstance(value, str):
        return []
    cleaned = value.strip().casefold()
    if not cleaned or len(cleaned) > MAX_VALUE_LENGTH:
        return []
    return [cleaned]


def _from_key(key: str) -> str | None:
    """Returns the attribute a `details` key names, if it names one."""
    lowered = key.strip().casefold()
    if any(needle in lowered for needle in _LOGISTICS):
        return None
    best: tuple[int, str] | None = None
    for attribute, needles in KEY_FAMILIES:
        for needle in needles:
            if needle in lowered and (best is None or len(needle) > best[0]):
                best = (len(needle), attribute)
    return best[1] if best else None
