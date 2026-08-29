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

from techjam.submission.src import text

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

# A refusal names an attribute the customer does not want. Anchored to the head
# of the constraint, or to the head of a `"Key: value"` payload, because an
# interior "not" almost always belongs to a marketing clause ("not sold
# separately") rather than to the customer, and flipping a whole constraint on
# one of those inverts a signal that was reading correctly.
#
# `no` and `non-` are deliberately absent, and the cue must be followed by
# whitespace rather than a hyphen. This catalog spells attribute *names* that
# way -- `Non-Polarized` 239 times, `No Closure closure` 192, `No-Tie Laces`,
# `NO SHOW ATHLETIC SOCKS` -- so reading them as refusals inverts 0.3% of all
# constraint text, 3 of which reach the public 200 (findings 3.31). The cost is
# that a customer who says "no wool" is not heard; the catalog says the
# ambiguity is not worth it.
# The master switch for reading refusals at all. Ships live: without it a
# refusal is scored as evidence *for* what the customer declined. Ablating
# `ranking.NEGATION_WEIGHT` alone measures only the penalty half, which is the
# small half; this is what turns the whole mechanism off (findings 3.31).
NEGATION = True

_NEGATION_RE = re.compile(
    r"^(?:not|without|avoid|excluding|other than|anything but|rather than|"
    r"don'?t want|do not want|doesn'?t have to be)\s+",
    re.IGNORECASE,
)

# Values longer than this are prose, not attribute vocabulary, and indexing them
# would teach the classifier whole marketing sentences.
MAX_VALUE_LENGTH = 30

# How much of a value's occurrences must agree before it is taught as that
# attribute's vocabulary. A plurality rather than unanimity: catalog rows are
# dirty, and "large" appears 86 times under `Size` against one stray row each
# under `Department` and `Style`, which should not disqualify it.
MIN_AGREEMENT = 0.7

# The token vocabulary is a second, looser pass over the same `details` values,
# and it exists only so free constraint text can be typed (findings 3.38). Its
# thresholds are separate from the ones above on purpose: `classify` matches a
# value outright and can afford to be strict, while a single word carries less
# evidence and needs a support floor instead.
MAX_TOKEN_VALUE_LENGTH = 60
MIN_TOKEN_LENGTH = 2
MIN_TOKEN_SUPPORT = 5

# Words that appear under every kind of key and so type nothing.
_TOKEN_STOPWORDS = frozenset(
    ("and", "the", "for", "with", "from", "not", "all", "one", "two")
)


def polarity(value: str) -> tuple[bool, str]:
    """Returns whether a constraint refuses something, and its text without
    the cue.

    The stripped text is what typing and retrieval want; the caller keeps the
    raw string, because the reply quotes the customer back to themselves and
    "polyester" is not what they said.
    """
    if not NEGATION:
        return False, value

    match = _KEY_RE.match(value)
    key, body = (match.group(1), match.group(2)) if match else ("", value)

    stripped = _NEGATION_RE.sub("", body.strip(), count=1)
    if stripped == body.strip():
        return False, value
    if not stripped:
        return False, value
    return True, f"{key}: {stripped}" if key else stripped


@dataclass(frozen=True)
class Slot:
    """One thing the customer has told us, and what it was about."""

    attribute: str
    value: str
    turn: int
    negated: bool = False


class Taxonomy:
    """Maps a constraint string to the attribute it describes."""

    def __init__(
        self,
        values: dict[str, str],
        declared: dict[str, int],
        documents: int,
        tokens: dict[str, str] | None = None,
    ) -> None:
        self._values = values
        self._declared = declared
        self._documents = documents or 1
        self._tokens = tokens or {}

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

        A refusal is typed by what it refuses, so "not polyester" is a material
        like "polyester" is. Typing it as the majority class instead would hide
        it from the targeted override, which supersedes by attribute.
        """
        _, value = polarity(value)
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

    def classify_text(self, value: str) -> str:
        """Types free text by the vocabulary its individual words land in.

        `classify` asks whether the whole string is a value this catalog uses.
        This asks the weaker question -- whether any word in it is -- which is
        what a marketing bullet needs, since it names an attribute in passing
        rather than stating it. The two are kept apart because `classify` feeds
        `NEGATION` and the targeted override, and loosening it would move both.

        Ties break toward the more specific attribute: a stated budget is more
        specific than a material, a material more than a style.
        """
        _, body = polarity(value)
        if _BUDGET_RE.search(body):
            return BUDGET
        hits = {
            self._tokens[token] for token in text.unique_tokens(body)
            if token in self._tokens
        }
        if not hits:
            return DEFAULT
        return min(hits, key=_specificity)

    def slots(
        self, constraints: tuple[str, ...], turn: int
    ) -> tuple[Slot, ...]:
        """Types a whole constraint list, keeping order."""
        return tuple(
            Slot(self.classify(value), value, turn, polarity(value)[0])
            for value in constraints
        )


class TaxonomyBuilder:
    """Accumulates attribute vocabulary while the catalog is read once."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._tokens: dict[str, dict[str, int]] = {}
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
                for word in _candidate_tokens(value):
                    tally = self._tokens.setdefault(word, {})
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
        tokens = {}
        for word, tally in self._tokens.items():
            total = sum(tally.values())
            if total < MIN_TOKEN_SUPPORT:
                continue
            attribute = max(tally, key=tally.get)
            if tally[attribute] / total >= MIN_AGREEMENT:
                tokens[word] = attribute
        return Taxonomy(
            learned, dict(self._declared), self._documents, tokens
        )


def _specificity(attribute: str) -> int:
    """Returns how narrow an attribute is, for breaking a multi-hit tie."""
    order = (BUDGET, MATERIAL, COLOR, SIZE, STYLE, USE_CASE)
    return order.index(attribute) if attribute in order else len(order)


def _candidate_values(value: object) -> list[str]:
    """Returns the vocabulary entries one `details` value contributes."""
    if not isinstance(value, str):
        return []
    cleaned = value.strip().casefold()
    if not cleaned or len(cleaned) > MAX_VALUE_LENGTH:
        return []
    return [cleaned]


def _candidate_tokens(value: object) -> list[str]:
    """Returns the single words one `details` value contributes.

    `_candidate_values` keeps the whole string, which is what `classify` wants:
    an outright-stated value should match outright. But free constraint text
    almost never restates a value verbatim -- a bullet reads "95% Cotton, 5%
    Spandex" where the vocabulary learned "cotton" -- so typing that text needs
    the same evidence one level down (findings 3.38).
    """
    if not isinstance(value, str) or len(value) > MAX_TOKEN_VALUE_LENGTH:
        return []
    return [
        token for token in text.unique_tokens(value)
        if len(token) > MIN_TOKEN_LENGTH and token not in _TOKEN_STOPWORDS
    ]


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
