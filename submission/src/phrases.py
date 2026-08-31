"""Whole-phrase evidence: exact constraint strings, weighted by rarity.

The simulator builds every constraint it utters by cleaning one `features`
bullet or one `details` value of the target's own catalog record, so a
constraint is usually an exact substring of exactly one product. 89.9% of the
225,684 phrases in this catalog belong to a single product (measurements 3.3).

That makes phrase evidence very sharp and very brittle at once, so it is used
only to reorder a slate that has already been chosen by the blend. A constraint
the index has never seen contributes nothing and the slate is returned
untouched, so this stage's own downside is zero (measurements 3.23).
"""

from __future__ import annotations

import re
from array import array

# `intent_card` truncates every constraint to 180 characters.
MAX_CONSTRAINT_LENGTH = 180

_WHITESPACE_RE = re.compile(r"\s+")
_STRIP_CHARACTERS = " -;,.\t\n"

# The two vocabularies `intent_card` prepends to its candidate list before the
# `features` and `details` values. Both produce very common phrases, so they
# score near zero once weighted by rarity, but indexing them keeps this module's
# vocabulary identical to the one the customer draws from.
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow"
    r"|orange)\b",
    re.IGNORECASE,
)


def normalize(value: str) -> str:
    """Returns a constraint string in canonical form.

    Collapses runs of whitespace, trims list punctuation from both ends, and
    caps the length. Index and query share this function, which is the only
    reason an exact lookup can succeed at all: the same words written with a
    stray trailing semicolon must reach the same key.
    """
    collapsed = _WHITESPACE_RE.sub(" ", value).strip(_STRIP_CHARACTERS)
    return collapsed[:MAX_CONSTRAINT_LENGTH].rstrip()


def flatten(value: object) -> list[str]:
    """Returns one candidate string per entry of a catalog field.

    A `details` mapping renders as `"Key: Value"`, which is how attributes are
    quoted in practice, so the key stays part of the phrase.
    """
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def candidates(product: dict, corpus: str) -> list[str]:
    """Returns the phrases a session about `product` could quote verbatim.

    The material and color arms matter more than their rarity weight suggests:
    `hard_constraints[0]` is a bare material word in 76% of buying sessions
    (measurements 3.18), so dropping them costs 0.003 even though a phrase held by
    thousands of products can barely move a ranking on its own.

    Args:
        product: One catalog record.
        corpus: Text the material and color patterns are matched against. The
          material and colour of a product are often stated somewhere other
          than the fields we index, so this is deliberately the wider text.
          Narrowing it to `title` + `features` costs 0.0001, which is not worth
          reading `description` for.
    """
    found = [
        *flatten(product.get("features")),
        *flatten(product.get("details")),
    ]
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        found.insert(0, material.group(1).lower())
    if color:
        found.insert(1, f"color: {color.group(1).lower()}")
    return found


class PhraseBuilder:
    """Accumulates documents in catalog order, then freezes into an index."""

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._phrase_ids = array("i")
        self._offsets = array("i", [0])
        self._document_frequency: list[int] = []

    def add(self, phrases: list[str]) -> None:
        """Appends one document. Call order defines document indices."""
        seen: set[int] = set()
        for phrase in phrases:
            cleaned = normalize(phrase)
            if not cleaned:
                continue
            phrase_id = self._vocabulary.get(cleaned)
            if phrase_id is None:
                phrase_id = len(self._vocabulary)
                self._vocabulary[cleaned] = phrase_id
                self._document_frequency.append(0)
            if phrase_id not in seen:
                seen.add(phrase_id)
                self._phrase_ids.append(phrase_id)
                self._document_frequency[phrase_id] += 1
        self._offsets.append(len(self._phrase_ids))

    def freeze(self) -> PhraseIndex:
        """Returns an immutable index."""
        return PhraseIndex(
            self._vocabulary,
            self._phrase_ids,
            self._offsets,
            self._document_frequency,
        )


class PhraseIndex:
    """A frozen index of the phrases each product could be described by."""

    def __init__(
        self,
        vocabulary: dict[str, int],
        phrase_ids: array,
        offsets: array,
        document_frequency: list[int],
    ) -> None:
        self._vocabulary = vocabulary
        self._phrase_ids = phrase_ids
        self._offsets = offsets
        self._document_frequency = document_frequency

    def query_ids(self, constraints: tuple[str, ...]) -> frozenset[int]:
        """Returns phrase ids, dropping constraints absent from the catalog."""
        ids = (
            self._vocabulary.get(normalize(constraint))
            for constraint in constraints
        )
        return frozenset(
            phrase_id for phrase_id in ids if phrase_id is not None
        )

    def evidence(
        self, document_index: int, query_ids: frozenset[int]
    ) -> float:
        """Returns how much rare phrase evidence names this document.

        A phrase held by one product is worth 1.0; one held by a thousand is
        worth 0.001, which is why a bare material word cannot move a ranking.
        """
        if not query_ids:
            return 0.0
        start = self._offsets[document_index]
        end = self._offsets[document_index + 1]
        return sum(
            1.0 / self._document_frequency[phrase_id]
            for phrase_id in self._phrase_ids[start:end]
            if phrase_id in query_ids
        )
