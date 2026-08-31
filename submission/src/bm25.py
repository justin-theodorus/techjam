"""BM25 over one flat token-id array with per-document offsets."""

from __future__ import annotations

import math
from array import array

# Not the textbook 1.2 / 0.75, and not the public-set optimum either. A document
# here is a title plus short attribute bullets, and the customer's constraints
# are cleaned substrings of the target's own bullets, so heavy term-frequency
# saturation and heavy length normalisation both work against the signal. Moving
# toward 0.6 / 0.3 is worth +0.030 of TechnicalScore.
#
# The axis has a cost at both ends, which is why these sit mid-range
# (measurements 3.22). Textbook values over-penalise long documents. Driving K1 and B to zero
# wins another 0.013 on the public set but leaks popularity into the lexical
# half, because a product carrying more bullets matches more distinct constraint
# terms and those are the heavily-reviewed ones -- and insuring against a change
# in target sampling is the one job this half of the blend has. These values
# were chosen against the counterfactual risk surface, not the public score.
K1 = 0.6
B = 0.3


class Bm25Builder:
    """Accumulates documents in catalog order, then freezes into an index."""

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._token_ids = array("i")
        self._offsets = array("i", [0])
        self._document_frequency: list[int] = []

    def add(self, tokens: list[str]) -> None:
        """Appends one document. Call order defines document indices."""
        seen: set[int] = set()
        for token in tokens:
            token_id = self._vocabulary.get(token)
            if token_id is None:
                token_id = len(self._vocabulary)
                self._vocabulary[token] = token_id
                self._document_frequency.append(0)
            self._token_ids.append(token_id)
            if token_id not in seen:
                seen.add(token_id)
                self._document_frequency[token_id] += 1
        self._offsets.append(len(self._token_ids))

    def freeze(self) -> Bm25Index:
        """Returns an immutable index.

        Raises:
            ValueError: If no documents were added.
        """
        document_count = len(self._offsets) - 1
        if document_count <= 0:
            raise ValueError("cannot build a BM25 index over an empty catalog")
        idf = [
            math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for frequency in self._document_frequency
        ]
        return Bm25Index(
            self._vocabulary,
            self._token_ids,
            self._offsets,
            idf,
            len(self._token_ids) / document_count,
        )


class Bm25Index:
    """A frozen BM25 index over the catalog's `title` and `features` text."""

    def __init__(
        self,
        vocabulary: dict[str, int],
        token_ids: array,
        offsets: array,
        idf: list[float],
        average_length: float,
    ) -> None:
        self._vocabulary = vocabulary
        self._token_ids = token_ids
        self._offsets = offsets
        self._idf = idf
        self._average_length = average_length

    def query_ids(self, tokens: list[str]) -> frozenset[int]:
        """Returns token ids, dropping tokens absent from the catalog."""
        ids = (self._vocabulary.get(token) for token in tokens)
        return frozenset(token_id for token_id in ids if token_id is not None)

    def tokens_of(self, document_index: int) -> frozenset[int]:
        """Returns the distinct token ids of one document.

        Exposed so a set-selection stage can measure how much two products say
        the same thing without a second index: the postings are already here.
        """
        start = self._offsets[document_index]
        end = self._offsets[document_index + 1]
        return frozenset(self._token_ids[start:end])

    def score(self, document_index: int, query_ids: frozenset[int]) -> float:
        """Returns the BM25 score of one document against a query."""
        start = self._offsets[document_index]
        end = self._offsets[document_index + 1]
        length = end - start
        if not length or not query_ids:
            return 0.0

        frequencies: dict[int, int] = {}
        for token_id in self._token_ids[start:end]:
            if token_id in query_ids:
                frequencies[token_id] = frequencies.get(token_id, 0) + 1
        if not frequencies:
            return 0.0

        normalizer = K1 * (1.0 - B + B * length / self._average_length)
        return sum(
            self._idf[token_id] * count * (K1 + 1.0) / (count + normalizer)
            for token_id, count in frequencies.items()
        )
