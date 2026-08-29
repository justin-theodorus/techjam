"""The dense retrieval track: a latent space over the catalog's own text.

A second retriever, not a second weighting. BM25 matches the words the
customer used; this matches the words the catalog uses *about* the same
things, which is the one failure mode lexical retrieval cannot survive. The
customer who says "trousers" about a product whose bullets say "pants" scores
zero on every term in `bm25.py` and near one here.

The space is latent semantic indexing over `title` + `features`, built
offline by `tools/build_dense.py` and bundled as a quantized asset. Two
consequences worth stating up front, because both are load-bearing:

  - **It is derived from the frozen catalog, not from vendored weights.**
    There is no model to download, no runtime to install and no license to
    argue about. Rebuilding it is one command against a file the organizer
    ships.
  - **Absence is not an error.** A missing, truncated or mismatched asset
    yields `None` and every caller degrades to the lexical blend. The dense
    track can never cost a session; at worst it is not there.

Scoring is document-major over an already category-filtered pool, mirroring
`bm25.py`, so cost scales with the bucket rather than the catalog: 0.4 ms at
the median bucket of 182 products and 6 ms at the largest pool any route can
assemble.
"""

from __future__ import annotations

import array
import hashlib
import pathlib
import struct
import sys
from typing import Iterable, Sequence

# Asset layout. `MAGIC` and `VERSION` exist so a stale asset is refused
# rather than misread; the header is fixed-width and everything after it is
# sized from the header's own counts.
MAGIC = b"TJDENSE1"
VERSION = 1
HEADER = struct.Struct("<8sIIIII32s")

# int8 codes run to +/- this. 127 rather than 128 keeps the range symmetric,
# so a coordinate and its negation quantize to codes of equal magnitude.
QUANTIZED_MAX = 127

DEFAULT_ASSET = pathlib.Path(__file__).resolve().parent.parent / "assets" / (
    "dense.bin"
)


def fingerprint(asins: Sequence[str]) -> bytes:
    """Returns a digest of the catalog identity this asset is bound to.

    Row offsets in the document block are catalog line numbers. An asset
    built against a different catalog would therefore score the wrong
    products under the right names, and nothing would raise. Comparing this
    digest is the only thing standing between that and a silent, unreadable
    regression.
    """
    digest = hashlib.sha256()
    for asin in asins:
        digest.update(asin.encode("utf-8"))
        digest.update(b"\0")
    return digest.digest()


def header(
    dimensions: int,
    document_count: int,
    term_count: int,
    vocabulary_bytes: int,
    fingerprint: bytes,
) -> bytes:
    """Returns the packed asset header. Shared with the offline builder."""
    return HEADER.pack(
        MAGIC,
        VERSION,
        dimensions,
        document_count,
        term_count,
        vocabulary_bytes,
        fingerprint,
    )


class DenseIndex:
    """A frozen latent space over the catalog, quantized to int8."""

    def __init__(
        self,
        dimensions: int,
        vocabulary: dict[str, int],
        document_codes: array.array,
        document_scales: array.array,
        term_codes: array.array,
        term_scales: array.array,
        catalog_fingerprint: bytes,
    ) -> None:
        self.dimensions = dimensions
        self._vocabulary = vocabulary
        self._document_codes = document_codes
        self._document_scales = document_scales
        self._term_codes = term_codes
        self._term_scales = term_scales
        self._fingerprint = catalog_fingerprint

    def matches(self, asins: Sequence[str]) -> bool:
        """Whether this asset was built against exactly this catalog."""
        return self._fingerprint == fingerprint(asins)

    def encode(self, tokens: Iterable[str]) -> tuple[float, ...] | None:
        """Returns the unit query vector, or `None` if nothing is known.

        `None` is the degradation path and the reason this tier cannot cost
        anything: a query of entirely unseen words produces no vector, the
        dense term contributes zero, and the blend is bit-identical to the
        lexical one.
        """
        dimensions = self.dimensions
        total = [0.0] * dimensions
        seen = False
        for token in tokens:
            position = self._vocabulary.get(token)
            if position is None:
                continue
            seen = True
            scale = self._term_scales[position]
            offset = position * dimensions
            codes = self._term_codes[offset:offset + dimensions]
            for axis in range(dimensions):
                total[axis] += codes[axis] * scale
        if not seen:
            return None
        length = sum(value * value for value in total) ** 0.5
        if length <= 0.0:
            return None
        return tuple(value / length for value in total)

    def score(
        self, document_index: int, query: Sequence[float]
    ) -> float:
        """Returns cosine similarity against one document, floored at zero.

        Both sides are unit length, so this is a cosine in [-1, 1]. A
        negative similarity is evidence *against* a product, but the blend
        max-normalizes each term over the pool, and a negative floor there
        would turn the least similar product into a positive contribution.
        Clamping is what keeps the term monotone in similarity.
        """
        dimensions = self.dimensions
        offset = document_index * dimensions
        codes = self._document_codes[offset:offset + dimensions]
        total = 0.0
        for axis in range(dimensions):
            total += codes[axis] * query[axis]
        total *= self._document_scales[document_index]
        return total if total > 0.0 else 0.0

    def nearest(
        self, query: Sequence[float], pool: Sequence[int], limit: int
    ) -> list[int]:
        """Returns the `limit` documents in `pool` closest to `query`.

        The retrieval half of the track, as opposed to the ranking half:
        this chooses *which* products a route considers, where `score`
        only weighs ones already chosen.
        """
        if limit <= 0 or not pool:
            return []
        scored = [(self.score(index, query), index) for index in pool]
        scored.sort(key=lambda pair: -pair[0])
        return [
            index for similarity, index in scored[:limit] if similarity > 0.0
        ]


def load(path: pathlib.Path | str = DEFAULT_ASSET) -> DenseIndex | None:
    """Reads the bundled asset, or returns `None` if it is not usable.

    Never raises. A submission bundle shipped without the asset, or with a
    truncated or stale one, must run exactly as it does today rather than
    fail at construction, so every failure here is a `None`.
    """
    try:
        blob = pathlib.Path(path).read_bytes()
    except OSError:
        return None
    if len(blob) < HEADER.size:
        return None

    magic, version, dimensions, documents, terms, vocabulary_bytes, digest = (
        HEADER.unpack_from(blob, 0)
    )
    if magic != MAGIC or version != VERSION:
        return None
    if dimensions <= 0 or documents <= 0 or terms <= 0:
        return None

    cursor = HEADER.size
    expected = (
        cursor
        + vocabulary_bytes
        + documents * dimensions + documents * 4
        + terms * dimensions + terms * 4
    )
    if len(blob) != expected:
        return None

    try:
        names = blob[cursor:cursor + vocabulary_bytes].decode("utf-8").split(
            "\n"
        )
    except UnicodeDecodeError:
        return None
    if len(names) != terms:
        return None
    cursor += vocabulary_bytes

    document_codes, cursor = _codes(blob, cursor, documents * dimensions)
    document_scales, cursor = _scales(blob, cursor, documents)
    term_codes, cursor = _codes(blob, cursor, terms * dimensions)
    term_scales, cursor = _scales(blob, cursor, terms)

    return DenseIndex(
        dimensions=dimensions,
        vocabulary={name: index for index, name in enumerate(names)},
        document_codes=document_codes,
        document_scales=document_scales,
        term_codes=term_codes,
        term_scales=term_scales,
        catalog_fingerprint=digest,
    )


def _codes(blob: bytes, cursor: int, count: int) -> tuple[array.array, int]:
    """Reads `count` int8 values starting at `cursor`."""
    values = array.array("b")
    values.frombytes(blob[cursor:cursor + count])
    return values, cursor + count


def _scales(blob: bytes, cursor: int, count: int) -> tuple[array.array, int]:
    """Reads `count` little-endian float32 values starting at `cursor`.

    The asset is written little-endian so it is portable; `byteswap` is the
    one line that makes it so on a big-endian host, where it is untested and
    cheap insurance rather than a supported configuration.
    """
    values = array.array("f")
    values.frombytes(blob[cursor:cursor + count * 4])
    if sys.byteorder != "little":
        values.byteswap()
    return values, cursor + count * 4
