"""Builds the bundled dense retrieval asset from the frozen catalog.

Offline, developer-only, and never imported at scoring time. It is the only
file in the project that uses numpy; `submission/` stays standard library
only, which is the Feasibility claim the submission is built on.

The space is latent semantic indexing over the catalog's own text: a TF-IDF
term-document matrix reduced by randomized SVD. That choice is not
incidental. It needs no vendored weights, no runtime, and no license
argument, and it is derived entirely from data the organizer supplies, so
regenerating it is one command against a file everyone already has.

Two invariants this file exists to hold, both of which fail silently if
broken:

  - The document text is built by `catalog._document_text` and tokenized by
    `text.tokens`, the same two functions the BM25 index uses. A dense space
    over a different field set would quietly contradict decision 7.
  - Documents are emitted in catalog line order, and a fingerprint over the
    parent ASIN sequence is written into the header. The loader refuses an
    asset whose fingerprint does not match the catalog it was built against,
    because mismatched row offsets would scramble every ranking without
    raising anything.

Usage:
    python3 tools/build_dense.py --out submission/assets/dense.bin
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys
import time

import numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from techjam.submission.src import catalog as catalog_module
from techjam.submission.src import dense
from techjam.submission.src import text

# Latent dimensions. 64 keeps the int8 document block at 3.2 MB and scores a
# median bucket in 0.4 ms; the singular values are already flat by 64.
DIMENSIONS = 64

# Terms rarer than this carry no reliable co-occurrence evidence and would
# spend asset bytes on noise. At 3 the vocabulary is 20,079 of 59,403.
MIN_DOCUMENT_FREQUENCY = 3

# Randomized SVD: extra columns to sample beyond `DIMENSIONS`, and how many
# power iterations to sharpen the spectrum with.
OVERSAMPLING = 16
POWER_ITERATIONS = 2

# Fixed, and part of the artifact's provenance: the asset must rebuild
# byte-identical from the same catalog.
SEED = 0


def read_catalog(path: pathlib.Path) -> tuple[list[str], list[list[str]]]:
    """Returns parent ASINs and tokenized documents, in catalog line order."""
    asins: list[str] = []
    documents: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asins.append(str(product["parent_asin"]))
            documents.append(
                text.tokens(catalog_module._document_text(product))
            )
    return asins, documents


def term_document(
    documents: list[list[str]],
) -> tuple[list[str], numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Returns the vocabulary and the row-normalized TF-IDF matrix as COO.

    Sparse by hand rather than by scipy: the whole point of this file is that
    it depends on numpy alone, so it can be re-run anywhere numpy installs.
    """
    frequency: collections.Counter[str] = collections.Counter()
    counted = []
    for tokens in documents:
        counts = collections.Counter(tokens)
        counted.append(counts)
        frequency.update(counts.keys())

    vocabulary = sorted(
        term for term, count in frequency.items()
        if count >= MIN_DOCUMENT_FREQUENCY
    )
    identifier = {term: index for index, term in enumerate(vocabulary)}
    total = len(documents)
    inverse = numpy.array(
        [math.log(total / frequency[term]) for term in vocabulary],
        dtype=numpy.float32,
    )

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for document, counts in enumerate(counted):
        for term, count in counts.items():
            column = identifier.get(term)
            if column is None:
                continue
            rows.append(document)
            columns.append(column)
            values.append(1.0 + math.log(count))

    row_index = numpy.array(rows, dtype=numpy.int32)
    column_index = numpy.array(columns, dtype=numpy.int32)
    weight = numpy.array(values, dtype=numpy.float32) * inverse[column_index]

    order = numpy.argsort(row_index, kind="stable")
    row_index = row_index[order]
    column_index = column_index[order]
    weight = weight[order]

    squared = numpy.bincount(
        row_index, weights=weight.astype(numpy.float64) ** 2, minlength=total
    )
    norm = numpy.sqrt(numpy.maximum(squared, 1e-12)).astype(numpy.float32)
    weight = (weight / norm[row_index]).astype(numpy.float32)
    return vocabulary, row_index, column_index, weight


class Sparse:
    """The TF-IDF matrix, with just the two products the SVD needs."""

    def __init__(self, rows, columns, values, shape):
        self._rows = rows
        self._columns = columns
        self._values = values
        self.shape = shape

    def dot(self, right: numpy.ndarray) -> numpy.ndarray:
        """Returns `self @ right`, chunked to bound peak memory."""
        result = numpy.zeros(
            (self.shape[0], right.shape[1]), dtype=numpy.float32
        )
        step = 1_000_000
        for start in range(0, len(self._values), step):
            stop = min(start + step, len(self._values))
            columns = self._columns[start:stop]
            numpy.add.at(
                result,
                self._rows[start:stop],
                self._values[start:stop, None] * right[columns],
            )
        return result

    def transpose_dot(self, right: numpy.ndarray) -> numpy.ndarray:
        """Returns `self.T @ right`, one output column at a time.

        `bincount` per dimension beats `numpy.add.at` on the transpose by
        roughly an order of magnitude, and the loop is only `DIMENSIONS` long.
        """
        result = numpy.zeros(
            (self.shape[1], right.shape[1]), dtype=numpy.float32
        )
        for dimension in range(right.shape[1]):
            contribution = self._values * right[self._rows, dimension]
            result[:, dimension] = numpy.bincount(
                self._columns,
                weights=contribution.astype(numpy.float64),
                minlength=self.shape[1],
            )
        return result


def latent_terms(matrix: Sparse) -> numpy.ndarray:
    """Returns the term-to-latent projection, `V` of the truncated SVD."""
    generator = numpy.random.default_rng(SEED)
    sampled = DIMENSIONS + OVERSAMPLING
    basis = matrix.dot(
        generator.standard_normal((matrix.shape[1], sampled)).astype(
            numpy.float32
        )
    )
    for _ in range(POWER_ITERATIONS):
        basis = numpy.linalg.qr(basis)[0].astype(numpy.float32)
        projected = numpy.linalg.qr(matrix.transpose_dot(basis))[0]
        basis = matrix.dot(projected.astype(numpy.float32))
    basis = numpy.linalg.qr(basis)[0].astype(numpy.float32)
    _, _, right = numpy.linalg.svd(
        matrix.transpose_dot(basis).T, full_matrices=False
    )
    return right[:DIMENSIONS].T.astype(numpy.float32)


def normalize(matrix: numpy.ndarray) -> numpy.ndarray:
    """Returns `matrix` with every row scaled to unit length."""
    length = numpy.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / numpy.maximum(length, 1e-9)


def quantize(matrix: numpy.ndarray) -> tuple[bytes, bytes]:
    """Returns int8 codes and the per-row float32 scale that decodes them.

    Per row rather than per matrix: rows are already unit length, but their
    coordinate ranges differ by an order of magnitude, and a shared scale
    would spend most of the int8 range on the widest row.
    """
    peak = numpy.maximum(numpy.abs(matrix).max(axis=1), 1e-9)
    codes = numpy.rint(matrix / peak[:, None] * dense.QUANTIZED_MAX)
    codes = numpy.clip(codes, -dense.QUANTIZED_MAX, dense.QUANTIZED_MAX)
    scale = (peak / dense.QUANTIZED_MAX).astype("<f4")
    return codes.astype(numpy.int8).tobytes(), scale.tobytes()


def write(path: pathlib.Path, asins, vocabulary, documents, terms) -> None:
    """Writes the asset in the layout `submission/src/dense.py` reads."""
    vocabulary_block = "\n".join(vocabulary).encode("utf-8")
    document_codes, document_scales = quantize(documents)
    term_codes, term_scales = quantize(terms)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(
            dense.header(
                dimensions=DIMENSIONS,
                document_count=len(asins),
                term_count=len(vocabulary),
                vocabulary_bytes=len(vocabulary_block),
                fingerprint=dense.fingerprint(asins),
            )
        )
        handle.write(vocabulary_block)
        handle.write(document_codes)
        handle.write(document_scales)
        handle.write(term_codes)
        handle.write(term_scales)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out", default="submission/assets/dense.bin")
    arguments = parser.parse_args()

    started = time.perf_counter()
    asins, documents = read_catalog(pathlib.Path(arguments.catalog))
    vocabulary, rows, columns, values = term_document(documents)
    matrix = Sparse(rows, columns, values, (len(asins), len(vocabulary)))
    print(
        f"{len(asins)} documents, {len(vocabulary)} terms, "
        f"{len(values)} postings, {time.perf_counter() - started:.1f}s"
    )

    terms = latent_terms(matrix)
    document_vectors = normalize(matrix.dot(terms))
    term_vectors = normalize(terms)

    path = pathlib.Path(arguments.out)
    write(path, asins, vocabulary, document_vectors, term_vectors)
    print(
        f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB) "
        f"in {time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
