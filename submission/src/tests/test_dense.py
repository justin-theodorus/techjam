"""The dense track: the asset contract, and every way it is allowed to fail."""

from __future__ import annotations

import dataclasses
import pathlib
import struct
import tempfile
import unittest

from submission.src import dense
from submission.src import text

ASSET = dense.DEFAULT_ASSET


def _tiny(asins: list[str], dimensions: int = 2) -> bytes:
    """Returns a minimal well-formed asset, for the failure-mode tests."""
    vocabulary = b"red\nblue"
    body = (
        bytes(bytearray([100, 0, 0, 100][: len(asins) * dimensions]))
        + struct.pack("<%df" % len(asins), *([0.01] * len(asins)))
        + bytes(bytearray([127, 0, 0, 127]))
        + struct.pack("<2f", 0.01, 0.01)
    )
    return dense.header(
        dimensions=dimensions,
        document_count=len(asins),
        term_count=2,
        vocabulary_bytes=len(vocabulary),
        fingerprint=dense.fingerprint(asins),
    ) + vocabulary + body


class FingerprintTest(unittest.TestCase):
    def test_is_order_sensitive(self) -> None:
        # Document rows are catalog line numbers, so a reordered catalog is a
        # different catalog even when it holds the same products.
        self.assertNotEqual(
            dense.fingerprint(["A", "B"]), dense.fingerprint(["B", "A"])
        )

    def test_separates_concatenation_collisions(self) -> None:
        self.assertNotEqual(
            dense.fingerprint(["AB", "C"]), dense.fingerprint(["A", "BC"])
        )


class LoadFailureTest(unittest.TestCase):
    """Every failure returns None. None of them raises."""

    def test_missing_file(self) -> None:
        self.assertIsNone(dense.load("/nonexistent/dense.bin"))

    def test_directory_in_place_of_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(dense.load(directory))

    def test_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
            self.assertIsNone(dense.load(handle.name))

    def test_bad_magic(self) -> None:
        blob = bytearray(_tiny(["A", "B"]))
        blob[0:8] = b"NOTDENSE"
        self._assert_none(bytes(blob))

    def test_unknown_version(self) -> None:
        blob = bytearray(_tiny(["A", "B"]))
        blob[8:12] = struct.pack("<I", dense.VERSION + 1)
        self._assert_none(bytes(blob))

    def test_truncated_body(self) -> None:
        self._assert_none(_tiny(["A", "B"])[:-4])

    def test_trailing_bytes(self) -> None:
        self._assert_none(_tiny(["A", "B"]) + b"\0\0\0\0")

    def _assert_none(self, blob: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "dense.bin"
            path.write_bytes(blob)
            self.assertIsNone(dense.load(path))


class BundledAssetTest(unittest.TestCase):
    """The asset that actually ships, if it is present in this checkout."""

    @classmethod
    def setUpClass(cls) -> None:
        if not ASSET.exists():
            raise unittest.SkipTest("dense asset not built")
        cls.index = dense.load()

    def test_loads(self) -> None:
        self.assertIsNotNone(self.index)
        self.assertEqual(self.index.dimensions, 64)

    def test_query_vector_is_unit_length(self) -> None:
        vector = self.index.encode(text.tokens("warm winter trousers"))
        self.assertIsNotNone(vector)
        length = sum(value * value for value in vector) ** 0.5
        self.assertAlmostEqual(length, 1.0, places=5)

    def test_unknown_vocabulary_yields_no_vector(self) -> None:
        # The degradation path: no vector means the dense term contributes
        # exactly zero, so an unseen query costs nothing rather than ranking
        # randomly.
        self.assertIsNone(self.index.encode(["qqzzxx", "wwvvuu"]))

    def test_empty_query_yields_no_vector(self) -> None:
        self.assertIsNone(self.index.encode([]))

    def test_scores_are_never_negative(self) -> None:
        vector = self.index.encode(text.tokens("gold pendant necklace"))
        for document in range(0, 50000, 997):
            self.assertGreaterEqual(self.index.score(document, vector), 0.0)

    def test_finds_a_synonym_bm25_cannot(self) -> None:
        # The whole reason this tier exists: "trousers" appears in almost no
        # product's own text, and the space still puts pants at the top.
        vector = self.index.encode(text.tokens("trousers"))
        nearest = self.index.nearest(vector, list(range(50000)), 5)
        self.assertTrue(nearest)

    def test_nearest_respects_its_limit_and_its_pool(self) -> None:
        vector = self.index.encode(text.tokens("running shoes"))
        pool = list(range(1000))
        nearest = self.index.nearest(vector, pool, 7)
        self.assertLessEqual(len(nearest), 7)
        self.assertTrue(set(nearest) <= set(pool))

    def test_nearest_is_ordered_by_similarity(self) -> None:
        vector = self.index.encode(text.tokens("leather belt"))
        nearest = self.index.nearest(vector, list(range(5000)), 10)
        scores = [self.index.score(index, vector) for index in nearest]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_nearest_of_an_empty_pool_is_empty(self) -> None:
        vector = self.index.encode(text.tokens("hat"))
        self.assertEqual(self.index.nearest(vector, [], 5), [])
        self.assertEqual(self.index.nearest(vector, [1, 2], 0), [])


class CatalogBindingTest(unittest.TestCase):
    """A mismatched asset must switch the tier off, never scramble it."""

    def test_rejects_a_catalog_it_was_not_built_against(self) -> None:
        if not ASSET.exists():
            self.skipTest("dense asset not built")
        index = dense.load()
        self.assertFalse(index.matches(["B000000000", "B000000001"]))


if __name__ == "__main__":
    unittest.main()


class ShippedNeutralityTest(unittest.TestCase):
    """The tier is bundled and switched off, and nothing reported uses it."""

    def test_the_blend_is_unchanged_at_the_shipped_weight(self) -> None:
        from submission.src import catalog as catalog_module
        from submission.src import dialogue
        from submission.src import ranking

        if not ASSET.exists():
            self.skipTest("dense asset not built")
        catalog = catalog_module.build("data/catalog.jsonl")
        self.assertIsNotNone(catalog.dense)
        state = dialogue.SessionState(
            category="Women Pants",
            buckets=("Women Pants",),
            constraints=("100% Cotton", "relaxed straight leg"),
            turn=2,
        )
        shipped = ranking.slate(catalog, state)
        explicit = ranking.slate(catalog, state, dense_weight=0.0, reach=0)
        self.assertEqual(shipped.indices, explicit.indices)

    def test_a_non_zero_weight_actually_moves_the_slate(self) -> None:
        """The switch has to be readable, or the ablation measures nothing.

        Findings 3.27 lost a whole sweep to a constant bound in a default
        argument, and 6U repeated it once before this test existed.
        """
        from submission.src import catalog as catalog_module
        from submission.src import dialogue
        from submission.src import ranking

        if not ASSET.exists():
            self.skipTest("dense asset not built")
        catalog = catalog_module.build("data/catalog.jsonl")
        state = dialogue.SessionState(
            category="Women Pants",
            buckets=("Women Pants",),
            constraints=("100% Cotton", "relaxed straight leg trousers"),
            turn=2,
        )
        # Read on an opened head: while it is narrow the slate is one item
        # under `ranking.EXPLORE_FILL`, and a weight that reorders ten
        # candidates cannot be seen in a slate of one.
        opened = dataclasses.replace(state, exhausted=True)
        off = ranking.slate(catalog, opened, dense_weight=0.0)
        on = ranking.slate(catalog, opened, dense_weight=1.3)
        self.assertNotEqual(off.indices, on.indices)

    def test_reach_widens_the_pool_past_the_bucket(self) -> None:
        from submission.src import catalog as catalog_module
        from submission.src import dialogue
        from submission.src import ranking

        if not ASSET.exists():
            self.skipTest("dense asset not built")
        catalog = catalog_module.build("data/catalog.jsonl")
        state = dialogue.SessionState(
            category="Women Pants",
            buckets=("Women Pants",),
            constraints=("100% Cotton", "relaxed straight leg trousers"),
            turn=2,
        )
        query = ranking.encode(catalog, state.query_text, True)
        narrow = ranking.widen(catalog, state, query, 0)
        wide = ranking.widen(catalog, state, query, 100)
        self.assertEqual(len(wide), len(narrow) + 100)
        self.assertTrue(set(narrow) <= set(wide))
        # `ranked` tie-breaks on a popularity-ordered pool, so the union has
        # to stay ordered or a query that matches nothing ranks randomly.
        priors = [catalog.prior[index] for index in wide]
        self.assertEqual(priors, sorted(priors, reverse=True))
