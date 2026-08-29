from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from techjam.submission.src import bm25
from techjam.submission.src import catalog as catalog_module
from techjam.submission.src import dialogue
from techjam.submission.src import ranking
from techjam.submission.src.tests import fixtures


class Bm25Test(unittest.TestCase):
    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = fixtures.write_catalog(Path(root.name))
        self.catalog = catalog_module.build(path)

    def _score(self, asin: str, *tokens: str) -> float:
        index = self.catalog.asins.index(asin)
        return self.catalog.index.score(
            index, self.catalog.index.query_ids(list(tokens))
        )

    def test_repetition_is_discounted_without_being_eliminated(self) -> None:
        # TF_REPEATER says "merino" eleven times across 55 tokens, TF_EXACT
        # once across three. Raw counting would pay the repeater 11x. The
        # shipped constants pay it under 1.3x, which is small enough that it
        # cannot overturn a real prior gap. It is not zero, and that is the
        # known cost of holding B off the floor; see the note in bm25.py.
        ratio = self._score("TF_REPEATER", "merino") / self._score(
            "TF_EXACT", "merino"
        )
        self.assertLess(ratio, 1.5)

    def test_a_query_term_absent_from_the_catalog_is_dropped(self) -> None:
        self.assertEqual(
            self.catalog.index.query_ids(["zzzzqqqq"]), frozenset()
        )

    def test_an_empty_query_scores_zero(self) -> None:
        self.assertEqual(self._score("TF_EXACT"), 0.0)

    def test_a_rarer_term_outscores_a_common_one(self) -> None:
        self.assertGreater(
            self._score("SNEAK_RARE", "hemp"),
            self._score("SNEAK_MID", "upper"),
        )

    def test_an_empty_catalog_cannot_be_frozen(self) -> None:
        with self.assertRaises(ValueError):
            bm25.Bm25Builder().freeze()

    def test_the_shipped_constants_stay_off_the_textbook_defaults(
        self,
    ) -> None:
        # Pinned deliberately: 1.2 / 0.75 costs 0.030 of TechnicalScore, and
        # K1 at zero leaks popularity into the lexical half (findings 3.22).
        self.assertEqual((bm25.K1, bm25.B), (0.6, 0.3))


if __name__ == "__main__":
    unittest.main()
