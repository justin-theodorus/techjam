"""Tests for the phrase index, including parity with the reference cleaning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from techjam.submission.src import catalog as catalog_module
from techjam.submission.src import phrases
from techjam.submission.src.tests import fixtures

try:
    from techjam.evaluator import local_evaluator
except ImportError:  # The submission bundle ships without organizer files.
    local_evaluator = None


class NormalizeTest(unittest.TestCase):
    CASES = (
        "  100% Cotton  ",
        "cotton\tblend\nlining",
        "-;,.Button closure.",
        "Imported",
        "",
        "   ",
        "a" * 250,
        "Machine Wash;",
    )

    def test_collapsing_and_stripping_match_the_documented_form(self) -> None:
        self.assertEqual(phrases.normalize("  100%   Cotton "), "100% Cotton")
        self.assertEqual(
            phrases.normalize("-;,.Button closure."), "Button closure"
        )
        self.assertEqual(phrases.normalize("   "), "")

    def test_a_long_phrase_is_truncated_then_right_stripped(self) -> None:
        cleaned = phrases.normalize("x " * 200)
        self.assertLessEqual(len(cleaned), phrases.MAX_CONSTRAINT_LENGTH)
        self.assertEqual(cleaned, cleaned.rstrip())

    @unittest.skipUnless(local_evaluator, "evaluator not importable")
    def test_parity_with_the_evaluators_own_cleaning(self) -> None:
        # The whole route depends on producing byte-identical strings to the
        # ones the simulator utters, so this is the load-bearing assertion.
        for case in self.CASES:
            with self.subTest(case=case):
                self.assertEqual(
                    phrases.normalize(case),
                    local_evaluator._clean_constraint(
                        case, phrases.MAX_CONSTRAINT_LENGTH
                    ),
                )


class FlattenTest(unittest.TestCase):
    def test_a_dict_becomes_key_colon_value_pairs(self) -> None:
        self.assertEqual(
            phrases.flatten({"Material": "Cotton", "Empty": ""}),
            ["Material: Cotton"],
        )

    def test_a_list_drops_empty_entries(self) -> None:
        self.assertEqual(phrases.flatten(["a", "", None, "b"]), ["a", "b"])

    def test_a_scalar_becomes_one_entry_and_none_becomes_zero(self) -> None:
        self.assertEqual(phrases.flatten("solo"), ["solo"])
        self.assertEqual(phrases.flatten(None), [])


class PhraseIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = fixtures.write_catalog(Path(root.name))
        self.catalog = catalog_module.build(path)
        self.index = self.catalog.phrases

    def _evidence(self, asin: str, *constraints: str) -> float:
        return self.index.evidence(
            self.catalog.asins.index(asin), self.index.query_ids(constraints)
        )

    def test_an_unknown_constraint_yields_no_query(self) -> None:
        self.assertEqual(
            self.index.query_ids(("never seen here",)), frozenset()
        )

    def test_a_phrase_unique_to_one_product_is_worth_its_full_weight(
        self,
    ) -> None:
        self.assertEqual(self._evidence("DEEP_15", "hemp footbed marker"), 1.0)

    def test_a_product_without_the_phrase_scores_nothing(self) -> None:
        self.assertEqual(self._evidence("DEEP_00", "hemp footbed marker"), 0.0)

    def test_a_shared_phrase_is_worth_less_than_a_unique_one(self) -> None:
        shared = self._evidence("DEEP_00", "synthetic footbed")
        self.assertGreater(shared, 0.0)
        self.assertLess(
            shared, self._evidence("DEEP_15", "hemp footbed marker")
        )

    def test_an_empty_query_scores_nothing(self) -> None:
        self.assertEqual(self._evidence("DEEP_15"), 0.0)

    def test_constraints_are_matched_after_normalization(self) -> None:
        self.assertEqual(
            self._evidence("DEEP_15", "  hemp   footbed marker. "), 1.0
        )


if __name__ == "__main__":
    unittest.main()
