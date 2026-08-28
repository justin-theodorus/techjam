"""Category-filter parity against the evaluator, plus catalog build checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluator import local_evaluator
from submission.src import catalog as catalog_module
from submission.src.tests import fixtures

CATALOG_PATH = Path("data/catalog.jsonl")

TRICKY_CATEGORY_PATHS = [
    [],
    ["Clothing"],
    ["Clothing, Shoes & Jewelry"],
    ["Clothing, Shoes & Jewelry", "Women", "Shoes, Sneakers"],
    ["Clothing Shoes & Jewelry", "Novelty & More", "Clothing"],
    ["Men", "Accessories", "Belts, Dress Belts"],
    ["  ", "Girls", "Dresses"],
]


class CoarseCategoryParityTest(unittest.TestCase):
    def test_matches_the_evaluator_on_hand_written_edge_cases(self) -> None:
        for values in TRICKY_CATEGORY_PATHS:
            with self.subTest(values=values):
                self.assertEqual(
                    catalog_module.coarse_category(values),
                    local_evaluator.coarse_category(values),
                )

    def test_an_empty_path_falls_back_to_the_evaluators_literal(self) -> None:
        self.assertEqual(catalog_module.coarse_category([]), "clothing item")

    @unittest.skipUnless(CATALOG_PATH.exists(), "run `make data` first")
    def test_matches_the_evaluator_on_the_whole_catalog(self) -> None:
        with CATALOG_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                categories = [
                    str(value) for value in product.get("categories") or []
                ]
                self.assertEqual(
                    catalog_module.coarse_category(categories),
                    local_evaluator.coarse_category(categories),
                )


class BuildTest(unittest.TestCase):
    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = fixtures.write_catalog(Path(root.name))
        self.catalog = catalog_module.build(path)

    def test_buckets_are_ordered_by_descending_popularity(self) -> None:
        pool = self.catalog.bucket(fixtures.SNEAKER_BUCKET)
        self.assertEqual(
            self.catalog.slate_of(pool),
            ["SNEAK_POP", "SNEAK_MID", "SNEAK_RARE"],
        )

    def test_an_unknown_category_yields_an_empty_pool(self) -> None:
        self.assertEqual(self.catalog.bucket("No Such Bucket"), ())
        self.assertEqual(self.catalog.bucket(None), ())

    def test_the_fallback_pool_is_the_coarser_group(self) -> None:
        pool = self.catalog.fallback_pool(fixtures.BOOT_BUCKET)
        self.assertEqual(self.catalog.slate_of(pool), ["BOOT_POP", "BOOT_RARE"])

    def test_description_is_not_indexed(self) -> None:
        self.assertEqual(self.catalog.index.query_ids(["prose"]), frozenset())
        canvas = self.catalog.index.query_ids(["canvas"])
        self.assertNotEqual(canvas, frozenset())

    def test_global_popularity_is_ordered_and_fills_a_slate(self) -> None:
        priors = [self.catalog.prior[index] for index in self.catalog.popular]
        self.assertEqual(priors, sorted(priors, reverse=True))
        self.assertGreaterEqual(len(self.catalog.popular), 10)


if __name__ == "__main__":
    unittest.main()
