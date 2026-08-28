from __future__ import annotations

import unittest

from submission.src import slots


def taxonomy(rows: list[tuple[dict, bool]]) -> slots.Taxonomy:
    builder = slots.TaxonomyBuilder()
    for details, has_features in rows:
        builder.add(details, has_features)
    return builder.freeze()


CATALOG = taxonomy([
    ({"Color": "black", "Material": "cotton", "Size": "large"}, True),
    ({"Color": "navy", "Material": "leather", "Size": "large"}, True),
    ({"Color": "black", "Occasion": "wedding"}, True),
    ({"Department": "womens", "Package Dimensions": "9 x 4 x 1 inches"}, False),
])


class KeyFamilyTest(unittest.TestCase):
    def test_a_key_names_the_attribute_its_value_describes(self) -> None:
        cases = (
            ("Fabric type: Polyester", slots.MATERIAL),
            ("Color: turquoise", slots.COLOR),
            ("Occasion: graduation", slots.USE_CASE),
            ("Neck Style: crew", slots.STYLE),
            ("Closure Type: magnetic", slots.FEATURE),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(CATALOG.classify(value), expected)

    def test_the_longest_matching_key_family_wins(self) -> None:
        """`fit type` is a size, not a `type` of something else."""
        self.assertEqual(CATALOG.classify("Fit Type: relaxed"), slots.SIZE)

    def test_logistics_keys_name_no_attribute(self) -> None:
        for value in (
            "Package Dimensions: 9 x 4 x 1 inches",
            "Item Weight: 3.2 ounces",
            "Date First Available: March 2019",
        ):
            with self.subTest(value=value):
                self.assertEqual(CATALOG.classify(value), slots.DEFAULT)


class LearnedVocabularyTest(unittest.TestCase):
    def test_a_bare_value_is_typed_from_what_the_catalog_calls_it(self) -> None:
        self.assertEqual(CATALOG.classify("black"), slots.COLOR)
        self.assertEqual(CATALOG.classify("cotton"), slots.MATERIAL)
        self.assertEqual(CATALOG.classify("large"), slots.SIZE)

    def test_a_stray_disagreement_does_not_disqualify_a_value(self) -> None:
        """Catalog rows are dirty; a plurality is the honest reading."""
        learned = taxonomy(
            [({"Size": "large"}, False)] * 9 + [({"Color": "large"}, False)]
        )

        self.assertEqual(learned.classify("large"), slots.SIZE)

    def test_a_genuinely_split_value_falls_back(self) -> None:
        learned = taxonomy(
            [({"Size": "solo"}, False)] * 5 + [({"Color": "solo"}, False)] * 5
        )

        self.assertEqual(learned.classify("solo"), slots.DEFAULT)

    def test_prose_is_not_learned_as_vocabulary(self) -> None:
        long_value = "x" * (slots.MAX_VALUE_LENGTH + 1)
        learned = taxonomy([({"Color": long_value}, False)])

        self.assertEqual(learned.classify(long_value), slots.DEFAULT)

    def test_an_unrecognized_string_is_a_feature(self) -> None:
        self.assertEqual(CATALOG.classify("Ribbed knit cuffs"), slots.DEFAULT)


class BudgetTest(unittest.TestCase):
    def test_a_stated_price_is_a_budget_however_it_is_phrased(self) -> None:
        for value in (
            "budget around $24.99", "under 30 dollars", "less than $15",
            "cheaper than $40",
        ):
            with self.subTest(value=value):
                self.assertEqual(CATALOG.classify(value), slots.BUDGET)

    def test_a_number_that_is_not_a_price_is_not_a_budget(self) -> None:
        self.assertNotEqual(CATALOG.classify("Size: 10 medium"), slots.BUDGET)


class PrevalenceTest(unittest.TestCase):
    def test_prevalence_counts_products_not_mentions(self) -> None:
        """Two of four products declare a size; `large` appearing twice is one
        product each, not four mentions."""
        self.assertAlmostEqual(CATALOG.prevalence(slots.SIZE), 0.5)
        self.assertAlmostEqual(CATALOG.prevalence(slots.COLOR), 0.75)

    def test_a_product_with_bullets_can_be_described_by_one(self) -> None:
        self.assertAlmostEqual(CATALOG.prevalence(slots.FEATURE), 0.75)

    def test_an_attribute_nothing_declares_has_no_prevalence(self) -> None:
        self.assertEqual(CATALOG.prevalence(slots.CATEGORY), 0.0)
        self.assertEqual(CATALOG.prevalence(slots.BUDGET), 0.0)

    def test_an_empty_catalog_does_not_divide_by_zero(self) -> None:
        self.assertEqual(taxonomy([]).prevalence(slots.COLOR), 0.0)


class SlotTest(unittest.TestCase):
    def test_typing_a_list_keeps_order_and_stamps_the_turn(self) -> None:
        typed = CATALOG.slots(("cotton", "black", "Ribbed cuffs"), 3)

        self.assertEqual(
            [slot.attribute for slot in typed],
            [slots.MATERIAL, slots.COLOR, slots.DEFAULT],
        )
        self.assertEqual([slot.turn for slot in typed], [3, 3, 3])
        self.assertEqual(typed[0].value, "cotton")


if __name__ == "__main__":
    unittest.main()
