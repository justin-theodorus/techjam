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


class PolarityTest(unittest.TestCase):
    """Reading a refusal, and refusing to read one that is not there."""

    def test_a_spoken_refusal_is_read_and_its_cue_stripped(self) -> None:
        for value, term in (
            ("not polyester", "polyester"),
            ("without leather", "leather"),
            ("avoid silk", "silk"),
            ("other than wool", "wool"),
            ("don't want spandex", "spandex"),
        ):
            with self.subTest(value=value):
                self.assertEqual(slots.polarity(value), (True, term))

    def test_a_cue_inside_a_clause_does_not_flip_it(self) -> None:
        value = "moisture wicking, not bulky"

        self.assertEqual(slots.polarity(value), (False, value))

    def test_an_attribute_named_negatively_is_not_a_refusal(self) -> None:
        """This catalog spells 239 `Non-Polarized` and 192 `No Closure
        closure`, 3 of which reach the public 200 (findings 3.31)."""
        for value in ("Non-Polarized", "No Closure closure", "No-Tie Laces",
                      "no wool"):
            with self.subTest(value=value):
                self.assertEqual(slots.polarity(value), (False, value))

    def test_a_refusal_inside_a_keyed_pair_keeps_its_key(self) -> None:
        self.assertEqual(
            slots.polarity("Fabric Type: not cotton"),
            (True, "Fabric Type: cotton"),
        )

    def test_a_cue_with_nothing_after_it_is_not_a_refusal(self) -> None:
        self.assertEqual(slots.polarity("not"), (False, "not"))

    def test_a_refusal_is_typed_by_what_it_refuses(self) -> None:
        """Typing it as the majority class would hide it from the targeted
        override, which supersedes by attribute."""
        self.assertEqual(CATALOG.classify("not cotton"), slots.MATERIAL)
        self.assertEqual(CATALOG.classify("cotton"), slots.MATERIAL)

    def test_typing_records_the_polarity_on_the_slot(self) -> None:
        typed = CATALOG.slots(("not cotton", "black"), turn=1)

        self.assertEqual([slot.negated for slot in typed], [True, False])
        self.assertEqual(typed[0].value, "not cotton")


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


class ClassifyTextTest(unittest.TestCase):
    """Typing free text, which `classify` cannot do (findings 3.38)."""

    def setUp(self) -> None:
        builder = slots.TaxonomyBuilder()
        for _ in range(8):
            builder.add({"Material": "Cotton", "Color": "Navy"}, True)
            builder.add({"Material": "Polyester", "Color": "Black"}, True)
        self.taxonomy = builder.freeze()

    def test_a_value_named_inside_a_sentence_is_still_typed(self) -> None:
        """The case `classify` misses: a bullet mentions, it does not state."""
        self.assertEqual(
            self.taxonomy.classify("95% Cotton, 5% Spandex"), slots.FEATURE
        )
        self.assertEqual(
            self.taxonomy.classify_text("95% Cotton, 5% Spandex"),
            slots.MATERIAL,
        )

    def test_text_naming_nothing_learned_falls_back(self) -> None:
        self.assertEqual(
            self.taxonomy.classify_text("Imported"), slots.DEFAULT
        )

    def test_a_stated_budget_outranks_any_other_word(self) -> None:
        self.assertEqual(
            self.taxonomy.classify_text("cotton, budget around $25"),
            slots.BUDGET,
        )

    def test_a_refusal_is_typed_by_what_it_refuses(self) -> None:
        self.assertEqual(
            self.taxonomy.classify_text("not cotton"), slots.MATERIAL
        )

    def test_the_vocabulary_is_learned_rather_than_declared(self) -> None:
        """Rename the attribute and the typing follows it."""
        builder = slots.TaxonomyBuilder()
        for _ in range(8):
            builder.add({"Color": "Cotton"}, True)
        renamed = builder.freeze()

        self.assertEqual(renamed.classify_text("100% cotton"), slots.COLOR)


if __name__ == "__main__":
    unittest.main()
