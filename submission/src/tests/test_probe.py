from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from submission.src import catalog as catalog_module
from submission.src import dialogue
from submission.src import probe
from submission.src.tests import fixtures
from submission.src import slots


def taxonomy(declared: dict[str, int], documents: int) -> slots.Taxonomy:
    return slots.Taxonomy({}, declared, documents)


CATALOG = taxonomy(
    {
        slots.FEATURE: 90, slots.STYLE: 88, slots.BRAND: 49,
        slots.COLOR: 5, slots.MATERIAL: 5, slots.USE_CASE: 4, slots.SIZE: 2,
    },
    100,
)


def stated(*attributes: str) -> tuple[slots.Slot, ...]:
    return tuple(
        slots.Slot(attribute, f"v{index}", 1)
        for index, attribute in enumerate(attributes)
    )


class DominanceTest(unittest.TestCase):
    """The wildcard's match set contains every specific arm's, by construction.

    The conclusion that it dominates is therefore derived here, not asserted,
    and it survives a catalog whose attributes are distributed differently.
    """

    def test_the_wildcard_is_never_worth_less_than_a_specific_arm(self) -> None:
        scores = probe.expected_yield(dialogue.SessionState(), CATALOG)

        for arm in probe.ARMS:
            with self.subTest(arm=arm):
                self.assertGreaterEqual(scores[probe.WILDCARD], scores[arm])

    def test_it_dominates_under_another_attribute_distribution(self) -> None:
        inverted = taxonomy({slots.SIZE: 95, slots.FEATURE: 2}, 100)
        scores = probe.expected_yield(dialogue.SessionState(), inverted)

        self.assertGreaterEqual(scores[probe.WILDCARD], max(
            scores[arm] for arm in probe.ARMS
        ))

    def test_a_fresh_session_asks_the_wildcard(self) -> None:
        self.assertEqual(
            probe.choose(dialogue.SessionState(), CATALOG), probe.WILDCARD
        )


class DeadArmTest(unittest.TestCase):
    def test_an_attribute_nothing_declares_is_worth_nothing(self) -> None:
        scores = probe.expected_yield(dialogue.SessionState(), CATALOG)

        self.assertEqual(scores[slots.CATEGORY], 0.0)
        self.assertEqual(scores[slots.BUDGET], 0.0)


class DecayTest(unittest.TestCase):
    def test_an_attribute_already_described_is_worth_less(self) -> None:
        fresh = probe.expected_yield(dialogue.SessionState(), CATALOG)
        heard = probe.expected_yield(
            dialogue.SessionState(slots=stated(slots.COLOR)), CATALOG
        )

        self.assertLess(heard[slots.COLOR], fresh[slots.COLOR])

    def test_it_decays_rather_than_zeroing(self) -> None:
        """One question rarely exhausts what a product has to say."""
        heard = probe.expected_yield(
            dialogue.SessionState(slots=stated(slots.COLOR)), CATALOG
        )

        self.assertGreater(heard[slots.COLOR], 0.0)


class RefusalTest(unittest.TestCase):
    def test_a_declined_attribute_is_not_asked_about_again(self) -> None:
        state = dialogue.SessionState(refused=(slots.MATERIAL,))
        scores = probe.expected_yield(state, CATALOG)

        self.assertEqual(scores[slots.MATERIAL], 0.0)

    def test_declining_one_attribute_does_not_end_the_talk(self) -> None:
        state = dialogue.SessionState(refused=(slots.MATERIAL,))

        self.assertEqual(probe.choose(state, CATALOG), probe.WILDCARD)


class ExhaustionTest(unittest.TestCase):
    def test_a_customer_out_of_preferences_is_not_asked_again(self) -> None:
        state = dialogue.SessionState(exhausted=True)

        self.assertIsNone(probe.choose(state, CATALOG))

    def test_every_arm_is_worthless_once_they_are_out(self) -> None:
        scores = probe.expected_yield(
            dialogue.SessionState(exhausted=True), CATALOG
        )

        self.assertEqual(set(scores.values()), {0.0})


class ContractTest(unittest.TestCase):
    ALLOWED = {
        "category", "material", "color", "size", "style", "brand",
        "budget", "feature", "use_case", "other",
    }

    def test_every_arm_is_inside_the_published_enum(self) -> None:
        self.assertTrue(set(probe.ARMS) <= self.ALLOWED)
        self.assertIn(probe.WILDCARD, self.ALLOWED)

    def test_the_choice_is_always_an_allowed_value_or_none(self) -> None:
        states = (
            dialogue.SessionState(),
            dialogue.SessionState(exhausted=True),
            dialogue.SessionState(refused=tuple(probe.ARMS)),
            dialogue.SessionState(slots=stated(*probe.ARMS)),
        )
        for state in states:
            with self.subTest(state=state):
                chosen = probe.choose(state, CATALOG)
                self.assertTrue(chosen is None or chosen in self.ALLOWED)


class SpecificArmTest(unittest.TestCase):
    """Scoring an arm against the pool actually in contention."""

    def setUp(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.catalog = catalog_module.build(fixtures.write_catalog(root))
        self.state = dialogue.SessionState(
            category=fixtures.SNEAKER_BUCKET,
            buckets=(fixtures.SNEAKER_BUCKET,),
        )

    def test_it_names_an_attribute_rather_than_the_wildcard(self) -> None:
        chosen = probe.specific(self.state, self.catalog)

        self.assertIsNotNone(chosen)
        self.assertNotEqual(chosen, probe.WILDCARD)
        self.assertIn(chosen, probe.ARMS)

    def test_an_empty_pool_falls_back_rather_than_guessing(self) -> None:
        nowhere = dialogue.SessionState(category="no such bucket")

        self.assertEqual(
            probe.specific(nowhere, self.catalog), probe.WILDCARD
        )

    def test_a_refused_arm_is_never_chosen(self) -> None:
        chosen = probe.specific(self.state, self.catalog)
        refused = replace(self.state, refused=(chosen,))

        self.assertNotEqual(probe.specific(refused, self.catalog), chosen)

    def test_an_attribute_the_customer_covered_is_asked_about_less(
        self
    ) -> None:
        """`ARM_DECAY` has to move the ordering, not merely the score."""
        chosen = probe.specific(self.state, self.catalog)
        spoken = replace(
            self.state, constraints=_lines_typed(self.catalog, chosen)
        )

        self.assertNotEqual(probe.specific(spoken, self.catalog), chosen)

    def test_a_line_the_customer_already_said_stops_counting(self) -> None:
        """Coverage must fall when the pool has nothing new left to offer."""
        every_line = tuple(self.catalog.offer_ids)
        said = replace(self.state, constraints=every_line)

        self.assertEqual(probe.specific(said, self.catalog), probe.WILDCARD)

    def test_a_thin_pool_falls_back_to_the_wildcard(self) -> None:
        """The whole point of the ratio: never ask what nobody can answer."""
        original = probe.WILDCARD_FALLBACK_RATIO
        probe.WILDCARD_FALLBACK_RATIO = 1.01
        self.addCleanup(
            setattr, probe, "WILDCARD_FALLBACK_RATIO", original
        )

        self.assertEqual(
            probe.specific(self.state, self.catalog), probe.WILDCARD
        )

    def test_no_fallback_ratio_always_names_an_attribute(self) -> None:
        original = probe.WILDCARD_FALLBACK_RATIO
        probe.WILDCARD_FALLBACK_RATIO = 0.0
        self.addCleanup(
            setattr, probe, "WILDCARD_FALLBACK_RATIO", original
        )

        self.assertNotEqual(
            probe.specific(self.state, self.catalog), probe.WILDCARD
        )

    def test_choose_ignores_the_pool_when_the_switch_is_off(self) -> None:
        original = probe.SPECIFIC_ARMS
        probe.SPECIFIC_ARMS = False
        self.addCleanup(setattr, probe, "SPECIFIC_ARMS", original)

        chosen = probe.choose(
            self.state, self.catalog.taxonomy, self.catalog
        )

        self.assertEqual(chosen, probe.WILDCARD)

    def test_an_exhausted_customer_is_not_asked_again(self) -> None:
        spent = replace(self.state, exhausted=True)

        self.assertIsNone(
            probe.choose(spent, self.catalog.taxonomy, self.catalog)
        )


def _lines_typed(catalog, arm: str) -> tuple[str, ...]:
    """Returns every lead line in the catalog that types as `arm`."""
    by_id = {index: value for value, index in catalog.offer_ids.items()}
    return tuple({
        by_id[value_id]
        for row in catalog.offers
        for offered, value_id, _ in row
        if offered == arm
    })


if __name__ == "__main__":
    unittest.main()
