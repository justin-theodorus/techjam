from __future__ import annotations

import unittest

from submission.src import dialogue
from submission.src import probe
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


if __name__ == "__main__":
    unittest.main()
