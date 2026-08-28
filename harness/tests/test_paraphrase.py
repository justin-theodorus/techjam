from __future__ import annotations

import random
import unittest

from evaluator import local_evaluator
from harness import paraphrase


def card(hard: list[str], soft: list[str]) -> dict:
    return {"target_category": "t", "hard_constraints": hard, "soft_preferences": soft}


def sample(scenario: str, hard: list[str], soft: list[str], override: dict | None = None) -> dict:
    row = {
        "sample_id": "s1",
        "scenario_type": scenario,
        "intent_card": card(hard, soft),
        "behavior": {"scenario_type": scenario},
    }
    if override:
        row["behavior"]["override"] = override
    return row


class CleanStyleParityTest(unittest.TestCase):
    """The `clean` column must be byte-identical to the shipped simulator.

    Every other column is only interpretable relative to this one, so parity here
    is the load-bearing assumption of the whole gate. If the evaluator's templates
    ever change, these assertions fail rather than the gate silently drifting.
    """

    def setUp(self) -> None:
        self.patched = paraphrase.patched(paraphrase.CLEAN)

    def test_browsing_opening_matches(self) -> None:
        row = sample("browsing", ["cotton"], ["breathable"])
        disclosed: set[str] = set()
        expected = local_evaluator.initial_message(row, "Shirts Tops", set())

        self.assertEqual(self.patched["initial_message"](row, "Shirts Tops", disclosed), expected)

    def test_buying_opening_matches_and_discloses_the_same_constraint(self) -> None:
        row = sample("buying", ["cotton", "color: black"], ["breathable"])
        reference: set[str] = set()
        expected = local_evaluator.initial_message(row, "Shirts Tops", reference)
        disclosed: set[str] = set()

        actual = self.patched["initial_message"](row, "Shirts Tops", disclosed)

        self.assertEqual(actual, expected)
        self.assertEqual(disclosed, reference)

    def test_override_opening_matches(self) -> None:
        row = sample(
            "intent_override", ["cotton"], ["breathable"],
            {"turn": 3, "old_value": "breathable", "new_value": "cotton", "message": "m"},
        )
        expected = local_evaluator.initial_message(row, "Shirts Tops", set())

        self.assertEqual(self.patched["initial_message"](row, "Shirts Tops", set()), expected)

    def test_every_customer_reply_branch_matches(self) -> None:
        cases = (
            ("boundary", "material", set(), False),
            ("browsing", None, set(), False),
            ("browsing", "other", set(), False),
            ("browsing", "not_an_enum_value", set(), False),
            ("browsing", "budget", set(), False),
            ("browsing", "other", {"cotton", "breathable"}, False),
        )
        for scenario, attribute, disclosed, used in cases:
            with self.subTest(scenario=scenario, attribute=attribute):
                row = sample(scenario, ["cotton"], ["breathable"])
                expected = local_evaluator.customer_reply(row, attribute, set(disclosed), used)
                actual = self.patched["customer_reply"](row, attribute, set(disclosed), used)

                self.assertEqual(actual, expected)

    def test_customer_reply_discloses_the_same_constraints(self) -> None:
        row = sample("browsing", ["cotton", "color: black"], ["breathable", "Imported"])
        reference: set[str] = set()
        local_evaluator.customer_reply(row, "other", reference, False)
        disclosed: set[str] = set()

        self.patched["customer_reply"](row, "other", disclosed, False)

        self.assertEqual(disclosed, reference)

    def test_pivot_message_matches(self) -> None:
        built = self.patched["behavior_for"](
            "intent_override", card(["cotton"], ["breathable"]), random.Random(1)
        )
        expected = local_evaluator.behavior_for(
            "intent_override", card(["cotton"], ["breathable"]), random.Random(1)
        )

        self.assertEqual(built["override"]["message"], expected["override"]["message"])

    def test_intent_card_is_untouched_when_not_substituting(self) -> None:
        product = {"title": "Tee", "features": ["100% Cotton"], "price": 9.99}

        self.assertEqual(
            self.patched["intent_card"](product), local_evaluator.intent_card(product)
        )


class OverrideTurnPreservationTest(unittest.TestCase):
    def test_every_style_keeps_the_pivot_on_the_turn_the_seed_chose(self) -> None:
        """A style may change the wording. It may not change when the pivot lands.

        `behavior_for` draws the turn from a seeded generator; the patched version
        delegates to the original before touching the message, so the draw order,
        and therefore the turn, is preserved.
        """
        for style in paraphrase.STYLES:
            with self.subTest(style=style.name):
                expected = local_evaluator.behavior_for(
                    "intent_override", card(["cotton"], ["breathable"]), random.Random("seed")
                )["override"]["turn"]
                actual = paraphrase.patched(style)["behavior_for"](
                    "intent_override", card(["cotton"], ["breathable"]), random.Random("seed")
                )["override"]["turn"]

                self.assertEqual(actual, expected)


class PerturbationTest(unittest.TestCase):
    def test_reworded_shares_no_opening_literal_with_the_shipped_template(self) -> None:
        row = sample("browsing", ["cotton"], ["breathable"])
        reworded = paraphrase.patched(paraphrase.REWORDED)["initial_message"](
            row, "Shirts Tops", set()
        )

        self.assertNotIn("I'm looking for", reworded)
        self.assertIn("Shirts Tops", reworded)

    def test_filler_keeps_the_payload_and_breaks_the_prefix_anchor(self) -> None:
        row = sample("browsing", ["cotton"], ["breathable"])
        filled = paraphrase.patched(paraphrase.FILLER)["initial_message"](
            row, "Shirts Tops", set()
        )

        self.assertFalse(filled.startswith("I'm looking for"))
        self.assertIn("I'm looking for Shirts Tops", filled)

    def test_substitution_rewrites_vocabulary_without_emptying_a_constraint(self) -> None:
        product = {"title": "Tee", "features": ["100% Cotton", "Machine wash cold"]}
        swapped = paraphrase.patched(paraphrase.SYNONYM)["intent_card"](product)
        values = [*swapped["hard_constraints"], *swapped["soft_preferences"]]

        self.assertTrue(all(values))
        self.assertNotEqual(values, local_evaluator.intent_card(product)["hard_constraints"])

    def test_a_style_is_reproducible_across_runs(self) -> None:
        row = sample("browsing", ["cotton"], ["breathable"])
        first = paraphrase.patched(paraphrase.REWORDED)["initial_message"](row, "Tops", set())
        second = paraphrase.patched(paraphrase.REWORDED)["initial_message"](row, "Tops", set())

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
