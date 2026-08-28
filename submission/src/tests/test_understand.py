from __future__ import annotations

import unittest

from submission.src import category
from submission.src import dialogue
from submission.src import understand

RESOLVER = category.build(
    ("Shoes Sneakers", "St. John Dresses", "Novelty Clothing", "Tops Blouses")
)


def general(message: str) -> dialogue.ParsedTurn:
    """Reads a message with the template shortcut switched off."""
    return understand.interpret(message, RESOLVER, fast_path=False)


class TemplateTest(unittest.TestCase):
    """The shipped templates, read by the fast path."""

    def test_buying_opening_yields_category_and_constraint(self) -> None:
        parsed = understand.interpret(
            "I'm looking for Shoes Sneakers. "
            "A key requirement is: 100% Leather.",
            RESOLVER,
        )

        self.assertEqual(parsed.category, "Shoes Sneakers")
        self.assertEqual(parsed.constraints, ("100% Leather",))
        self.assertEqual(parsed.scenario_hint, dialogue.BUYING)
        self.assertEqual(parsed.confidence, understand.EXACT)

    def test_exploring_opening_yields_only_the_category(self) -> None:
        parsed = understand.interpret(
            "I'm looking for Novelty Clothing, but I'm still exploring.",
            RESOLVER,
        )

        self.assertEqual(parsed.category, "Novelty Clothing")
        self.assertEqual(parsed.constraints, ())
        self.assertEqual(parsed.scenario_hint, dialogue.EXPLORING)

    def test_override_opening_splits_on_the_longest_known_bucket(self) -> None:
        parsed = understand.interpret(
            "I'm looking for St. John Dresses. Ribbed knit cuffs", RESOLVER
        )

        self.assertEqual(parsed.category, "St. John Dresses")
        self.assertEqual(parsed.constraints, ("Ribbed knit cuffs",))
        self.assertEqual(parsed.scenario_hint, dialogue.OVERRIDE)

    def test_pivot_is_flagged_and_carries_the_new_constraint(self) -> None:
        parsed = understand.interpret(
            "Actually, ignore my earlier preference. What I need is: cotton.",
            RESOLVER,
        )

        self.assertTrue(parsed.pivot)
        self.assertEqual(parsed.constraints, ("cotton",))
        self.assertEqual(parsed.act, dialogue.ACT_RESET)

    def test_disclosure_yields_both_constraints(self) -> None:
        parsed = understand.interpret(
            "For that, what matters is: cotton; color: black.", RESOLVER
        )

        self.assertEqual(parsed.constraints, ("cotton", "color: black"))
        self.assertEqual(parsed.act, dialogue.ACT_DISCLOSE)

    def test_content_free_templates_yield_no_constraint_text(self) -> None:
        refusal = understand.interpret(
            "I don't have a preference for other; please use your judgment.",
            RESOLVER,
        )
        exhausted = understand.interpret(
            "I don't have an additional preference for other.", RESOLVER
        )
        rejection = understand.interpret(
            "Those options are not quite right yet. "
            "Ask me about one specific attribute.",
            RESOLVER,
        )

        self.assertTrue(refusal.boundary_refusal)
        self.assertTrue(exhausted.exhausted)
        for parsed in (refusal, exhausted, rejection):
            self.assertEqual(parsed.constraints, ())
            self.assertIsNone(parsed.category)

    def test_a_non_string_message_yields_an_empty_turn(self) -> None:
        empty = dialogue.ParsedTurn()

        self.assertEqual(understand.interpret(None, RESOLVER), empty)
        self.assertEqual(understand.interpret("   ", RESOLVER), empty)


class TemplateValidationTest(unittest.TestCase):
    """A template that matches but yields nonsense must defer, not guess.

    This is the punctuation failure: the prefix matched, the suffix did not, and
    the remainder of the sentence was returned as a category. An unknown
    category is an empty pool, which is worse than the wider pool the general
    path would have produced (findings 3.24).
    """

    def test_a_broken_suffix_falls_through_to_the_general_path(self) -> None:
        parsed = understand.interpret(
            "I'm looking for Tops Blouses but I'm still exploring", RESOLVER
        )

        self.assertEqual(parsed.category, "Tops Blouses")
        self.assertEqual(parsed.confidence, understand.CUED)

    def test_an_unknown_category_never_reaches_the_state(self) -> None:
        parsed = understand.interpret(
            "I'm looking for Nonexistent Bucket, but I'm still exploring.",
            RESOLVER,
        )

        self.assertEqual(parsed.buckets, ())


class GeneralPathTest(unittest.TestCase):
    """Everything above, with the template shortcut switched off."""

    def test_a_reworded_browsing_opening_yields_the_category(self) -> None:
        parsed = general(
            "Show me some Tops Blouses, I'm just browsing for now."
        )

        self.assertEqual(parsed.buckets, ("Tops Blouses",))
        self.assertEqual(parsed.constraints, ())
        self.assertEqual(parsed.scenario_hint, dialogue.EXPLORING)

    def test_a_reworded_buying_opening_yields_the_requirement(self) -> None:
        parsed = general("I need Shoes Sneakers. It has to be 100% Leather.")

        self.assertEqual(parsed.buckets, ("Shoes Sneakers",))
        self.assertEqual(parsed.constraints, ("100% Leather",))

    def test_the_rightmost_cue_introduces_the_requirement(self) -> None:
        """`I need` precedes the category; `must be` introduces the payload."""
        parsed = general("I need Shoes Sneakers and it must be leather")

        self.assertEqual(parsed.constraints, ("leather",))

    def test_an_uncued_requirement_is_taken_from_what_the_category_omits(
        self,
    ) -> None:
        parsed = general("I want Tops Blouses. color: black")

        self.assertEqual(parsed.buckets, ("Tops Blouses",))
        self.assertEqual(parsed.constraints, ("color: black",))

    def test_naming_only_the_category_yields_no_requirement(self) -> None:
        parsed = general("I want Tops Blouses.")

        self.assertEqual(parsed.constraints, ())

    def test_padding_does_not_hide_the_category(self) -> None:
        parsed = general(
            "Hi there. I'm looking for Tops Blouses, "
            "but I'm still exploring. Thanks!"
        )

        self.assertEqual(parsed.buckets, ("Tops Blouses",))
        self.assertEqual(parsed.constraints, ())


class ActDetectionTest(unittest.TestCase):
    def test_reset_phrasings(self) -> None:
        messages = (
            "Actually, ignore my earlier preference. What I need is: cotton.",
            "Actually, please ignore my earlier preference.",
            "Actually, scratch that. I really need cotton.",
            "Wait, forget what I said. It has to be cotton.",
            "Change of mind, I actually need cotton.",
            "Actually ignore my earlier preference, what I need is cotton",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertTrue(general(message).pivot)

    def test_reset_extracts_the_replacement_requirement(self) -> None:
        for message in (
            "Actually, scratch that. I really need cotton.",
            "Wait, forget what I said. It has to be cotton.",
            "Change of mind, I actually need cotton.",
        ):
            with self.subTest(message=message):
                self.assertEqual(general(message).constraints, ("cotton",))

    def test_a_reset_with_no_replacement_still_erases(self) -> None:
        parsed = general("Actually, please ignore my earlier preference.")

        self.assertTrue(parsed.pivot)
        self.assertEqual(parsed.constraints, ())

    def test_refusal_phrasings(self) -> None:
        messages = (
            "I don't have a preference for material; please use your judgment.",
            "No strong feelings on material, you decide.",
            "material doesn't matter to me, whatever you think is best.",
            "I'm not fussed about material.",
        )
        for message in messages:
            with self.subTest(message=message):
                parsed = general(message)
                self.assertTrue(parsed.boundary_refusal)
                self.assertEqual(parsed.constraints, ())

    def test_exhaustion_phrasings_are_not_read_as_refusals(self) -> None:
        messages = (
            "I don't have an additional preference for material.",
            "Nothing else on material.",
            "That's all I can think of for material.",
            "No more preferences there.",
        )
        for message in messages:
            with self.subTest(message=message):
                parsed = general(message)
                self.assertTrue(parsed.exhausted)
                self.assertFalse(parsed.boundary_refusal)

    def test_rejection_phrasings_yield_nothing(self) -> None:
        for message in (
            "Those options are not quite right yet. Ask about one attribute.",
            "Hmm, none of those look right. Ask me something specific.",
            "Not quite. What do you want to know?",
        ):
            with self.subTest(message=message):
                parsed = general(message)
                self.assertEqual(parsed.constraints, ())
                self.assertEqual(parsed.act, dialogue.ACT_REJECT)

    def test_a_reset_wins_over_the_need_cue_it_contains(self) -> None:
        """Every reset sentence also states a need; the reset is the event."""
        parsed = general("Actually, scratch that. I really need cotton.")

        self.assertEqual(parsed.act, dialogue.ACT_RESET)


class DisclosureTest(unittest.TestCase):
    def test_reworded_disclosures_split_on_their_joiner(self) -> None:
        cases = (
            ("What I care about is cotton and color: black.",
             ("cotton", "color: black")),
            ("Mainly cotton and color: black.", ("cotton", "color: black")),
            ("It should be cotton.", ("cotton",)),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(general(message).constraints, expected)

    def test_a_comma_inside_a_constraint_is_not_a_separator(self) -> None:
        """Real constraints carry commas, so splitting on them over-splits."""
        parsed = general("What I care about is Machine Wash, Tumble Dry")

        self.assertEqual(parsed.constraints, ("Machine Wash, Tumble Dry",))


if __name__ == "__main__":
    unittest.main()
