from __future__ import annotations

import unittest

from submission.src import category
from submission.src import dialogue
from submission.src import ranking
from submission.src import slots
from submission.src import understand

RESOLVER = category.build(
    ("Shoes Sneakers", "St. John Dresses", "Novelty Clothing")
)

_BUILDER = slots.TaxonomyBuilder()
for _row in (
    {"Material": "cotton", "Color": "black"},
    {"Material": "leather", "Color": "navy"},
):
    _BUILDER.add(_row, True)
TAXONOMY = _BUILDER.freeze()

BUYING_OPENING = (
    "I'm looking for Shoes Sneakers. A key requirement is: 100% Leather."
)
EXPLORING_OPENING = "I'm looking for Novelty Clothing, but I'm still exploring."
OVERRIDE_OPENING = "I'm looking for St. John Dresses. Ribbed knit cuffs"
PIVOT = "Actually, ignore my earlier preference. What I need is: cotton."
DISCLOSURE = "For that, what matters is: cotton; color: black."
REFUSAL = "I don't have a preference for other; please use your judgment."
EXHAUSTED = "I don't have an additional preference for other."


class StateTest(unittest.TestCase):
    def _drive(
        self, messages: list[str], asked: str | None = "other"
    ) -> dialogue.SessionState:
        state = dialogue.SessionState()
        for message in messages:
            parsed = understand.interpret(message, RESOLVER)
            state = dialogue.update(state, parsed, asked)
        return state

    def test_constraints_accumulate_without_duplicates(self) -> None:
        state = self._drive([
            BUYING_OPENING,
            DISCLOSURE,
            "For that, what matters is: cotton; Imported.",
        ])

        self.assertEqual(
            state.constraints,
            ("100% Leather", "cotton", "color: black", "Imported"),
        )

    def test_category_is_locked_at_turn_one_and_never_revised(self) -> None:
        state = self._drive([EXPLORING_OPENING, BUYING_OPENING])

        self.assertEqual(state.category, "Novelty Clothing")
        self.assertEqual(state.pool_keys, ("Novelty Clothing",))

    def test_the_pivot_erases_prior_constraints(self) -> None:
        state = self._drive([BUYING_OPENING, DISCLOSURE, PIVOT])

        self.assertEqual(state.constraints, ("cotton",))
        self.assertIn("100% Leather", state.superseded)
        self.assertIn("color: black", state.superseded)
        self.assertTrue(state.pivoted)
        self.assertEqual(state.scenario, dialogue.OVERRIDE)

    def test_constraints_after_the_pivot_still_accumulate(self) -> None:
        state = self._drive([
            BUYING_OPENING, PIVOT, "For that, what matters is: Imported."
        ])

        self.assertEqual(state.constraints, ("cotton", "Imported"))

    def test_boundary_refusal_does_not_pollute_the_query(self) -> None:
        state = self._drive([EXPLORING_OPENING, REFUSAL])

        self.assertEqual(state.scenario, dialogue.BOUNDARY)
        self.assertEqual(state.query_text, "")

    def test_a_refusal_is_attributed_to_the_attribute_that_was_asked(
        self,
    ) -> None:
        state = self._drive([EXPLORING_OPENING, REFUSAL], asked="material")

        self.assertEqual(state.refused, ("material",))

    def test_a_refusal_with_nothing_asked_records_no_attribute(self) -> None:
        state = self._drive([EXPLORING_OPENING, REFUSAL], asked=None)

        self.assertEqual(state.refused, ())

    def test_exhaustion_is_recorded_once_the_simulator_runs_dry(self) -> None:
        state = self._drive([EXPLORING_OPENING, EXHAUSTED])

        self.assertTrue(state.exhausted)
        self.assertEqual(state.query_text, "")

    def test_update_returns_a_new_state(self) -> None:
        before = dialogue.SessionState()
        parsed = understand.interpret(DISCLOSURE, RESOLVER)
        after = dialogue.update(before, parsed)

        self.assertEqual(before.constraints, ())
        self.assertEqual(after.constraints, ("cotton", "color: black"))
        self.assertIsNot(before, after)


class NegationTest(unittest.TestCase):
    """A refusal must leave the positive query and enter the negative one."""

    def _state(self, *constraints: str) -> dialogue.SessionState:
        taxonomy = slots.TaxonomyBuilder()
        taxonomy.add({"Material": "cotton"}, True)
        return dialogue.update(
            dialogue.SessionState(),
            dialogue.ParsedTurn(constraints=constraints),
            taxonomy=taxonomy.freeze(),
        )

    def test_a_refusal_stays_out_of_the_positive_query(self) -> None:
        state = self._state("black", "not cotton")

        self.assertEqual(state.query_text, "black")

    def test_a_refusal_reaches_the_negative_query_without_its_cue(self) -> None:
        state = self._state("black", "not cotton")

        self.assertEqual(state.excluded_text, "cotton")

    def test_the_raw_wording_is_kept_so_the_reply_can_quote_it(self) -> None:
        state = self._state("not cotton")

        self.assertEqual(state.constraints, ("not cotton",))

    def test_a_session_with_no_refusal_has_no_negative_query(self) -> None:
        state = self._state("black", "leather")

        self.assertEqual(state.excluded_text, "")
        self.assertEqual(state.query_text, "black leather")


class ShownTest(unittest.TestCase):
    """What has been served, and the one thing that un-serves it."""

    def test_the_shown_set_accumulates_across_turns(self) -> None:
        state = dialogue.SessionState().with_slate(("A", "B"))
        state = state.with_slate(("B", "C"))

        self.assertEqual(state.shown, frozenset({"A", "B", "C"}))

    def test_an_ordinary_turn_carries_the_shown_set_forward(self) -> None:
        state = dialogue.SessionState().with_slate(("A", "B"))

        folded = dialogue.update(state, dialogue.ParsedTurn(constraints=("x",)))

        self.assertEqual(folded.shown, frozenset({"A", "B"}))

    def test_a_pivot_clears_it_because_nothing_was_ever_tested(self) -> None:
        """`override_applied` gates scoring, so a pre-pivot impression was
        never checked against the new target and may well be it."""
        state = dialogue.SessionState().with_slate(("A", "B"))

        folded = dialogue.update(state, dialogue.ParsedTurn(pivot=True))

        self.assertEqual(folded.shown, frozenset())


class PoolKeysTest(unittest.TestCase):
    def test_a_state_with_no_category_retrieves_from_nothing(self) -> None:
        self.assertEqual(dialogue.SessionState().pool_keys, ())

    def test_a_category_alone_is_its_own_pool(self) -> None:
        state = dialogue.SessionState(category="Shoes Sneakers")

        self.assertEqual(state.pool_keys, ("Shoes Sneakers",))

    def test_resolved_buckets_win_over_the_primary_category(self) -> None:
        state = dialogue.SessionState(
            category="Shoes Sneakers",
            buckets=("Shoes Sneakers", "Novelty Clothing"),
        )

        self.assertEqual(
            state.pool_keys, ("Shoes Sneakers", "Novelty Clothing")
        )


class HeadSizeTest(unittest.TestCase):
    """`head_size` is pure in the state, so it needs no catalog."""

    def test_a_fresh_session_serves_only_the_best_guess(self) -> None:
        state = dialogue.SessionState(turn=1)

        self.assertEqual(ranking.head_size(state, 10), ranking.HEAD_SIZE)

    def test_exhaustion_opens_the_slate(self) -> None:
        state = dialogue.SessionState(turn=1, exhausted=True)

        self.assertEqual(ranking.head_size(state, 10), 10)

    def test_full_disclosure_opens_the_slate(self) -> None:
        state = dialogue.SessionState(
            turn=1, constraints=tuple(str(i) for i in range(4))
        )

        self.assertEqual(ranking.head_size(state, 10), 10)

    def test_a_customer_who_never_finishes_still_opens_the_slate(self) -> None:
        state = dialogue.SessionState(turn=ranking.MAX_DEFER_TURNS + 1)

        self.assertEqual(ranking.head_size(state, 10), 10)

    def test_the_head_never_exceeds_the_requested_size(self) -> None:
        state = dialogue.SessionState(turn=1)

        self.assertEqual(ranking.head_size(state, 1), 1)


class TurnCounterTest(unittest.TestCase):
    def test_the_counter_starts_at_zero_and_advances_per_fold(self) -> None:
        state = dialogue.SessionState()

        self.assertEqual(state.turn, 0)
        for expected in (1, 2, 3):
            state = dialogue.update(state, dialogue.ParsedTurn())
            self.assertEqual(state.turn, expected)

    def test_a_pivot_advances_the_counter_rather_than_resetting_it(
        self,
    ) -> None:
        state = dialogue.SessionState(turn=3, constraints=("cotton",))
        pivoted = dialogue.update(
            state, dialogue.ParsedTurn(constraints=("leather",), pivot=True)
        )

        self.assertEqual(pivoted.turn, 4)


class ScopedExhaustionTest(unittest.TestCase):
    """"Nothing more about X" is not "nothing more at all" (findings 3.37)."""

    EMPTY = dialogue.ParsedTurn(exhausted=True)

    def test_an_empty_specific_answer_retires_only_that_arm(self) -> None:
        state = dialogue.update(
            dialogue.SessionState(), self.EMPTY, asked="material"
        )

        self.assertFalse(state.exhausted)
        self.assertIn("material", state.refused)

    def test_an_empty_wildcard_answer_still_ends_the_asking(self) -> None:
        state = dialogue.update(
            dialogue.SessionState(), self.EMPTY, asked=dialogue.WILDCARD
        )

        self.assertTrue(state.exhausted)

    def test_the_attribute_the_customer_named_wins_over_what_was_asked(
        self
    ) -> None:
        """The reply says which thing they are out of; believe the reply."""
        named = dialogue.ParsedTurn(
            exhausted=True, exhausted_arm=dialogue.WILDCARD
        )

        state = dialogue.update(
            dialogue.SessionState(), named, asked="material"
        )

        self.assertTrue(state.exhausted)
        self.assertNotIn("material", state.refused)

    def test_an_empty_answer_to_no_question_still_ends_it(self) -> None:
        state = dialogue.update(dialogue.SessionState(), self.EMPTY)

        self.assertTrue(state.exhausted)

    def test_exhaustion_once_reached_is_never_taken_back(self) -> None:
        spent = dialogue.SessionState(exhausted=True)

        state = dialogue.update(spent, self.EMPTY, asked="color")

        self.assertTrue(state.exhausted)

    def test_switching_the_scope_off_restores_the_old_reading(self) -> None:
        original = dialogue.SCOPED_EXHAUSTION
        dialogue.SCOPED_EXHAUSTION = False
        self.addCleanup(
            setattr, dialogue, "SCOPED_EXHAUSTION", original
        )

        state = dialogue.update(
            dialogue.SessionState(), self.EMPTY, asked="material"
        )

        self.assertTrue(state.exhausted)


if __name__ == "__main__":
    unittest.main()


class TargetedOverrideTest(unittest.TestCase):
    """A replacement erases what it contradicts, not everything.

    Worth +0.016 overall and it converts every override session on the public
    set, 0.900 to 1.000 hit@10 (findings 3.26). The earlier comparison only ever
    tested erase-everything against keep-everything.
    """

    def _pivot(self, before: tuple[str, ...], after: tuple[str, ...]):
        taxonomy = TAXONOMY
        state = dialogue.SessionState()
        state = dialogue.update(
            state, dialogue.ParsedTurn(constraints=before), None, taxonomy
        )
        return dialogue.update(
            state,
            dialogue.ParsedTurn(constraints=after, pivot=True),
            None,
            taxonomy,
        )

    def test_a_contradicted_attribute_is_replaced(self) -> None:
        state = self._pivot(("cotton",), ("leather",))

        self.assertEqual(state.constraints, ("leather",))
        self.assertEqual(state.superseded, ("cotton",))

    def test_an_untouched_attribute_survives(self) -> None:
        state = self._pivot(("cotton", "black"), ("leather",))

        self.assertIn("black", state.constraints)
        self.assertIn("leather", state.constraints)
        self.assertEqual(state.superseded, ("cotton",))

    def test_the_replacement_is_recorded_as_the_pivot_turn(self) -> None:
        state = self._pivot(("cotton",), ("leather",))

        self.assertEqual(state.pivot_turn, 2)
        self.assertTrue(state.pivoted)

    def test_total_erasure_is_still_reachable_for_the_ablation(self) -> None:
        original = dialogue.TARGETED_OVERRIDE
        try:
            dialogue.TARGETED_OVERRIDE = False
            state = self._pivot(("cotton", "black"), ("leather",))
            self.assertEqual(state.constraints, ("leather",))
        finally:
            dialogue.TARGETED_OVERRIDE = original


class SlotTypingTest(unittest.TestCase):
    def test_constraints_are_typed_as_they_arrive(self) -> None:
        state = dialogue.update(
            dialogue.SessionState(),
            dialogue.ParsedTurn(constraints=("cotton", "black")),
            None,
            TAXONOMY,
        )

        self.assertEqual(
            [slot.attribute for slot in state.slots],
            [slots.MATERIAL, slots.COLOR],
        )

    def test_without_a_taxonomy_a_session_still_tracks_what_was_said(
        self,
    ) -> None:
        state = dialogue.update(
            dialogue.SessionState(),
            dialogue.ParsedTurn(constraints=("cotton",)),
        )

        self.assertEqual(state.constraints, ("cotton",))
        self.assertEqual(state.slots[0].attribute, slots.DEFAULT)
