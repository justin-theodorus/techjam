from __future__ import annotations

import unittest

from submission.src import dialogue
from submission.src import policy
from submission.src import slots


class SelectionOrderTest(unittest.TestCase):
    """The order is the design, so each rule is pinned against the one below.

    Every assertion here is a claim about precedence rather than about a single
    condition: a state that satisfies two rules must resolve to the higher one,
    which is the only part of a priority list that can silently rot.
    """

    def test_a_session_with_nothing_said_is_discovery(self) -> None:
        self.assertEqual(policy.select(dialogue.SessionState()),
                         policy.DISCOVERY)

    def test_one_constraint_is_still_discovery(self) -> None:
        state = dialogue.SessionState(constraints=("cotton",), turn=1)

        self.assertEqual(policy.select(state), policy.DISCOVERY)

    def test_two_constraints_earn_precision(self) -> None:
        state = dialogue.SessionState(
            constraints=("cotton", "color: black"), turn=2
        )

        self.assertEqual(policy.select(state), policy.PRECISION)

    def test_a_decline_outranks_precision(self) -> None:
        state = dialogue.SessionState(
            constraints=("cotton", "color: black"),
            declined=(slots.MATERIAL,), turn=3,
        )

        self.assertEqual(policy.select(state), policy.BOUNDARY)

    def test_running_dry_on_one_arm_is_not_a_decline(self) -> None:
        """`refused` merges both readings; only `declined` is a refusal.

        This is the defect measurements 3.46 measured: a scoped exhaustion put the
        spent arm into `refused`, and reading that as a refusal sent 74% of
        `compound_hard` turns down the boundary branch.
        """
        state = dialogue.SessionState(
            constraints=("cotton", "color: black"),
            refused=(slots.MATERIAL,), turn=3,
        )

        self.assertEqual(policy.select(state), policy.PRECISION)

    def test_repeated_empty_answers_outrank_a_decline(self) -> None:
        state = dialogue.SessionState(
            constraints=("cotton", "color: black"),
            declined=(slots.MATERIAL,),
            idle=dialogue.STAGNATION_TURNS, turn=4,
        )

        self.assertEqual(policy.select(state), policy.STAGNATION)

    def test_one_empty_answer_is_not_yet_stagnation(self) -> None:
        state = dialogue.SessionState(constraints=("cotton",), idle=1, turn=3)

        self.assertEqual(policy.select(state), policy.DISCOVERY)

    def test_running_out_of_turns_outranks_stagnation(self) -> None:
        state = dialogue.SessionState(
            constraints=("cotton",), idle=3, turn=policy.COVERAGE_TURN
        )

        self.assertEqual(policy.select(state), policy.COVERAGE)

    def test_an_exhausted_customer_is_coverage_whatever_the_turn(self) -> None:
        state = dialogue.SessionState(exhausted=True, turn=2)

        self.assertEqual(policy.select(state), policy.COVERAGE)

    def test_a_redirect_outranks_everything(self) -> None:
        state = dialogue.SessionState(
            pivoted=True, pivot_turn=3, exhausted=True,
            declined=(slots.MATERIAL,), idle=4, turn=9,
        )

        self.assertEqual(policy.select(state), policy.RECOVERY)


class DecisionReadinessTest(unittest.TestCase):
    def test_vague_exploration_is_low_readiness(self) -> None:
        state = dialogue.SessionState(
            category="Furniture", scenario=dialogue.EXPLORING, turn=1
        )

        self.assertLess(
            policy.decision_readiness(state, candidate_count=500),
            policy.PARTIAL_READINESS_THRESHOLD,
        )

    def test_concrete_current_constraints_are_high_readiness(self) -> None:
        state = dialogue.SessionState(
            category="Furniture",
            constraints=("Budget: under $500", "Size: queen"),
            slots=(
                slots.Slot(slots.BUDGET, "Budget: under $500", 2,
                           strength=slots.HARD),
                slots.Slot(slots.SIZE, "Size: queen", 2,
                           strength=slots.HARD),
            ),
            turn=2,
            confidence=1.0,
        )

        self.assertGreaterEqual(
            policy.decision_readiness(state, candidate_count=40),
            policy.PRECISION_READINESS_THRESHOLD,
        )

    def test_decision_exposes_readiness_for_traceability(self) -> None:
        state = dialogue.SessionState(
            category="Furniture",
            constraints=("Budget: under $500", "Size: queen"),
            slots=(
                slots.Slot(slots.BUDGET, "Budget: under $500", 2,
                           strength=slots.HARD),
                slots.Slot(slots.SIZE, "Size: queen", 2,
                           strength=slots.HARD),
            ),
            turn=2,
            confidence=1.0,
        )

        decision = policy.decide(state, candidate_count=40)

        self.assertEqual(decision.name, policy.PRECISION)
        self.assertGreaterEqual(
            decision.readiness, policy.PRECISION_READINESS_THRESHOLD
        )


class ReadinessRecurrenceTest(unittest.TestCase):
    """`D_t = 0.7 * current + 0.3 * D_{t-1}`, the self-evolving update."""

    def _decisive(self, turn: int) -> dialogue.SessionState:
        return dialogue.SessionState(
            category="Furniture",
            constraints=("Budget: under $500", "Size: queen"),
            slots=(
                slots.Slot(slots.BUDGET, "Budget: under $500", turn,
                           strength=slots.HARD),
                slots.Slot(slots.SIZE, "Size: queen", turn,
                           strength=slots.HARD),
            ),
            turn=turn,
            confidence=1.0,
        )

    def test_the_carried_estimate_is_weighted_at_three_tenths(self) -> None:
        state = self._decisive(2)
        current = policy._current_readiness(state, 40, 0)

        readiness = policy.decision_readiness(
            state, candidate_count=40, previous=0.0
        )

        self.assertAlmostEqual(readiness, round(0.7 * current, 3), places=3)

    def test_a_low_prior_is_not_erased_by_one_decisive_turn(self) -> None:
        """Most of the way, but not all of it: that is what 0.3 buys."""
        state = self._decisive(2)

        cold = policy.decision_readiness(state, candidate_count=40,
                                         previous=0.0)
        warm = policy.decision_readiness(state, candidate_count=40,
                                         previous=1.0)

        self.assertLess(cold, warm)
        self.assertAlmostEqual(warm - cold, 0.3, places=2)

    def test_readiness_climbs_across_a_narrowing_session(self) -> None:
        vague = dialogue.SessionState(
            category="Furniture", scenario=dialogue.EXPLORING, turn=1
        )
        first = policy.decision_readiness(vague, candidate_count=500)

        second = policy.decision_readiness(
            self._decisive(2), candidate_count=40, previous=first
        )

        self.assertLess(first, policy.PARTIAL_READINESS_THRESHOLD)
        self.assertGreater(second, first)

    def test_readiness_falls_back_when_the_customer_stalls(self) -> None:
        """The score is the turn's, so it has to be able to go down."""
        stalled = dialogue.SessionState(
            category="Furniture",
            constraints=("Budget: under $500",),
            slots=(
                slots.Slot(slots.BUDGET, "Budget: under $500", 2,
                           strength=slots.HARD),
            ),
            declined=("material",),
            idle=2,
            turn=4,
        )

        self.assertLess(
            policy.decision_readiness(stalled, candidate_count=500,
                                      previous=0.9),
            0.9,
        )

    def test_an_absent_prior_is_derived_rather_than_zero(self) -> None:
        state = self._decisive(2)

        self.assertGreater(
            policy.decision_readiness(state, candidate_count=40),
            policy.decision_readiness(state, candidate_count=40, previous=0.0),
        )


class UrgencyTermTest(unittest.TestCase):
    """The readiness term that reads deadline language off a constraint."""

    def test_deadline_phrasing_reads_as_urgency(self) -> None:
        for phrase in ("need it urgently", "ASAP please", "by friday",
                       "within 2 weeks", "in 3 days", "last-minute gift",
                       "as soon as possible", "right away", "by tomorrow"):
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(policy._URGENCY_RE.search(phrase))

    def test_product_vocabulary_is_not_urgency(self) -> None:
        """The trap the first draft fell into: catalogue nouns are not dates."""
        for phrase in ("moving blankets", "ships in 2 business days",
                       "arrives well packed", "shipped from the US",
                       "soon to be a classic", "queen size", "under $500",
                       "waterproof material", "delivery bag"):
            with self.subTest(phrase=phrase):
                self.assertIsNone(policy._URGENCY_RE.search(phrase))

    def test_urgency_raises_readiness(self) -> None:
        def state(value: str) -> dialogue.SessionState:
            return dialogue.SessionState(
                category="Furniture",
                constraints=(value,),
                slots=(slots.Slot(slots.SIZE, value, 2,
                                  strength=slots.HARD),),
                turn=2,
            )

        self.assertGreater(
            policy.decision_readiness(state("Size: queen, needed by friday"),
                                      candidate_count=120, previous=0.4),
            policy.decision_readiness(state("Size: queen"),
                                      candidate_count=120, previous=0.4),
        )


class HybridTieTest(unittest.TestCase):
    """One hard constraint over a small pool, with readiness just under the
    precision threshold: the real crossover band, not an invented one."""

    def _crossover(self) -> dialogue.SessionState:
        return dialogue.SessionState(
            category="Furniture",
            constraints=("Budget: under $500",),
            slots=(
                slots.Slot(slots.BUDGET, "Budget: under $500", 2,
                           strength=slots.HARD),
            ),
            turn=2,
            confidence=1.0,
        )

    def test_a_close_second_is_reported_as_a_tie(self) -> None:
        decision = policy.decide(self._crossover(), 40, 0, 0.5)

        self.assertLessEqual(decision.margin, policy.HYBRID_MARGIN)
        self.assertTrue(decision.hybrid)
        self.assertEqual(decision.runner_up, policy.PRECISION)
        self.assertTrue(decision.hybrid)
        self.assertNotEqual(decision.runner_up, "")
        self.assertNotEqual(decision.runner_up, decision.name)

    def test_a_clear_winner_is_not_a_tie(self) -> None:
        state = dialogue.SessionState(
            category="Furniture", pivoted=True, pivot_turn=3, turn=3
        )

        decision = policy.decide(state)

        self.assertEqual(decision.name, policy.RECOVERY)
        self.assertFalse(decision.hybrid)

    def test_framing_ships_neutral(self) -> None:
        """The tie changes the trace, not the turn, until a sweep moves it."""
        decision = policy.decide(self._crossover(), 40, 0, 0.5)

        self.assertTrue(decision.hybrid)
        self.assertEqual(policy.framing(decision), decision.name)

    def test_the_shipped_agent_does_not_steer_on_readiness(self) -> None:
        """`READINESS_STEERS` ships off; readiness reports, it does not pick.

        Measured, not assumed: on `compound_hard`'s 1514 turns, steering flips
        zero policies, because the precision/discovery margin is a median ~1.5
        and the bump is worth at most 0.75. See `policy.READINESS_STEERS`.
        """
        self.assertFalse(policy.READINESS_STEERS)
        state = self._crossover()

        cold = policy.decide(state, 40, 0, 0.0)
        warm = policy.decide(state, 40, 0, 1.0)

        self.assertGreater(warm.readiness, cold.readiness)
        self.assertEqual(warm.name, cold.name)

    def test_the_steering_arm_works_when_switched_on(self) -> None:
        """The sweep arm has to be able to move something, or it measures air."""
        state = self._crossover()
        original = policy.READINESS_STEERS
        policy.READINESS_STEERS = True
        try:
            tied = policy.decide(state, 40, 0, 0.5)
            resolved = policy.decide(state, 40, 0, 1.0)
        finally:
            policy.READINESS_STEERS = original

        self.assertEqual(tied.name, policy.DISCOVERY)
        self.assertEqual(resolved.name, policy.PRECISION)


if __name__ == "__main__":
    unittest.main()
