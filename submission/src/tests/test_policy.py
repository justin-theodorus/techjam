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

        This is the defect findings 3.46 measured: a scoped exhaustion put the
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


if __name__ == "__main__":
    unittest.main()
