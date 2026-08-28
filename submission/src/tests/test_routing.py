from __future__ import annotations

import unittest

from submission.src import dialogue
from submission.src import ranking
from submission.src import routing
from submission.src import slots


class RouteChoiceTest(unittest.TestCase):
    def test_a_session_with_nothing_said_is_discovery(self) -> None:
        self.assertEqual(routing.choose(dialogue.SessionState()).name,
                         routing.DISCOVERY)

    def test_one_constraint_is_still_discovery(self) -> None:
        """A bare material word matches half its bucket (findings 3.18)."""
        state = dialogue.SessionState(constraints=("cotton",))

        self.assertEqual(routing.choose(state).name, routing.DISCOVERY)

    def test_several_constraints_earn_the_precision_route(self) -> None:
        state = dialogue.SessionState(constraints=("cotton", "color: black"))

        self.assertEqual(routing.choose(state).name, routing.PRECISION)

    def test_a_refusal_routes_to_boundary(self) -> None:
        state = dialogue.SessionState(refused=(slots.MATERIAL,))

        self.assertEqual(routing.choose(state).name, routing.BOUNDARY)

    def test_a_redirect_routes_to_recovery_and_outranks_the_rest(self) -> None:
        state = dialogue.SessionState(
            pivoted=True, pivot_turn=3, refused=(slots.MATERIAL,),
            constraints=("cotton", "color: black"),
        )

        self.assertEqual(routing.choose(state).name, routing.RECOVERY)


class NeutralityTest(unittest.TestCase):
    """Every route ships at the shared constants.

    Route-conditional `alpha` was built and measured: it gains 0.014 on the dev
    half and loses on the held-out half, so it is reported as a negative result
    rather than shipped. These assertions are what keep it that way by accident
    becoming deliberate.
    """

    def test_no_route_deviates_on_alpha(self) -> None:
        states = (
            dialogue.SessionState(),
            dialogue.SessionState(constraints=("a", "b")),
            dialogue.SessionState(refused=("material",)),
            dialogue.SessionState(pivoted=True, pivot_turn=3),
        )
        for state in states:
            with self.subTest(state=state):
                self.assertEqual(routing.choose(state).alpha, ranking.ALPHA)

    def test_no_route_deviates_on_the_turn_budget(self) -> None:
        states = (
            dialogue.SessionState(),
            dialogue.SessionState(constraints=("a", "b")),
            dialogue.SessionState(pivoted=True, pivot_turn=4),
        )
        for state in states:
            with self.subTest(state=state):
                self.assertEqual(
                    routing.choose(state).defer_turns, ranking.MAX_DEFER_TURNS
                )


class RecoveryRestartTest(unittest.TestCase):
    def test_the_restart_switch_extends_the_budget_from_the_redirect(
        self,
    ) -> None:
        state = dialogue.SessionState(pivoted=True, pivot_turn=4)
        original = routing.RECOVERY_RESTART
        try:
            routing.RECOVERY_RESTART = 1
            self.assertEqual(
                routing.choose(state).defer_turns,
                ranking.MAX_DEFER_TURNS + 4,
            )
        finally:
            routing.RECOVERY_RESTART = original


if __name__ == "__main__":
    unittest.main()
