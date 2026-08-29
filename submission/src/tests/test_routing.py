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

    def test_no_route_deviates_on_the_dense_track(self) -> None:
        """The dense track is bundled, loaded, and off on every route.

        `None` means "defer to `ranking.DENSE_WEIGHT`", which is zero, so no
        route runs the dense retriever and no reported number depends on the
        bundled asset (findings 3.35).
        """
        states = (
            dialogue.SessionState(),
            dialogue.SessionState(constraints=("a",)),
            dialogue.SessionState(constraints=("a", "b")),
            dialogue.SessionState(refused=("material",)),
            dialogue.SessionState(pivoted=True, pivot_turn=3),
        )
        for state in states:
            with self.subTest(state=state):
                route = routing.choose(state)
                self.assertIsNone(route.dense_weight)
                self.assertEqual(route.reach, 0)

    def test_the_shared_dense_weight_is_zero(self) -> None:
        self.assertEqual(ranking.DENSE_WEIGHT, 0.0)
        self.assertEqual(ranking.DENSE_NEGATION_WEIGHT, 0.0)

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


class RouteDiversityTest(unittest.TestCase):
    """Spreading the slate per route: built, measured, shipped neutral.

    The brief's dual-track claim taken literally, and the only route-conditional
    setting whose argument does not reduce to `alpha` in a disguise, because it
    changes how the slate is selected rather than how the pool is scored. It
    costs 0.035 to 0.042 on the public 200 at every weight measured, which is
    barely less than the unconditional weight findings 3.27 rejected
    (findings 3.43).
    """

    def test_no_route_deviates_on_diversification(self) -> None:
        """`None` defers to `ranking.DIVERSITY`, which is zero."""
        states = (
            dialogue.SessionState(),
            dialogue.SessionState(constraints=("a",)),
            dialogue.SessionState(constraints=("a", "b")),
            dialogue.SessionState(refused=("material",)),
            dialogue.SessionState(pivoted=True, pivot_turn=3),
        )
        for state in states:
            with self.subTest(state=state):
                self.assertIsNone(routing.choose(state).diversity)
        self.assertIsNone(routing.DISCOVERY_DIVERSITY)
        self.assertEqual(ranking.DIVERSITY, 0.0)

    def test_the_switch_reaches_only_the_discovery_route(self) -> None:
        original = routing.DISCOVERY_DIVERSITY
        try:
            routing.DISCOVERY_DIVERSITY = 0.5
            reached = (
                dialogue.SessionState(),
                dialogue.SessionState(constraints=("cotton",)),
            )
            for state in reached:
                with self.subTest(state=state):
                    self.assertEqual(routing.choose(state).diversity, 0.5)
            spared = (
                dialogue.SessionState(constraints=("a", "b")),
                dialogue.SessionState(refused=("material",)),
                dialogue.SessionState(pivoted=True, pivot_turn=2),
            )
            for state in spared:
                with self.subTest(state=state):
                    self.assertIsNone(routing.choose(state).diversity)
        finally:
            routing.DISCOVERY_DIVERSITY = original

    def test_the_route_gate_is_not_a_scenario_split(self) -> None:
        """A buying session opens inside this gate, so it acts from turn 2.

        Buying discloses one constraint before turn 1 and the precision
        threshold is two, so every scenario takes discovery on the opening
        turn. Reading this switch as "browsing gets variety" is wrong.
        """
        state = dialogue.SessionState(constraints=("cotton",))

        self.assertGreater(routing.MIN_PRECISION_CONSTRAINTS, 1)
        self.assertEqual(routing.choose(state).name, routing.DISCOVERY)


if __name__ == "__main__":
    unittest.main()
