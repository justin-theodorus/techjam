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
        state = dialogue.SessionState(declined=(slots.MATERIAL,))

        self.assertEqual(routing.choose(state).name, routing.BOUNDARY)

    def test_running_dry_on_one_arm_is_not_a_refusal(self) -> None:
        """A scoped exhaustion retires an arm; it does not decline one.

        `dialogue.SCOPED_EXHAUSTION` puts the spent arm into `refused` so the
        probe stops asking about it. Reading that as a refusal routed 74% of
        `compound_hard` turns to boundary and left the precision branch firing
        on 1.6% of them (findings 3.46).
        """
        state = dialogue.SessionState(
            refused=(slots.MATERIAL,),
            constraints=("cotton", "color: black"),
        )

        self.assertEqual(routing.choose(state).name, routing.PRECISION)

    def test_a_redirect_routes_to_recovery_and_outranks_the_rest(self) -> None:
        state = dialogue.SessionState(
            pivoted=True, pivot_turn=3, refused=(slots.MATERIAL,),
            constraints=("cotton", "color: black"),
        )

        self.assertEqual(routing.choose(state).name, routing.RECOVERY)


class PolicyScoringTest(unittest.TestCase):
    def test_policy_scores_explain_the_chosen_route(self) -> None:
        state = dialogue.SessionState(constraints=("cotton", "color: black"))
        route = routing.choose(state)
        scores = dict(route.policy_scores)

        self.assertEqual(route.name, routing.PRECISION)
        self.assertGreater(scores[routing.PRECISION],
                           scores[routing.DISCOVERY])
        self.assertGreater(route.policy_confidence, 0.0)
        self.assertGreater(route.policy_margin, 0.0)

    def test_policy_can_change_as_session_state_changes(self) -> None:
        states = (
            dialogue.SessionState(),
            dialogue.SessionState(constraints=("cotton", "color: black")),
            dialogue.SessionState(declined=(slots.MATERIAL,)),
            dialogue.SessionState(pivoted=True, pivot_turn=4),
        )
        names = [routing.choose(state).name for state in states]

        self.assertEqual(
            names,
            [
                routing.DISCOVERY,
                routing.PRECISION,
                routing.BOUNDARY,
                routing.RECOVERY,
            ],
        )

    def test_a_long_vague_session_can_route_to_stagnation(self) -> None:
        shown = frozenset(f"ASIN_{index}" for index in range(30))
        state = dialogue.SessionState(turn=6, shown=shown)

        route = routing.choose(
            state, candidate_count=500, previous_contenders=10
        )

        self.assertEqual(route.name, routing.STAGNATION)

    def test_an_exhausted_session_routes_to_coverage(self) -> None:
        state = dialogue.SessionState(
            turn=3,
            exhausted=True,
            constraints=("cotton", "color: black"),
        )

        self.assertEqual(routing.choose(state).name, routing.COVERAGE)


class NeutralityTest(unittest.TestCase):
    """Every route ships at the shared constants but the deferral window.

    Route-conditional `alpha` was built and measured: it gains 0.014 on the dev
    half and loses on the held-out half, so it is reported as a negative result
    rather than shipped. These assertions are what keep it that way by accident
    becoming deliberate.

    The one deliberate exception is the deferral pair, which ships ordered on
    an argument about shoppers rather than a sweep result, and costs nothing:
    see `test_only_the_two_ordered_routes_deviate_on_the_turn_budget`.
    """

    def test_no_route_deviates_on_alpha(self) -> None:
        states = (
            dialogue.SessionState(),
            dialogue.SessionState(constraints=("a", "b")),
            dialogue.SessionState(declined=("material",)),
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
            dialogue.SessionState(declined=("material",)),
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

    def test_only_the_two_ordered_routes_deviate_on_the_turn_budget(
        self,
    ) -> None:
        """Discovery opens sooner than precision; nothing else deviates.

        The pair is the one place the policy layer reaches retrieval, and it
        is ordered rather than tuned: a customer who has told us least must not
        be the one narrowed hardest.
        """
        self.assertLess(routing.DISCOVERY_DEFER, routing.PRECISION_DEFER)
        self.assertEqual(routing.PRECISION_DEFER, ranking.MAX_DEFER_TURNS)

        silent = dialogue.SessionState()
        spoken = dialogue.SessionState(constraints=("a", "b"))
        self.assertEqual(
            routing.choose(silent).defer_turns, routing.DISCOVERY_DEFER
        )
        self.assertEqual(
            routing.choose(spoken).defer_turns, routing.PRECISION_DEFER
        )

        pivoted = dialogue.SessionState(pivoted=True, pivot_turn=4)
        self.assertEqual(
            routing.choose(pivoted).defer_turns, ranking.MAX_DEFER_TURNS
        )

    def test_the_discovery_window_is_unreachable_on_the_public_set(
        self,
    ) -> None:
        """Declared, and measured neutral: no discovery turn reaches turn 4.

        Ordering the pair costs exactly nothing on the public 200 because the
        route it shortens is a turn-1 route. It fires on a customer who keeps
        browsing without disclosing, which this evaluator never simulates, so
        it ships as a stated invariant rather than as a score claim.
        """
        self.assertGreaterEqual(routing.DISCOVERY_DEFER, 3)

    def test_the_route_head_ships_neutral(self) -> None:
        """Widening a browsing slate is a measured negative (-0.0101 at two)."""
        self.assertIsNone(routing.DISCOVERY_HEAD)
        self.assertIsNone(routing.PRECISION_HEAD)


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
            dialogue.SessionState(declined=("material",)),
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
                dialogue.SessionState(declined=("material",)),
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
