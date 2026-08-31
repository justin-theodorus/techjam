from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from submission.src import catalog as catalog_module
from submission.src import dialogue
from submission.src import orchestrate
from submission.src import policy
from submission.src import ranking
from submission.src.tests import fixtures


class ShippedSwitchTest(unittest.TestCase):
    """Pins every switch, so a sweep value cannot be left behind."""

    def test_the_controller_ships_on(self) -> None:
        self.assertTrue(orchestrate.ENABLED)

    def test_the_shipped_trigger_and_selection_rule(self) -> None:
        self.assertEqual(orchestrate.SPENT_RATIO, 0.5)
        self.assertEqual(orchestrate.MIN_REFUTED, 1)
        self.assertEqual(orchestrate.HORIZON, ranking.WINDOW)

    def test_every_control_ships_off(self) -> None:
        self.assertEqual(orchestrate.SCHEDULE, 0)
        self.assertFalse(orchestrate.BLIND)
        self.assertFalse(orchestrate.FRESHEST)

    def test_candidates_run_most_specific_evidence_first(self) -> None:
        """The order is an argument about evidence, not a fit to a column."""
        self.assertEqual(
            orchestrate.CANDIDATES,
            (ranking.BLEND, ranking.PHRASE, ranking.PRIOR, ranking.LEXICAL),
        )


class RefutedTest(unittest.TestCase):
    def test_a_first_turn_has_disproven_nothing(self) -> None:
        self.assertEqual(orchestrate.refuted(dialogue.SessionState(turn=1)), 0)

    def test_every_continued_turn_disproves_the_slate_before_it(self) -> None:
        self.assertEqual(orchestrate.refuted(dialogue.SessionState(turn=4)), 3)

    def test_slates_served_before_a_pivot_prove_nothing(self) -> None:
        """`override_applied` gates scoring, so a pre-pivot slate was never
        checked against the target that counts (measurements 3.32)."""
        state = dialogue.SessionState(turn=6, pivoted=True, pivot_turn=4)

        self.assertEqual(orchestrate.refuted(state), 2)


class ControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.catalog = catalog_module.build(fixtures.write_catalog(
            Path(root.name)))
        self.deep = self.catalog.slate_of(
            self.catalog.pool((fixtures.DEEP_BUCKET,))
        )

    def _state(self, shown: int, turn: int = 3) -> dialogue.SessionState:
        return dialogue.SessionState(
            category=fixtures.DEEP_BUCKET,
            constraints=("hemp footbed marker",),
            turn=turn,
            shown=frozenset(self.deep[:shown]),
        )

    def _choose(self, state: dialogue.SessionState):
        return orchestrate.choose(self.catalog, state, policy.DISCOVERY)

    def test_nothing_disproven_yet_keeps_the_shipped_blend(self) -> None:
        workflow = self._choose(self._state(shown=0, turn=1))

        self.assertEqual(workflow.ordering, ranking.BLEND)
        self.assertFalse(workflow.switched)

    def test_a_fresh_blend_head_keeps_the_turn(self) -> None:
        workflow = self._choose(self._state(shown=2))

        self.assertEqual(workflow.ordering, ranking.BLEND)

    def test_a_disproven_head_switches_to_one_still_unspent(self) -> None:
        """The blend's pick is disproven and the phrase ordering names the
        same product, so the only candidate left with an unserved head is the
        prior. A candidate whose own head is spent is skipped, not taken."""
        self.addCleanup(setattr, orchestrate, "HORIZON", orchestrate.HORIZON)
        orchestrate.HORIZON = 1
        state = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET,
            constraints=("hemp footbed marker",),
            turn=3,
            shown=frozenset({"DEEP_15"}),
        )

        workflow = self._choose(state)

        self.assertEqual(workflow.ordering, ranking.PRIOR)
        self.assertTrue(workflow.switched)

    def test_nothing_fresher_anywhere_keeps_the_blend(self) -> None:
        """Every ordering re-sorts the same pool, so once the session has seen
        all of it there is nowhere to switch to and thrashing buys nothing."""
        workflow = self._choose(self._state(shown=len(self.deep)))

        self.assertEqual(workflow.ordering, ranking.BLEND)

    def test_an_ordering_with_no_evidence_is_never_offered(self) -> None:
        """Without this the emptiest candidate wins, because an ordering that
        returns the pool untouched has a head nobody has served (3.50)."""
        state = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET,
            turn=3,
            shown=frozenset(self.deep),
        )

        self.assertNotEqual(self._choose(state).ordering, ranking.PHRASE)

    def test_the_master_switch_returns_the_shipped_workflow(self) -> None:
        self.addCleanup(setattr, orchestrate, "ENABLED", orchestrate.ENABLED)
        orchestrate.ENABLED = False

        workflow = self._choose(self._state(shown=len(self.deep)))

        self.assertEqual(workflow.ordering, ranking.BLEND)
        self.assertEqual(workflow.reason, "shipped")


class PhraseOrderingTest(unittest.TestCase):
    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.catalog = catalog_module.build(fixtures.write_catalog(
            Path(root.name)))
        self.pool = self.catalog.pool((fixtures.DEEP_BUCKET,))

    def test_a_rare_phrase_reaches_a_product_the_prior_buries(self) -> None:
        state = dialogue.SessionState(constraints=("hemp footbed marker",))

        ordered, _ = ranking.phrase_ordered(self.catalog, self.pool, state)

        self.assertEqual(self.catalog.slate_of(ordered)[0], "DEEP_15")

    def test_no_phrase_evidence_degrades_to_the_popularity_order(self) -> None:
        """Findings 3.1 measured this route degrading to noise as a session's
        primary retriever. As a fallback it degrades to the prior."""
        state = dialogue.SessionState(constraints=("nothing in the index",))

        ordered, _ = ranking.phrase_ordered(self.catalog, self.pool, state)

        self.assertEqual(ordered, list(self.pool))

    def test_an_unknown_ordering_is_a_programming_error(self) -> None:
        with self.assertRaises(ValueError):
            ranking.alternative(
                "nonsense", self.catalog, self.pool,
                dialogue.SessionState(),
            )


if __name__ == "__main__":
    unittest.main()
