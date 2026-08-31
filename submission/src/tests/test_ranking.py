from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from submission.src import catalog as catalog_module
from submission.src import dialogue
from submission.src import ranking
from submission.src import slots
from submission.src.tests import fixtures


class RankingTest(unittest.TestCase):
    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = fixtures.write_catalog(Path(root.name))
        self.catalog = catalog_module.build(path)

    def _slate(self, category: str | None, *constraints: str) -> list[str]:
        """The slate a still-narrowing turn serves: the committed head alone."""
        state = dialogue.SessionState(
            category=category, constraints=constraints
        )
        return self.catalog.slate_of(ranking.slate(self.catalog, state).indices)

    def _open_slate(self, category: str | None, *constraints: str) -> list[str]:
        """The slate once the head has opened, which is where an ordering reads.

        `EXPLORE_FILL` withholds every slot below the head, so a test about
        *ranking* has to open the head first or it is reading a one-item slate.
        """
        state = dialogue.SessionState(
            category=category, constraints=constraints, exhausted=True
        )
        return self.catalog.slate_of(ranking.slate(self.catalog, state).indices)

    def test_an_empty_query_ranks_the_bucket_by_popularity_alone(self) -> None:
        slate = self._open_slate(fixtures.SNEAKER_BUCKET)
        self.assertEqual(slate[:3], ["SNEAK_POP", "SNEAK_MID", "SNEAK_RARE"])

    def test_a_query_matching_nothing_falls_back_to_popularity(self) -> None:
        slate = self._open_slate(fixtures.SNEAKER_BUCKET, "zzzz qqqq")
        self.assertEqual(slate[:3], ["SNEAK_POP", "SNEAK_MID", "SNEAK_RARE"])

    def test_lexical_evidence_lifts_an_unpopular_product(self) -> None:
        slate = self._slate(
            fixtures.SNEAKER_BUCKET, "hemp upper", "cork footbed"
        )
        self.assertEqual(slate[0], "SNEAK_RARE")

    def test_the_prior_decides_when_lexical_evidence_is_shared(self) -> None:
        slate = self._slate(fixtures.SNEAKER_BUCKET, "upper")
        self.assertEqual(slate[0], "SNEAK_POP")

    def test_the_hard_filter_keeps_other_buckets_out_of_the_head(self) -> None:
        ranked = ranking.rank(
            self.catalog,
            self.catalog.bucket(fixtures.SNEAKER_BUCKET),
            self.catalog.index.query_ids(["wool", "lining"]),
        )
        self.assertNotIn("BOOT_POP", self.catalog.slate_of(ranked))

    def test_a_short_bucket_is_padded_to_a_full_slate(self) -> None:
        slate = self._open_slate(fixtures.BOOT_BUCKET)
        self.assertEqual(len(slate), ranking.SLATE_SIZE)
        self.assertEqual(len(set(slate)), ranking.SLATE_SIZE)
        self.assertEqual(slate[:2], ["BOOT_POP", "BOOT_RARE"])
        self.assertTrue(set(slate) <= self.catalog.ids)

    def test_an_unknown_category_still_produces_a_full_slate(self) -> None:
        slate = self._open_slate("No Such Bucket")
        self.assertEqual(len(slate), ranking.SLATE_SIZE)
        self.assertTrue(set(slate) <= self.catalog.ids)

    def test_alpha_zero_ranks_on_lexical_evidence_alone(self) -> None:
        state = dialogue.SessionState(
            category=fixtures.SNEAKER_BUCKET, constraints=("hemp",)
        )
        chosen = ranking.slate(self.catalog, state, alpha=0.0).indices
        slate = self.catalog.slate_of(chosen)
        self.assertEqual(slate[0], "SNEAK_RARE")

    def test_profile_weighting_ships_live_but_gated(self) -> None:
        """Not a score claim: +0.0002 in the mean, measurably not harmful.

        Ungated the same weight is monotonically negative, so the gate is the
        load-bearing half of the pair (findings 3.43).
        """
        self.assertGreater(ranking.PROFILE_WEIGHT, 0.0)
        self.assertEqual(ranking.PROFILE_MAX_CONSTRAINTS, 0)

    def test_the_profile_is_withheld_once_the_customer_has_spoken(self) -> None:
        tags = frozenset(self.catalog.index.query_ids(["hemp", "cork"]))
        silent = dialogue.SessionState(category=fixtures.SNEAKER_BUCKET)
        spoken = dialogue.SessionState(
            category=fixtures.SNEAKER_BUCKET, constraints=("upper",)
        )

        self.assertEqual(ranking.personalised(silent, tags), tags)
        self.assertEqual(ranking.personalised(spoken, tags), frozenset())

    def test_a_spoken_turn_ranks_as_though_no_profile_existed(self) -> None:
        """The gate must be a true no-op, not a weakened profile."""
        tags = frozenset(self.catalog.index.query_ids(["hemp", "cork"]))
        spoken = dialogue.SessionState(
            category=fixtures.SNEAKER_BUCKET, constraints=("upper",)
        )

        self.assertEqual(
            ranking.slate(self.catalog, spoken, profile_ids=tags).indices,
            ranking.slate(self.catalog, spoken).indices,
        )

    def test_the_ungated_switch_restores_the_rejected_configuration(
        self,
    ) -> None:
        tags = frozenset(self.catalog.index.query_ids(["hemp", "cork"]))
        spoken = dialogue.SessionState(
            category=fixtures.SNEAKER_BUCKET, constraints=("upper",)
        )
        original = ranking.PROFILE_MAX_CONSTRAINTS
        try:
            ranking.PROFILE_MAX_CONSTRAINTS = -1
            self.assertEqual(ranking.personalised(spoken, tags), tags)
        finally:
            ranking.PROFILE_MAX_CONSTRAINTS = original

    def test_the_switch_lets_the_profile_move_a_ranking(self) -> None:
        state = dialogue.SessionState(category=fixtures.SNEAKER_BUCKET)
        pool = self.catalog.pool(state.pool_keys)
        tags = frozenset(self.catalog.index.query_ids(["hemp", "cork"]))
        original = ranking.PROFILE_WEIGHT
        try:
            ranking.PROFILE_WEIGHT = 2.0
            ordered, _ = ranking.ranked(
                self.catalog, pool, frozenset(), ranking.ALPHA, tags)
        finally:
            ranking.PROFILE_WEIGHT = original
        self.assertEqual(self.catalog.slate_of(ordered)[0], "SNEAK_RARE")

    def test_rank_over_an_empty_pool_returns_nothing(self) -> None:
        self.assertEqual(ranking.rank(self.catalog, (), frozenset()), [])

    def test_a_narrow_head_withholds_every_slot_below_it(self) -> None:
        """The withheld slots stay empty rather than reaching past the slate.

        `compose` used to spend them on ranks 10-19 on the theory that a rank
        no turn would otherwise reach is free to serve. The evaluator scores
        the rank the target occupied on the turn it first appeared, so serving
        one from that band converts it at the band's position and ends the
        session; see `ranking.EXPLORE_FILL`.
        """
        slate = self._slate(fixtures.DEEP_BUCKET)
        self.assertEqual(slate, ["DEEP_00"])

    def test_a_wide_head_serves_the_top_ten(self) -> None:
        state = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET, exhausted=True
        )
        served = ranking.slate(self.catalog, state)
        slate = self.catalog.slate_of(served.indices)
        self.assertEqual(slate, [f"DEEP_{n:02d}" for n in range(10)])

    def test_the_withheld_ranks_return_once_disclosure_finishes(self) -> None:
        narrow = self._slate(fixtures.DEEP_BUCKET)
        self.assertNotIn("DEEP_05", narrow)
        wide = self.catalog.slate_of(ranking.slate(
            self.catalog,
            dialogue.SessionState(
                category=fixtures.DEEP_BUCKET, exhausted=True
            ),
        ).indices)
        self.assertIn("DEEP_05", wide)

    def test_a_narrow_head_emits_exactly_the_head(self) -> None:
        for category in (fixtures.DEEP_BUCKET, fixtures.BOOT_BUCKET, None):
            with self.subTest(category=category):
                slate = self._slate(category)
                self.assertEqual(len(slate), ranking.HEAD_SIZE)
                self.assertEqual(len(set(slate)), len(slate))
                self.assertTrue(set(slate) <= self.catalog.ids)

    def test_an_opened_head_still_emits_a_full_unique_slate(self) -> None:
        for category in (fixtures.DEEP_BUCKET, fixtures.BOOT_BUCKET, None):
            with self.subTest(category=category):
                slate = self._open_slate(category)
                self.assertEqual(len(slate), ranking.SLATE_SIZE)
                self.assertEqual(len(set(slate)), ranking.SLATE_SIZE)
                self.assertTrue(set(slate) <= self.catalog.ids)

    def test_compose_never_exceeds_the_slate_size(self) -> None:
        ordered = list(range(50))
        for head in (1, 3, 10, 25):
            with self.subTest(head=head):
                chosen = ranking.compose(ordered, head, ranking.SLATE_SIZE)
                self.assertLessEqual(len(chosen), ranking.SLATE_SIZE)
                self.assertEqual(len(chosen), len(set(chosen)))

    def test_compose_on_a_shallow_pool_returns_only_the_head(self) -> None:
        self.assertEqual(ranking.compose([7, 8, 9], 1, 10), [7])

    def test_rerank_is_always_a_permutation(self) -> None:
        # This is the entire safety argument for the stage. The evaluator ends
        # a session on the first turn the target appears anywhere in the slate,
        # so as long as membership is preserved, reordering cannot touch
        # hit@10 or MTTC and can only move MRR.
        states = (
            dialogue.SessionState(category=fixtures.DEEP_BUCKET),
            dialogue.SessionState(
                category=fixtures.DEEP_BUCKET,
                constraints=("hemp footbed marker",),
            ),
            dialogue.SessionState(
                category=fixtures.DEEP_BUCKET,
                constraints=("synthetic footbed", "nothing matches this"),
            ),
            dialogue.SessionState(
                category=fixtures.BOOT_BUCKET, constraints=("wool lining",)
            ),
        )
        for state in states:
            with self.subTest(constraints=state.constraints):
                chosen = ranking.pad(
                    self.catalog,
                    ranking.compose(
                        ranking.order(
                            self.catalog,
                            self.catalog.bucket(state.category),
                            frozenset(),
                        ),
                        ranking.head_size(state),
                        ranking.SLATE_SIZE,
                    ),
                    state.category,
                    ranking.SLATE_SIZE,
                )
                reordered = ranking.rerank(self.catalog, chosen, state)
                self.assertEqual(sorted(reordered), sorted(chosen))

    def test_rerank_promotes_a_uniquely_named_product(self) -> None:
        state = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET,
            constraints=("hemp footbed marker",),
        )
        served = ranking.slate(self.catalog, state)
        slate = self.catalog.slate_of(served.indices)
        self.assertEqual(slate[0], "DEEP_15")

    def test_rerank_leaves_the_blend_alone_without_evidence(self) -> None:
        blend = self._slate(fixtures.DEEP_BUCKET)
        unmatched = self.catalog.slate_of(ranking.slate(
            self.catalog,
            dialogue.SessionState(
                category=fixtures.DEEP_BUCKET,
                constraints=("no product says this",),
            ),
        ).indices)
        self.assertEqual(unmatched, blend)


if __name__ == "__main__":
    unittest.main()


class DeferredCommitmentTest(unittest.TestCase):
    """The alternative to holding slots back, measured rather than argued.

    Maximal marginal relevance is the standard answer to a slate full of
    near-duplicates. It is not the problem here: an impression is irreversible,
    so the cost is committing a slot to a product the ranking is unsure about,
    not repeating one. Every diversity weight above zero costs (findings 3.27),
    and these assertions keep the alternative runnable for the ablation.
    """

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.catalog = catalog_module.build(
            fixtures.write_catalog(Path(self.directory.name))
        )
        self.addCleanup(self.directory.cleanup)
        self.state = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET, turn=1
        )

    def _ranked(self):
        return ranking.ranked(
            self.catalog, self.catalog.pool(self.state.pool_keys), frozenset()
        )

    def test_no_diversity_reproduces_the_plain_top_slate(self) -> None:
        ordered, scores = self._ranked()

        self.assertEqual(
            ranking.diversify(self.catalog, ordered, scores, 1, 10, 0.0),
            ordered[:10],
        )

    def test_diversity_reaches_past_what_rank_alone_would_serve(self) -> None:
        ordered, scores = self._ranked()
        picked = ranking.diversify(self.catalog, ordered, scores, 1, 10, 0.9)

        self.assertEqual(len(picked), 10)
        self.assertEqual(len(set(picked)), 10)
        self.assertEqual(picked[0], ordered[0])
        self.assertTrue(any(index not in ordered[:10] for index in picked))

    def test_similarity_is_bounded_and_reflexive(self) -> None:
        left = frozenset({1, 2, 3})

        self.assertEqual(ranking._similarity(left, left), 1.0)
        self.assertEqual(ranking._similarity(left, frozenset({4, 5})), 0.0)
        self.assertEqual(ranking._similarity(left, frozenset()), 0.0)

    def test_diversification_is_switched_off(self) -> None:
        """Measured: it loses 0.005, and improves as relevance is ignored."""
        self.assertEqual(ranking.DIVERSITY, 0.0)

    def test_the_shipped_slate_holds_the_middle_ranks_back(self) -> None:
        served = ranking.slate(self.catalog, self.state)
        ordered, _ = self._ranked()

        self.assertEqual(served.indices[0], ordered[0])
        for index in ordered[1:10]:
            self.assertNotIn(index, served.indices)


class ExploreBandTest(unittest.TestCase):
    """Choosing the exploration slots by marginal relevance instead of by rank.

    Ships live. It is the same commit-then-explore shape `compose` already had
    and the same withheld band; only the reach changes, from a fixed offset to
    one that goes deeper when the scores are flat. It costs 0.0017 on the
    saturated public 200 and gains on every risk column and the worst
    paraphrase column (findings 3.45).
    """

    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.catalog = catalog_module.build(
            fixtures.write_catalog(Path(root.name))
        )
        self.state = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET, turn=1
        )
        self.ordered, self.scores = ranking.ranked(
            self.catalog, self.catalog.pool(self.state.pool_keys), frozenset()
        )

    def _explore(self, **overrides) -> list[int]:
        """Exercises the band with `EXPLORE_FILL` on, which is the only state
        that reaches it. The stage ships dormant -- see
        `test_the_band_is_dormant_as_shipped` -- and is kept live for the
        configuration that reverts to filling the withheld slots.
        """
        originals = (ranking.EXPLORE_DIVERSITY, ranking.EXPLORE_SORT,
                     ranking.EXPLORE_FILL)
        try:
            ranking.EXPLORE_FILL = True
            for name, value in overrides.items():
                setattr(ranking, name, value)
            return ranking.explore(
                self.catalog, self.ordered, self.scores, 1, 10
            )
        finally:
            (ranking.EXPLORE_DIVERSITY, ranking.EXPLORE_SORT,
             ranking.EXPLORE_FILL) = originals

    def test_the_band_is_dormant_as_shipped(self) -> None:
        """`EXPLORE_FILL` is off, so nothing below the head is served at all."""
        self.assertFalse(ranking.EXPLORE_FILL)
        self.assertEqual(
            ranking.explore(self.catalog, self.ordered, self.scores, 1, 10),
            self.ordered[:1],
        )

    def test_the_stage_ships_live(self) -> None:
        self.assertGreater(ranking.EXPLORE_DIVERSITY, 0.0)
        self.assertTrue(ranking.EXPLORE_SORT)

    def test_the_withheld_band_is_never_served(self) -> None:
        """The invariant that separates this from `DIVERSITY`.

        A target inside ranks 2..10 converts *later at rank 1*, so spending a
        slot on it now is what made diversifying the whole head cost 100% of
        its loss in MRR (findings 3.45).
        """
        for weight in (0.5, 0.95, 1.0):
            with self.subTest(weight=weight):
                served = self._explore(EXPLORE_DIVERSITY=weight)
                for index in self.ordered[1:10]:
                    self.assertNotIn(index, served)

    def test_the_committed_head_is_untouched(self) -> None:
        served = self._explore(EXPLORE_DIVERSITY=0.95)

        self.assertEqual(served[0], self.ordered[0])
        self.assertEqual(len(served), 10)
        self.assertEqual(len(set(served)), 10)

    def test_disabled_reproduces_the_fixed_offset(self) -> None:
        original = ranking.EXPLORE_FILL
        try:
            ranking.EXPLORE_FILL = True
            fixed = ranking.compose(self.ordered, 1, 10)
        finally:
            ranking.EXPLORE_FILL = original
        self.assertEqual(self._explore(EXPLORE_DIVERSITY=0.0), fixed)

    def test_marginal_relevance_promotes_the_distinctive_product(self) -> None:
        """`DEEP_15` is the one product in the band with its own vocabulary."""
        served = self._explore(EXPLORE_DIVERSITY=0.95, EXPLORE_SORT=False)
        slate = self.catalog.slate_of(served)

        self.assertEqual(slate[2], "DEEP_15")
        self.assertNotEqual(
            served, ranking.compose(self.ordered, 1, 10)
        )

    def test_sorting_restores_score_order(self) -> None:
        """A pure permutation: membership is coverage, order is MRR."""
        unsorted = self._explore(EXPLORE_DIVERSITY=0.95, EXPLORE_SORT=False)
        served = self._explore(EXPLORE_DIVERSITY=0.95, EXPLORE_SORT=True)

        self.assertEqual(set(served), set(unsorted))
        rank = {index: position
                for position, index in enumerate(self.ordered)}
        tail = [rank[index] for index in served[1:]]
        self.assertEqual(tail, sorted(tail))

    def test_a_shallow_band_falls_back_to_the_fixed_offset(self) -> None:
        shallow = self.ordered[:14]
        served = ranking.explore(
            self.catalog, shallow, self.scores[:14], 1, 10
        )

        self.assertEqual(served, ranking.compose(shallow, 1, 10))


class FlatnessTest(unittest.TestCase):
    """The statistic on its own, before anything reads it."""

    def test_an_undifferentiated_slate_reads_one(self) -> None:
        self.assertEqual(ranking.flatness([1.0] * 10, 10), 1.0)

    def test_a_decided_slate_reads_small(self) -> None:
        self.assertAlmostEqual(ranking.flatness([1.0] + [0.1] * 9, 10), 0.1)

    def test_it_reads_the_tail_not_the_crowd_at_the_head(self) -> None:
        """The distinction from `contention`, which is why this exists.

        A slate whose second product already sits outside the 2% margin can
        still be flat across ten, and a count near the leader cannot say so.
        """
        scores = [1.0 - 0.03 * step for step in range(10)]

        self.assertEqual(ranking.contention(scores), 1)
        self.assertAlmostEqual(ranking.flatness(scores, 10), 0.73)

    def test_a_non_positive_leader_reads_as_maximally_separated(self) -> None:
        """`ranked` subtracts for refusals, so a whole pool can go negative."""
        self.assertEqual(ranking.flatness([], 10), 0.0)
        self.assertEqual(ranking.flatness([0.0] * 10, 10), 0.0)
        self.assertEqual(ranking.flatness([-1.0, -2.0, -3.0], 10), 0.0)

    def test_a_negative_tail_never_reads_as_flat(self) -> None:
        self.assertEqual(ranking.flatness([1.0] * 9 + [-1.0], 10), 0.0)

    def test_a_short_list_reads_its_deepest_score(self) -> None:
        self.assertAlmostEqual(ranking.flatness([1.0, 0.4], 10), 0.4)

    def test_the_reading_is_never_negative(self) -> None:
        """Why a zero threshold is neutral by arithmetic, not by branch."""
        for scores in ([1.0] * 10, [1.0] * 9 + [-5.0], [-1.0] * 10, []):
            with self.subTest(scores=scores):
                self.assertGreaterEqual(ranking.flatness(scores, 10), 0.0)


class DiversityGateTest(unittest.TestCase):
    """Diversifying only while the customer still has something to say.

    Findings 3.30 swept `DIVERSITY` on every turn of every route and read the
    result as a verdict on the mechanism. It was a verdict on one configuration:
    the sessions it lost on are the ones carrying constraints, which this gate
    can exclude. Measured again gated, it still loses (findings 3.43), and these
    assertions keep the alternative runnable for the ablation.
    """

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.catalog = catalog_module.build(
            fixtures.write_catalog(Path(self.directory.name))
        )
        self.addCleanup(self.directory.cleanup)
        self.state = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET, turn=1
        )

    def test_both_gates_ship_disabled(self) -> None:
        """Measured: gating shrinks the loss without turning one positive."""
        self.assertEqual(ranking.DIVERSITY_MAX_CONSTRAINTS, -1)
        self.assertEqual(ranking.FLATNESS_GATE, 0.0)

    def test_an_open_gate_admits_every_session(self) -> None:
        spoken = dialogue.SessionState(constraints=("cotton", "black"))

        for state in (dialogue.SessionState(), spoken):
            for scores in ([1.0] * 10, [1.0] + [0.001] * 9, [], [-1.0] * 10):
                with self.subTest(state=state, scores=scores):
                    self.assertTrue(
                        ranking.worth_diversifying(state, list(scores), 10)
                    )

    def test_a_gate_can_only_veto_never_enable(self) -> None:
        """What makes them safe to ship reachable but unreached."""
        originals = (
            ranking.FLATNESS_GATE, ranking.DIVERSITY_MAX_CONSTRAINTS
        )
        try:
            ranking.FLATNESS_GATE = 0.01
            ranking.DIVERSITY_MAX_CONSTRAINTS = 9
            self.assertEqual(
                ranking.slate(self.catalog, self.state).indices,
                ranking.slate(
                    self.catalog, self.state, diversity=0.0
                ).indices,
            )
        finally:
            (ranking.FLATNESS_GATE,
             ranking.DIVERSITY_MAX_CONSTRAINTS) = originals

    def test_the_switch_admits_only_the_undisclosed(self) -> None:
        silent = dialogue.SessionState()
        spoken = dialogue.SessionState(constraints=("cotton",))
        original = ranking.DIVERSITY_MAX_CONSTRAINTS
        try:
            ranking.DIVERSITY_MAX_CONSTRAINTS = 0
            self.assertTrue(ranking.worth_diversifying(silent, [1.0], 10))
            self.assertFalse(ranking.worth_diversifying(spoken, [1.0], 10))
        finally:
            ranking.DIVERSITY_MAX_CONSTRAINTS = original

    def test_the_two_gates_veto_independently(self) -> None:
        """A conjunction: either one refuses what the other admits."""
        silent = dialogue.SessionState()
        spoken = dialogue.SessionState(constraints=("cotton",))
        originals = (
            ranking.FLATNESS_GATE, ranking.DIVERSITY_MAX_CONSTRAINTS
        )
        try:
            ranking.FLATNESS_GATE = 0.65
            ranking.DIVERSITY_MAX_CONSTRAINTS = 0
            self.assertTrue(
                ranking.worth_diversifying(silent, [1.0] * 10, 10))
            self.assertFalse(
                ranking.worth_diversifying(silent, [1.0] + [0.1] * 9, 10))
            self.assertFalse(
                ranking.worth_diversifying(spoken, [1.0] * 10, 10))
        finally:
            (ranking.FLATNESS_GATE,
             ranking.DIVERSITY_MAX_CONSTRAINTS) = originals

    def test_an_unreachable_flatness_gate_restores_the_shipped_slate(
        self,
    ) -> None:
        plain = ranking.slate(self.catalog, self.state).indices
        original = ranking.FLATNESS_GATE
        try:
            ranking.FLATNESS_GATE = 1.01
            self.assertEqual(
                ranking.slate(
                    self.catalog, self.state, diversity=0.9
                ).indices,
                plain,
            )
        finally:
            ranking.FLATNESS_GATE = original

    def test_a_closed_gate_restores_the_shipped_slate(self) -> None:
        """A gated-out turn must be the deferred slate, not a weakened one."""
        spoken = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET, turn=1, constraints=("cotton",)
        )
        original = ranking.DIVERSITY_MAX_CONSTRAINTS
        try:
            ranking.DIVERSITY_MAX_CONSTRAINTS = 0
            gated = ranking.slate(self.catalog, spoken, diversity=0.9)
        finally:
            ranking.DIVERSITY_MAX_CONSTRAINTS = original

        self.assertEqual(
            gated.indices, ranking.slate(self.catalog, spoken).indices
        )

    def test_the_weight_reaches_the_slate_when_the_gate_is_open(self) -> None:
        """Read with `EXPLORE_FILL` on, which is the only state it reaches.

        Diversity selects the slots below the committed head, and `diversify`
        short-circuits once the head is the whole slate -- so with the withheld
        slots empty there is nothing for the weight to choose. See
        `test_the_weight_is_dormant_as_shipped`.
        """
        original = ranking.EXPLORE_FILL
        try:
            ranking.EXPLORE_FILL = True
            varied = ranking.slate(self.catalog, self.state, diversity=0.9)
            plain = ranking.slate(self.catalog, self.state).indices
        finally:
            ranking.EXPLORE_FILL = original

        self.assertNotEqual(varied.indices, plain)

    def test_the_weight_is_dormant_as_shipped(self) -> None:
        """`DIVERSITY` chooses withheld slots, and they are no longer served."""
        self.assertFalse(ranking.EXPLORE_FILL)
        self.assertEqual(
            ranking.slate(self.catalog, self.state, diversity=0.9).indices,
            ranking.slate(self.catalog, self.state).indices,
        )

    def test_an_absent_weight_defers_to_the_module(self) -> None:
        """`None` means "whatever ships", which is what the sweep patches."""
        self.assertEqual(
            ranking.slate(self.catalog, self.state, diversity=None).indices,
            ranking.slate(self.catalog, self.state).indices,
        )


class NegationPenaltyTest(unittest.TestCase):
    """Refused terms subtract. The switch ships live, not neutral."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.catalog = catalog_module.build(
            fixtures.write_catalog(Path(self.directory.name))
        )
        self.addCleanup(self.directory.cleanup)
        self.pool = self.catalog.pool((fixtures.SNEAKER_BUCKET,))

    def _order(self, negative: tuple[str, ...] = ()) -> list[str]:
        ordered, _ = ranking.ranked(
            self.catalog, self.pool, frozenset(), ranking.ALPHA, frozenset(),
            self.catalog.index.query_ids(list(negative)),
        )
        return self.catalog.slate_of(ordered)

    def test_the_penalty_ships_live_rather_than_neutral(self) -> None:
        """The behaviour it replaces is inverted, not neutral: the refused
        material was served at 2.3x its shelf rate (findings 3.31)."""
        self.assertGreater(ranking.NEGATION_WEIGHT, 0.0)
        self.assertTrue(slots.NEGATION)

    def test_the_master_switch_stops_refusals_being_read_at_all(self) -> None:
        """Ablating the weight alone leaves refusals out of the positive
        query, which is the half that pays. This is the whole feature."""
        original = slots.NEGATION
        try:
            slots.NEGATION = False
            self.assertEqual(slots.polarity("not cotton"),
                             (False, "not cotton"))
        finally:
            slots.NEGATION = original

    def test_a_refused_term_pushes_its_product_down(self) -> None:
        """`SNEAK_POP` leads on popularity alone; refusing what it is made of
        must cost it that lead."""
        self.assertEqual(self._order()[0], "SNEAK_POP")

        self.assertNotEqual(self._order(("cotton", "canvas"))[0], "SNEAK_POP")

    def test_a_refusal_matching_nothing_changes_no_ranking(self) -> None:
        self.assertEqual(self._order(("unobtainium",)), self._order())

    def test_the_penalty_is_a_no_op_when_switched_off(self) -> None:
        original = ranking.NEGATION_WEIGHT
        try:
            ranking.NEGATION_WEIGHT = 0.0
            self.assertEqual(self._order(("cotton", "canvas")), self._order())
        finally:
            ranking.NEGATION_WEIGHT = original


class SkipShownTest(unittest.TestCase):
    """A product already served cannot convert, so it must not hold a slot."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.catalog = catalog_module.build(
            fixtures.write_catalog(Path(self.directory.name))
        )
        self.addCleanup(self.directory.cleanup)
        self.state = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET, turn=1
        )

    def test_skipping_the_shown_ships_live(self) -> None:
        """62.9% of `thin_cards` impressions were repeats, and 60 of its 78
        misses were reachable on the slots they wasted (findings 3.32)."""
        self.assertTrue(ranking.SKIP_SHOWN)

    def test_a_second_slate_repeats_nothing_from_the_first(self) -> None:
        first = ranking.slate(self.catalog, self.state)
        shown = tuple(self.catalog.slate_of(first.indices))
        second = ranking.slate(self.catalog, self.state.with_slate(shown))

        self.assertEqual(len(second.indices), len(first.indices))
        served = set(self.catalog.slate_of(second.indices))
        self.assertFalse(served & set(shown))

    def test_the_switch_restores_the_repeats(self) -> None:
        first = ranking.slate(self.catalog, self.state)
        shown = tuple(self.catalog.slate_of(first.indices))
        original = ranking.SKIP_SHOWN
        try:
            ranking.SKIP_SHOWN = False
            second = ranking.slate(self.catalog, self.state.with_slate(shown))
        finally:
            ranking.SKIP_SHOWN = original

        self.assertEqual(second.indices, first.indices)

    def test_a_pool_too_small_to_refill_serves_repeats_over_gaps(self) -> None:
        """An *opened* slate emits repeats rather than empty slots.

        Read with the head opened: while it is narrow the slate is deliberately
        short, so a length check there would be measuring `EXPLORE_FILL` rather
        than `SKIP_SHOWN`.
        """
        state = dialogue.SessionState(
            category=fixtures.SNEAKER_BUCKET, turn=1, exhausted=True
        )
        served = ranking.slate(self.catalog, state)
        shown = tuple(self.catalog.slate_of(served.indices))

        again = ranking.slate(self.catalog, state.with_slate(shown))

        self.assertEqual(len(again.indices), ranking.SLATE_SIZE)


class ContentionTest(unittest.TestCase):
    def test_a_flat_ranking_leaves_everything_in_contention(self) -> None:
        self.assertEqual(ranking.contention([1.0, 1.0, 1.0]), 3)

    def test_a_decisive_leader_stands_alone(self) -> None:
        self.assertEqual(ranking.contention([1.0, 0.5, 0.4]), 1)

    def test_an_empty_ranking_has_no_contenders(self) -> None:
        self.assertEqual(ranking.contention([]), 0)

    def test_a_scoreless_pool_is_entirely_in_contention(self) -> None:
        self.assertEqual(ranking.contention([0.0, 0.0]), 2)

    def test_converging_early_is_switched_off(self) -> None:
        """Measured: it lifts hit@10 to 0.995 and drops MRR to 0.781."""
        self.assertEqual(ranking.CONVERGE_AT, 0)

    def test_the_switch_widens_the_slate_when_enabled(self) -> None:
        """A gated-out turn falls through to the derived head, not to one."""
        state = dialogue.SessionState(turn=1)
        original = ranking.CONVERGE_AT
        try:
            ranking.CONVERGE_AT = 1
            self.assertEqual(ranking.head_size(state, 10, 3, 1), 10)
            self.assertEqual(ranking.head_size(state, 10, 3, 5), 5)
        finally:
            ranking.CONVERGE_AT = original

    def test_the_head_is_derived_from_contention(self) -> None:
        """Commit to what is still competing, and to one once it has decided."""
        self.assertTrue(ranking.HEAD_FROM_CONTENTION)
        state = dialogue.SessionState(turn=1)
        for contenders, expected in ((1, 1), (2, 2), (4, 4), (20, 10)):
            with self.subTest(contenders=contenders):
                self.assertEqual(
                    ranking.head_size(state, 10, 6, contenders), expected
                )

    def test_head_size_never_falls_below_the_floor(self) -> None:
        """`contenders == 0` means the caller did not measure it."""
        state = dialogue.SessionState(turn=1)
        self.assertEqual(ranking.head_size(state, 10, 6, 0), ranking.HEAD_SIZE)

    def test_the_margin_is_read_at_call_time(self) -> None:
        """Bound as a default it was fixed at import, so no sweep could move it.

        The same defect findings 3.27 caught in `slate`: every earlier sweep of
        `CONTENTION_MARGIN` silently measured the shipped value.
        """
        scores = [1.0, 0.999, 0.99, 0.5]
        original = ranking.CONTENTION_MARGIN
        try:
            ranking.CONTENTION_MARGIN = 0.0
            tight = ranking.contention(scores)
            ranking.CONTENTION_MARGIN = 0.05
            loose = ranking.contention(scores)
        finally:
            ranking.CONTENTION_MARGIN = original
        self.assertEqual(tight, 1)
        self.assertEqual(loose, 3)
        self.assertGreater(loose, tight)


class PhrasePromotionTest(unittest.TestCase):
    """Pins `PHRASE_POOL`, which decides the head rather than decorating it."""

    # `fixtures.CATALOG_ROWS` gives this phrase to DEEP_15 alone, and orders
    # the Sandals bucket by review count so prior rank is exactly the index.
    # It therefore sits at rank 15: inside the shipped window, outside a
    # narrow one, which is what makes the reach testable at all.
    RARE = "hemp footbed marker"

    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = fixtures.write_catalog(Path(root.name))
        self.catalog = catalog_module.build(path)

    def _promoted(self, *constraints: str, width: int | None = None) -> str:
        """The asin the head would commit to, at `width`."""
        state = dialogue.SessionState(
            category=fixtures.DEEP_BUCKET, constraints=constraints
        )
        original = ranking.PHRASE_POOL
        try:
            if width is not None:
                ranking.PHRASE_POOL = width
            served = ranking.slate(self.catalog, state)
        finally:
            ranking.PHRASE_POOL = original
        return self.catalog.asins[served.indices[0]]

    def test_it_ships_live(self) -> None:
        """So its sweep row reads backwards, like `NEGATION_WEIGHT`'s."""
        self.assertEqual(ranking.PHRASE_POOL, 20)

    def test_rare_evidence_decides_the_committed_head(self) -> None:
        """The whole point: `rerank` cannot do this once the head is one wide."""
        self.assertEqual(self._promoted(self.RARE), "DEEP_15")

    def _sandals(self) -> list[int]:
        """The Sandals bucket in prior order, so DEEP_15 sits at rank 15."""
        by_asin = {asin: index for index, asin in enumerate(self.catalog.asins)}
        return [by_asin[f"DEEP_{n:02d}"] for n in range(fixtures.DEEP_SIZE)]

    def _reached(self, width: int) -> str:
        """The head `phrase_promoted` yields over a fixed prior ordering.

        Driven at the function rather than through `slate`, because the blend
        also ranks the only product holding these tokens first: an end-to-end
        assertion could not tell the two signals apart.
        """
        pool = self._sandals()
        state = dialogue.SessionState(constraints=(self.RARE,))
        original = ranking.PHRASE_POOL
        try:
            ranking.PHRASE_POOL = width
            promoted, _ = ranking.phrase_promoted(
                self.catalog, pool, [0.0] * len(pool), state
            )
        finally:
            ranking.PHRASE_POOL = original
        return self.catalog.asins[promoted[0]]

    def test_the_window_bounds_the_reach(self) -> None:
        """Rank 15 is past a ten-wide window, so nothing promotes it there."""
        self.assertEqual(self._reached(10), "DEEP_00")
        self.assertEqual(self._reached(20), "DEEP_15")

    def test_zero_restores_the_pre_promotion_behaviour(self) -> None:
        self.assertEqual(self._reached(0), "DEEP_00")

    def test_a_constraint_no_product_names_leaves_the_blend_alone(self) -> None:
        """It degrades to the prior rather than to noise."""
        self.assertEqual(self._promoted("nothing says this"), "DEEP_00")

    def test_scores_travel_with_the_products_they_belong_to(self) -> None:
        """A slate whose scores stayed put would misreport every row."""
        pool = tuple(range(len(self.catalog.asins)))
        scores = [float(len(pool) - rank) for rank in range(len(pool))]
        state = dialogue.SessionState(constraints=(self.RARE,))
        before = dict(zip(pool, scores))
        promoted, moved = ranking.phrase_promoted(
            self.catalog, list(pool), scores, state
        )
        self.assertCountEqual(promoted, pool)
        for index, score in zip(promoted, moved):
            self.assertEqual(score, before[index])
