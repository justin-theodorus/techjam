from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from submission.src import catalog as catalog_module
from submission.src import dialogue
from submission.src import ranking
from submission.src.tests import fixtures


class RankingTest(unittest.TestCase):
    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = fixtures.write_catalog(Path(root.name))
        self.catalog = catalog_module.build(path)

    def _slate(self, category: str | None, *constraints: str) -> list[str]:
        state = dialogue.SessionState(
            category=category, constraints=constraints
        )
        return self.catalog.slate_of(ranking.slate(self.catalog, state).indices)

    def test_an_empty_query_ranks_the_bucket_by_popularity_alone(self) -> None:
        slate = self._slate(fixtures.SNEAKER_BUCKET)
        self.assertEqual(slate[:3], ["SNEAK_POP", "SNEAK_MID", "SNEAK_RARE"])

    def test_a_query_matching_nothing_falls_back_to_popularity(self) -> None:
        slate = self._slate(fixtures.SNEAKER_BUCKET, "zzzz qqqq")
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
        slate = self._slate(fixtures.BOOT_BUCKET)
        self.assertEqual(len(slate), ranking.SLATE_SIZE)
        self.assertEqual(len(set(slate)), ranking.SLATE_SIZE)
        self.assertEqual(slate[:2], ["BOOT_POP", "BOOT_RARE"])
        self.assertTrue(set(slate) <= self.catalog.ids)

    def test_an_unknown_category_still_produces_a_full_slate(self) -> None:
        slate = self._slate("No Such Bucket")
        self.assertEqual(len(slate), ranking.SLATE_SIZE)
        self.assertTrue(set(slate) <= self.catalog.ids)

    def test_alpha_zero_ranks_on_lexical_evidence_alone(self) -> None:
        state = dialogue.SessionState(
            category=fixtures.SNEAKER_BUCKET, constraints=("hemp",)
        )
        chosen = ranking.slate(self.catalog, state, alpha=0.0).indices
        slate = self.catalog.slate_of(chosen)
        self.assertEqual(slate[0], "SNEAK_RARE")

    def test_rank_over_an_empty_pool_returns_nothing(self) -> None:
        self.assertEqual(ranking.rank(self.catalog, (), frozenset()), [])

    def test_a_narrow_head_explores_past_the_slate_not_below_it(self) -> None:
        slate = self._slate(fixtures.DEEP_BUCKET)
        self.assertEqual(slate[0], "DEEP_00")
        self.assertEqual(
            slate[1:], [f"DEEP_{n:02d}" for n in range(10, 19)]
        )

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

    def test_a_narrow_head_still_emits_a_full_unique_slate(self) -> None:
        for category in (fixtures.DEEP_BUCKET, fixtures.BOOT_BUCKET, None):
            with self.subTest(category=category):
                slate = self._slate(category)
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

    def test_the_shipped_slate_holds_the_middle_ranks_back(self) -> None:
        served = ranking.slate(self.catalog, self.state)
        ordered, _ = self._ranked()

        self.assertEqual(served.indices[0], ordered[0])
        for index in ordered[1:10]:
            self.assertNotIn(index, served.indices)


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
        state = dialogue.SessionState(turn=1)
        original = ranking.CONVERGE_AT
        try:
            ranking.CONVERGE_AT = 1
            self.assertEqual(ranking.head_size(state, 10, 3, 1), 10)
            self.assertEqual(ranking.head_size(state, 10, 3, 5), 1)
        finally:
            ranking.CONVERGE_AT = original
