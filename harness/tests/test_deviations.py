from __future__ import annotations

import unittest

from harness import deviations
from submission.src import dialogue
from submission.src import memory
from submission.src import orchestrate
from submission.src import policy
from submission.src import probe
from submission.src import ranking
from submission.src import slots
from submission.src import routing

MODULES = {"dialogue": dialogue, "memory": memory,
           "orchestrate": orchestrate, "policy": policy, "probe": probe,
           "ranking": ranking, "routing": routing, "slots": slots}


def constant(dotted: str):
    module_name, _, attribute = dotted.rpartition(".")
    return getattr(MODULES[module_name], attribute)


class PatcherTest(unittest.TestCase):
    """The patcher is the gate. If it leaks, every cell below it is void."""

    def test_the_constant_is_set_inside_the_block(self) -> None:
        with deviations.patched({"ranking.DIVERSITY": 0.5}):
            self.assertEqual(ranking.DIVERSITY, 0.5)

    def test_the_constant_is_restored_after_the_block(self) -> None:
        original = ranking.DIVERSITY
        with deviations.patched({"ranking.DIVERSITY": 0.5}):
            pass
        self.assertEqual(ranking.DIVERSITY, original)

    def test_the_constant_is_restored_after_an_exception(self) -> None:
        original = ranking.CONVERGE_AT
        with self.assertRaises(RuntimeError):
            with deviations.patched({"ranking.CONVERGE_AT": 3}):
                raise RuntimeError("boom")
        self.assertEqual(ranking.CONVERGE_AT, original)

    def test_several_constants_across_two_modules_move_together(self) -> None:
        assignments = {
            "routing.PRECISION_ALPHA": 0.4,
            "routing.DISCOVERY_ALPHA": 1.3,
            "ranking.PROFILE_WEIGHT": 0.05,
        }
        originals = (routing.PRECISION_ALPHA, routing.DISCOVERY_ALPHA,
                     ranking.PROFILE_WEIGHT)
        with deviations.patched(assignments):
            self.assertEqual(routing.PRECISION_ALPHA, 0.4)
            self.assertEqual(routing.DISCOVERY_ALPHA, 1.3)
            self.assertEqual(ranking.PROFILE_WEIGHT, 0.05)
        self.assertEqual(
            (routing.PRECISION_ALPHA, routing.DISCOVERY_ALPHA,
             ranking.PROFILE_WEIGHT),
            originals,
        )

    def test_a_constant_that_does_not_exist_raises(self) -> None:
        """A typo in the sweep table must not read as a run of null results."""
        with self.assertRaises(AttributeError):
            with deviations.patched({"ranking.NO_SUCH_CONSTANT": 1}):
                pass

    def test_an_earlier_constant_is_restored_when_a_later_fails(self) -> None:
        original = ranking.DIVERSITY
        with self.assertRaises(AttributeError):
            with deviations.patched({"ranking.DIVERSITY": 0.5,
                                     "ranking.NO_SUCH_CONSTANT": 1}):
                pass
        self.assertEqual(ranking.DIVERSITY, original)


class SweepTableTest(unittest.TestCase):

    def test_every_swept_constant_exists_and_is_at_its_shipped_value(
        self,
    ) -> None:
        """The gate reads every component against its shipped setting.

        Three of them read the same slate-spreading weight under a route switch
        and two vetoes, none of which can switch it on.

        Most ship switched off and are ablated upward; `NEGATION_WEIGHT`,
        `SKIP_SHOWN` and the gated `PROFILE_WEIGHT` ship live and are ablated
        back down (findings 3.31, 3.32, 3.43). `submission/src/tests` pins each
        constant individually;
        this asserts the sweep table names the same ones and nothing that has
        been renamed away.
        """
        neutral = {
            "routing.PRECISION_ALPHA": ranking.ALPHA,
            "routing.DISCOVERY_ALPHA": ranking.ALPHA,
            "routing.RECOVERY_RESTART": 0,
            "ranking.DIVERSITY": 0.0,
            # Ships live, so its sweep reads backwards: the 0 point is the
            # deviation and a negative delta there argues for keeping it.
            "ranking.PHRASE_POOL": 20,
            "routing.DISCOVERY_DIVERSITY": None,
            "ranking.DIVERSITY_MAX_CONSTRAINTS": -1,
            "ranking.FLATNESS_GATE": 0.0,
            # The head is derived rather than asserted; the margin is the
            # width of the band it counts, and ships mid-plateau.
            "ranking.HEAD_FROM_CONTENTION": True,
            "ranking.CONTENTION_MARGIN": 0.0005,
            # The route-conditional pair. Ships ordered rather than neutral,
            # and its sweep deviates back toward a single shared window.
            "routing.DISCOVERY_DEFER": 3,
            "routing.PRECISION_DEFER": 6,
            "routing.DISCOVERY_HEAD": None,
            # Ships off, so its sweep reads backwards: the deviation is the
            # old behaviour of filling the withheld slots from ranks 11-19.
            "ranking.EXPLORE_FILL": False,
            "ranking.EXPLORE_DIVERSITY": 0.95,
            "ranking.EXPLORE_SORT": True,
            "ranking.CONVERGE_AT": 0,
            "ranking.PROFILE_WEIGHT": 0.02,
            "ranking.PROFILE_MAX_CONSTRAINTS": 0,
            "ranking.NEGATION_WEIGHT": ranking.NEGATION_WEIGHT,
            "ranking.DENSE_WEIGHT": 0.0,
            "ranking.DENSE_NEGATION_WEIGHT": 0.0,
            "routing.PRECISION_DENSE": None,
            "routing.DISCOVERY_DENSE": None,
            "routing.DISCOVERY_REACH": 0,
            "ranking.SKIP_SHOWN": True,
            "slots.NEGATION": True,
            "ranking.LLM_RERANK": 0,
            # These two ship on, so their sweeps deviate toward the old
            # behaviour rather than away from a neutral one (findings 3.41).
            "probe.SPECIFIC_ARMS": True,
            "probe.WILDCARD_FALLBACK_RATIO": 0.2,
            "dialogue.SCOPED_EXHAUSTION": True,
            # Phase 6W's two dialogue switches. Both ship on, so their sweeps
            # also read backwards (findings 3.47).
            "probe.STAGNATION_ESCAPE": True,
            "probe.COVERAGE_SILENCE": True,
            "dialogue.STAGNATION_TURNS": 2,
            # Phase 6W's memory gates. The first four ship on and read
            # backwards; `CARRY_POSITIVES` is the one shipped off, and none of
            # them can fire without an identity the organizer never supplies.
            "memory.ENABLED": True,
            "memory.CARRY_REFUSALS": True,
            "memory.CARRY_ARMS": True,
            "memory.CARRY_BUCKETS": True,
            "memory.CARRY_POSITIVES": False,
            # Phase 6Y. `ENABLED` ships on, so the mechanism row reads
            # backwards; the three controls ship off and their rows read
            # forwards (findings 3.50).
            "orchestrate.ENABLED": True,
            "orchestrate.SPENT_RATIO": 0.5,
            "orchestrate.CANDIDATES": orchestrate.CANDIDATES,
            "orchestrate.SCHEDULE": 0,
            "orchestrate.BLIND": False,
            "orchestrate.FRESHEST": False,
            "policy.RECOVERY_TURNS": 0,
            # Phase 6Z. `READINESS_STEERS` ships on, so its row reads
            # backwards; the thresholds and the recurrence weight are the
            # curve it is read off. `HYBRID_FRAMING` ships off and reads
            # forwards.
            "policy.READINESS_STEERS": False,
            "policy.PRECISION_READINESS_THRESHOLD": 0.7,
            "policy.PARTIAL_READINESS_THRESHOLD": 0.3,
            "policy.READINESS_CURRENT_WEIGHT": 0.7,
            "policy.HYBRID_FRAMING": False,
            "policy.HYBRID_MARGIN": 0.5,
        }
        swept = {
            dotted
            for deviation in deviations.DEVIATIONS
            for _, assignments in deviation.points
            for dotted in assignments
        }
        self.assertEqual(swept, set(neutral))
        for dotted, value in neutral.items():
            self.assertEqual(constant(dotted), value)

    def test_no_point_reproduces_the_neutral_setting(self) -> None:
        """A sweep point equal to the shipped value measures nothing."""
        for deviation in deviations.DEVIATIONS:
            for label, assignments in deviation.points:
                self.assertTrue(
                    any(constant(dotted) != value
                        for dotted, value in assignments.items()),
                    f"{deviation.name} point {label} is the shipped setting",
                )

    def test_component_names_are_unique(self) -> None:
        names = [deviation.name for deviation in deviations.DEVIATIONS]
        self.assertEqual(len(names), len(set(names)))

    def test_chosen_rejects_an_unknown_component(self) -> None:
        with self.assertRaises(SystemExit):
            deviations._chosen("no_such_component")

    def test_chosen_with_no_names_returns_every_offline_component(self) -> None:
        self.assertEqual(len(deviations._chosen("")),
                         len(deviations.DEVIATIONS) - len(deviations.OPT_IN))

    def test_the_default_sweep_cannot_reach_the_network(self) -> None:
        """`make deviations` must stay free, and free means offline.

        A component that calls a model once per turn would be roughly 400,000
        requests inside the default grid, so this is a cost guard rather than
        a preference and belongs in the suite rather than in a comment.
        """
        default = {item.name for item in deviations._chosen("")}
        for name in deviations.OPT_IN:
            self.assertNotIn(name, default)
            self.assertIn(name, {item.name for item in deviations.DEVIATIONS})


def artifact(ranks: dict[str | None, int], score: float,
             sessions: list[dict]) -> dict:
    return {
        "metrics": {
            "recommended_technical_score": score,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": 0.0,
            "scenario_metrics": {},
        },
        "health": {"agent_exceptions": 0, "discarded_responses": 0,
                   "dropped_slots": 0, "short_slates": 0,
                   "wasted_pre_pivot_hits": 0},
        "ranks": ranks,
        "sessions": sessions,
    }


def session(sample_id: str, best_rank: int | None, turn: int = 1) -> dict:
    return {
        "sample_id": sample_id,
        "hit": best_rank is not None,
        "best_rank": best_rank,
        "first_hit_turn": turn,
    }


class ReadingTest(unittest.TestCase):

    def test_a_set_at_the_threshold_counts_as_saturated(self) -> None:
        rows = [session(f"s{index}", 1) for index in range(85)]
        rows += [session(f"s{index}", 4) for index in range(85, 100)]
        item = artifact({1: 85, 4: 15, None: 0}, 0.9, rows)
        self.assertEqual(deviations.rank1(item), 0.85)
        self.assertTrue(deviations.is_saturated(item))

    def test_a_set_below_the_threshold_is_readable(self) -> None:
        rows = [session(f"s{index}", 1) for index in range(84)]
        rows += [session(f"s{index}", 4) for index in range(84, 100)]
        item = artifact({1: 84, 4: 16, None: 0}, 0.9, rows)
        self.assertFalse(deviations.is_saturated(item))

    def test_a_change_that_wins_two_and_loses_two_is_not_a_no_op(self) -> None:
        """The reason flips are reported at all: the delta hides this."""
        before = artifact({}, 0.9, [session("a", 1), session("b", 1),
                                    session("c", 5), session("d", 5)])
        after = artifact({}, 0.9, [session("a", 5), session("b", 5),
                                   session("c", 1), session("d", 1)])
        moved = deviations.flips(before, after)
        self.assertEqual(sorted(moved["up"]), ["c", "d"])
        self.assertEqual(sorted(moved["down"]), ["a", "b"])

    def test_a_lost_hit_counts_as_a_demotion(self) -> None:
        before = artifact({}, 0.9, [session("a", 3)])
        after = artifact({}, 0.8, [session("a", None)])
        self.assertEqual(deviations.flips(before, after)["down"], ["a"])

    def test_an_all_identical_sweep_is_reported_as_suspect(self) -> None:
        """Findings 3.27 lost a whole sweep to a default argument doing this."""
        baseline = {"mirror": artifact({}, 0.9, [])}
        results = {"0.3": {"mirror": artifact({}, 0.9, [])},
                   "0.5": {"mirror": artifact({}, 0.9, [])}}
        self.assertTrue(deviations.unmoved(baseline, results))

    def test_a_sweep_that_moves_one_cell_is_not_suspect(self) -> None:
        baseline = {"mirror": artifact({}, 0.9, [])}
        results = {"0.3": {"mirror": artifact({}, 0.9, [])},
                   "0.5": {"mirror": artifact({}, 0.8, [])}}
        self.assertFalse(deviations.unmoved(baseline, results))


if __name__ == "__main__":
    unittest.main()
