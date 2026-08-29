from __future__ import annotations

import unittest

from techjam.harness.diff import compare
from techjam.harness.report import technical_score


def artifact(sessions: list[dict], metrics: dict | None = None) -> dict:
    return {
        "metrics": metrics or {
            "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": 11.0,
            "recommended_technical_score": 0.0, "scenario_metrics": {},
        },
        "sessions": sessions,
    }


def session(sample_id: str, hit: bool, rank: int | None = None, turn: int | None = None) -> dict:
    return {
        "sample_id": sample_id, "scenario_type": "buying", "hit": hit,
        "best_rank": rank, "first_hit_turn": turn,
    }


class DiffTest(unittest.TestCase):
    def test_reports_flips_in_both_directions(self) -> None:
        before = artifact([session("a", True, 1, 1), session("b", False)])
        after = artifact([session("a", False), session("b", True, 3, 2)])

        result = compare(before, after)

        self.assertEqual(result["lost"], ["a"])
        self.assertEqual(result["gained"], ["b"])

    def test_separates_rank_and_turn_movement_among_sessions_that_hit_in_both(self) -> None:
        before = artifact([session("a", True, 5, 4), session("b", True, 2, 2)])
        after = artifact([session("a", True, 2, 4), session("b", True, 2, 5)])

        result = compare(before, after)

        self.assertEqual(result["rank_better"], ["a"])
        self.assertEqual(result["rank_worse"], [])
        self.assertEqual(result["turn_worse"], ["b"])

    def test_sessions_present_in_only_one_run_are_reported_not_silently_skipped(self) -> None:
        result = compare(artifact([session("a", True, 1, 1)]), artifact([session("c", True, 1, 1)]))

        self.assertEqual(result["only_in_before"], ["a"])
        self.assertEqual(result["only_in_after"], ["c"])
        self.assertEqual(result["gained"], [])

    def test_technical_score_matches_the_evaluators_formula(self) -> None:
        metrics = {"hit_rate_at_10": 0.125, "mrr": 0.068034, "mttc": 9.81}

        self.assertAlmostEqual(technical_score(metrics), 0.10671, places=5)

    def test_efficiency_is_clipped_at_both_ends(self) -> None:
        self.assertAlmostEqual(technical_score({"hit_rate_at_10": 0, "mrr": 0, "mttc": 11.0}), 0.0)
        self.assertAlmostEqual(technical_score({"hit_rate_at_10": 0, "mrr": 0, "mttc": 0.5}), 0.2)


if __name__ == "__main__":
    unittest.main()
