from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate
from harness.analysis import analyze, health_summary, latency_summary
from harness.record import RecordingAgent
from harness.tests.fixtures import ConstantAgent, sample, write_catalog


def run(agent, samples):
    with tempfile.TemporaryDirectory() as directory:
        catalog_ids, categories, products = catalog_index(write_catalog(Path(directory)))
        recorder = RecordingAgent(agent)
        result = evaluate(recorder, samples, catalog_ids, categories, products)
        return result, analyze(recorder.sessions, samples, result, catalog_ids)


class AnalysisTest(unittest.TestCase):
    def test_trace_agrees_with_the_evaluator_verdict_on_a_hit(self) -> None:
        result, sessions = run(ConstantAgent(["A", "B"]), [sample("s1", "buying", "A")])

        self.assertEqual(result["hit_rate_at_10"], 1.0)
        self.assertEqual(sessions[0]["first_hit_turn"], 1)
        self.assertEqual(sessions[0]["turns"][0]["target_rank"], 1)
        self.assertTrue(sessions[0]["turns"][0]["scorable"])

    def test_out_of_catalog_and_duplicate_entries_are_counted_as_dropped_slots(self) -> None:
        _, sessions = run(ConstantAgent(["A", "A", "ZZZ", "B"]), [sample("s1", "buying", "B")])

        turn = sessions[0]["turns"][0]
        self.assertEqual(turn["slate"], ["A", "B"])
        self.assertEqual(turn["dropped_slots"], 2)

    def test_override_pre_pivot_hits_are_marked_wasted_and_pivot_turn_is_recovered(self) -> None:
        _, sessions = run(ConstantAgent(["A"]), [sample("s1", "intent_override", "A")])

        session = sessions[0]
        self.assertIn(session["pivot_turn"], (3, 4))
        self.assertEqual(session["first_hit_turn"], session["pivot_turn"])
        wasted = [turn["turn"] for turn in session["turns"] if turn["wasted_hit"]]
        self.assertEqual(wasted, list(range(1, session["pivot_turn"])))

    def test_boundary_refusal_turn_is_flagged(self) -> None:
        _, sessions = run(ConstantAgent(["B"]), [sample("s1", "boundary", "A")])

        self.assertTrue(sessions[0]["turns"][1]["boundary_refusal"])

    def test_disclosed_constraints_are_extracted_from_the_reply_template(self) -> None:
        _, sessions = run(ConstantAgent(["B"]), [sample("s1", "browsing", "A")])

        disclosed = [turn["disclosed"] for turn in sessions[0]["turns"] if turn["disclosed"]]
        self.assertTrue(disclosed)
        self.assertNotIn(";", disclosed[0][0])

    def test_health_summary_surfaces_what_the_evaluator_swallows(self) -> None:
        _, sessions = run(ConstantAgent(["A"]), [sample("s1", "intent_override", "A")])

        health = health_summary(sessions)
        self.assertEqual(health["agent_exceptions"], 0)
        self.assertGreater(health["short_slates"], 0)
        self.assertGreater(health["wasted_pre_pivot_hits"], 0)

    def test_latency_summary_covers_every_recorded_turn(self) -> None:
        _, sessions = run(ConstantAgent(["B"]), [sample("s1", "browsing", "A")])

        self.assertEqual(latency_summary(sessions)["turn_count"], sessions[0]["turn_count"])

    def test_length_mismatch_between_recording_and_results_is_fatal(self) -> None:
        with self.assertRaises(RuntimeError):
            analyze([], [sample("s1", "buying", "A")], {"sessions": []}, {"A"})


if __name__ == "__main__":
    unittest.main()
