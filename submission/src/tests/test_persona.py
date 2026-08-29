from __future__ import annotations

import unittest

from techjam.submission.src import context_distiller
from techjam.submission.src import dialogue
from techjam.submission.src import intent_detector
from techjam.submission.src import outcome_tracker
from techjam.submission.src import persona_classifier
from techjam.submission.src import response


class IntentTest(unittest.TestCase):
    def test_current_pivot_uses_structured_state(self) -> None:
        state = dialogue.SessionState(
            pivoted=True, turn=3, pivot_turn=3, constraints=("leather",)
        )

        decision = intent_detector.IntentDetector().detect(
            "Please make it leather", state, []
        )

        self.assertEqual(
            decision.intent_type, intent_detector.IntentType.INTENT_OVERRIDE
        )

    def test_empty_opening_is_not_mislabeled_as_buying(self) -> None:
        decision = intent_detector.IntentDetector().detect(
            "hello", dialogue.SessionState(turn=1), []
        )

        self.assertEqual(decision.intent_type, intent_detector.IntentType.BROWSING)


class PersonaTest(unittest.TestCase):
    def test_classifier_accepts_external_pool_and_profile_context(self) -> None:
        intent = intent_detector.IntentDecision(
            intent_detector.IntentType.BUYING, 0.9, "specific", ["need"]
        )
        state = dialogue.SessionState(turn=1, constraints=("black",))

        match = persona_classifier.PersonaClassifier().classify(
            intent, state, candidate_count=300,
            user_profile={"rating_style": "critical"},
        )

        self.assertEqual(
            match.persona_type,
            persona_classifier.PersonaType.EARLY_BUYER_SPECIFIC,
        )
        self.assertEqual(match.context_signals["products_remaining"], 300)

    def test_distiller_uses_explicit_context_not_missing_state_fields(self) -> None:
        context = context_distiller.ContextDistiller().distill(
            dialogue.SessionState(turn=2, constraints=("cotton",)),
            candidate_count=42,
            user_profile={"rating_style": "mixed"},
            asked="budget",
        )

        self.assertIn("42", context.products_status)
        self.assertIn("budget", context.estimated_opportunity)

    def test_question_wording_matches_the_structured_probe(self) -> None:
        state = dialogue.SessionState(turn=1)
        match = persona_classifier.PersonaMatch(
            persona_classifier.PersonaType.MID_BROWSER_VAGUE,
            0.8, "vague", {}, None,
        )

        text = response.compose_with_persona(
            state, dialogue.ParsedTurn(), 8, 1, 10,
            "budget", "show me options", persona_match=match,
        )

        self.assertIn("budget", text.lower())
        self.assertIn("?", text)

    def test_no_probe_means_no_question(self) -> None:
        text = response.compose_with_persona(
            dialogue.SessionState(exhausted=True, turn=10),
            dialogue.ParsedTurn(exhausted=True), 1, 10, 10,
            None, "that is all",
        )

        self.assertNotIn("?", text)


class TrackerTest(unittest.TestCase):
    def test_default_tracker_is_in_memory_only(self) -> None:
        tracker = outcome_tracker.OutcomeTracker()
        match = persona_classifier.PersonaMatch(
            persona_classifier.PersonaType.MID_BROWSER_VAGUE,
            0.8, "vague", {}, None,
        )

        tracker.record_turn(
            "s", 1, match, "cotton", "material?", [], ["cotton"],
            "mixed", 100, 20,
        )

        self.assertIsNone(tracker.log_path)
        self.assertTrue(tracker.current_session_records[0].led_to_constraint)


if __name__ == "__main__":
    unittest.main()
