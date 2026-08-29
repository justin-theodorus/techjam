from __future__ import annotations

import unittest

from techjam.harness.record import RecordingAgent


class CrashingAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        raise ValueError("boom")


class DebugAgent:
    def __init__(self) -> None:
        self.debug: dict = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.debug = {"pool": 182, "category": "shoes"}
        return {"message": "ok", "ask_attribute": "other", "recommendations": ["A", {"parent_asin": "B"}]}


class BadShapeAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {"message": None, "recommendations": ["A"]}


class RecordingAgentTest(unittest.TestCase):
    def test_records_turn_and_normalizes_mixed_recommendation_shapes(self) -> None:
        recorder = RecordingAgent(DebugAgent())
        recorder.reset("s1", {})
        recorder.respond("s1", "hello", 1, 10)

        turn = recorder.sessions[0]["turns"][0]
        self.assertEqual(turn["raw_recommendations"], ["A", "B"])
        self.assertEqual(turn["ask_attribute"], "other")
        self.assertFalse(turn["discarded"])
        self.assertIsNone(turn["error"])

    def test_captures_agent_debug_snapshot(self) -> None:
        recorder = RecordingAgent(DebugAgent())
        recorder.reset("s1", {})
        recorder.respond("s1", "hello", 1, 10)

        self.assertEqual(recorder.sessions[0]["turns"][0]["debug"], {"pool": 182, "category": "shoes"})

    def test_records_exception_then_reraises_so_evaluator_behaviour_is_unchanged(self) -> None:
        recorder = RecordingAgent(CrashingAgent())
        recorder.reset("s1", {})

        with self.assertRaises(ValueError):
            recorder.respond("s1", "hello", 1, 10)

        turn = recorder.sessions[0]["turns"][0]
        self.assertIn("ValueError: boom", turn["error"])
        self.assertEqual(turn["raw_recommendations"], [])

    def test_flags_response_the_evaluator_would_discard_whole(self) -> None:
        recorder = RecordingAgent(BadShapeAgent())
        recorder.reset("s1", {})
        recorder.respond("s1", "hello", 1, 10)

        self.assertTrue(recorder.sessions[0]["turns"][0]["discarded"])

    def test_respond_before_reset_is_an_explicit_error(self) -> None:
        recorder = RecordingAgent(DebugAgent())

        with self.assertRaises(RuntimeError):
            recorder.respond("s1", "hello", 1, 10)


if __name__ == "__main__":
    unittest.main()
