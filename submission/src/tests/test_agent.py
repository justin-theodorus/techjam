from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from techjam.submission.src import agent as agent_module
from techjam.submission.src import probe
from techjam.submission.src import ranking
from techjam.submission.src.tests import fixtures

OPENING = f"I'm looking for {fixtures.SNEAKER_BUCKET}, but I'm still exploring."

# `local_evaluator.ALLOWED_ATTRIBUTES`, restated rather than imported: the
# submission bundle ships without the evaluator.
ALLOWED_ATTRIBUTES = frozenset((
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
))
DISCLOSURE = "For that, what matters is: hemp upper; cork footbed."


def _asins(response: dict) -> list[str]:
    return [item["parent_asin"] for item in response["recommendations"]]


def _explode(*args: object, **kwargs: object) -> list[int]:
    raise RuntimeError("ranking blew up")


class AgentTest(unittest.TestCase):
    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = fixtures.write_catalog(Path(root.name))
        self.agent = agent_module.Agent(str(path))

    def _break_ranking(self) -> None:
        original = ranking.slate
        ranking.slate = _explode
        self.addCleanup(setattr, ranking, "slate", original)

    def test_a_turn_returns_exactly_the_four_contract_keys(self) -> None:
        self.agent.reset("s1", {})
        response = self.agent.respond("s1", OPENING, 1, 10)
        self.assertEqual(
            set(response),
            {"message", "ask_attribute", "recommendations", "usage"},
        )
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES)
        self.assertEqual(
            response["usage"], {"prompt_tokens": 0, "completion_tokens": 0}
        )

    def test_every_turn_emits_a_full_slate_of_unique_ids(self) -> None:
        self.agent.reset("s1", {})
        messages = [
            OPENING,
            DISCLOSURE,
            "I don't have an additional preference for other.",
        ]
        for turn, message in enumerate(messages, start=1):
            asins = _asins(self.agent.respond("s1", message, turn, 10))
            self.assertEqual(len(asins), ranking.SLATE_SIZE)
            self.assertEqual(len(set(asins)), ranking.SLATE_SIZE)
            self.assertTrue(set(asins) <= self.agent.catalog.ids)

    def test_recommendation_items_carry_only_contract_fields(self) -> None:
        self.agent.reset("s1", {})
        for item in self.agent.respond("s1", OPENING, 1, 10)["recommendations"]:
            self.assertEqual(set(item), {"parent_asin", "score"})
            self.assertIsInstance(item["score"], float)

    def test_every_probe_names_an_attribute_the_evaluator_accepts(
        self
    ) -> None:
        """A word outside the enum is silently read as the wildcard."""
        self.agent.reset("s1", {})
        for turn in range(1, 11):
            response = self.agent.respond("s1", OPENING, turn, 10)
            with self.subTest(turn=turn):
                self.assertIn(
                    response["ask_attribute"], ALLOWED_ATTRIBUTES | {None}
                )

    def test_the_probe_never_reaches_for_the_wildcard(self) -> None:
        """What `probe.SPECIFIC_ARMS` buys, and the whole reason it costs."""
        self.agent.reset("s1", {})
        asked = [
            self.agent.respond("s1", OPENING, turn, 10)["ask_attribute"]
            for turn in range(1, 6)
        ]

        self.assertNotIn(probe.WILDCARD, asked)

    def test_switching_specific_arms_off_restores_the_wildcard(self) -> None:
        """The reported 0.9554 is this branch, so it has to stay reachable."""
        original = probe.SPECIFIC_ARMS
        probe.SPECIFIC_ARMS = False
        self.addCleanup(setattr, probe, "SPECIFIC_ARMS", original)

        self.agent.reset("s1", {})
        response = self.agent.respond("s1", OPENING, 1, 10)

        self.assertEqual(response["ask_attribute"], probe.WILDCARD)

    def test_a_customer_out_of_preferences_stops_being_asked(self) -> None:
        self.agent.reset("s1", {})
        self.agent.respond("s1", OPENING, 1, 10)
        exhausted = "I don't have an additional preference for other."
        response = self.agent.respond("s1", exhausted, 2, 10)

        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(len(_asins(response)), ranking.SLATE_SIZE)

    def test_disclosed_constraints_move_the_slate(self) -> None:
        self.agent.reset("s1", {})
        before = _asins(self.agent.respond("s1", OPENING, 1, 10))
        after = _asins(self.agent.respond("s1", DISCLOSURE, 2, 10))
        self.assertEqual(before[0], "SNEAK_POP")
        self.assertEqual(after[0], "SNEAK_RARE")

    def test_reset_clears_state_between_sessions(self) -> None:
        self.agent.reset("s1", {})
        self.agent.respond("s1", OPENING, 1, 10)
        self.agent.respond("s1", DISCLOSURE, 2, 10)
        self.agent.reset("s2", {})
        served = _asins(self.agent.respond("s2", OPENING, 1, 10))
        self.assertEqual(served[0], "SNEAK_POP")

    def test_an_unknown_session_id_starts_clean(self) -> None:
        response = self.agent.respond("never-reset", OPENING, 1, 10)
        self.assertEqual(len(_asins(response)), ranking.SLATE_SIZE)

    def test_a_failing_stage_degrades_to_the_previous_slate(self) -> None:
        self.agent.reset("s1", {})
        expected = _asins(self.agent.respond("s1", OPENING, 1, 10))
        self._break_ranking()
        served = _asins(self.agent.respond("s1", OPENING, 2, 10))
        self.assertEqual(served, expected)
        self.assertEqual(self.agent.debug["degraded"], "last_slate")

    def test_a_failure_after_a_short_slate_degrades_to_the_pool(self) -> None:
        self.agent.reset("s1", {})
        opening = (
            f"I'm looking for {fixtures.FILLER_BUCKET}, "
            "but I'm still exploring."
        )
        self.agent.respond("s1", opening, 1, 3)
        self._break_ranking()
        response = self.agent.respond("s1", opening, 2, 10)
        self.assertEqual(self.agent.debug["degraded"], "pool")
        self.assertEqual(len(_asins(response)), ranking.SLATE_SIZE)

    def test_a_failure_on_turn_one_degrades_to_global_popularity(self) -> None:
        self.agent.reset("s1", {})
        self._break_ranking()
        response = self.agent.respond("s1", OPENING, 1, 10)
        self.assertEqual(self.agent.debug["degraded"], "global")
        self.assertEqual(len(_asins(response)), ranking.SLATE_SIZE)
        self.assertIsInstance(response["message"], str)

    def test_a_malformed_message_never_raises_and_still_scores(self) -> None:
        self.agent.reset("s1", {})
        for message in (None, "", 42, "   ", " garbage"):
            response = self.agent.respond("s1", message, 1, 10)
            self.assertIsInstance(response["message"], str)
            self.assertEqual(len(_asins(response)), ranking.SLATE_SIZE)

    def test_the_debug_dict_carries_flat_scalars_for_the_trace(self) -> None:
        self.agent.reset("s1", {})
        self.agent.respond("s1", OPENING, 1, 10)
        self.assertEqual(self.agent.debug["bucket"], fixtures.SNEAKER_BUCKET)
        self.assertEqual(self.agent.debug["pool"], 3)
        for value in self.agent.debug.values():
            self.assertIsInstance(value, (str, int, float, bool))


if __name__ == "__main__":
    unittest.main()


class PayloadTest(unittest.TestCase):
    """Every slot is a free chance at a hit, so none may be lost quietly."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.agent = agent_module.Agent(
            fixtures.write_catalog(Path(self.directory.name))
        )
        self.addCleanup(self.directory.cleanup)

    def test_a_short_scores_list_does_not_truncate_the_slate(self) -> None:
        self.agent.reset("s1", {})
        self.agent.respond("s1", OPENING, 1, 10)
        self.agent._scores = (1.0,)

        payload = self.agent._payload(("A", "B", "C"))

        self.assertEqual(len(payload), 3)
        self.assertEqual(payload[1]["score"], 0.0)

    def test_the_profile_is_read_and_survives_a_malformed_one(self) -> None:
        self.agent.reset("s1", {"preference_tags": ["cotton", "canvas"]})
        self.assertTrue(self.agent._profile_ids)

        for profile in ({}, {"preference_tags": "cotton"}, None, []):
            with self.subTest(profile=profile):
                self.agent.reset("s2", profile)
                self.assertEqual(self.agent._profile_ids, frozenset())

    def test_a_profile_carrying_no_catalog_vocabulary_is_empty(self) -> None:
        self.agent.reset("s1", {"preference_tags": ["zzzznotaword"]})

        self.assertEqual(self.agent._profile_ids, frozenset())

    def test_the_message_is_grounded_in_what_was_understood(self) -> None:
        self.agent.reset("s1", {})
        response_body = self.agent.respond("s1", OPENING, 1, 10)

        self.assertIn(fixtures.SNEAKER_BUCKET.lower(),
                      response_body["message"].lower())
