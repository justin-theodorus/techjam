from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from submission.src import agent as agent_module
from submission.src import probe
from submission.src import dialogue
from submission.src import memory
from submission.src import ranking
from submission.src.tests import fixtures

OPENING = f"I'm looking for {fixtures.SNEAKER_BUCKET}, but I'm still exploring."
# The evaluator's own boundary refusal, restated rather than imported:
# the submission bundle ships without `evaluator/`.
REFUSAL = ("I don't have a preference for material; "
           "please use your judgment.")

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

    def test_every_turn_emits_a_valid_non_empty_slate(self) -> None:
        """Length is the head's to decide; validity and uniqueness are not.

        A narrowing turn serves its committed head alone under
        `ranking.EXPLORE_FILL`, so the invariant is that every slate is
        non-empty, within budget, free of repeats and drawn from the catalog.
        """
        self.agent.reset("s1", {})
        messages = [
            OPENING,
            DISCLOSURE,
            "I don't have an additional preference for other.",
        ]
        for turn, message in enumerate(messages, start=1):
            asins = _asins(self.agent.respond("s1", message, turn, 10))
            self.assertGreaterEqual(len(asins), 1)
            self.assertLessEqual(len(asins), ranking.SLATE_SIZE)
            self.assertEqual(len(set(asins)), len(asins))
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
        self.assertEqual(len(_asins(response)), ranking.HEAD_SIZE)

    def test_a_failing_stage_degrades_to_the_previous_slate(self) -> None:
        """The rung needs a full previous slate, so open the head first.

        A narrowing turn now serves its committed head alone, and `_degrade`
        takes this rung only for a slate of `SLATE_SIZE` -- below that the pool
        offers more slots, and on an error path every slot is a free chance at
        a hit.
        """
        self.agent.reset("s1", {})
        self.agent.respond("s1", OPENING, 1, 10)
        expected = _asins(self.agent.respond(
            "s1", "I don't have an additional preference for other.", 2, 10
        ))
        self.assertEqual(len(expected), ranking.SLATE_SIZE)
        self._break_ranking()
        served = _asins(self.agent.respond("s1", OPENING, 3, 10))
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
            self.assertGreaterEqual(len(_asins(response)), 1)

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


class MemoryTest(unittest.TestCase):
    """Per-person memory, and the guarantee that it is off by default.

    The organizer's harness never names a shopper, so the reported score must
    not depend on this existing. That is asserted here as "not reached" rather
    than as "neutral", which is the stronger of the two claims (findings 3.33).
    """

    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.path = fixtures.write_catalog(Path(root.name))
        self.agent = agent_module.Agent(str(self.path))

    def _drive(self, session: str, *messages: str) -> list[dict]:
        self.agent.reset(session, {})
        return [
            self.agent.respond(session, message, turn, 10)
            for turn, message in enumerate(messages, start=1)
        ]

    def test_no_identity_opens_at_the_default_state(self) -> None:
        self.agent.reset("s1", {})

        self.assertEqual(self.agent._state, dialogue.SessionState())

    def test_an_unnamed_run_serves_what_an_unnamed_run_served(self) -> None:
        first = self._drive("s1", OPENING, "cotton upper")
        second = self._drive("s2", OPENING, "cotton upper")

        self.assertEqual(first, second)

    def test_the_same_profile_twice_is_two_strangers(self) -> None:
        """The blurb repeats across public sessions; identity does not."""
        profile = {"preference_tags": ["cotton"]}
        self.agent.reset("s1", profile)
        self.agent.respond("s1", OPENING, 1, 10)

        self.agent.reset("s2", profile)

        self.assertEqual(self.agent._state, dialogue.SessionState())

    def test_a_returning_shopper_opens_on_what_it_learned(self) -> None:
        self.agent.remember("alice")
        self._drive("s1", OPENING, REFUSAL)

        self.agent.remember("alice")
        self.agent.reset("s2", {})

        self.assertTrue(self.agent._state.carried_arms)

    def test_forgetting_returns_the_agent_to_an_unnamed_one(self) -> None:
        self.agent.remember("alice")
        self._drive("s1", OPENING, REFUSAL)

        self.agent.forget()
        self.agent.reset("s2", {})

        self.assertEqual(self.agent._state, dialogue.SessionState())

    def test_memory_switched_off_serves_the_unnamed_slate(self) -> None:
        self.addCleanup(setattr, memory, "ENABLED", memory.ENABLED)
        self.agent.remember("alice")
        self._drive("s1", OPENING, REFUSAL)
        memory.ENABLED = False

        self.agent.remember("alice")
        self.agent.reset("s2", {})

        self.assertEqual(self.agent._state, dialogue.SessionState())

    def test_a_session_nobody_opened_is_shopped_anonymously(self) -> None:
        """The caller lost the boundary, so its last identity is not trusted."""
        self.agent.remember("alice")
        self._drive("s1", OPENING, REFUSAL)
        self.agent.remember("alice")

        self.agent.respond("s2", OPENING, 1, 10)

        self.assertEqual(self.agent._state.carried_arms, ())
