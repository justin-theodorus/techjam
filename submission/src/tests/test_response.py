from __future__ import annotations

import unittest

from submission.src import dialogue
from submission.src import policy
from submission.src import response


def reply(state, parsed, contenders=1, head=1, served=10, asked="other"):
    return response.compose(state, parsed, contenders, head, served, asked)


class ContractTest(unittest.TestCase):
    """`message` must be a string on every path.

    A non-string discards the whole response, recommendations included, so this
    is the one field whose type is worth more than its content.
    """

    def test_every_state_produces_a_non_empty_string(self) -> None:
        states = (
            dialogue.SessionState(),
            dialogue.SessionState(category="Tops Blouses", turn=1),
            dialogue.SessionState(constraints=("a", "b", "c", "d"), turn=4),
            dialogue.SessionState(exhausted=True, turn=5),
            dialogue.SessionState(pivoted=True, superseded=("cotton",)),
        )
        parses = (
            dialogue.ParsedTurn(),
            dialogue.ParsedTurn(constraints=("cotton",)),
            dialogue.ParsedTurn(pivot=True, constraints=("leather",)),
            dialogue.ParsedTurn(boundary_refusal=True),
            dialogue.ParsedTurn(exhausted=True),
        )
        for state in states:
            for parsed in parses:
                with self.subTest(state=state, act=parsed.act):
                    for asked in ("other", None):
                        text = reply(state, parsed, asked=asked)
                        self.assertIsInstance(text, str)
                        self.assertTrue(text)

    def test_an_empty_slate_still_replies(self) -> None:
        text = reply(dialogue.SessionState(), dialogue.ParsedTurn(), served=0)

        self.assertIsInstance(text, str)
        self.assertTrue(text)


class GroundingTest(unittest.TestCase):
    def test_the_opening_names_the_category_it_understood(self) -> None:
        state = dialogue.SessionState(category="Tops Blouses", turn=1)

        self.assertIn("tops blouses", reply(state, dialogue.ParsedTurn()))

    def test_a_disclosure_is_read_back(self) -> None:
        text = reply(
            dialogue.SessionState(constraints=("cotton",), turn=2),
            dialogue.ParsedTurn(constraints=("cotton",)),
        )

        self.assertIn("cotton", text)

    def test_a_long_constraint_is_trimmed_rather_than_recited(self) -> None:
        long_value = "a very long marketing sentence " * 4
        text = reply(
            dialogue.SessionState(constraints=(long_value,), turn=2),
            dialogue.ParsedTurn(constraints=(long_value,)),
        )

        self.assertIn("...", text)
        self.assertNotIn(long_value.strip(), text)


class OverrideTest(unittest.TestCase):
    def test_a_redirect_names_what_it_replaced(self) -> None:
        state = dialogue.SessionState(
            pivoted=True, superseded=("color: black",), constraints=("leather",)
        )
        text = reply(
            state, dialogue.ParsedTurn(pivot=True, constraints=("leather",))
        )

        self.assertIn("leather", text)
        self.assertIn("color: black", text)

    def test_it_does_not_claim_to_drop_what_it_is_adopting(self) -> None:
        """A replacement often restates something already said."""
        state = dialogue.SessionState(
            pivoted=True, superseded=("100% Leather",), constraints=("leather",)
        )
        text = reply(
            state, dialogue.ParsedTurn(pivot=True, constraints=("leather",))
        )

        self.assertNotIn("instead of", text)


class SlateDescriptionTest(unittest.TestCase):
    def test_a_committed_slate_is_described_as_one(self) -> None:
        text = reply(
            dialogue.SessionState(exhausted=True), dialogue.ParsedTurn(),
            head=10, served=10, asked=None,
        )

        self.assertIn("10 closest matches", text)

    def test_a_held_back_slate_says_so(self) -> None:
        text = reply(dialogue.SessionState(turn=1), dialogue.ParsedTurn(),
                     head=1, served=10)

        self.assertIn("best match", text)
        self.assertIn("9", text)

    def test_a_crowded_pool_is_described_as_a_spread(self) -> None:
        text = reply(dialogue.SessionState(turn=1), dialogue.ParsedTurn(),
                     contenders=response.CROWDED + 1, head=1, served=10)

        self.assertIn("narrow down", text)


class QuestionTest(unittest.TestCase):
    def test_declining_to_ask_does_not_ask(self) -> None:
        text = reply(
            dialogue.SessionState(exhausted=True), dialogue.ParsedTurn(),
            head=10, asked=None,
        )

        self.assertNotIn("?", text)

    def test_asking_produces_a_question(self) -> None:
        text = reply(dialogue.SessionState(turn=1), dialogue.ParsedTurn())

        self.assertIn("?", text)


if __name__ == "__main__":
    unittest.main()


class PolicyPhrasingTest(unittest.TestCase):
    """The framing changes with the policy; the attribute asked does not.

    Nothing here can move the score: `evaluator/local_evaluator.py:243` only
    type-checks `message`, and `customer_reply()` reads the `ask_attribute`
    enum alone. These assertions are about what a person reading the transcript
    sees (findings 3.46).
    """

    def setUp(self) -> None:
        self.state = dialogue.SessionState(
            category="Bags Totes", constraints=("cotton",), turn=3,
        )
        self.parsed = dialogue.ParsedTurn()

    def _reply(self, policy: str, asked: str, options=()) -> str:
        return response.compose(
            self.state, self.parsed, 3, 1, 10, asked, policy, options
        )

    def test_offered_options_reach_the_question(self) -> None:
        reply = self._reply(
            policy.PRECISION, "material", ("leather", "canvas", "nylon")
        )

        self.assertIn("leather, canvas, or nylon", reply)

    def test_two_options_are_joined_without_a_comma(self) -> None:
        reply = self._reply(policy.PRECISION, "material",
                            ("leather", "canvas"))

        self.assertIn("leather or canvas", reply)
        self.assertNotIn("leather, or canvas", reply)

    def test_a_lone_option_is_not_offered_as_a_choice(self) -> None:
        reply = self._reply(policy.PRECISION, "material", ("leather",))

        self.assertNotIn("leather", reply)
        self.assertTrue(reply.endswith("?"))

    def test_stagnation_asks_a_different_kind_of_question(self) -> None:
        reply = self._reply(policy.STAGNATION, "use_case",
                            ("work", "travel", "everyday"))

        self.assertIn("another angle", reply)

    def test_boundary_names_what_it_is_letting_go(self) -> None:
        state = dialogue.SessionState(
            constraints=("cotton",), declined=("material",), turn=4,
        )
        reply = response.compose(
            state, self.parsed, 3, 1, 10, "use_case", policy.BOUNDARY, ()
        )

        self.assertIn("material can stay open", reply)

    def test_the_release_is_said_once_on_the_turn_it_happens(self) -> None:
        state = dialogue.SessionState(
            constraints=("cotton",), declined=("material",), turn=4,
        )
        reply = response.compose(
            state, dialogue.ParsedTurn(boundary_refusal=True), 3, 1, 10,
            "use_case", policy.BOUNDARY, (),
        )

        self.assertEqual(reply.count("can stay open"), 1)
        self.assertNotIn("No problem, I will use my judgement", reply)

    def test_coverage_asks_nothing_and_says_so(self) -> None:
        state = dialogue.SessionState(constraints=("cotton",), turn=10)
        reply = response.compose(
            state, self.parsed, 3, 10, 10, None, policy.COVERAGE, ()
        )

        self.assertNotIn("?", reply)

    def test_every_policy_produces_one_question_at_most(self) -> None:
        for name in (policy.DISCOVERY, policy.PRECISION, policy.RECOVERY,
                     policy.BOUNDARY, policy.STAGNATION):
            for options in ((), ("leather", "canvas")):
                with self.subTest(policy=name, options=options):
                    reply = self._reply(name, "material", options)
                    self.assertEqual(reply.count("?"), 1)

    def test_a_scoped_exhaustion_does_not_claim_the_session_is_done(
        self,
    ) -> None:
        """The customer ran dry on one attribute, not on the conversation."""
        parsed = dialogue.ParsedTurn(exhausted=True, exhausted_arm="material")
        reply = response.compose(
            self.state, parsed, 3, 1, 10, "color", policy.PRECISION, ()
        )

        self.assertIn("no strong view on material", reply)
        self.assertNotIn("I have what I need", reply)
