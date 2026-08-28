from __future__ import annotations

import unittest

from submission.src import dialogue
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
