from __future__ import annotations

import unittest

from submission.src import dialogue
from submission.src import policy
from submission.src import response
from submission.src import slots


def reply(state, parsed, contenders=1, head=1, served=10, asked="other",
          policy_name=policy.DISCOVERY, options=(), names=(), size=10):
    return response.compose(state, parsed, contenders, head, served, asked,
                            policy_name, options, names, size)


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

    def test_a_held_back_slate_never_calls_one_item_matches(self) -> None:
        """The bug this replaced: `head >= served` was true on every turn once
        `ranking.EXPLORE_FILL` was switched off, so a single-item slate
        described itself as "the 1 closest matches" on 429 of 482 turns."""
        text = reply(dialogue.SessionState(turn=1), dialogue.ParsedTurn(),
                     head=1, served=1)

        self.assertNotIn("1 closest matches", text)
        self.assertIn("closest match I can justify so far", text)

    def test_a_held_back_slate_names_what_it_is_showing(self) -> None:
        text = reply(dialogue.SessionState(turn=1), dialogue.ParsedTurn(),
                     head=1, served=1,
                     names=("Columbia Men's BugabootPlus III Omni Boot",))

        self.assertIn("Columbia Men's BugabootPlus III Omni", text)

    def test_a_short_slate_names_all_of_it(self) -> None:
        text = reply(dialogue.SessionState(turn=1), dialogue.ParsedTurn(),
                     head=2, served=2,
                     names=("NIKE Men's Layup 2 Shorts", "Pro Club Mesh Tee"))

        self.assertIn("NIKE Men's Layup 2 Shorts", text)
        self.assertIn("Pro Club Mesh Tee", text)

    def test_variants_of_one_product_are_collapsed(self) -> None:
        """Two ASINs sharing a title is a colour/size variant, not a choice."""
        title = "Columbia Men's BugabootPlus III Omni Cold-Weather Boot"
        text = reply(dialogue.SessionState(turn=1), dialogue.ParsedTurn(),
                     head=2, served=2, names=(title, title))

        self.assertIn("(2 variants)", text)
        self.assertEqual(text.count("BugabootPlus"), 1)

    def test_a_committed_slate_names_only_the_first(self) -> None:
        text = reply(dialogue.SessionState(exhausted=True), dialogue.ParsedTurn(),
                     head=10, served=10, asked=None,
                     names=tuple(f"Product {i}" for i in range(10)))

        self.assertIn("all 10 closest matches", text)
        self.assertIn("Product 0", text)
        self.assertNotIn("Product 5", text)

    def test_a_bare_brand_token_is_not_used_as_a_name(self) -> None:
        text = reply(dialogue.SessionState(turn=1), dialogue.ParsedTurn(),
                     head=1, served=1, names=("MxG - Womens Cotton Tank Top",))

        self.assertNotIn(": MxG.", text)


class ThreeLineShapeTest(unittest.TestCase):
    """Acknowledgement, slate and question each get their own line."""

    def test_the_question_is_on_its_own_line(self) -> None:
        text = reply(
            dialogue.SessionState(category="Tops Blouses", turn=1),
            dialogue.ParsedTurn(constraints=("leather",)),
            head=1, served=1, names=("Lavemi Leather Belt",),
        )
        lines = text.split("\n")

        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("Got it,"))
        self.assertIn("Lavemi", lines[1])
        self.assertTrue(lines[2].endswith("?"))

    def test_an_absent_part_leaves_no_blank_line(self) -> None:
        text = reply(dialogue.SessionState(), dialogue.ParsedTurn(),
                     head=10, served=10, asked=None)

        self.assertNotIn("\n\n", text)
        self.assertFalse(text.startswith("\n"))


class ReadableTextTest(unittest.TestCase):
    """Catalog text is spaced for reading, but never stripped of meaning.

    The acknowledgement is the only place a customer can catch a misparse, so
    a field prefix is kept: `Item model number: G796` without its prefix is
    just `G796`.
    """

    def test_a_field_prefix_survives(self) -> None:
        self.assertEqual(response._short("Material:alloy"), "material: alloy")

    def test_a_prefix_that_carries_the_meaning_survives(self) -> None:
        self.assertIn("item model number",
                      response._short("Item model number: G796"))

    def test_jammed_commas_are_spaced(self) -> None:
        self.assertEqual(response._short("polyester,cotton"),
                         "polyester, cotton")

    def test_a_thousands_separator_is_left_alone(self) -> None:
        self.assertEqual(response._short("3,000 count"), "3,000 count")

    def test_a_decimal_is_left_alone(self) -> None:
        self.assertIn("8.37", response._short('8.37" shaft'))

    def test_a_jammed_percentage_is_spaced(self) -> None:
        self.assertEqual(response._short("95%cotton"), "95% cotton")

    def test_an_over_long_value_is_cut_at_a_clause(self) -> None:
        text = response._short(
            "Solids: 100% Cotton; heathers: 75% cotton, 25% polyester"
        )

        self.assertEqual(text, "solids: 100% cotton")
        self.assertNotIn("...", text)

    def test_prefixed_values_are_joined_without_ambiguity(self) -> None:
        """"material: alloy and buckle closure" reads as one material."""
        self.assertIn(";", response._listed(("Material:alloy", "Buckle closure")))
        self.assertIn(" and ", response._listed(("rubber sole", "buckle closure")))


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
    sees (measurements 3.46).
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


class PunctuationTest(unittest.TestCase):
    """A trimmed value carries its own ellipsis; a full stop after it reads
    as "coverage....", which appeared on five turns before `_stopped`."""

    def test_a_trimmed_constraint_is_not_double_stopped(self) -> None:
        long_value = "long torso camisole for extra coverage and a bit more"
        text = reply(
            dialogue.SessionState(turn=1),
            dialogue.ParsedTurn(constraints=(long_value,)),
            head=1, served=1,
        )

        self.assertNotIn("....", text)

    def test_a_trimmed_pivot_is_not_double_stopped(self) -> None:
        long_value = "long torso camisole for extra coverage and a bit more"
        text = reply(
            dialogue.SessionState(turn=3, superseded=("cotton",)),
            dialogue.ParsedTurn(constraints=(long_value,), pivot=True),
            head=1, served=1,
        )

        self.assertNotIn("....", text)


class AttributeLabelTest(unittest.TestCase):
    """The acknowledgement names the attribute each constraint was filed
    under, so a misfiling is visible on the turn it happens.

    "leather" alone does not show whether it was read as a material or a
    brand; "material: leather" does.
    """

    def _state(self, slots):
        return dialogue.SessionState(
            turn=1, constraints=tuple(s.value for s in slots), slots=tuple(slots)
        )

    def test_a_constraint_is_named_by_its_attribute(self) -> None:
        slot = slots.Slot("material", "leather", 1)
        text = reply(self._state([slot]),
                     dialogue.ParsedTurn(constraints=("leather",)),
                     head=1, served=1)

        self.assertIn("material: leather", text)

    def test_a_value_with_its_own_prefix_gains_no_second_one(self) -> None:
        slot = slots.Slot("feature", "Material:alloy", 1)
        text = reply(self._state([slot]),
                     dialogue.ParsedTurn(constraints=("Material:alloy",)),
                     head=1, served=1)

        self.assertIn("material: alloy", text)
        self.assertNotIn("feature: material", text)

    def test_values_sharing_an_attribute_are_named_once(self) -> None:
        pair = [slots.Slot("feature", "water resistant", 1),
                slots.Slot("feature", "3 year battery", 1)]
        text = reply(self._state(pair),
                     dialogue.ParsedTurn(
                         constraints=("water resistant", "3 year battery")),
                     head=1, served=1)

        self.assertIn("feature: water resistant and 3 year battery", text)
        self.assertEqual(text.count("feature:"), 1)

    def test_a_restated_constraint_still_finds_its_attribute(self) -> None:
        """A repeat keeps the turn it first arrived on, so the lookup must
        span every slot rather than only this turn's."""
        old = slots.Slot("feature", "stainless steel band", 1)
        new = slots.Slot("feature", "day / date indicator", 3)
        state = dialogue.SessionState(
            turn=3, constraints=(old.value, new.value), slots=(old, new))
        text = reply(state, dialogue.ParsedTurn(
            constraints=("day / date indicator", "stainless steel band")),
            head=1, served=1)

        self.assertIn("feature: day / date indicator and stainless steel band",
                      text)

    def test_an_unclassified_constraint_is_still_spoken(self) -> None:
        text = reply(dialogue.SessionState(turn=1, constraints=("leather",)),
                     dialogue.ParsedTurn(constraints=("leather",)),
                     head=1, served=1)

        self.assertIn("leather", text)
