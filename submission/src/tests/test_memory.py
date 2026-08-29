from __future__ import annotations

import dataclasses
import unittest

from submission.src import dialogue
from submission.src import memory
from submission.src import policy
from submission.src import slots


def state(**fields) -> dialogue.SessionState:
    return dialogue.SessionState(**fields)


def refusal(attribute: str, value: str) -> slots.Slot:
    return slots.Slot(attribute, value, 1, True)


def preference(attribute: str, value: str) -> slots.Slot:
    return slots.Slot(attribute, value, 1, False)


class IdentityTest(unittest.TestCase):
    """The key is the person, never the profile blurb (findings 3.33).

    `user_profile` repeats across 75 of 200 public sessions, but those are
    distinct people whose aggregates coincide, so a store keyed on the blurb
    would merge 26 strangers into one customer. These two tests are the
    assertion that it is not, because it is the mistake the data invites.
    """

    def setUp(self) -> None:
        self.store = memory.Store()

    def _visit(self, shopper: str, value: str) -> None:
        self.store.remember(shopper)
        self.store.observe(state(slots=(refusal("material", value),)))

    def test_a_second_shopper_recalls_nothing_of_the_first(self) -> None:
        self._visit("alice", "not polyester")

        self.store.remember("bob")

        self.assertIsNone(self.store.recall())

    def test_one_shopper_recalls_across_visits(self) -> None:
        self._visit("alice", "not polyester")

        self.store.remember("alice")

        recalled = self.store.recall()
        self.assertIsNotNone(recalled)
        self.assertEqual(recalled.visits, 1)

    def test_an_anonymous_session_recalls_nothing(self) -> None:
        self._visit("alice", "not polyester")

        self.store.remember(None)

        self.assertIsNone(self.store.recall())

    def test_an_anonymous_session_writes_nothing(self) -> None:
        self.store.remember(None)

        self.store.observe(state(slots=(refusal("material", "not wool"),)))

        self.store.remember("alice")
        self.assertIsNone(self.store.recall())

    def test_forgetting_drops_every_shopper(self) -> None:
        self._visit("alice", "not polyester")

        self.store.forget()

        self.store.remember("alice")
        self.assertIsNone(self.store.recall())


class WriteTest(unittest.TestCase):
    """There is no end-of-session callback, so the write repeats per turn."""

    def setUp(self) -> None:
        self.store = memory.Store()
        self.store.remember("alice")

    def test_writing_the_same_visit_twice_does_not_count_it_twice(self) -> None:
        session = state(slots=(refusal("material", "not polyester"),))

        self.store.observe(session)
        self.store.observe(session)

        self.store.remember("alice")
        self.assertEqual(self.store.recall().visits, 1)

    def test_a_later_turn_replaces_what_an_earlier_one_recorded(self) -> None:
        self.store.observe(state(slots=(refusal("material", "not wool"),)))
        self.store.observe(state(slots=(
            refusal("material", "not wool"), refusal("color", "not red"),
        )))

        self.store.remember("alice")
        self.assertEqual(len(self.store.recall().refusals), 2)

    def test_a_second_visit_decays_the_first(self) -> None:
        self.store.observe(state(slots=(refusal("material", "not wool"),)))
        self.store.remember("alice")
        self.store.observe(state())

        self.store.remember("alice")
        weight = self.store.recall().refusals[0][2]
        self.assertAlmostEqual(weight, memory.DECAY)

    def test_a_memory_below_the_floor_is_dropped(self) -> None:
        self.store.observe(state(slots=(refusal("material", "not wool"),)))
        for _ in range(6):
            self.store.remember("alice")
            self.store.observe(state())

        self.store.remember("alice")
        self.assertEqual(self.store.recall().refusals, ())


class BoundsTest(unittest.TestCase):
    """Nothing here may grow across 800 sessions."""

    def test_the_store_evicts_its_oldest_shopper(self) -> None:
        store = memory.Store()
        for number in range(memory.MAX_SHOPPERS + 1):
            store.remember(f"shopper_{number}")
            store.observe(state(buckets=("Shirts",)))

        store.remember("shopper_0")
        self.assertIsNone(store.recall())

    def test_a_record_holds_no_more_than_the_cap(self) -> None:
        store = memory.Store()
        store.remember("alice")
        store.observe(state(slots=tuple(
            refusal("material", f"not fibre{number}")
            for number in range(memory.MAX_ENTRIES + 5)
        )))

        store.remember("alice")
        self.assertEqual(len(store.recall().refusals), memory.MAX_ENTRIES)

    def test_the_cap_is_taken_in_a_declared_order(self) -> None:
        """Never in set or dict order: hashing is salted per process."""
        store = memory.Store()
        store.remember("alice")
        store.observe(state(buckets=("b", "a", "c")))

        store.remember("alice")
        self.assertEqual(
            [key for key, _ in store.recall().buckets], ["a", "b", "c"]
        )


class SeedTest(unittest.TestCase):
    """What a remembered shopper opens a session on."""

    def test_an_empty_recall_opens_the_default_session(self) -> None:
        """The bit-identical guarantee, expressed as a shape."""
        self.assertEqual(memory.seed(None), dialogue.SessionState())

    def _seeded(self, session: dialogue.SessionState) -> dialogue.SessionState:
        store = memory.Store()
        store.remember("alice")
        store.observe(session)
        store.remember("alice")
        return memory.seed(store.recall())

    def test_a_remembered_refusal_arrives_as_a_negated_slot(self) -> None:
        opened = self._seeded(state(slots=(refusal("material", "not wool"),)))

        self.assertEqual(len(opened.carried), 1)
        self.assertTrue(opened.carried[0].negated)
        self.assertIn("wool", opened.excluded_text)

    def test_a_remembered_refusal_is_not_a_constraint(self) -> None:
        """Otherwise turn one opens in `precision` with nothing disclosed."""
        opened = self._seeded(state(slots=(
            refusal("material", "not wool"), refusal("color", "not red"),
        )))

        self.assertEqual(opened.constraints, ())
        self.assertEqual(policy.select(opened), policy.DISCOVERY)

    def test_a_remembered_dimension_is_not_a_declined_one(self) -> None:
        """`policy` reads `declined`; only `probe` may read what is carried."""
        opened = self._seeded(state(declined=("material",)))

        self.assertEqual(opened.carried_arms, ("material",))
        self.assertEqual(opened.declined, ())
        self.assertEqual(policy.select(opened), policy.DISCOVERY)

    def test_a_remembered_preference_is_withheld_by_default(self) -> None:
        opened = self._seeded(state(slots=(preference("material", "wool"),)))

        self.assertEqual(opened.carried_positives, ())

    def test_a_carried_preference_speaks_only_before_the_customer(self) -> None:
        self.addCleanup(
            setattr, memory, "CARRY_POSITIVES", memory.CARRY_POSITIVES)
        memory.CARRY_POSITIVES = True
        opened = self._seeded(state(slots=(preference("material", "wool"),)))

        self.assertIn("wool", opened.query_text)

        spoken = dataclasses.replace(
            opened, slots=(preference("color", "red"),)
        )
        self.assertNotIn("wool", spoken.query_text)


class GateTest(unittest.TestCase):
    """Every switch, and the branch behind it."""

    def test_the_shipped_gates_are_the_ones_measured(self) -> None:
        self.assertTrue(memory.ENABLED)
        self.assertTrue(memory.CARRY_REFUSALS)
        self.assertTrue(memory.CARRY_ARMS)
        self.assertTrue(memory.CARRY_BUCKETS)
        self.assertFalse(memory.CARRY_POSITIVES)

    def _recall(self) -> memory.Shopper:
        store = memory.Store()
        store.remember("alice")
        store.observe(state(
            slots=(refusal("material", "not wool"),),
            declined=("color",),
            buckets=("Shirts",),
        ))
        store.remember("alice")
        return store.recall()

    def test_disabling_memory_recalls_nothing(self) -> None:
        self.addCleanup(setattr, memory, "ENABLED", memory.ENABLED)
        memory.ENABLED = False

        store = memory.Store()
        store.remember("alice")
        store.observe(state(buckets=("Shirts",)))
        store.remember("alice")

        self.assertIsNone(store.recall())

    def test_refusals_can_be_withheld(self) -> None:
        self.addCleanup(
            setattr, memory, "CARRY_REFUSALS", memory.CARRY_REFUSALS)
        memory.CARRY_REFUSALS = False

        self.assertEqual(memory.seed(self._recall()).carried, ())

    def test_dimensions_can_be_withheld(self) -> None:
        self.addCleanup(setattr, memory, "CARRY_ARMS", memory.CARRY_ARMS)
        memory.CARRY_ARMS = False

        self.assertEqual(memory.seed(self._recall()).carried_arms, ())

    def test_bucket_affinity_can_be_withheld(self) -> None:
        self.addCleanup(setattr, memory, "CARRY_BUCKETS", memory.CARRY_BUCKETS)
        memory.CARRY_BUCKETS = False

        self.assertEqual(memory.affinity(self._recall()), {})

    def test_bucket_affinity_reports_what_was_shopped(self) -> None:
        self.assertEqual(list(memory.affinity(self._recall())), ["Shirts"])


class PivotTest(unittest.TestCase):
    """A redirect outranks memory for the reason it outranks this session."""

    def test_a_pivot_drops_every_carried_field(self) -> None:
        opened = dialogue.SessionState(
            carried=(refusal("material", "not wool"),),
            carried_arms=("color",),
            carried_positives=("wool",),
        )

        after = dialogue.update(opened, dialogue.ParsedTurn(pivot=True))

        self.assertEqual(after.carried, ())
        self.assertEqual(after.carried_arms, ())
        self.assertEqual(after.carried_positives, ())

    def test_an_ordinary_turn_keeps_them(self) -> None:
        opened = dialogue.SessionState(
            carried=(refusal("material", "not wool"),),
            carried_arms=("color",),
        )

        after = dialogue.update(opened, dialogue.ParsedTurn())

        self.assertEqual(len(after.carried), 1)
        self.assertEqual(after.carried_arms, ("color",))


if __name__ == "__main__":
    unittest.main()
