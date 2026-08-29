from __future__ import annotations

import unittest

from harness import deviations
from harness import returning


def rows(count: int, visits: int = 3) -> list[dict]:
    """`count` shoppers laid out visit-major, as the axis emits them."""
    return [
        {
            "shopper_id": f"shopper_{position % count:04d}",
            "visit": position // count + 1,
        }
        for position in range(count * visits)
    ]


class ScrambleTest(unittest.TestCase):
    """The control, and the property that makes it one.

    It must differ from the reading it controls in exactly one variable: who
    wrote the memory each session reads. A separately seeded set would differ
    in its whole draw as well, which is why the control is a permutation of
    these rows rather than a second frozen set.
    """

    def test_the_rows_keep_their_order_and_their_visits(self) -> None:
        original = rows(4)

        rotated = returning.scrambled(original)

        self.assertEqual(
            [row["visit"] for row in rotated],
            [row["visit"] for row in original],
        )

    def test_every_shopper_still_owns_one_session_per_block(self) -> None:
        rotated = returning.scrambled(rows(4))
        seen: dict[str, list[int]] = {}
        for row in rotated:
            seen.setdefault(row["shopper_id"], []).append(row["visit"])

        self.assertEqual(len(seen), 4)
        for shopper, visits in seen.items():
            with self.subTest(shopper=shopper):
                self.assertEqual(sorted(visits), [1, 2, 3])

    def test_no_shopper_keeps_its_own_later_visits(self) -> None:
        original = rows(4)

        rotated = returning.scrambled(original)

        moved = [
            after["shopper_id"] != before["shopper_id"]
            for before, after in zip(original, rotated)
            if before["visit"] > 1
        ]
        self.assertTrue(all(moved))

    def test_the_first_visit_block_is_left_alone(self) -> None:
        """Nothing is remembered yet, so rotating it would change nothing."""
        original = rows(4)

        rotated = returning.scrambled(original)

        self.assertEqual(
            [row["shopper_id"] for row in rotated[:4]],
            [row["shopper_id"] for row in original[:4]],
        )

    def test_rows_naming_nobody_are_returned_unchanged(self) -> None:
        plain = [{"sample_id": "a"}, {"sample_id": "b"}]

        self.assertEqual(returning.scrambled(plain), plain)


class BlockTest(unittest.TestCase):
    """One visit block's four numbers."""

    def test_a_block_that_never_converts_reports_the_turn_ceiling(self) -> None:
        block = returning._block([
            {"hit": False, "first_hit_turn": None, "reciprocal_rank": 0.0,
             "best_rank": None},
        ])

        self.assertEqual(block["hit"], 0.0)
        self.assertEqual(block["mttc"], 10)

    def test_a_block_averages_only_the_sessions_that_converted(self) -> None:
        block = returning._block([
            {"hit": True, "first_hit_turn": 2, "reciprocal_rank": 1.0,
             "best_rank": 1},
            {"hit": False, "first_hit_turn": None, "reciprocal_rank": 0.0,
             "best_rank": None},
        ])

        self.assertEqual(block["mttc"], 2)
        self.assertEqual(block["hit"], 0.5)
        self.assertEqual(block["rank1"], 0.5)
        self.assertEqual(block["mrr"], 0.5)


class SweepTest(unittest.TestCase):
    def test_the_sweep_reads_every_gate_one_at_a_time(self) -> None:
        """An isolated cell is what makes a gate's zero mean "no input"."""
        memory = next(
            item for item in deviations.DEVIATIONS if item.name == "memory")

        for label, assignments in memory.points:
            with self.subTest(point=label):
                self.assertEqual(len(assignments), 1)


class ReadingTest(unittest.TestCase):
    def test_the_control_carries_the_switches_it_controls(self) -> None:
        """A control for a switch that does nothing would prove nothing."""
        readings = dict(
            (name, assignments) for name, assignments, _ in returning.READINGS
        )

        self.assertEqual(readings["scrambled"], readings["positives"])

    def test_exactly_one_reading_rotates_the_identities(self) -> None:
        rotated = [name for name, _, rotate in returning.READINGS if rotate]

        self.assertEqual(rotated, ["scrambled"])


if __name__ == "__main__":
    unittest.main()
