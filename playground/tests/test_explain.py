"""The playground's own health line.

The explanation on screen is a replay of the agent's stages, not a reading of
them, so it is only worth anything while the replay still reproduces what was
served. These tests are what makes that a claim rather than a hope: if a future
change to `ranking` or `probe` moves a stage the replay does not follow, the
`verified` flag goes false here before it goes false on camera.
"""

from __future__ import annotations

import unittest

from evaluator import local_evaluator
from playground import driver
from playground import explain
from submission.src import probe

# Enough sessions to cover every scenario type and the pivot, few enough that
# the suite stays in the same time class as the rest of `make test`.
SAMPLE_COUNT = 12

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"


class ExplainTest(unittest.TestCase):
    """Replaying the pipeline must reproduce the pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = local_evaluator.load_jsonl(DATASET)
        cls.dataset = local_evaluator.catalog_index(CATALOG)
        cls.agent = explain.ExplainingAgent(CATALOG)
        cls.replays = [
            driver.replay(cls.agent, sample, cls.dataset)
            for sample in cls._spread(cls.samples, SAMPLE_COUNT)
        ]

    @staticmethod
    def _spread(samples: list[dict], count: int) -> list[dict]:
        """Takes an even stride, so every scenario type is represented."""
        step = max(len(samples) // count, 1)
        return samples[::step][:count]

    def turns(self):
        for replay in self.replays:
            for turn in replay["session"]["turns"]:
                yield replay, turn

    def test_replay_reproduces_the_served_slate(self):
        for replay, turn in self.turns():
            with self.subTest(sample=replay["session"]["sample_id"],
                              turn=turn["turn"]):
                ranking = turn["explain"]["ranking"]
                self.assertTrue(
                    ranking["verified"],
                    "the re-derived slate diverged from the served one",
                )

    def test_score_breakdowns_sum_to_the_served_score(self):
        for replay, turn in self.turns():
            for slot in turn["explain"]["ranking"]["slots"]:
                with self.subTest(asin=slot["asin"], turn=turn["turn"]):
                    self.assertTrue(
                        slot["agrees"],
                        f"{slot['breakdown']} does not sum to {slot['score']}",
                    )

    def test_only_unpooled_slots_carry_no_score(self):
        """`slate` reports zero for a slot the blend never scored, and only that.

        Padding draws from the coarser group, which overlaps the ranked pool,
        so a padded slot often does have a real score. Conflating the two
        would put a false zero on screen.
        """
        for _, turn in self.turns():
            for slot in turn["explain"]["ranking"]["slots"]:
                with self.subTest(asin=slot["asin"], source=slot["source"]):
                    if slot["pooled"]:
                        self.assertTrue(slot["agrees"])
                    else:
                        self.assertEqual(slot["score"], 0.0)

    def test_probe_table_argmax_matches_the_arm_actually_asked(self):
        """The arm table is reproduced, so it must agree with `probe.choose`."""
        for _, turn in self.turns():
            asked = turn["ask_attribute"]
            table = turn["explain"]["probe"]["table"]
            if asked is None or asked == probe.WILDCARD or not table:
                continue
            with self.subTest(turn=turn["turn"], asked=asked):
                self.assertEqual(table[0]["arm"], asked)

    def test_every_turn_explains_itself(self):
        for _, turn in self.turns():
            explained = turn["explain"]
            self.assertFalse(explained["degraded"], "a turn degraded")
            for panel in ("understand", "state", "policy", "route",
                          "ranking", "probe", "message"):
                self.assertIn(panel, explained)

    def test_the_slate_on_screen_is_the_slate_scored(self):
        """The panel and the evaluator must not be able to disagree."""
        for _, turn in self.turns():
            served = [
                slot["asin"] for slot in turn["explain"]["ranking"]["slots"]
            ]
            self.assertEqual(served, turn["slate"])

    def test_the_target_rank_on_screen_is_the_evaluator_s(self):
        for replay, turn in self.turns():
            goal = turn["explain"]["ranking"]["goal"]
            self.assertEqual(goal["asin"], replay["target"])
            if turn["target_rank"] is not None:
                position = turn["slate"].index(replay["target"]) + 1
                self.assertEqual(position, turn["target_rank"])


if __name__ == "__main__":
    unittest.main()
