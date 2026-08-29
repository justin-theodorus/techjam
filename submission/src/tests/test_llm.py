"""Tier 2: the permutation property, and every way it is allowed to fail."""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path

from techjam.submission.src import catalog as catalog_module
from techjam.submission.src import dialogue
from techjam.submission.src import llm
from techjam.submission.src import ranking
from techjam.submission.src.tests import fixtures


@contextlib.contextmanager
def switched(value: str | None):
    """Sets the opt-in environment variable for the block."""
    original = os.environ.get(llm.ENV_SWITCH)
    try:
        if value is None:
            os.environ.pop(llm.ENV_SWITCH, None)
        else:
            os.environ[llm.ENV_SWITCH] = value
        yield
    finally:
        os.environ.pop(llm.ENV_SWITCH, None)
        if original is not None:
            os.environ[llm.ENV_SWITCH] = original


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Usage:
    def __init__(self, prompt: int, completion: int, cached: int = 0) -> None:
        self.input_tokens = prompt
        self.output_tokens = completion
        self.cache_read_input_tokens = cached
        self.cache_creation_input_tokens = 0


class _Response:
    def __init__(self, text: str, usage: _Usage | None = None) -> None:
        self.content = [_Block(text)]
        self.usage = usage or _Usage(100, 20)


class _Messages:
    """Stands in for `client.messages`, returning a scripted reply."""

    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls = 0
        self.last: dict = {}

    def create(self, **kwargs) -> _Response:
        self.calls += 1
        self.last = kwargs
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _Client:
    def __init__(self, reply: object) -> None:
        self.messages = _Messages(reply)


def _reranker(reply: object) -> llm.LLMReranker:
    return llm.LLMReranker(_Client(reply))


_STATE = dialogue.SessionState(constraints=("cotton canvas",))


def _catalog(case: unittest.TestCase) -> catalog_module.Catalog:
    """A fixture catalog keeping its product text, cleaned up with the case."""
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    return catalog_module.build(
        fixtures.write_catalog(Path(directory.name)), cards=True
    )


class SwitchTest(unittest.TestCase):
    """Tier 2 ships off, and both gates are needed to reach it."""

    def test_the_switch_is_off(self) -> None:
        """A permutation cannot lift coverage, and can lose precision."""
        self.assertEqual(ranking.LLM_RERANK, 0)

    def test_nothing_is_built_without_the_opt_in(self) -> None:
        with switched(None):
            self.assertIsNone(llm.build())
            self.assertFalse(llm.wanted())

    def test_a_client_without_a_credential_is_not_a_stage(self) -> None:
        """The SDK defers auth to request time; this gate does not.

        A keyless client constructs happily and raises on every call, which
        would show up as several hundred counted failures instead of a tier
        that simply is not there.
        """
        class _Keyless:
            api_key = None
            auth_token = None

        self.assertFalse(llm._credentialed(_Keyless()))
        self.assertTrue(llm._credentialed(_Client(None)) is False)

    def test_a_resolved_credential_passes_the_gate(self) -> None:
        class _Keyed:
            api_key = "sk-ant-example"

        self.assertTrue(llm._credentialed(_Keyed()))

    def test_a_falsy_value_fails_closed(self) -> None:
        for value in ("0", "false", "", "yes", "true"):
            with self.subTest(value=value), switched(value):
                self.assertIsNone(llm.build())

    def test_a_built_stage_is_ignored_while_the_switch_is_off(self) -> None:
        built = _reranker(_Response('{"order": [2, 1]}'))
        chosen = [0, 1]
        self.assertEqual(
            ranking.reranked(_catalog(self), chosen, _STATE, built), chosen
        )
        self.assertEqual(built.take()["calls"], 0)

    def test_the_switch_selects_a_built_stage(self) -> None:
        built = _reranker(_Response('{"order": [2, 1]}'))
        original = ranking.LLM_RERANK
        try:
            ranking.LLM_RERANK = 1
            catalog = _catalog(self)
            self.assertEqual(
                ranking.reranked(catalog, [0, 1], _STATE, built), [1, 0]
            )
            # The switch alone is not enough: with nothing built there is
            # still only the offline stage to run.
            self.assertEqual(
                ranking.reranked(catalog, [0, 1], _STATE, None), [0, 1]
            )
        finally:
            ranking.LLM_RERANK = original

    def test_the_model_composes_over_the_offline_stage(self) -> None:
        """A failed call must cost the model's judgement and nothing else.

        Replacing the phrase reranker rather than composing with it would
        make a timeout also throw away the phrase evidence, so a declined
        turn would be served worse than the offline agent serves it.
        """
        original = ranking.LLM_RERANK
        try:
            ranking.LLM_RERANK = 1
            catalog = _catalog(self)
            chosen = [0, 1, 2, 3, 4]
            offline = ranking.rerank(catalog, chosen, _STATE)
            broken = _reranker(RuntimeError("timeout"))
            self.assertEqual(
                ranking.reranked(catalog, chosen, _STATE, broken), offline
            )
            self.assertEqual(broken.take()["failures"], 1)
        finally:
            ranking.LLM_RERANK = original


class PermutationTest(unittest.TestCase):
    """Whatever the model says, the served set does not change.

    This is the entire safety argument for the stage: membership fixes
    HitRate and MTTC, so a repair that always returns a permutation makes
    those two immune to anything the model does.
    """

    CHOSEN = [11, 22, 33, 44, 55]

    def test_a_full_ordering_is_applied(self) -> None:
        self.assertEqual(
            llm._permute(self.CHOSEN, [5, 4, 3, 2, 1]),
            [55, 44, 33, 22, 11],
        )

    def test_malformed_orderings_still_permute(self) -> None:
        cases = {
            "empty": (),
            "duplicates": (1, 1, 1, 2),
            "out of range low": (0, -3, 2),
            "out of range high": (99, 6, 3),
            "partial": (4,),
            "over long": tuple(range(1, 50)),
            "reversed with junk": (5, 0, 4, 99, 3, 3, 2, 1),
        }
        for name, order in cases.items():
            with self.subTest(order=name):
                result = llm._permute(self.CHOSEN, order)
                self.assertEqual(sorted(result), sorted(self.CHOSEN))
                self.assertEqual(len(result), len(self.CHOSEN))

    def test_the_unranked_tail_keeps_the_blend_order(self) -> None:
        # The model named only the fourth candidate, so everything it did not
        # rank stays exactly as the blend left it, behind that one.
        self.assertEqual(
            llm._permute(self.CHOSEN, [4]), [44, 11, 22, 33, 55]
        )


class FailureTest(unittest.TestCase):
    """Every failure returns the slate unchanged. None of them raises."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = fixtures.write_catalog(Path(directory.name))
        self.catalog = catalog_module.build(path, cards=True)
        self.state = dialogue.SessionState(constraints=("cotton canvas",))
        self.chosen = list(range(5))

    def _assert_untouched(self, stage: llm.LLMReranker) -> None:
        self.assertEqual(stage(self.catalog, self.chosen, self.state),
                         self.chosen)

    def test_a_raising_client_is_counted_not_propagated(self) -> None:
        stage = _reranker(RuntimeError("connection reset"))
        self._assert_untouched(stage)
        self.assertEqual(stage.take()["failures"], 1)

    def test_unparseable_text_is_a_failure(self) -> None:
        stage = _reranker(_Response("not json at all"))
        self._assert_untouched(stage)
        self.assertEqual(stage.take()["failures"], 1)

    def test_a_reply_without_the_key_is_not_a_failure(self) -> None:
        # Well-formed JSON that simply says nothing useful is the model
        # declining to reorder, not the call going wrong.
        stage = _reranker(_Response('{"order": "nonsense"}'))
        self._assert_untouched(stage)
        self.assertEqual(stage.take()["failures"], 0)

    def test_no_constraints_means_no_call(self) -> None:
        stage = _reranker(_Response('{"order": [5, 4, 3, 2, 1]}'))
        blank = dialogue.SessionState()
        self.assertEqual(stage(self.catalog, self.chosen, blank), self.chosen)
        self.assertEqual(stage.take()["calls"], 0)

    def test_a_catalog_without_cards_means_no_call(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = fixtures.write_catalog(Path(directory.name))
        bare = catalog_module.build(path)
        stage = _reranker(_Response('{"order": [5, 4, 3, 2, 1]}'))
        self.assertIsNone(bare.cards)
        self.assertEqual(stage(bare, self.chosen, self.state), self.chosen)
        self.assertEqual(stage.take()["calls"], 0)


class UsageTest(unittest.TestCase):
    """The counts the evaluator sums must be integers it will accept."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = fixtures.write_catalog(Path(directory.name))
        self.catalog = catalog_module.build(path, cards=True)
        self.state = dialogue.SessionState(constraints=("cotton canvas",))

    def test_counts_accumulate_then_clear(self) -> None:
        stage = _reranker(_Response('{"order": [2, 1]}', _Usage(120, 30)))
        stage(self.catalog, [0, 1], self.state)
        counts = stage.take()
        self.assertEqual(counts["prompt_tokens"], 120)
        self.assertEqual(counts["completion_tokens"], 30)
        self.assertEqual(counts["calls"], 1)
        self.assertEqual(stage.take()["prompt_tokens"], 0)

    def test_cached_tokens_are_counted_as_prompt_tokens(self) -> None:
        # `input_tokens` excludes what the cache served, so reporting it alone
        # would understate what we actually sent.
        stage = _reranker(
            _Response('{"order": [2, 1]}', _Usage(20, 30, cached=100))
        )
        stage(self.catalog, [0, 1], self.state)
        counts = stage.take()
        self.assertEqual(counts["prompt_tokens"], 120)
        self.assertEqual(counts["cached_tokens"], 100)

    def test_every_reported_count_is_a_non_negative_int(self) -> None:
        stage = _reranker(_Response('{"order": [2, 1]}'))
        stage(self.catalog, [0, 1], self.state)
        for key, value in stage.take().items():
            if key == "milliseconds":
                continue
            with self.subTest(key=key):
                self.assertIsInstance(value, int)
                self.assertGreaterEqual(value, 0)

    def test_no_usage_matches_the_counter_shape(self) -> None:
        stage = _reranker(_Response('{"order": [2, 1]}'))
        self.assertEqual(set(llm.no_usage()), set(stage.take()))

    def test_cost_is_arithmetic_over_the_published_rates(self) -> None:
        self.assertAlmostEqual(
            llm.cost(1_000_000, 1_000_000),
            llm.PRICE_IN_PER_MTOK + llm.PRICE_OUT_PER_MTOK,
        )


class PromptTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = fixtures.write_catalog(Path(directory.name))
        self.catalog = catalog_module.build(path, cards=True)

    def test_refusals_reach_the_prompt_as_refusals(self) -> None:
        # A refusal scored as a preference is the defect Phase 6T existed to
        # catch; the model must not be handed one as a want.
        state = dialogue.update(
            dialogue.SessionState(category=fixtures.SNEAKER_BUCKET),
            dialogue.ParsedTurn(constraints=("not polyester",)),
            None,
            self.catalog.taxonomy,
        )
        prompt = llm._prompt(state, ["a canvas shoe", "a polyester shoe"])
        self.assertIn("Refused:", prompt)
        self.assertIn("polyester", prompt.split("Refused:")[1])

    def test_candidates_are_numbered_from_one(self) -> None:
        state = dialogue.SessionState(constraints=("canvas",))
        prompt = llm._prompt(state, ["first card", "second card"])
        self.assertIn("1. first card", prompt)
        self.assertIn("2. second card", prompt)

    def test_cards_are_truncated(self) -> None:
        cards = llm._cards(self.catalog, [0, 1])
        self.assertIsNotNone(cards)
        for card in cards:
            self.assertLessEqual(len(card), llm.MAX_CARD_CHARS)


if __name__ == "__main__":
    unittest.main()
