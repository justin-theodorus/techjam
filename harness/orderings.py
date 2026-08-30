"""Phase 6Y.0: whether a portfolio of orderings exists to switch between.

Adaptive orchestration needs somewhere to switch *to*. Every previous attempt
in this project varied a constant inside one ordering -- route `alpha` (3.30),
dense weight (3.35), diversity (3.43), head size (3.46), reach (3.32) -- and
none of them asked whether a *different* ordering of the same pool would have
found the target the shipped one missed.

Three diagnostics, run before any controller is written, because decision 23
says measure the mechanism before the lever and decision 27 says locate the
failure before designing the fix:

  D1  For every missed session, at its last turn (the most the agent will ever
      know), where does the target rank under each candidate ordering? The
      number that matters is the *union*: if no ordering reaches a target the
      blend missed, there is no portfolio and the phase stops here.
  D2  How different are the orderings from each other? Overlap@40 per turn.
      Highly correlated orderings cannot form a portfolio whatever D1 says.
  D3  How often would a controller fire, and on which sets? Decision 29: log
      which branch fires and on what share of turns before attaching behaviour
      to it.

Reads the agent through a subclass of the shipped one that overrides `_record`
and nothing else, so the dialogue, the slates and the scoring are the real
ones. The orderings are computed beside the turn, never in place of it.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import statistics
from pathlib import Path
from time import perf_counter

from evaluator import local_evaluator
from harness import identity
from harness import run
from harness import session_axes
from harness import sessions
from submission.src import agent as agent_module
from submission.src import ranking
from submission.src import text

# The band a controller would serve its nine exploration slots from, and the
# whole budget a ten-turn session can physically show once `SKIP_SHOWN` stops
# it spending a slot twice (findings 3.44).
HORIZON = 40
BUDGET = 100

# How much of an ordering's head must already have been served and disproven
# before that ordering counts as spent for this session.
SPENT_RATIO = 0.5

PUBLIC = "public 200"


@dataclasses.dataclass(frozen=True)
class Reading:
    """One turn, priced under every candidate ordering."""

    turn: int
    ranks: dict[str, int | None]
    heads: dict[str, tuple[int, ...]]
    spent: dict[str, float]


def _positions(scores: list[float], pool: tuple[int, ...]) -> list[int]:
    """Returns `pool` best first, ties falling back to the prior.

    The pool arrives popularity-ordered and Python's sort is stable, which is
    the same tie-break `ranking.ranked` relies on.
    """
    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    return [pool[position] for position in order]


def _blend(catalog, turn: dict) -> list[int]:
    return ranking.ranked(
        catalog, turn["pool"], turn["query_ids"], ranking.ALPHA,
        turn["profile_ids"], turn["negative_ids"],
    )[0]


def _lexical(catalog, turn: dict) -> list[int]:
    """The blend with the popularity prior switched off entirely."""
    return ranking.ranked(
        catalog, turn["pool"], turn["query_ids"], 0.0,
        turn["profile_ids"], turn["negative_ids"],
    )[0]


def _prior(catalog, turn: dict) -> list[int]:
    """The blend with the customer's words switched off entirely."""
    return ranking.ranked(
        catalog, turn["pool"], frozenset(), ranking.ALPHA,
        frozenset(), turn["negative_ids"],
    )[0]


def _dense(catalog, turn: dict) -> list[int]:
    """The bundled latent space alone, with no lexical term at all."""
    if turn["dense_query"] is None:
        return list(turn["pool"])
    return ranking.ranked(
        catalog, turn["pool"], frozenset(), 0.0, frozenset(),
        turn["negative_ids"], dense_query=turn["dense_query"],
        dense_weight=1.0,
    )[0]


def _phrase(catalog, turn: dict) -> list[int]:
    """Rare whole-phrase evidence alone, over the pool rather than the slate.

    `ranking.rerank` only ever permutes the served ten. This asks what the same
    evidence would say about the whole pool. With no phrase evidence it
    degenerates to the popularity order, which is `_prior`, and the D2 overlap
    table is where that shows up.
    """
    phrase_ids = catalog.phrases.query_ids(turn["constraints"])
    if not phrase_ids:
        return list(turn["pool"])
    evidence = [
        catalog.phrases.evidence(index, phrase_ids)
        for index in turn["pool"]
    ]
    return _positions(evidence, turn["pool"])


ORDERINGS = (
    ("blend", _blend),
    ("lexical", _lexical),
    ("prior", _prior),
    ("dense", _dense),
    ("phrase", _phrase),
)


class ProbingAgent(agent_module.Agent):
    """The shipped agent, plus the orderings it did not use.

    `_record` is the seam because it is handed the state `ranking.slate` was
    actually called with, before this turn's slate joins `shown`. Nothing else
    is overridden, so the served slate and the score stay the shipped ones.
    """

    def __init__(self, catalog_path: str, rows: list[dict]) -> None:
        super().__init__(catalog_path)
        self._rows = rows
        self._served = 0
        self._goal: int | None = None
        self.readings: list[list[Reading]] = []
        # `Catalog` keeps `asins` for slate assembly and never needs to go the
        # other way; following one named target down a ranking does.
        self.index_of = {
            asin: index for index, asin in enumerate(self.catalog.asins)
        }

    def rows(self, rows: list[dict]) -> None:
        """Points the probe at a new set without rebuilding the indexes."""
        self._rows = rows
        self._served = 0
        self.readings = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self.readings.append([])
        # `evaluate()` calls reset exactly once per row, in row order, which is
        # the same assumption `harness/identity.py` is built on.
        row = self._rows[self._served] if self._served < len(self._rows) else {}
        self._served += 1
        target = str(row.get("ground_truth", {}).get("parent_asin", ""))
        self._goal = self.index_of.get(target)

    def _record(self, state, parsed, route, served, asins) -> None:
        super()._record(state, parsed, route, served, asins)
        if self.readings:
            self.readings[-1].append(self._read(state))

    def _read(self, state) -> Reading:
        """Prices this turn under every candidate ordering."""
        turn = {
            "pool": self.catalog.pool(state.pool_keys),
            "query_ids": self.catalog.index.query_ids(
                text.unique_tokens(state.query_text)
            ),
            "negative_ids": self.catalog.index.query_ids(
                text.unique_tokens(state.excluded_text)
            ),
            "profile_ids": ranking.personalised(state, self._profile_ids),
            "dense_query": ranking.encode(self.catalog, state.query_text, True),
            "constraints": state.constraints,
        }
        ranks: dict[str, int | None] = {}
        heads: dict[str, tuple[int, ...]] = {}
        spent: dict[str, float] = {}
        for name, ordering in ORDERINGS:
            ordered = ordering(self.catalog, turn)
            ranks[name] = _rank_of(ordered, self._goal)
            heads[name] = tuple(ordered[:HORIZON])
            spent[name] = _spent_share(heads[name], state.shown, self.catalog)
        return Reading(state.turn, ranks, heads, spent)


def _rank_of(ordered: list[int], goal: int | None) -> int | None:
    if goal is None:
        return None
    try:
        return ordered.index(goal) + 1
    except ValueError:
        return None


def _spent_share(head: tuple[int, ...], shown, catalog) -> float:
    """How much of an ordering's head has already been served and disproven.

    A slate that was served and did not end the session is provably wrong
    (findings 3.32), so this is the one quantity available to the agent that is
    about correctness rather than about how confident the text looks.
    """
    if not head:
        return 0.0
    asins = catalog.slate_of(head)
    return sum(1 for asin in asins if asin in shown) / len(head)


def measure(agent: ProbingAgent, rows: list[dict], catalog_ids: set[str],
            categories: dict[str, list[str]],
            by_asin: dict[str, dict]) -> dict:
    """Scores one set through the real `evaluate()` and keeps the readings."""
    agent.rows(rows)
    # Wrapped for the same reason `deviations.score` wraps: a set whose rows
    # name a shopper is scored with those identities supplied, and every set
    # starts from an empty store because the agent is never rebuilt.
    recorder = identity.ReturningAgent(agent, rows)
    result = local_evaluator.evaluate(
        recorder, rows, catalog_ids, categories, by_asin)
    return {"result": result, "readings": list(agent.readings)}


def _last_readings(artifact: dict, missed: bool) -> list[Reading]:
    """The final turn of every session, where the agent knows the most."""
    verdicts = artifact["result"]["sessions"]
    return [
        readings[-1]
        for verdict, readings in zip(verdicts, artifact["readings"])
        if readings and verdict["hit"] is not missed
    ]


def _within(reading: Reading, name: str, depth: int) -> bool:
    rank = reading.ranks[name]
    return rank is not None and rank <= depth


def _share(readings: list[Reading], name: str, depth: int) -> float | None:
    """`None` when the set had no misses, which is not the same as 0%.

    Decision 21: a reading taken on a set with no headroom is a reading about
    the set. An empty cell has to look empty.
    """
    if not readings:
        return None
    hits = sum(1 for reading in readings if _within(reading, name, depth))
    return hits / len(readings)


def _union_share(readings: list[Reading], depth: int) -> float | None:
    """Share reached by *any* ordering. The number the phase turns on."""
    if not readings:
        return None
    hits = sum(
        1 for reading in readings
        if any(_within(reading, name, depth) for name, _ in ORDERINGS)
    )
    return hits / len(readings)


def _percent(value: float | None, width: int) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.1%}"


def d1_table(measured: list[tuple[str, dict]]) -> list[str]:
    """Where a missed target ranks under each ordering, at the last turn."""
    names = [name for name, _ in ORDERINGS]
    header = (f"{'set':<26}{'miss':>5}  "
              + "".join(f"{name:>9}" for name in names)
              + f"{'ANY@40':>9}{'ANY@100':>9}")
    lines = [header, "-" * len(header)]
    for label, artifact in measured:
        readings = _last_readings(artifact, missed=True)
        cells = "".join(
            _percent(_share(readings, name, HORIZON), 9) for name in names
        )
        lines.append(
            f"{label:<26}{len(readings):>5}  {cells}"
            f"{_percent(_union_share(readings, HORIZON), 9)}"
            f"{_percent(_union_share(readings, BUDGET), 9)}"
        )
    return lines


def _overlap(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 1.0
    return len(set(left) & set(right)) / len(left)


def d2_table(measured: list[tuple[str, dict]]) -> list[str]:
    """How different the orderings are from the shipped blend, per turn."""
    others = [name for name, _ in ORDERINGS if name != "blend"]
    header = (f"{'set':<26}{'turns':>6}  "
              + "".join(f"{name:>10}" for name in others))
    lines = [header, "-" * len(header)]
    for label, artifact in measured:
        every = [
            reading for readings in artifact["readings"] for reading in readings
        ]
        cells = []
        for name in others:
            scores = [
                _overlap(reading.heads["blend"], reading.heads[name])
                for reading in every
            ]
            mean = statistics.fmean(scores) if scores else 0.0
            cells.append(f"{mean:>10.2f}")
        lines.append(f"{label:<26}{len(every):>6}  " + "".join(cells))
    return lines


def d3_table(measured: list[tuple[str, dict]]) -> list[str]:
    """How often the blend's own head is already spent, and where."""
    header = (f"{'set':<26}{'turns':>6}{'would fire':>12}"
              f"{'median spent':>14}{'spent at end':>14}")
    lines = [header, "-" * len(header)]
    for label, artifact in measured:
        every = [
            reading for readings in artifact["readings"] for reading in readings
        ]
        spent = [reading.spent["blend"] for reading in every]
        firing = sum(1 for value in spent if value >= SPENT_RATIO)
        ends = [
            reading.spent["blend"]
            for reading in _last_readings(artifact, missed=True)
        ]
        lines.append(
            f"{label:<26}{len(every):>6}"
            f"{firing / max(1, len(every)):>12.1%}"
            f"{statistics.median(spent) if spent else 0.0:>14.2f}"
            f"{statistics.median(ends) if ends else 0.0:>14.2f}"
        )
    return lines


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6Y.0: is there a portfolio of orderings?")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--set", default="", dest="sets",
                        help="comma-separated subset of frozen set names")
    parser.add_argument("--no-public", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_path = Path(args.catalog)
    run.require_catalog(catalog_path)
    recipes = sessions.chosen(args.sets)

    started = perf_counter()
    products = sessions.load_products(catalog_path)
    facts = session_axes.survey(products)
    catalog_ids, categories, by_asin = local_evaluator.catalog_index(
        catalog_path)
    public = local_evaluator.load_jsonl(args.dataset)
    public_profiles = [row["user_profile"] for row in public]
    agent = ProbingAgent(str(catalog_path), [])
    setup_s = round(perf_counter() - started, 2)

    corpus = [] if args.no_public else [(PUBLIC, public)]
    for recipe in recipes:
        rows = sessions.generate(recipe, products, facts, public_profiles)
        for row in rows:
            sessions.validate_row(row, catalog_ids)
        corpus.append((recipe.name, rows))

    started = perf_counter()
    measured = [
        (label, measure(agent, rows, catalog_ids, categories, by_asin))
        for label, rows in corpus
    ]
    print(f"setup {setup_s}s, {len(corpus)} sets in "
          f"{round(perf_counter() - started, 2)}s\n")

    print("D1  a missed target's rank under each ordering, at the last turn")
    print(f"    share within the top {HORIZON}; ANY is the union\n")
    print("\n".join(d1_table(measured)))
    print(f"\n\nD2  mean overlap@{HORIZON} with the blend, every turn")
    print("    1.00 means the same head, so no portfolio exists there\n")
    print("\n".join(d2_table(measured)))
    print("\n\nD3  how much of the blend's own head is already disproven")
    print(f"    'would fire' is the share of turns at or past {SPENT_RATIO}\n")
    print("\n".join(d3_table(measured)))
    print(
        "\n\nthe gate: build the controller only if ANY@40 beats blend@40 on "
        "at least three\nreadable sets, and D2 falls meaningfully below 1.00 "
        "somewhere real. Otherwise\nthere is nowhere to switch to and the "
        "phase is a recorded negative (decision 23)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
