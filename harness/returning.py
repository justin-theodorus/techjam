"""The returning-shopper gate: does visit two go better than visit one?

The organizer's harness cannot answer that. `evaluate()` mints a fresh
`public_{uuid4().hex}` per session and the published contract closes both
`reset_request` and `user_profile` with `additionalProperties: false`, so no
identity reaches the agent and a per-person memory is written once and read
never (measurements 3.33). This module supplies the identity the interface has no
field for, and it does so beside the real `evaluate()` rather than instead of
it, so the dialogue and the score still come from the organizer's own loop.

Three readings, over one set of rows:

    on          memory enabled, every shopper reading its own earlier visits
    off         memory disabled, the same rows
    scrambled   memory enabled, identities rotated by visit so every shopper
                reads a stranger's

`off` says the visit-block gap is memory rather than the recipe. `scrambled`
says it is a memory of *this person* rather than a prior anyone would have
supplied: it must not improve, and if it does, the mechanism is cohort
inference wearing memory's clothes, which is already measured and dead at
0.66x lift. Neither number is a TechnicalScore and neither belongs in a table
beside one.
"""

from __future__ import annotations

import argparse
import collections
import pathlib

from evaluator import local_evaluator

from harness import analysis
from harness import deviations
from harness import identity
from harness import record
from harness import run
from harness import session_axes
from harness import session_sets
from harness import sessions

SET_NAME = "returning_shopper"
# Each reading is a name, the constants to patch, and whether the identities
# are rotated. The scrambled row carries the same switches as the row it
# controls, because a control for a switch that does nothing proves nothing.
READINGS = (
    ("shipped", {}, False),
    ("off", {"memory.ENABLED": False}, False),
    ("positives", {"memory.CARRY_POSITIVES": True}, False),
    ("scrambled", {"memory.CARRY_POSITIVES": True}, True),
)


def scrambled(rows: list[dict]) -> list[dict]:
    """Returns the same rows with every shopper reading a stranger.

    Rotating by the visit index rather than shuffling is what keeps this an
    exact control: each identity still owns one session per visit block, in
    the same row positions, against the same targets and the same profiles.
    The only thing that changes is which sessions share an identity.
    """
    count = len({row["shopper_id"] for row in rows if "shopper_id" in row})
    if not count:
        return list(rows)
    rotated = []
    for row in rows:
        identity = row.get("shopper_id")
        if identity is None:
            rotated.append(row)
            continue
        owner = (int(identity.rsplit("_", 1)[1]) + row["visit"] - 1) % count
        rotated.append({**row, "shopper_id": f"shopper_{owner:04d}"})
    return rotated


def measure(agent: object, rows: list[dict], catalog_ids: set[str],
            categories: dict[str, list[str]],
            by_asin: dict[str, dict]) -> dict:
    """Scores one reading and returns its per-visit breakdown."""
    recorder = record.RecordingAgent(identity.ReturningAgent(agent, rows))
    result = local_evaluator.evaluate(
        recorder, rows, catalog_ids, categories, by_asin)
    analysed = analysis.analyze(
        recorder.sessions, rows, result, catalog_ids)
    visits: dict[int, list[dict]] = collections.defaultdict(list)
    for row, session in zip(rows, analysed):
        visits[row.get("visit", 0)].append(session)
    return {
        "score": result["recommended_technical_score"],
        "health": analysis.health_summary(analysed),
        "visits": {visit: _block(group) for visit, group in visits.items()},
    }


def _block(group: list[dict]) -> dict:
    """The four numbers one visit block is compared on."""
    size = max(1, len(group))
    hits = [session for session in group if session["hit"]]
    turns = [session["first_hit_turn"] for session in hits]
    return {
        "n": len(group),
        "hit": len(hits) / size,
        "mrr": sum(session["reciprocal_rank"] for session in group) / size,
        "mttc": sum(turns) / len(turns) if turns else float(
            local_evaluator.MAX_TURNS),
        "rank1": sum(
            1 for session in group if session["best_rank"] == 1) / size,
    }


def table(readings: dict[str, dict]) -> list[str]:
    """The deliverable, and it is a difference rather than a level."""
    lines = [
        f"{'reading':<12}{'visit':>6}{'n':>6}{'hit@10':>9}"
        f"{'MRR':>9}{'MTTC':>8}{'rank1':>8}",
        "-" * 58,
    ]
    for name, reading in readings.items():
        for visit in sorted(reading["visits"]):
            block = reading["visits"][visit]
            lines.append(
                f"{name:<12}{visit:>6}{block['n']:>6}{block['hit']:>9.3f}"
                f"{block['mrr']:>9.3f}{block['mttc']:>8.2f}"
                f"{block['rank1']:>8.3f}"
            )
        lines.append("-" * 58)
    return lines


def gaps(readings: dict[str, dict]) -> list[str]:
    """Visit one against the visits that could have remembered it."""
    lines = ["", "later visits minus the first, in MTTC turns and in MRR", ""]
    for name, reading in readings.items():
        blocks = reading["visits"]
        first = blocks.get(1)
        later = [blocks[visit] for visit in sorted(blocks) if visit > 1]
        if first is None or not later:
            continue
        mttc = sum(block["mttc"] for block in later) / len(later)
        mrr = sum(block["mrr"] for block in later) / len(later)
        lines.append(
            f"  {name:<12} MTTC {mttc - first['mttc']:+.3f}   "
            f"MRR {mrr - first['mrr']:+.4f}"
        )
    lines.append("")
    lines.append(
        "  `off` is the control that says a gap is memory and not the recipe."
    )
    lines.append(
        "  `scrambled` runs `positives` with every shopper reading a "
        "stranger."
    )
    lines.append(
        "  It must not match it: a gain that survives scrambling is a generic"
    )
    lines.append(
        "  prior, which is the cohort mechanism already dead at 0.66x (3.33)."
    )
    return lines


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="the returning-shopper gate")
    parser.add_argument("--agent", default=run.DEFAULT_AGENT)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_path = pathlib.Path(args.catalog)
    run.require_catalog(catalog_path)
    products = sessions.load_products(catalog_path)
    catalog_ids, categories, by_asin = local_evaluator.catalog_index(
        catalog_path)
    facts = session_axes.survey(products)
    profiles = [
        row["user_profile"]
        for row in local_evaluator.load_jsonl(args.dataset)
    ]
    recipe = next(
        item for item in session_sets.MANIFEST if item.name == SET_NAME)
    rows = sessions.generate(recipe, products, facts, profiles)
    for row in rows:
        sessions.validate_row(row, catalog_ids)

    agent = run.load_agent_class(args.agent)(str(catalog_path))
    readings = {}
    for name, assignments, rotate in READINGS:
        served = scrambled(rows) if rotate else rows
        with deviations.patched(assignments):
            readings[name] = measure(
                agent, served, catalog_ids, categories, by_asin)

    print(f"{SET_NAME}: {len(rows)} sessions, "
          f"{session_axes.VISITS_PER_SHOPPER} visits each")
    print()
    for line in table(readings):
        print(line)
    for line in gaps(readings):
        print(line)
    print()
    print("Not a TechnicalScore. The organizer's harness supplies no second")
    print("visit, so this capability is demonstrable here and scoreable")
    print("nowhere (measurements 3.33).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
