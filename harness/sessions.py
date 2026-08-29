"""Manufacture session sets the public 200 cannot be.

176 of the 200 public sessions already convert at rank 1, so the set holds 24
sessions of headroom against 176 of downside and can no longer separate a good
idea from a bad one. It is also uniform where a private set need not be: one
`category_bucket`, one `purchase_frequency`, a nine-word tag vocabulary, and
not one target with an empty `features` list against 10.4% of the catalog.

`materialize_hidden_fields()` (`local_evaluator.py:204`) hands back an authored
`intent_card` and `behavior` verbatim whenever a row carries both, so a set can
be manufactured with any targets, any constraint wording and any pivot turn and
still be scored by the organizer's own `evaluate()` unmodified.

    python3 -m harness.sessions                        # every frozen set
    python3 -m harness.sessions --set thin_cards
    python3 -m harness.sessions --set mirror --write mirror.jsonl

This module builds the instrument and reports what the instrument *is*. It
takes no reading about any agent feature. That is a later session's job, kept
separate so a set can never be shaped until a feature looks good on it.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import random
import statistics
from pathlib import Path
from time import perf_counter

from evaluator import local_evaluator

from harness import analysis
from harness import counterfactual
from harness import record
from harness import run
from harness import session_axes
from harness import session_sets

SAMPLE_PREFIX = "syn"

# The five keys `docs/agent_api_contract.json` requires of `user_profile`, with
# `additionalProperties: false`. Repeated rather than read at import so the
# generator has no runtime dependency on a docs file; the test asserts parity.
PROFILE_KEYS = frozenset({
    "purchase_frequency", "average_prior_rating", "rating_style",
    "preference_tags", "summary",
})

# `evaluate():256-259` breaks at MAX_TURNS before testing the pivot, so the
# check only ever sees `turn` in 1..9. A pivot outside this window never fires
# and the session can never score.
MIN_PIVOT_TURN = 2
MAX_PIVOT_TURN = 10

RANK_BUCKETS = (("r1", (1, 1)), ("r2", (2, 2)), ("r3", (3, 3)),
                ("r4-6", (4, 6)), ("r7-10", (7, 10)))


Recipe = session_axes.Recipe


def scale_mix(mix: tuple[tuple[str, int], ...],
              count: int) -> list[tuple[str, int]]:
    """Returns `mix` scaled to `count`, the largest scenario taking the rest."""
    total = sum(size for _, size in mix)
    if total <= 0 or count <= 0:
        raise ValueError(f"cannot scale mix {mix!r} to {count} sessions")
    scaled = [(name, size * count // total) for name, size in mix]
    shortfall = count - sum(size for _, size in scaled)
    if shortfall:
        widest = max(range(len(scaled)), key=lambda index: scaled[index][1])
        scaled[widest] = (scaled[widest][0], scaled[widest][1] + shortfall)
    return [(name, size) for name, size in scaled if size > 0]


def _draw(products: list[dict], weights: list[float], rng: random.Random,
          count: int) -> list[dict]:
    """Inverse-CDF sampling with replacement, as `counterfactual.draw_targets`.

    Reimplemented rather than reused because that one fixes the count at 200
    and seeds on a bare integer, and this generator needs a per-axis string
    seed so the target draw cannot move when another axis changes.
    """
    cumulative, total = [], 0.0
    for weight in weights:
        total += weight
        cumulative.append(total)
    if total <= 0.0:
        raise ValueError("every candidate weighs zero; the pool is unusable")
    return [
        products[bisect.bisect_left(cumulative, rng.random() * total)]
        for _ in range(count)
    ]


def _author(recipe: Recipe, scenario: str, sample_id: str, product: dict,
            facts: session_axes.CatalogFacts) -> tuple[dict, dict]:
    """Reproduces the evaluator's derivation exactly, then applies axes 2 and 5.

    Order matters. The card is shaped and reworded first so `behavior_for` sees
    what the customer will actually say, and the behaviour seed is the
    evaluator's own, so a recipe that transforms nothing lands on the very same
    pivot turn the derived path would have chosen.
    """
    card = local_evaluator.intent_card(product)
    dialogue_rng = random.Random(
        f"{recipe.name}\0{recipe.seed}\0dialogue\0{sample_id}")
    card = session_axes.shape_card(recipe.dialogue, card, dialogue_rng)

    text_rng = random.Random(f"{recipe.name}\0{recipe.seed}\0text\0{sample_id}")
    hard = card["hard_constraints"]
    reworded = session_axes.constraints(
        recipe.text, [*hard, *card["soft_preferences"]], product, text_rng)
    card = {
        "target_category": card["target_category"],
        "hard_constraints": reworded[:len(hard)],
        "soft_preferences": reworded[len(hard):],
    }

    behavior = local_evaluator.behavior_for(
        scenario, card, random.Random(f"{sample_id}\0{scenario}"))
    behavior = session_axes.shape_behavior(
        recipe.dialogue, behavior, product, facts, dialogue_rng)
    return card, behavior


def generate(recipe: Recipe, products: list[dict],
             facts: session_axes.CatalogFacts,
             public_profiles: list[dict]) -> list[dict]:
    """Returns one session set in the exact public row format.

    Rows are grouped by scenario rather than interleaved, matching the public
    file, so `harness.run.split_samples` still stratifies them correctly.
    """
    candidates = session_axes.pool(recipe.pool, products, facts)
    if not candidates:
        raise ValueError(f"pool {recipe.pool!r} selected no products")
    if recipe.weights not in session_axes.WEIGHT_NAMES:
        raise ValueError(f"unknown weights {recipe.weights!r}")
    weights = counterfactual.weight_schemes(candidates)[recipe.weights]

    counts = scale_mix(recipe.mix, recipe.count)
    targets = _draw(
        candidates, weights,
        random.Random(f"{recipe.name}\0{recipe.seed}\0targets"),
        sum(size for _, size in counts),
    )
    profile_rng = random.Random(f"{recipe.name}\0{recipe.seed}\0profiles")

    rows, index = [], 0
    for scenario, size in counts:
        for _ in range(size):
            product = targets[index]
            sample_id = f"{SAMPLE_PREFIX}_{recipe.name}_{index:04d}"
            row = {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "category_bucket": "clothing",
                "difficulty_bucket": "easy",
                "user_profile": session_axes.profile(
                    recipe.profiles, profile_rng, public_profiles, product),
                "ground_truth": {"parent_asin": str(product["parent_asin"])},
            }
            if recipe.is_authored:
                card, behavior = _author(
                    recipe, scenario, sample_id, product, facts)
                row = {**row, "intent_card": card, "behavior": behavior}
            rows.append(row)
            index += 1
    return rows


def _check_override(row: dict) -> None:
    """Requires all four override keys, which is stricter than crash-safety.

    Only `old_value` can crash: `initial_message():161` indexes it with no
    default. The other three have defaults at `evaluate():259-264`, but two of
    those defaults are silent mishandles rather than safe fallbacks. A missing
    `message` falls back to "Actually, **please** ignore my earlier
    preference.", one word off the literal both `analysis.py:14` and the agent
    match on, so the pivot fires without anything detecting it (3.24). A
    missing `new_value` discloses nothing and leaves the redirect pointing at
    no replacement. An authored override that omits either would read as a
    difficulty result when it is really a broken row.
    """
    override = row["behavior"].get("override")
    if not isinstance(override, dict):
        raise ValueError(
            f"{row['sample_id']}: intent_override needs behavior.override; "
            "initial_message() indexes it outside the try block"
        )
    for key in ("turn", "old_value", "new_value", "message"):
        if key not in override:
            raise ValueError(f"{row['sample_id']}: override lacks {key!r}")
    turn = override["turn"]
    if isinstance(turn, bool) or not isinstance(turn, int):
        raise ValueError(f"{row['sample_id']}: override turn must be an int")
    if not MIN_PIVOT_TURN <= turn <= MAX_PIVOT_TURN:
        raise ValueError(
            f"{row['sample_id']}: override turn {turn} is outside "
            f"{MIN_PIVOT_TURN}..{MAX_PIVOT_TURN} and could never fire"
        )


def validate_row(row: dict, catalog_ids: set[str]) -> None:
    """Raises on anything `evaluate()` would crash on or silently mishandle."""
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"row has no sample_id: {row!r}")
    if sample_id.startswith("public_"):
        raise ValueError(f"{sample_id}: a synthetic id must not look public")
    target = row.get("ground_truth")
    if not isinstance(target, dict) or "parent_asin" not in target:
        raise ValueError(f"{sample_id}: row has no ground_truth.parent_asin")
    if str(target["parent_asin"]) not in catalog_ids:
        raise ValueError(f"{sample_id}: target is not in the catalog")
    profile = row.get("user_profile")
    if not isinstance(profile, dict) or set(profile) != PROFILE_KEYS:
        raise ValueError(
            f"{sample_id}: user_profile is not the contract's "
            f"{sorted(PROFILE_KEYS)}"
        )
    authored = [key for key in ("intent_card", "behavior") if key in row]
    if len(authored) == 1:
        raise ValueError(
            f"{sample_id}: carries {authored[0]!r} alone, so "
            "materialize_hidden_fields() discards it and derives both"
        )
    if not authored:
        return
    card = row["intent_card"]
    if not isinstance(card.get("target_category"), str):
        raise ValueError(f"{sample_id}: intent_card needs a target_category")
    for key in ("hard_constraints", "soft_preferences"):
        if not isinstance(card.get(key), list):
            raise ValueError(f"{sample_id}: intent_card {key!r} must be a list")
        for value in card[key]:
            # customer_reply() joins these straight into the reply, so an empty
            # one discloses nothing while still consuming its slot.
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{sample_id}: {key} holds an empty constraint")
    if row["scenario_type"] == "intent_override":
        _check_override(row)


def _effective_card(row: dict, by_asin: dict[str, dict]) -> dict:
    if "intent_card" in row:
        return row["intent_card"]
    return local_evaluator.intent_card(
        by_asin[str(row["ground_truth"]["parent_asin"])])


def difficulty(rows: list[dict], by_asin: dict[str, dict],
               facts: session_axes.CatalogFacts) -> dict:
    """Returns what the set is, measured before any agent has seen it."""
    if not rows:
        raise ValueError("cannot profile an empty set")
    targets = [by_asin[str(row["ground_truth"]["parent_asin"])]
               for row in rows]
    authored = [row for row in rows if "intent_card" in row]
    moved = sum(
        1 for row in authored
        if row["intent_card"] != local_evaluator.intent_card(
            by_asin[str(row["ground_truth"]["parent_asin"])])
    )
    silent_buying = sum(
        1 for row in rows
        if row["scenario_type"] == "buying"
        and not _effective_card(row, by_asin)["hard_constraints"]
    )
    return {
        "count": len(rows),
        "reviews": statistics.median(
            float(item.get("rating_number") or 0) for item in targets),
        "bucket": statistics.median(
            facts.bucket_size[facts.bucket[str(item["parent_asin"])]]
            for item in targets),
        "features": statistics.median(
            len(item.get("features") or []) for item in targets),
        "twins": sum(1 for item in targets
                     if str(item["parent_asin"]) in facts.twins) / len(rows),
        "silent_buying": silent_buying,
        "moved": moved / len(rows) if authored else None,
    }


def measure(agent, rows: list[dict], catalog_ids: set[str],
            categories: dict[str, list[str]],
            by_asin: dict[str, dict]) -> dict:
    """Returns the rank distribution and the health counters for one set.

    Health matters as much as rank here. A generator that emits a row the agent
    cannot survive would otherwise read as a hard set rather than a broken one.
    """
    recorder = record.RecordingAgent(agent)
    result = local_evaluator.evaluate(
        recorder, rows, catalog_ids, categories, by_asin)
    sessions = analysis.analyze(recorder.sessions, rows, result, catalog_ids)
    ranks = collections.Counter(
        item["best_rank"] for item in result["sessions"])
    return {
        "score": result["recommended_technical_score"],
        "ranks": ranks,
        "rank1": ranks[1] / len(rows),
        "health": analysis.health_summary(sessions),
    }


def _difficulty_table(reports: list[tuple[str, dict]]) -> list[str]:
    lines = [
        f"{'set':<26}{'n':>5}{'medRN':>8}{'medBkt':>8}{'medFt':>7}"
        f"{'twin%':>7}{'buy>brw':>9}{'moved%':>8}",
        "-" * 78,
    ]
    for name, item in reports:
        moved = "-" if item["moved"] is None else f"{item['moved']:.0%}"
        lines.append(
            f"{name:<26}{item['count']:>5}{item['reviews']:>8.0f}"
            f"{item['bucket']:>8.0f}{item['features']:>7.1f}"
            f"{item['twins']:>7.0%}{item['silent_buying']:>9}{moved:>8}"
        )
    return lines


def _rank_table(reports: list[tuple[str, dict]]) -> list[str]:
    header = f"{'set':<26}"
    for label, _ in RANK_BUCKETS:
        header += f"{label:>7}"
    lines = [header + f"{'miss':>7}{'rank1%':>9}{'score':>9}", "-" * 78]
    for name, item in reports:
        ranks = item["ranks"]
        row = f"{name:<26}"
        for _, (low, high) in RANK_BUCKETS:
            row += f"{sum(ranks[rank] for rank in range(low, high + 1)):>7}"
        lines.append(
            row + f"{ranks[None]:>7}{item['rank1']:>9.1%}{item['score']:>9.4f}"
        )
    return lines


def render(profiles: list[tuple[str, dict]],
           scores: list[tuple[str, dict]]) -> str:
    """Returns the two tables plus the health verdict, as one string."""
    lines = _difficulty_table(profiles)
    if not scores:
        return "\n".join(lines)
    lines.extend(["", *_rank_table(scores), ""])
    broken = [
        name for name, item in scores
        if item["health"]["agent_exceptions"]
        or item["health"]["discarded_responses"]
    ]
    exceptions = sum(item["health"]["agent_exceptions"] for _, item in scores)
    discarded = sum(item["health"]["discarded_responses"]
                    for _, item in scores)
    short = sum(item["health"]["short_slates"] for _, item in scores)
    marker = "OK  " if not broken else "FAIL"
    lines.append(
        f"health {marker} exceptions={exceptions} discarded={discarded} "
        f"short_slates={short} over {len(scores)} sets"
    )
    if broken:
        lines.append(
            f"  a generated row broke the agent in: {', '.join(broken)}")
    return "\n".join(lines)


def load_products(catalog_path: Path) -> list[dict]:
    """Returns the catalog as a flat list, in file order.

    Line-wise, not splitlines(): product text carries Unicode separators that
    splitlines() would break on mid-string.
    """
    with catalog_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chosen(names: str) -> list[Recipe]:
    """Returns the frozen recipes named, or all of them."""
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    if not wanted:
        return list(session_sets.MANIFEST)
    known = {recipe.name: recipe for recipe in session_sets.MANIFEST}
    missing = [name for name in wanted if name not in known]
    if missing:
        raise SystemExit(
            f"unknown set(s) {', '.join(missing)}. Known: "
            f"{', '.join(recipe.name for recipe in session_sets.MANIFEST)}"
        )
    return [known[name] for name in wanted]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Synthetic session generator and its difficulty profile")
    parser.add_argument("--agent", default="submission.agent:Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--set", default="", dest="sets",
                        help="comma-separated subset of frozen set names")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N sessions of each set")
    parser.add_argument("--write", default=None,
                        help="emit the selected set as JSONL and stop")
    parser.add_argument("--list", action="store_true",
                        help="print the frozen set names and stop")
    return parser.parse_args(argv)


def _write(path: str, recipes: list[Recipe], products, facts, public_profiles,
           catalog_ids) -> int:
    if len(recipes) != 1:
        raise SystemExit("--write needs exactly one --set")
    rows = generate(recipes[0], products, facts, public_profiles)
    for row in rows:
        validate_row(row, catalog_ids)
    text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )
    Path(path).write_text(text, encoding="utf-8")
    print(f"{len(rows)} sessions written to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for recipe in session_sets.MANIFEST:
            print(f"{recipe.name:<26}seed {recipe.seed}")
        return 0

    catalog_path = Path(args.catalog)
    run.require_catalog(catalog_path)
    recipes = chosen(args.sets)

    started = perf_counter()
    products = load_products(catalog_path)
    facts = session_axes.survey(products)
    catalog_ids, categories, by_asin = local_evaluator.catalog_index(
        catalog_path)
    survey_s = round(perf_counter() - started, 2)
    public_profiles = [row["user_profile"]
                       for row in local_evaluator.load_jsonl(args.dataset)]

    if args.write:
        return _write(args.write, recipes, products, facts, public_profiles,
                      catalog_ids)

    started = perf_counter()
    agent = run.load_agent_class(args.agent)(str(catalog_path))
    build_s = round(perf_counter() - started, 2)
    print(f"{args.agent}   catalog survey {survey_s}s, agent build {build_s}s")
    print(f"{len(recipes)} frozen sets\n")

    profiles, scores = [], []
    for recipe in recipes:
        rows = generate(recipe, products, facts, public_profiles)
        for row in rows:
            validate_row(row, catalog_ids)
        if args.limit is not None:
            rows = rows[: args.limit]
        profiles.append((recipe.name, difficulty(rows, by_asin, facts)))
        scores.append((recipe.name,
                       measure(agent, rows, catalog_ids, categories, by_asin)))
    print(render(profiles, scores))
    print(
        "\nthese are properties of the instrument, not results about any "
        "feature.\nthe rank column is the shipped agent at its shipped "
        "defaults, nothing else."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
