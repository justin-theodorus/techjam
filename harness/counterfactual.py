"""Risk B gate: score the agent against other target distributions.

The headline score is conditional on how sessions were sampled. `README.md:111`
says targets come from review records, which makes target probability
proportional to `rating_number` and the popularity prior close to optimal. If
that is wrong on the private set, the prior is the component that swings.

`materialize_hidden_fields()` derives every session from the target's catalog
record, so a 200-session set can be manufactured with any targets at all. This
builds three, holding the 80/80/30/10 scenario mix and resampling user profiles
from the public rows, and scores each through the real `evaluate()`.

    python3 -m harness.counterfactual [--agent module:Class] [--seeds 3]

Read the `uniform` column as a pessimistic bound rather than a forecast: a
uniformly drawn target is not merely less popular, it is a genuinely harder
product with thinner text (findings 3.12, 3.20).
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
from pathlib import Path
from time import perf_counter

from techjam.evaluator.local_evaluator import evaluate, load_jsonl

from techjam.harness.run import catalog_index, load_agent_class, require_catalog

SCENARIO_MIX = (("buying", 80), ("browsing", 80), ("intent_override", 30), ("boundary", 10))
SESSION_COUNT = sum(count for _, count in SCENARIO_MIX)


def review_count(product: dict) -> float:
    value = product.get("rating_number")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if value > 0 else 0.0


def weight_schemes(products: list[dict]) -> dict[str, list[float]]:
    """The documented sampling rule, a weakened one, and the adversarial one."""
    counts = [review_count(product) for product in products]
    return {
        "size-biased": counts,
        "sqrt": [math.sqrt(count) for count in counts],
        "uniform": [1.0] * len(counts),
    }


def draw_targets(products: list[dict], weights: list[float], seed: int) -> list[dict]:
    cumulative, total = [], 0.0
    for weight in weights:
        total += weight
        cumulative.append(total)
    rng = random.Random(seed)
    return [
        products[bisect.bisect_left(cumulative, rng.random() * total)]
        for _ in range(SESSION_COUNT)
    ]


def build_samples(products: list[dict], weights: list[float], profiles: list[dict], seed: int) -> list[dict]:
    """Returns a synthetic session set with the public scenario mix."""
    targets = draw_targets(products, weights, seed)
    rng = random.Random(seed + 7)
    samples, index = [], 0
    for scenario, count in SCENARIO_MIX:
        for _ in range(count):
            samples.append({
                "sample_id": f"cf_{seed}_{index:04d}",
                "scenario_type": scenario,
                "category_bucket": "clothing",
                "difficulty_bucket": "easy",
                "user_profile": rng.choice(profiles),
                "ground_truth": {"parent_asin": str(targets[index]["parent_asin"])},
            })
            index += 1
    return samples


def surface(agent, products, profiles, catalog_ids, categories, by_asin, seeds: int) -> dict:
    """Returns TechnicalScore per distribution, averaged over `seeds` draws."""
    scores = {}
    for name, weights in weight_schemes(products).items():
        values = []
        for seed in range(1, seeds + 1):
            samples = build_samples(products, weights, profiles, seed)
            values.append(evaluate(agent, samples, catalog_ids, categories, by_asin)["recommended_technical_score"])
        scores[name] = sum(values) / len(values)
    return scores


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Counterfactual target-distribution gate")
    parser.add_argument("--agent", default="submission.agent:Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--seeds", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_path = Path(args.catalog)
    require_catalog(catalog_path)

    # Line-wise, not splitlines(): product text carries Unicode separators that
    # splitlines() would break on mid-string.
    with catalog_path.open(encoding="utf-8") as handle:
        products = [json.loads(line) for line in handle if line.strip()]
    public = load_jsonl(args.dataset)
    profiles = [row["user_profile"] for row in public]
    catalog_ids, categories, by_asin = catalog_index(catalog_path)

    started = perf_counter()
    agent = load_agent_class(args.agent)(str(catalog_path))
    build_s = round(perf_counter() - started, 2)

    real = evaluate(agent, public, catalog_ids, categories, by_asin)["recommended_technical_score"]
    scores = surface(agent, products, profiles, catalog_ids, categories, by_asin, args.seeds)

    print(f"{args.agent}   catalog build {build_s}s, {args.seeds} seeds per column\n")
    header = f"{'public (real)':>15}" + "".join(f"{name:>15}" for name in scores)
    print(header + f"{'worst':>15}")
    row = f"{real:>15.4f}" + "".join(f"{value:>15.4f}" for value in scores.values())
    print(row + f"{min(scores.values()):>15.4f}")
    print("\nread `uniform` as a pessimistic bound, not a forecast (findings 3.20)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
