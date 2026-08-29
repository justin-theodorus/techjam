"""Run-to-run comparison.

Aggregate deltas hide compensating changes: a tweak that wins five sessions and
loses five reads as a no-op. This reports which sessions flipped, in both
directions, so that never passes silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from techjam.harness.report import SCENARIO_ORDER, technical_score

METRIC_KEYS = ("hit_rate_at_10", "mrr", "mttc")
LISTED_FLIPS = 12


def _by_id(artifact: dict) -> dict[str, dict]:
    return {session["sample_id"]: session for session in artifact["sessions"]}


def compare(before: dict, after: dict) -> dict:
    old, new = _by_id(before), _by_id(after)
    shared = [key for key in new if key in old]
    return {
        "overall": {
            "hit_rate_at_10": (before["metrics"]["hit_rate_at_10"], after["metrics"]["hit_rate_at_10"]),
            "mrr": (before["metrics"]["mrr"], after["metrics"]["mrr"]),
            "mttc": (before["metrics"]["mttc"], after["metrics"]["mttc"]),
            "score": (
                before["metrics"]["recommended_technical_score"],
                after["metrics"]["recommended_technical_score"],
            ),
        },
        "scenarios": _scenario_deltas(before["metrics"], after["metrics"]),
        "gained": [key for key in shared if new[key]["hit"] and not old[key]["hit"]],
        "lost": [key for key in shared if old[key]["hit"] and not new[key]["hit"]],
        "rank_better": [
            key for key in shared
            if old[key]["hit"] and new[key]["hit"] and new[key]["best_rank"] < old[key]["best_rank"]
        ],
        "rank_worse": [
            key for key in shared
            if old[key]["hit"] and new[key]["hit"] and new[key]["best_rank"] > old[key]["best_rank"]
        ],
        "turn_better": [
            key for key in shared
            if old[key]["hit"] and new[key]["hit"]
            and new[key]["first_hit_turn"] < old[key]["first_hit_turn"]
        ],
        "turn_worse": [
            key for key in shared
            if old[key]["hit"] and new[key]["hit"]
            and new[key]["first_hit_turn"] > old[key]["first_hit_turn"]
        ],
        "only_in_before": sorted(key for key in old if key not in new),
        "only_in_after": sorted(key for key in new if key not in old),
    }


def _scenario_deltas(before: dict, after: dict) -> list[dict]:
    names = set(before["scenario_metrics"]) | set(after["scenario_metrics"])
    ordered = [name for name in SCENARIO_ORDER if name in names]
    ordered += sorted(name for name in names if name not in SCENARIO_ORDER)
    rows = []
    for name in ordered:
        old = before["scenario_metrics"].get(name)
        new = after["scenario_metrics"].get(name)
        if not old or not new:
            continue
        rows.append({
            "name": name,
            **{key: (old[key], new[key]) for key in METRIC_KEYS},
            "score": (technical_score(old), technical_score(new)),
        })
    return rows


def _delta(pair: tuple[float, float], width: int = 8, digits: int = 4) -> str:
    old, new = pair
    change = new - old
    sign = "+" if change >= 0 else ""
    return f"{new:>{width}.{digits}f} ({sign}{change:.{digits}f})"


def _flips(label: str, ids: list[str]) -> str:
    if not ids:
        return f"{label:<14} 0"
    listed = ", ".join(ids[:LISTED_FLIPS])
    suffix = f" (+{len(ids) - LISTED_FLIPS} more)" if len(ids) > LISTED_FLIPS else ""
    return f"{label:<14} {len(ids):<4} {listed}{suffix}"


def render(result: dict, before_label: str, after_label: str) -> str:
    lines = [f"diff  {before_label} -> {after_label}", ""]
    lines.append(f"{'scenario':<16}{'hit@10':>20}{'MRR':>20}{'MTTC':>20}{'score':>20}")
    lines.append("-" * 96)
    for row in result["scenarios"]:
        lines.append(
            f"{row['name']:<16}{_delta(row['hit_rate_at_10'], 8, 3):>20}"
            f"{_delta(row['mrr'], 8, 3):>20}{_delta(row['mttc'], 8, 2):>20}"
            f"{_delta(row['score'], 8, 4):>20}"
        )
    lines.append("-" * 96)
    overall = result["overall"]
    lines.append(
        f"{'OVERALL':<16}{_delta(overall['hit_rate_at_10'], 8, 3):>20}"
        f"{_delta(overall['mrr'], 8, 3):>20}{_delta(overall['mttc'], 8, 2):>20}"
        f"{_delta(overall['score'], 8, 4):>20}"
    )
    lines.append("")
    lines.append(_flips("miss -> hit", result["gained"]))
    lines.append(_flips("hit -> miss", result["lost"]))
    lines.append(_flips("rank better", result["rank_better"]))
    lines.append(_flips("rank worse", result["rank_worse"]))
    lines.append(_flips("turn better", result["turn_better"]))
    lines.append(_flips("turn worse", result["turn_worse"]))
    for label, key in (("only in before", "only_in_before"), ("only in after", "only_in_after")):
        if result[key]:
            lines.append(_flips(label, result[key]))
    return "\n".join(lines)


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two harness runs")
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    before, after = load(args.before), load(args.after)
    result = compare(before, after)
    print(render(result, args.before, args.after))
    if args.fail_on_regression and result["overall"]["score"][1] < result["overall"]["score"][0]:
        sys.exit(1)


if __name__ == "__main__":
    main()
