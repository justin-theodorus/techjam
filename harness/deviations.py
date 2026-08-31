"""Phase 6S-B gate: re-read the five switched-off components off the public 200.

Five components were built, measured and shipped switched off (findings 3.26 to
3.28). Every one of those verdicts was taken on a set where 176 of 200 sessions
already convert at rank 1, so it had 24 sessions of upside against 176 of
downside: enough to detect harm, nowhere near enough to detect benefit. This
sweeps the same switches over the same ranges against the frozen synthetic sets,
which run from 88% rank-1 down to 9%, and asks which verdicts survive.

    python3 -m harness.deviations
    python3 -m harness.deviations --component converge_at
    python3 -m harness.deviations --component diversity --set twin_cards --full

Three rules this gate exists to enforce, none of which a single score can:

- A set already at rank 1 nearly everywhere says nothing when a change does
  nothing to it. Those sets are marked `sat` and their cells are reported, never
  counted as evidence.
- A change that wins five sessions and loses five is not a no-op, so every cell
  carries its flips and not just its delta.
- A score above a non-zero health counter is meaningless, so the counters are
  summed across every point of every sweep.

`harness/sessions.py` and `harness/session_sets.py` are consumed **read-only**.
The seeds were frozen before any feature was scored against them; a set reshaped
because of what it showed about a feature is worthless as evidence.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from evaluator import local_evaluator

from harness import analysis
from harness import diff
from harness import identity
from harness import record
from harness import run
from harness import session_axes
from harness import sessions

# Where the modules holding the switches live. Resolved from a string so the
# sweep table below stays data, and so `harness/` keeps naming the submission
# package in exactly one place.
MODULES = {
    "dialogue": "submission.src.dialogue",
    "memory": "submission.src.memory",
    "orchestrate": "submission.src.orchestrate",
    "policy": "submission.src.policy",
    "probe": "submission.src.probe",
    "ranking": "submission.src.ranking",
    "routing": "submission.src.routing",
    "slots": "submission.src.slots",
}

PUBLIC = "public 200"

# A set whose target already sits at rank 1 this often has no ranking headroom
# left for a ranking change to be measured with. Reading "no effect" off such a
# set is reading "no room": the profile ablation moved 0 sessions up and 3 down
# not because the signal was absent but because there was nothing left to fix
# (findings 3.28, 3.29).
SATURATED = 0.85

# `RECOVERY_RESTART` is the one switch findings 3.26 measured across paired
# seeds, and it reports a seed-to-seed range this wide. Each frozen set is a
# single deterministic draw, so no single set can carry a claim smaller than
# this. The control available here is instead cross-set consistency: a real
# effect moves a coherent group of the 22 draws in one direction.
NOISE_FLOOR = 0.04

RANK_BUCKETS = sessions.RANK_BUCKETS


@dataclass(frozen=True)
class Deviation:
    """One switched-off component and the settings to read it at."""

    name: str
    switch: str
    neutral: str
    points: tuple[tuple[str, dict[str, float]], ...]


def _alpha(precision: float, discovery: float) -> dict[str, float]:
    return {
        "routing.PRECISION_ALPHA": precision,
        "routing.DISCOVERY_ALPHA": discovery,
    }


# The ranges are the ones findings 3.26 to 3.28 already swept on the public 200,
# so every cell below is directly comparable to a published number rather than
# to a fresh grid nobody has a reference for.
DEVIATIONS = (
    Deviation(
        "route_alpha",
        "routing.PRECISION_ALPHA / routing.DISCOVERY_ALPHA",
        "both at ranking.ALPHA",
        (
            ("0.4/1.0", _alpha(0.4, 1.0)),
            ("0.4/1.3", _alpha(0.4, 1.3)),
            ("0.7/1.0", _alpha(0.7, 1.0)),
            ("0.7/1.3", _alpha(0.7, 1.3)),
        ),
    ),
    Deviation(
        "phrase_pool",
        "ranking.PHRASE_POOL",
        "20 (live, so 0 is the deviation)",
        tuple(
            (f"{w:d}", {"ranking.PHRASE_POOL": w})
            for w in (0, 10, 15, 25, 40)
        ),
    ),
    Deviation(
        "recovery_restart",
        "routing.RECOVERY_RESTART",
        "0",
        (("1", {"routing.RECOVERY_RESTART": 1}),),
    ),
    Deviation(
        "diversity",
        "ranking.DIVERSITY",
        "0.0",
        tuple(
            (f"{weight:g}", {"ranking.DIVERSITY": weight})
            for weight in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
        ),
    ),
    Deviation(
        "head_contention",
        "ranking.HEAD_FROM_CONTENTION / ranking.CONTENTION_MARGIN",
        "True at 0.0005 (live): the head is the contention count, floor HEAD_SIZE",
        (
            ("asserted", {"ranking.HEAD_FROM_CONTENTION": False}),
            ("m .001", {"ranking.CONTENTION_MARGIN": 0.001}),
            ("m .01", {"ranking.CONTENTION_MARGIN": 0.01}),
            ("m .02", {"ranking.CONTENTION_MARGIN": 0.02}),
            ("m .05", {"ranking.CONTENTION_MARGIN": 0.05}),
        ),
    ),
    Deviation(
        "route_defer",
        "routing.DISCOVERY_DEFER / routing.PRECISION_DEFER",
        "3 and 6 (live): discovery opens sooner than precision",
        (
            ("neutral", {"routing.DISCOVERY_DEFER": None,
                         "routing.PRECISION_DEFER": None}),
            ("disc 2", {"routing.DISCOVERY_DEFER": 2}),
            ("disc 1", {"routing.DISCOVERY_DEFER": 1}),
            ("prec 8", {"routing.PRECISION_DEFER": 8}),
        ),
    ),
    Deviation(
        "route_head",
        "routing.DISCOVERY_HEAD",
        "None (defers to ranking.HEAD_SIZE, which is 1)",
        tuple((f"disc {h}", {"routing.DISCOVERY_HEAD": h})
              for h in (2, 3, 5, 10)),
    ),
    Deviation(
        "explore_fill",
        "ranking.EXPLORE_FILL",
        "False (live): the head is served alone and the withheld slots stay empty",
        (("on", {"ranking.EXPLORE_FILL": True}),),
    ),
    Deviation(
        "explore_band",
        "ranking.EXPLORE_DIVERSITY / ranking.EXPLORE_SORT",
        "0.95 sorted (live); 0.0 restores compose's fixed ranks 11-19",
        (
            ("off", {"ranking.EXPLORE_DIVERSITY": 0.0}),
            ("unsorted", {"ranking.EXPLORE_SORT": False}),
            ("w.5", {"ranking.EXPLORE_DIVERSITY": 0.5}),
            ("w.9", {"ranking.EXPLORE_DIVERSITY": 0.9}),
            ("w.99", {"ranking.EXPLORE_DIVERSITY": 0.99}),
        ),
    ),
    Deviation(
        "route_diversity",
        "routing.DISCOVERY_DIVERSITY",
        "None (defers to ranking.DIVERSITY, which is 0.0)",
        tuple(
            (f"disc {weight:g}", {"routing.DISCOVERY_DIVERSITY": weight})
            for weight in (0.3, 0.5, 0.7, 0.9)
        ),
    ),
    Deviation(
        "diversity_gate",
        "ranking.DIVERSITY_MAX_CONSTRAINTS, over a live ranking.DIVERSITY",
        "-1 (never vetoes, so an enabled weight spreads every turn)",
        (
            ("w.3 n0", {"ranking.DIVERSITY": 0.3,
                        "ranking.DIVERSITY_MAX_CONSTRAINTS": 0}),
            ("w.5 n0", {"ranking.DIVERSITY": 0.5,
                        "ranking.DIVERSITY_MAX_CONSTRAINTS": 0}),
            ("w.7 n0", {"ranking.DIVERSITY": 0.7,
                        "ranking.DIVERSITY_MAX_CONSTRAINTS": 0}),
            ("w.5 n1", {"ranking.DIVERSITY": 0.5,
                        "ranking.DIVERSITY_MAX_CONSTRAINTS": 1}),
            ("disc.5 n0", {"routing.DISCOVERY_DIVERSITY": 0.5,
                           "ranking.DIVERSITY_MAX_CONSTRAINTS": 0}),
        ),
    ),
    Deviation(
        "flatness_gate",
        "ranking.FLATNESS_GATE, over a live ranking.DIVERSITY",
        "0.0 (never vetoes)",
        (
            ("w.5 f.50", {"ranking.DIVERSITY": 0.5,
                          "ranking.FLATNESS_GATE": 0.50}),
            ("w.5 f.65", {"ranking.DIVERSITY": 0.5,
                          "ranking.FLATNESS_GATE": 0.65}),
            ("w.5 f.80", {"ranking.DIVERSITY": 0.5,
                          "ranking.FLATNESS_GATE": 0.80}),
            ("w.9 f.65", {"ranking.DIVERSITY": 0.9,
                          "ranking.FLATNESS_GATE": 0.65}),
        ),
    ),
    Deviation(
        "converge_at",
        "ranking.CONVERGE_AT",
        "0 (never)",
        tuple(
            (str(count), {"ranking.CONVERGE_AT": count})
            for count in (1, 2, 3)
        ),
    ),
    Deviation(
        "negation",
        "slots.NEGATION / ranking.NEGATION_WEIGHT",
        "read, penalty 0.5 (live)",
        (
            ("unread", {"slots.NEGATION": False}),
            ("w=0", {"ranking.NEGATION_WEIGHT": 0.0}),
            ("w=0.25", {"ranking.NEGATION_WEIGHT": 0.25}),
            ("w=1", {"ranking.NEGATION_WEIGHT": 1.0}),
            ("w=2", {"ranking.NEGATION_WEIGHT": 2.0}),
        ),
    ),
    Deviation(
        "dialogue_policy",
        "probe.STAGNATION_ESCAPE / probe.COVERAGE_SILENCE",
        "both True (live)",
        (
            ("no escape", {"probe.STAGNATION_ESCAPE": False}),
            ("no silence", {"probe.COVERAGE_SILENCE": False}),
            ("neither", {"probe.STAGNATION_ESCAPE": False,
                         "probe.COVERAGE_SILENCE": False}),
            ("escape n1", {"dialogue.STAGNATION_TURNS": 1}),
            ("escape n3", {"dialogue.STAGNATION_TURNS": 3}),
        ),
    ),
    Deviation(
        "skip_shown",
        "ranking.SKIP_SHOWN",
        "True (live)",
        (("off", {"ranking.SKIP_SHOWN": False}),),
    ),
    Deviation(
        "dense",
        "ranking.DENSE_WEIGHT",
        "0.0 (asset bundled and loaded, contributes nothing)",
        tuple(
            (f"{weight:g}", {"ranking.DENSE_WEIGHT": weight})
            for weight in (0.2, 0.4, 0.6, 0.9, 1.3)
        ),
    ),
    Deviation(
        "dense_negation",
        "ranking.DENSE_NEGATION_WEIGHT",
        "0.0, with ranking.DENSE_WEIGHT at 0.6",
        tuple(
            (f"{weight:g}", {
                "ranking.DENSE_WEIGHT": 0.6,
                "ranking.DENSE_NEGATION_WEIGHT": weight,
            })
            for weight in (0.0, 0.25, 0.5, 1.0)
        ),
    ),
    Deviation(
        "dense_route",
        "routing.PRECISION_DENSE / DISCOVERY_DENSE / DISCOVERY_REACH",
        "all None, reach 0 (no route specialises)",
        (
            ("prec 0.6", {"routing.PRECISION_DENSE": 0.6}),
            ("disc 0.6", {"routing.DISCOVERY_DENSE": 0.6}),
            ("reach 20", {"routing.DISCOVERY_REACH": 20}),
            ("reach 100", {"routing.DISCOVERY_REACH": 100}),
            ("disc 0.6 + reach 100", {
                "routing.DISCOVERY_DENSE": 0.6,
                "routing.DISCOVERY_REACH": 100,
            }),
        ),
    ),
    # Phase 6Y. `orchestration` ships *on*, so its row reads backwards like
    # `skip_shown` and `negation` do: a positive delta is a reason to switch it
    # off. `orchestration_control` is the pair of controls that could have
    # refuted the phase, plus the selection rule it replaced (findings 3.50,
    # decision 31).
    Deviation(
        "orchestration",
        "orchestrate.ENABLED / SPENT_RATIO / CANDIDATES",
        "True at 0.5 over four candidates (shipped on)",
        (
            ("off", {"orchestrate.ENABLED": False}),
            ("fire at 0.3", {"orchestrate.SPENT_RATIO": 0.3}),
            ("fire at 0.7", {"orchestrate.SPENT_RATIO": 0.7}),
            ("phrase only", {"orchestrate.CANDIDATES": ("blend", "phrase")}),
            ("prior only", {"orchestrate.CANDIDATES": ("blend", "prior")}),
            ("lexical only",
             {"orchestrate.CANDIDATES": ("blend", "lexical")}),
        ),
    ),
    # Phase 6Z. Decision readiness ships *off* as a controller and on as a
    # trace, so this row reads forwards: a positive delta for "steer on it" is
    # what would justify switching it on. Every cell measured neutral; the
    # threshold arms bracket 0.7/0.3 on both sides so that verdict is read off
    # a curve rather than a point (see `policy.READINESS_STEERS`).
    Deviation(
        "readiness",
        "policy.READINESS_STEERS / *_READINESS_THRESHOLD / CURRENT_WEIGHT",
        "steering off; readiness is reported but does not pick the policy",
        (
            ("steer on it", {"policy.READINESS_STEERS": True}),
            ("commit at 0.5", {"policy.READINESS_STEERS": True,
                               "policy.PRECISION_READINESS_THRESHOLD": 0.5}),
            ("commit at 0.9", {"policy.READINESS_STEERS": True,
                               "policy.PRECISION_READINESS_THRESHOLD": 0.9}),
            ("partial at 0.15", {"policy.READINESS_STEERS": True,
                                 "policy.PARTIAL_READINESS_THRESHOLD": 0.15}),
            ("partial at 0.45", {"policy.READINESS_STEERS": True,
                                 "policy.PARTIAL_READINESS_THRESHOLD": 0.45}),
            ("weight 0.5", {"policy.READINESS_STEERS": True,
                            "policy.READINESS_CURRENT_WEIGHT": 0.5}),
            ("weight 0.9", {"policy.READINESS_STEERS": True,
                            "policy.READINESS_CURRENT_WEIGHT": 0.9}),
            ("memoryless", {"policy.READINESS_STEERS": True,
                            "policy.READINESS_CURRENT_WEIGHT": 1.0}),
        ),
    ),
    Deviation(
        "hybrid_framing",
        "policy.HYBRID_FRAMING / policy.HYBRID_MARGIN",
        "off; a near-tie is reported but the winner still owns the turn",
        (
            ("runner-up framing", {"policy.HYBRID_FRAMING": True}),
            ("tie under 0.25", {"policy.HYBRID_FRAMING": True,
                                "policy.HYBRID_MARGIN": 0.25}),
            ("tie under 1.0", {"policy.HYBRID_FRAMING": True,
                               "policy.HYBRID_MARGIN": 1.0}),
        ),
    ),
    Deviation(
        "recovery_window",
        "policy.RECOVERY_TURNS",
        "0, meaning recovery holds for the rest of the session",
        (
            ("1 turn", {"policy.RECOVERY_TURNS": 1}),
            ("2 turns", {"policy.RECOVERY_TURNS": 2}),
            ("3 turns", {"policy.RECOVERY_TURNS": 3}),
        ),
    ),
    Deviation(
        "orchestration_control",
        "orchestrate.SCHEDULE / orchestrate.BLIND / orchestrate.FRESHEST",
        "off; the controller reads the refuted set",
        (
            ("schedule 2", {"orchestrate.SCHEDULE": 2}),
            ("schedule 3", {"orchestrate.SCHEDULE": 3}),
            ("schedule 5", {"orchestrate.SCHEDULE": 5}),
            ("blind", {"orchestrate.BLIND": True}),
            ("freshest", {"orchestrate.FRESHEST": True}),
        ),
    ),
    Deviation(
        "llm_rerank",
        "ranking.LLM_RERANK",
        "0",
        (("on", {"ranking.LLM_RERANK": 1}),),
    ),
    Deviation(
        "profile_weight",
        "ranking.PROFILE_WEIGHT / ranking.PROFILE_MAX_CONSTRAINTS",
        "0.02 gated to no-attribute turns (live)",
        (
            ("off", {"ranking.PROFILE_WEIGHT": 0.0}),
            ("ungated", {"ranking.PROFILE_MAX_CONSTRAINTS": -1}),
            ("0.05", {"ranking.PROFILE_WEIGHT": 0.05}),
            ("0.1", {"ranking.PROFILE_WEIGHT": 0.1}),
            ("0.2", {"ranking.PROFILE_WEIGHT": 0.2}),
            ("0.4 open", {"ranking.PROFILE_WEIGHT": 0.4,
                          "ranking.PROFILE_MAX_CONSTRAINTS": -1}),
        ),
    ),
    # The two below ship *on*, so their sweeps read backwards from every other
    # row here: the deviation is the old behaviour, and a positive delta is a
    # reason to switch them back off (findings 3.37).
    Deviation(
        "specific_arms",
        "probe.SPECIFIC_ARMS",
        "True (shipped on)",
        (("wildcard", {"probe.SPECIFIC_ARMS": False}),),
    ),
    Deviation(
        "wildcard_fallback",
        "probe.WILDCARD_FALLBACK_RATIO",
        "0.2 (shipped on)",
        tuple(
            (f"{ratio:g}", {"probe.WILDCARD_FALLBACK_RATIO": ratio})
            for ratio in (0.0, 0.35, 0.5, 0.8, 1.01)
        ),
    ),
    Deviation(
        "scoped_exhaustion",
        "dialogue.SCOPED_EXHAUSTION",
        "True (shipped on)",
        (("whole session", {"dialogue.SCOPED_EXHAUSTION": False}),),
    ),
    Deviation(
        "memory",
        "memory.ENABLED and its four read gates",
        "enabled, preferences off (shipped on); zero on every set whose rows "
        "name no shopper, which is the bit-identical claim taken through the "
        "sweep. `make memory` is where the component is actually read",
        (
            ("off", {"memory.ENABLED": False}),
            ("-refuse", {"memory.CARRY_REFUSALS": False}),
            ("-arms", {"memory.CARRY_ARMS": False}),
            ("-buckets", {"memory.CARRY_BUCKETS": False}),
            ("+prefer", {"memory.CARRY_POSITIVES": True}),
        ),
    ),
)

# Components that cost money and need a network, and are therefore reachable
# only by naming them. A default sweep is 61 points over 23 sets; putting a
# per-turn model call in that grid would be roughly 400,000 requests, so the
# exclusion is a correctness property of `make deviations`, not a preference.
OPT_IN = ("llm_rerank",)


@contextlib.contextmanager
def patched(assignments: dict[str, float]):
    """Sets module constants for the block, and puts them back afterwards.

    Every constant these sweeps touch is a module global read at call time, so
    assigning to the module is enough and the agent is never rebuilt. Restoring
    in a `finally` is what stops one sweep point leaking into the next.

    Args:
        assignments: `{"module.CONSTANT": value}`, keyed into `MODULES`.

    Raises:
        AttributeError: the sweep named a constant that does not exist, which is
            a sweep bug and must not be swallowed into a run of null results.
    """
    originals = []
    try:
        for dotted, value in assignments.items():
            module_name, _, attribute = dotted.rpartition(".")
            module = importlib.import_module(MODULES[module_name])
            originals.append((module, attribute, getattr(module, attribute)))
            setattr(module, attribute, value)
        yield
    finally:
        for module, attribute, value in originals:
            setattr(module, attribute, value)


def score(agent, rows: list[dict], catalog_ids: set[str],
          categories: dict[str, list[str]],
          by_asin: dict[str, dict]) -> dict:
    """Returns one run artifact: metrics, health, ranks, and every session.

    `sessions.measure` reports the aggregate rank distribution and drops the
    per-session identity, which is what a flip is made of, and that module is
    frozen for this phase. The artifact is therefore assembled here, in the
    shape `harness/diff.py` already consumes.
    """
    # Wrapped so a set whose rows name a shopper is scored with those
    # identities supplied, and so every sweep point starts from an
    # empty store: `patched` never rebuilds the agent, so without the
    # proxy's own reset the cell order would decide the result.
    recorder = record.RecordingAgent(identity.ReturningAgent(agent, rows))
    result = local_evaluator.evaluate(
        recorder, rows, catalog_ids, categories, by_asin)
    analyzed = analysis.analyze(recorder.sessions, rows, result, catalog_ids)
    return {
        "metrics": {key: value for key, value in result.items()
                    if key != "sessions"},
        "health": analysis.health_summary(analyzed),
        "latency": analysis.latency_summary(analyzed),
        "usage": analysis.usage_summary(analyzed),
        "ranks": collections.Counter(
            item["best_rank"] for item in result["sessions"]),
        "sessions": analyzed,
    }


def flips(before: dict, after: dict) -> dict:
    """Returns which sessions the change helped and which it hurt.

    A change that wins five sessions and loses five has the same delta as one
    that touches nothing, and they are not the same change.
    """
    verdict = diff.compare(before, after)
    return {
        "up": verdict["gained"] + verdict["rank_better"],
        "down": verdict["lost"] + verdict["rank_worse"],
        "sooner": verdict["turn_better"],
        "later": verdict["turn_worse"],
    }


def rank1(artifact: dict) -> float:
    """Share of sessions whose target converted at rank 1."""
    return artifact["ranks"][1] / max(1, len(artifact["sessions"]))


def is_saturated(artifact: dict) -> bool:
    return rank1(artifact) >= SATURATED


def unmoved(baseline: dict, results: dict) -> bool:
    """Whether a whole sweep reproduced the neutral score in every cell.

    Findings 3.27 lost a sweep to a default argument that bound the constant at
    definition, and every row came back identical. A sweep whose rows are all
    identical is a bug until proven otherwise.
    """
    return all(
        cell["metrics"]["recommended_technical_score"]
        == baseline[name]["metrics"]["recommended_technical_score"]
        for column in results.values()
        for name, cell in column.items()
    )


def _health_total(artifacts) -> dict:
    total: dict[str, int] = collections.Counter()
    for artifact in artifacts:
        total.update(artifact["health"])
    return dict(total)


def _spend_line(cells: list[dict]) -> str:
    """What a model-backed sweep cost, across every cell of it.

    Silent when nothing called a model, so the nine offline components keep
    the output they have always had.
    """
    usage = [cell["usage"] for cell in cells if cell["usage"]["model"]]
    if not usage:
        return ""
    prompt = sum(item["prompt_tokens"] for item in usage)
    completion = sum(item["completion_tokens"] for item in usage)
    calls = sum(item["calls"] for item in usage)
    failures = sum(item["failures"] for item in usage)
    seconds = sum(item["model_ms"] for item in usage) / 1000.0
    dollars = sum(item["cost_usd"] for item in usage)
    return (
        f"model    {usage[0]['model']} calls={calls} failures={failures} "
        f"tokens={prompt}/{completion} cost=${dollars:.4f} "
        f"model_time={seconds:.0f}s"
    )


def _health_line(health: dict) -> str:
    critical = (health.get("agent_exceptions", 0)
                + health.get("discarded_responses", 0))
    marker = "OK  " if critical == 0 else "FAIL"
    return (
        f"health {marker} exceptions={health.get('agent_exceptions', 0)} "
        f"discarded={health.get('discarded_responses', 0)} "
        f"dropped_slots={health.get('dropped_slots', 0)} "
        f"short_slates={health.get('short_slates', 0)}"
    )


def _rank_cells(artifact: dict) -> str:
    ranks = artifact["ranks"]
    cells = "".join(
        f"{sum(ranks[rank] for rank in range(low, high + 1)):>6}"
        for _, (low, high) in RANK_BUCKETS
    )
    return cells + f"{ranks[None]:>6}"


def baseline_table(baseline: dict) -> list[str]:
    """The rank distribution every sweep below is read against."""
    header = "".join(f"{label:>6}" for label, _ in RANK_BUCKETS)
    title = (f"{'set':<24}{header}{'miss':>6}{'rank1%':>9}{'miss%':>8}"
             f"{'score':>9}  headroom")
    lines = [title, "-" * len(title)]
    for name, artifact in baseline.items():
        total = max(1, len(artifact["sessions"]))
        misses = artifact["ranks"][None] / total
        lines.append(
            f"{name:<24}{_rank_cells(artifact)}"
            f"{rank1(artifact) * 100:>8.1f}%{misses * 100:>7.1f}%"
            f"{artifact['metrics']['recommended_technical_score']:>9.4f}"
            f"  {'saturated' if is_saturated(artifact) else 'readable'}"
        )
    return lines


def _delta(cell: dict, base: dict) -> float:
    return (cell["metrics"]["recommended_technical_score"]
            - base["metrics"]["recommended_technical_score"])


def delta_table(deviation: Deviation, baseline: dict,
                results: dict) -> list[str]:
    """Score change per set per sweep point, against the neutral run."""
    labels = [label for label, _ in deviation.points]
    lines = [
        f"{'set':<24}{'neutral':>9}"
        + "".join(f"{label:>10}" for label in labels),
        "-" * (33 + 10 * len(labels)),
    ]
    for name, base in baseline.items():
        marker = "*" if is_saturated(base) else " "
        cells = "".join(
            f"{_delta(results[label][name], base):>+10.4f}"
            for label in labels
        )
        lines.append(
            f"{name:<23}{marker}"
            f"{base['metrics']['recommended_technical_score']:>9.4f}{cells}"
        )
    lines.append(
        "* saturated: no headroom left, so that row is not evidence")
    return lines


def flip_table(deviation: Deviation, baseline: dict,
               results: dict) -> list[str]:
    """Sessions helped and hurt per set per sweep point."""
    labels = [label for label, _ in deviation.points]
    lines = [
        f"{'set':<24}" + "".join(f"{label:>10}" for label in labels),
        "-" * (24 + 10 * len(labels)),
    ]
    for name, base in baseline.items():
        cells = ""
        for label in labels:
            moved = flips(base, results[label][name])
            pair = f"+{len(moved['up'])}/-{len(moved['down'])}"
            cells += f"{pair:>10}"
        lines.append(f"{name:<24}{cells}")
    lines.append("cells are sessions the change promoted / demoted, by rank")
    return lines


def _listed(ids: list[str], limit: int = 8) -> str:
    if not ids:
        return "none"
    suffix = f" (+{len(ids) - limit} more)" if len(ids) > limit else ""
    return ", ".join(ids[:limit]) + suffix


def detail_lines(deviation: Deviation, baseline: dict,
                 results: dict) -> list[str]:
    """Per-session ids behind every non-empty cell, for `--full`."""
    lines = []
    for label, _ in deviation.points:
        for name, base in baseline.items():
            moved = flips(base, results[label][name])
            if not any(moved.values()):
                continue
            lines.append(f"  {deviation.name} {label} on {name}")
            for key in ("up", "down", "sooner", "later"):
                if moved[key]:
                    lines.append(f"    {key:<8}{_listed(moved[key])}")
    return lines


def sweep(agent, deviation: Deviation, corpus: list[tuple[str, list[dict]]],
          catalog_ids, categories, by_asin) -> dict:
    """Returns `{point label: {set name: artifact}}` for one component."""
    results = {}
    for label, assignments in deviation.points:
        with patched(assignments):
            results[label] = {
                name: score(agent, rows, catalog_ids, categories, by_asin)
                for name, rows in corpus
            }
    return results


def _chosen(names: str) -> list[Deviation]:
    """Returns the components named, or all of them."""
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    if not wanted:
        return [item for item in DEVIATIONS if item.name not in OPT_IN]
    known = {item.name: item for item in DEVIATIONS}
    missing = [name for name in wanted if name not in known]
    if missing:
        raise SystemExit(
            f"unknown component(s) {', '.join(missing)}. Known: "
            f"{', '.join(item.name for item in DEVIATIONS)}"
        )
    return [known[name] for name in wanted]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Phase 6S-B gate: the switched-off components, re-read")
    parser.add_argument("--agent", default="submission.agent:Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--component", default="", dest="components",
                        help="comma-separated subset of component names")
    parser.add_argument("--set", default="", dest="sets",
                        help="comma-separated subset of frozen set names")
    parser.add_argument("--no-public", action="store_true",
                        help="skip the real public 200 row")
    parser.add_argument("--full", action="store_true",
                        help="list the session ids behind every flip")
    return parser.parse_args(argv)


def build_corpus(recipes, products, facts, public_profiles, catalog_ids,
                 public: list[dict] | None) -> list[tuple[str, list[dict]]]:
    """Generates every set once, so a sweep re-scores rather than re-draws."""
    corpus = [] if public is None else [(PUBLIC, public)]
    for recipe in recipes:
        rows = sessions.generate(recipe, products, facts, public_profiles)
        for row in rows:
            sessions.validate_row(row, catalog_ids)
        corpus.append((recipe.name, rows))
    return corpus


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_path = Path(args.catalog)
    run.require_catalog(catalog_path)
    deviations = _chosen(args.components)
    recipes = sessions.chosen(args.sets)

    started = perf_counter()
    products = sessions.load_products(catalog_path)
    facts = session_axes.survey(products)
    catalog_ids, categories, by_asin = local_evaluator.catalog_index(
        catalog_path)
    public = local_evaluator.load_jsonl(args.dataset)
    public_profiles = [row["user_profile"] for row in public]
    survey_s = round(perf_counter() - started, 2)

    started = perf_counter()
    agent = run.load_agent_class(args.agent)(str(catalog_path))
    build_s = round(perf_counter() - started, 2)

    corpus = build_corpus(recipes, products, facts, public_profiles,
                          catalog_ids, None if args.no_public else public)
    baseline = {
        name: score(agent, rows, catalog_ids, categories, by_asin)
        for name, rows in corpus
    }

    print(f"{args.agent}   catalog survey {survey_s}s, agent build {build_s}s")
    print(f"{len(deviations)} components over {len(corpus)} sets\n")
    print("BASELINE  every switch at its shipped value")
    print("\n".join(baseline_table(baseline)))
    print(_health_line(_health_total(baseline.values())))

    for deviation in deviations:
        results = sweep(agent, deviation, corpus, catalog_ids, categories,
                        by_asin)
        print(f"\n\n{deviation.name.upper()}  {deviation.switch}"
              f"   neutral {deviation.neutral}")
        print("\n".join(delta_table(deviation, baseline, results)))
        print()
        print("\n".join(flip_table(deviation, baseline, results)))
        cells = [cell for column in results.values()
                 for cell in column.values()]
        print(_health_line(_health_total(cells)))
        spend = _spend_line(cells)
        if spend:
            print(spend)
        if unmoved(baseline, results):
            print("WARNING every cell reproduced the neutral score. A sweep "
                  "whose rows are all identical is a bug until proven "
                  "otherwise (findings 3.27).")
        if args.full:
            print("\n".join(detail_lines(deviation, baseline, results)))

    print(
        "\n\nthe neutral column must reproduce `make sessions` set for set, "
        "and `make eval`\non the public row; if it does not, nothing above it "
        "is readable.\na set marked saturated cannot distinguish 'no effect' "
        "from 'no headroom'.\nno single set carries a claim below the "
        f"{NOISE_FLOOR} seed range of findings 3.26."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
