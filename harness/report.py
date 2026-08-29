"""Terminal rendering of a run artifact."""

from __future__ import annotations

import json
from pathlib import Path

SCENARIO_ORDER = ("buying", "browsing", "intent_override", "boundary")
BASELINE_PATH = Path("docs/baseline_results.json")


def baseline() -> dict | None:
    """The organizer's frozen reference, read rather than hardcoded so it cannot drift."""
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def efficiency(mttc: float) -> float:
    return max(0.0, min(1.0, (11.0 - mttc) / 10.0))


def technical_score(metrics: dict) -> float:
    return (
        0.50 * metrics["hit_rate_at_10"]
        + 0.30 * metrics["mrr"]
        + 0.20 * efficiency(metrics["mttc"])
    )


def _scenario_rows(metrics: dict) -> list[tuple]:
    scenarios = metrics["scenario_metrics"]
    total = sum(item["sample_count"] for item in scenarios.values()) or 1
    ordered = [name for name in SCENARIO_ORDER if name in scenarios]
    ordered += [name for name in sorted(scenarios) if name not in SCENARIO_ORDER]
    return [
        (
            name,
            scenarios[name]["sample_count"],
            scenarios[name]["sample_count"] / total,
            scenarios[name]["hit_rate_at_10"],
            scenarios[name]["mrr"],
            scenarios[name]["mttc"],
            technical_score(scenarios[name]),
        )
        for name in ordered
    ]


def render(artifact: dict) -> str:
    metrics = artifact["metrics"]
    lines = [
        f"agent      {artifact['agent']}",
        f"dataset    {artifact['dataset']}  ({metrics['sample_count']} sessions in {artifact['duration_s']}s)",
        "",
        f"{'scenario':<16}{'n':>5}{'share':>8}{'hit@10':>9}{'MRR':>9}{'MTTC':>8}{'score':>9}",
        "-" * 64,
    ]
    for name, count, share, hit, mrr, mttc, score in _scenario_rows(metrics):
        lines.append(
            f"{name:<16}{count:>5}{share:>8.0%}{hit:>9.3f}{mrr:>9.3f}{mttc:>8.2f}{score:>9.4f}"
        )
    lines.append("-" * 64)
    lines.append(
        f"{'OVERALL':<16}{metrics['sample_count']:>5}{1.0:>8.0%}"
        f"{metrics['hit_rate_at_10']:>9.3f}{metrics['mrr']:>9.3f}"
        f"{metrics['mttc']:>8.2f}{metrics['recommended_technical_score']:>9.4f}"
    )
    reference = baseline()
    if reference:
        lines.append(
            f"{'baseline':<16}{reference['sample_count']:>5}{1.0:>8.0%}"
            f"{reference['hit_rate_at_10']:>9.3f}{reference['mrr']:>9.3f}"
            f"{reference['mttc']:>8.2f}{reference['technical_score']:>9.4f}"
        )
    lines.extend(["", _health_line(artifact["health"]), _latency_line(artifact["latency"])])
    usage = _usage_line(artifact.get("usage"))
    if usage:
        lines.append(usage)
    return "\n".join(lines)


def _health_line(health: dict) -> str:
    critical = health["agent_exceptions"] + health["discarded_responses"]
    marker = "OK  " if critical == 0 else "FAIL"
    return (
        f"health {marker} exceptions={health['agent_exceptions']} "
        f"discarded={health['discarded_responses']} dropped_slots={health['dropped_slots']} "
        f"short_slates={health['short_slates']} wasted_pre_pivot_hits={health['wasted_pre_pivot_hits']}"
    )


def _usage_line(usage: dict | None) -> str:
    """The model tier's spend, or nothing at all on the offline path.

    Silent rather than zeroed when no model ran: the offline configuration's
    honest zero is already the `usage` field in every recorded turn, and a
    permanent "$0.00" line would read as an omission rather than a result.
    """
    if not usage or not usage.get("model"):
        return ""
    return (
        f"model    {usage['model']} calls={usage['calls']} "
        f"failures={usage['failures']} "
        f"tokens={usage['prompt_tokens']}/{usage['completion_tokens']} "
        f"cost=${usage['cost_usd']:.4f} "
        f"model_time={usage['model_ms'] / 1000.0:.1f}s"
    )


def _latency_line(latency: dict) -> str:
    if not latency.get("turn_count"):
        return "latency  no turns recorded"
    return (
        f"latency  p50={latency['p50_ms']}ms p95={latency['p95_ms']}ms "
        f"max={latency['max_ms']}ms over {latency['turn_count']} turns"
    )
