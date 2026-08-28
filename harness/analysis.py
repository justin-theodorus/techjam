"""Post-hoc analysis of a recorded run.

Turns the raw proxy recording plus the evaluator's own session results into one
diffable artifact: per session, per turn, what was emitted and whether the target
was in it.
"""

from __future__ import annotations

import statistics

from evaluator.local_evaluator import normalize_recommendations

PIVOT_PREFIX = "Actually, ignore my earlier preference."
DISCLOSURE_PREFIX = "For that, what matters is: "
NO_PREFERENCE_PREFIX = "I don't have a preference for "


def _disclosed_constraints(user_message: str) -> list[str]:
    if not user_message.startswith(DISCLOSURE_PREFIX):
        return []
    body = user_message[len(DISCLOSURE_PREFIX):].rstrip(".")
    return [part.strip() for part in body.split(";") if part.strip()]


def analyze_session(record: dict, sample: dict, result: dict, catalog_ids: set[str]) -> dict:
    """Merge one recorded session with the evaluator's verdict for it."""
    target = str(sample["ground_truth"]["parent_asin"])
    is_override = sample["scenario_type"] == "intent_override"
    scorable = not is_override
    pivot_turn: int | None = None
    turns: list[dict] = []

    for entry in record["turns"]:
        if entry["user_message"].startswith(PIVOT_PREFIX):
            scorable = True
            pivot_turn = entry["turn"]
        ranked = normalize_recommendations(entry["raw_recommendations"], catalog_ids)
        rank = ranked.index(target) + 1 if target in ranked else None
        turns.append({
            **entry,
            "slate": ranked,
            "dropped_slots": len(entry["raw_recommendations"]) - len(ranked),
            "target_rank": rank,
            "scorable": scorable,
            "wasted_hit": rank is not None and not scorable,
            "disclosed": _disclosed_constraints(entry["user_message"]),
            "boundary_refusal": entry["user_message"].startswith(NO_PREFERENCE_PREFIX),
        })

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "target": target,
        "hit": result["hit"],
        "first_hit_turn": result["first_hit_turn"],
        "best_rank": result["best_rank"],
        "reciprocal_rank": result["reciprocal_rank"],
        "pivot_turn": pivot_turn,
        "turn_count": len(turns),
        "errors": sum(1 for turn in turns if turn["error"]),
        "discarded_turns": sum(1 for turn in turns if turn["discarded"]),
        "turns": turns,
    }


def analyze(records: list[dict], samples: list[dict], result: dict, catalog_ids: set[str]) -> list[dict]:
    if not (len(records) == len(samples) == len(result["sessions"])):
        raise RuntimeError(
            f"recording/sample/result length mismatch: "
            f"{len(records)}/{len(samples)}/{len(result['sessions'])}"
        )
    sessions = [
        analyze_session(record, sample, session_result, catalog_ids)
        for record, sample, session_result in zip(records, samples, result["sessions"])
    ]
    for session, session_result in zip(sessions, result["sessions"]):
        if session["sample_id"] != session_result["sample_id"]:
            raise RuntimeError("recording and evaluator session order diverged")
    return sessions


def latency_summary(sessions: list[dict]) -> dict:
    latencies = sorted(turn["latency_ms"] for session in sessions for turn in session["turns"])
    if not latencies:
        return {"turn_count": 0}
    return {
        "turn_count": len(latencies),
        "p50_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 3),
        "max_ms": round(latencies[-1], 3),
    }


def health_summary(sessions: list[dict]) -> dict:
    """Failures the evaluator hides. A non-zero count here invalidates the score."""
    return {
        "agent_exceptions": sum(session["errors"] for session in sessions),
        "discarded_responses": sum(session["discarded_turns"] for session in sessions),
        "dropped_slots": sum(
            turn["dropped_slots"] for session in sessions for turn in session["turns"]
        ),
        "short_slates": sum(
            1
            for session in sessions
            for turn in session["turns"]
            if len(turn["slate"]) < 10
        ),
        "wasted_pre_pivot_hits": sum(
            1 for session in sessions for turn in session["turns"] if turn["wasted_hit"]
        ),
    }
