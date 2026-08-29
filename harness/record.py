"""Transparent recording proxy around the agent under evaluation.

Wrapping the agent rather than re-implementing `evaluate()`'s dialogue loop is
the point: score and trace then come from the same run and cannot disagree.

The evaluator swallows agent exceptions into an empty turn, so a crash is
invisible in the score. The proxy records the traceback before re-raising, which
leaves evaluator behaviour identical but makes the failure visible in the trace.
"""

from __future__ import annotations

import traceback
from time import perf_counter

MESSAGE_PREVIEW_CHARS = 160


def _asin_of(item: object) -> str:
    value = item.get("parent_asin", "") if isinstance(item, dict) else item
    return str(value).strip()


def _raw_asins(response: object) -> list[str]:
    if not isinstance(response, dict):
        return []
    payload = response.get("recommendations")
    if not isinstance(payload, list):
        return []
    return [_asin_of(item) for item in payload]


def _is_discarded(response: object) -> bool:
    """Mirror of evaluator lines 243-244: the whole response is dropped."""
    return not isinstance(response, dict) or not isinstance(response.get("message"), str)


def _reported_usage(response: object) -> dict:
    """The turn's declared token usage, read exactly as the evaluator reads it.

    Mirror of evaluator lines 245-250: only non-negative `int` counts are
    accepted, so what the trace totals is what the organizer would total, and
    a count typed in the wrong shape shows here as a zero rather than hiding.
    """
    counts = {"prompt_tokens": 0, "completion_tokens": 0}
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return counts
    for field in counts:
        value = usage.get(field)
        ok = isinstance(value, int) and not isinstance(value, bool)
        if ok and value >= 0:
            counts[field] = value
    return counts


def _debug_snapshot(agent: object) -> dict | None:
    """Optional agent-side diagnostics.

    An agent may expose a plain `debug` dict describing the turn it just served
    (extracted slots, pool size after filtering, why an attribute was chosen).
    Absent or malformed, the trace simply carries no agent internals.
    """
    debug = getattr(agent, "debug", None)
    return dict(debug) if isinstance(debug, dict) else None


class RecordingAgent:
    def __init__(self, agent: object) -> None:
        self.agent = agent
        self.sessions: list[dict] = []

    @property
    def _current_turns(self) -> list[dict]:
        if not self.sessions:
            raise RuntimeError("respond() was called before reset()")
        return self.sessions[-1]["turns"]

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions.append({
            "session_id": session_id,
            "user_profile": user_profile,
            "turns": [],
        })
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> object:
        started = perf_counter()
        try:
            response = self.agent.respond(session_id, user_message, turn, top_k)
        except Exception:
            self._record(turn, user_message, None, traceback.format_exc(), started)
            raise
        self._record(turn, user_message, response, None, started)
        return response

    def _record(
        self,
        turn: int,
        user_message: str,
        response: object,
        error: str | None,
        started: float,
    ) -> None:
        message = response.get("message") if isinstance(response, dict) else None
        self._current_turns.append({
            "turn": turn,
            "user_message": user_message,
            "message": message[:MESSAGE_PREVIEW_CHARS] if isinstance(message, str) else None,
            "ask_attribute": response.get("ask_attribute") if isinstance(response, dict) else None,
            "raw_recommendations": _raw_asins(response),
            "discarded": _is_discarded(response),
            "error": error,
            "latency_ms": round((perf_counter() - started) * 1000.0, 3),
            "usage": _reported_usage(response),
            "debug": _debug_snapshot(self.agent),
        })
