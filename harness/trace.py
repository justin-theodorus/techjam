"""Per-turn trace reader.

    python3 -m harness.trace --misses --limit 5
    python3 -m harness.trace --sample-id public_0042 --full
"""

from __future__ import annotations

import argparse

from techjam.harness.diff import load

DEFAULT_ARTIFACT = "runs/latest.json"
MESSAGE_WIDTH = 96


def _clip(text: str, width: int = MESSAGE_WIDTH) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _debug(turn: dict) -> str:
    if not turn.get("debug"):
        return ""
    pairs = " ".join(f"{key}={_clip(value, 40)}" for key, value in turn["debug"].items())
    return f"  [{pairs}]"


def render_session(session: dict, full: bool = False) -> str:
    verdict = (
        f"HIT turn {session['first_hit_turn']} rank {session['best_rank']}"
        if session["hit"] else "MISS"
    )
    lines = [
        f"{session['sample_id']}  {session['scenario_type']}  "
        f"target {session['target']}  {verdict}"
    ]
    for turn in session["turns"]:
        marks = []
        if turn["error"]:
            marks.append("EXCEPTION")
        if turn["discarded"]:
            marks.append("DISCARDED")
        if turn["boundary_refusal"]:
            marks.append("refusal")
        if turn["wasted_hit"]:
            marks.append("pre-pivot hit wasted")
        if turn["dropped_slots"]:
            marks.append(f"{turn['dropped_slots']} slots dropped")
        rank = f"@{turn['target_rank']}" if turn["target_rank"] else "absent"
        lines.append(f"  T{turn['turn']} < {_clip(turn['user_message'])}")
        lines.append(
            f"       > ask={turn['ask_attribute']} slate={len(turn['slate'])} target={rank}"
            f"{' ' + ' '.join(marks) if marks else ''}{_debug(turn)}"
        )
        if full:
            lines.append(f"         msg: {_clip(turn['message'] or '')}")
            lines.append(f"         slate: {', '.join(turn['slate'])}")
            if turn["error"]:
                lines.append("         " + turn["error"].replace("\n", "\n         "))
    return "\n".join(lines)


def select(sessions: list[dict], args) -> list[dict]:
    chosen = sessions
    if args.sample_id:
        chosen = [item for item in chosen if item["sample_id"] in set(args.sample_id)]
    if args.scenario:
        chosen = [item for item in chosen if item["scenario_type"] == args.scenario]
    if args.misses:
        chosen = [item for item in chosen if not item["hit"]]
    if args.hits:
        chosen = [item for item in chosen if item["hit"]]
    if args.failures:
        chosen = [item for item in chosen if item["errors"] or item["discarded_turns"]]
    return chosen[: args.limit] if args.limit else chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Read per-turn traces from a run artifact")
    parser.add_argument("artifact", nargs="?", default=DEFAULT_ARTIFACT)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--scenario")
    parser.add_argument("--misses", action="store_true")
    parser.add_argument("--hits", action="store_true")
    parser.add_argument("--failures", action="store_true", help="only sessions that crashed or were discarded")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--full", action="store_true", help="include agent message and full slate")
    args = parser.parse_args()

    sessions = select(load(args.artifact)["sessions"], args)
    if not sessions:
        print("no sessions matched")
        return
    print("\n\n".join(render_session(session, args.full) for session in sessions))
    print(f"\n{len(sessions)} session(s) shown")


if __name__ == "__main__":
    main()
