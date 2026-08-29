"""Saying what the agent just did, in the customer's own terms.

The reply is composed from the session state rather than drawn from a rotation,
so it names the category it understood, the requirements it is matching on, what
a redirect replaced, and how much the slate has narrowed. It costs no tokens and
no network: this is template composition over live state, not generation.

Two reasons it is worth the module. The contract calls this field customer-
facing natural language and lists transparent recommendation explanations as a
goal, and a per-item explanation has nowhere to live -- the response schema
allows a recommendation only `parent_asin` and `score` -- so the explanation
belongs here. And a reply that states what was understood is the only place a
customer can catch the agent misunderstanding them.
"""

from __future__ import annotations

from techjam.submission.src import dialogue

# Reading a long constraint back verbatim is worse than not reading it back.
MAX_QUOTED = 42
MAX_LISTED = 2

# Above this many products still in contention, the slate is not yet a
# recommendation and the reply should say so rather than imply confidence.
CROWDED = 5

FALLBACK = "Here are some options. Anything specific you need?"


def compose(
    state: dialogue.SessionState,
    parsed: dialogue.ParsedTurn,
    contenders: int,
    head: int,
    served: int,
    asked: str | None,
) -> str:
    """Returns the customer-facing reply for the turn just served.

    Args:
        state: The session after this turn's message was folded in.
        parsed: What this turn's message yielded, for what changed just now.
        contenders: How many products are still scoring near the leader.
        head: How many of the recommendations the ranking is committing to.
        served: How many recommendations the slate carries.
        asked: The attribute being probed, or None if nothing is.
    """
    parts = [
        _acknowledge(state, parsed),
        _slate(contenders, head, served),
        _question(state, asked),
    ]
    reply = " ".join(part for part in parts if part)
    return reply or FALLBACK


def _acknowledge(
    state: dialogue.SessionState, parsed: dialogue.ParsedTurn
) -> str:
    """Returns what the agent understood from this turn."""
    if parsed.pivot:
        replacing = _listed(parsed.constraints)
        dropped = _listed(_distinct(state.superseded, parsed.constraints))
        if replacing and dropped:
            return f"Understood, {replacing} instead of {dropped}."
        if replacing:
            return f"Understood, {replacing} it is."
        return "Understood, starting over on that."

    if parsed.boundary_refusal:
        return "No problem, I will use my judgement there."

    if parsed.exhausted:
        return "Thanks, I think I have what I need."

    if parsed.constraints:
        return f"Got it: {_listed(parsed.constraints)}."

    if state.turn <= 1 and state.category:
        return f"Happy to help you find {state.category.lower()}."
    return ""


def _slate(contenders: int, head: int, served: int) -> str:
    """Returns what the recommendations on this turn are, and are not.

    The distinction is real, not decoration: a committed slate is the ranking's
    ten best, while a held-back one is one pick plus a spread it has not
    committed to. Telling the customer which they are looking at is what makes
    the second honest rather than merely odd.
    """
    if not served:
        return ""
    if head >= served:
        return f"Here are the {served} closest matches, best first."
    if contenders > CROWDED:
        return (
            f"My best match is first; the rest are a spread of "
            f"{served - head} others while we narrow down."
        )
    return f"Here is my best match, with {served - head} more to compare."


def _question(state: dialogue.SessionState, asked: str | None) -> str:
    """Returns the clarifying question, or a closing line when none helps."""
    if asked is None:
        return "Let me know if none of these are right."
    if state.constraints:
        return "Is there another detail I should match on?"
    return "Is there anything specific you need from it?"


def _distinct(
    dropped: tuple[str, ...], replacing: tuple[str, ...]
) -> tuple[str, ...]:
    """Returns what a redirect really discarded.

    A replacement often restates something already said, so the raw superseded
    list would have the reply announce it is dropping the very thing it is
    adopting. Only what the new requirement does not already contain is news.
    """
    incoming = [value.casefold() for value in replacing]
    return tuple(
        value for value in dropped
        if not any(_overlapping(value.casefold(), new) for new in incoming)
    )


def _overlapping(left: str, right: str) -> bool:
    """Returns whether two constraints are saying the same thing."""
    return left in right or right in left


def _listed(values: tuple[str, ...]) -> str:
    """Returns a short readable list of constraint strings."""
    quoted = [_short(value) for value in values[:MAX_LISTED]]
    quoted = [value for value in quoted if value]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return quoted[0]
    return " and ".join(quoted)


def _short(value: str) -> str:
    """Returns a constraint trimmed to something worth reading aloud."""
    cleaned = " ".join(value.split())
    if len(cleaned) <= MAX_QUOTED:
        return cleaned.lower()
    cut = cleaned[:MAX_QUOTED].rsplit(" ", 1)[0]
    return f"{cut.lower()}..." if cut else ""
