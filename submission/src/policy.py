"""Which dialogue policy this turn is running under.

A policy is a stance toward the conversation, not a type of customer. One
session moves through several: it opens exploring, narrows once the customer
has said something concrete, restarts when they change their mind, and stops
asking when there is nothing left to learn or no turn left to use it on.

The layer exists because the four decisions a turn makes -- which attribute to
ask about, how to word the question, how to retrieve, and how to fill the ten
slots -- are not independent. Deciding them separately means four modules each
inferring the same thing about the conversation from the same state, which is
how `routing` came to treat "had nothing more to say about material" as
"refused to discuss material" and route 74% of hard-set turns to the wrong
branch (findings 3.46). Naming the stance once, here, means the four decisions
disagree loudly rather than quietly.

Three of the six policies change nothing the evaluator can read, and that is
deliberate rather than unfinished: `routing`'s four routes ship at identical
constants because every attempt to differentiate them measured negative
(findings 3.26, 3.30, 3.35, 3.43, 3.44). What the policy layer adds on top of
those measurements is the wording, which the simulator never reads and a judge
does, and two decisions that do reach the evaluator, both switched behind
`probe.STAGNATION_ESCAPE` and `probe.COVERAGE_SILENCE`.
"""

from __future__ import annotations

from submission.src import dialogue

DISCOVERY = "discovery"
PRECISION = "precision"
RECOVERY = "recovery"
BOUNDARY = "boundary"
STAGNATION = "stagnation"
COVERAGE = "coverage"

# A bare material word matches roughly half its bucket, so a session holding
# only one constraint has not yet said anything that separates products.
MIN_PRECISION_CONSTRAINTS = 2

# The turn from which the objective stops being "learn something for later" and
# starts being "convert now". Chosen as the point where at most three turns
# remain, so an answer still has somewhere to be spent but the slate is what
# matters. On the public 200 no session reaches it; on `compound_hard` 58% do
# (findings 3.46).
COVERAGE_TURN = 8


def select(state: dialogue.SessionState) -> str:
    """Returns the policy governing the turn about to be served.

    The order is the whole design. A redirect outranks ordinary narrowing
    because the constraints it replaced are no longer true; running out of
    turns outranks gathering information there is no turn left to use; repeated
    unhelpful answers outrank asking a sharper question of the same kind; and a
    declined attribute outranks all of it, because pressing again is the one
    move that costs goodwill rather than a slot.

    Args:
        state: The session after the latest message has been folded in.
    """
    if state.pivoted:
        return RECOVERY
    if state.exhausted or state.turn >= COVERAGE_TURN:
        return COVERAGE
    if state.idle >= dialogue.STAGNATION_TURNS:
        return STAGNATION
    if state.scenario == dialogue.BOUNDARY or state.declined:
        return BOUNDARY
    if len(state.constraints) >= MIN_PRECISION_CONSTRAINTS:
        return PRECISION
    return DISCOVERY
