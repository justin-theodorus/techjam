"""Choosing how to retrieve, from what the conversation looks like so far.

Four routes over one pipeline. The route does not change *what* is retrieved --
that is always the resolved category buckets scored by the blend -- it changes
how much the ranking trusts the customer's words against the popularity prior,
and how long the slate stays narrow before it opens.

Every route ships at the same constants until a measurement moves it, so
routing costs nothing until it earns something. A route whose specialisation
does not pay stays neutral and is reported as a negative result rather than
quietly deleted.
"""

from __future__ import annotations

from dataclasses import dataclass

from submission.src import dialogue
from submission.src import ranking

PRECISION = "precision"
DISCOVERY = "discovery"
RECOVERY = "recovery"
BOUNDARY = "boundary"

# A bare material word matches roughly half its bucket, so a session holding
# only one constraint has not yet said anything that separates products.
MIN_PRECISION_CONSTRAINTS = 2

# How much weight the popularity prior carries on each route. The argument for
# splitting it is good: a session that has said little has said nothing worth
# displacing the prior with, and at low `alpha` one weak constraint costs buying
# 0.40 of turn-1 hit@10 (findings 3.16).
#
# It does not survive contact with a held-out split. The dev optimum sits at
# 0.4/1.3 and is worth +0.014 there; on the held-out half the same setting is
# 0.002 *worse* than a single shared `alpha`, and the dev surface is non-
# monotone, which is the shape of noise rather than signal. Both routes
# therefore ship at `ranking.ALPHA` and this is reported as a negative result
# (findings 3.26).
DISCOVERY_ALPHA = ranking.ALPHA
PRECISION_ALPHA = ranking.ALPHA

# Whether a redirect restarts the turn budget for narrowing the slate. The
# argument for it is that a replacement resets what the customer has told us.
# Measured at zero once a redirect stopped erasing the constraints it does not
# contradict: the surviving evidence is what converts, and re-narrowing only
# delays it (findings 3.26).
RECOVERY_RESTART = 0


@dataclass(frozen=True)
class Route:
    """How this turn should be retrieved."""

    name: str
    alpha: float
    defer_turns: int


def choose(state: dialogue.SessionState) -> Route:
    """Returns the retrieval policy for the turn about to be served.

    Args:
        state: The session after the latest message has been folded in.
    """
    if state.pivoted:
        return _recovery(state)
    if state.scenario == dialogue.BOUNDARY or state.refused:
        return Route(BOUNDARY, ranking.ALPHA, ranking.MAX_DEFER_TURNS)
    if len(state.constraints) >= MIN_PRECISION_CONSTRAINTS:
        return Route(PRECISION, PRECISION_ALPHA, ranking.MAX_DEFER_TURNS)
    return Route(DISCOVERY, DISCOVERY_ALPHA, ranking.MAX_DEFER_TURNS)


def _recovery(state: dialogue.SessionState) -> Route:
    """Returns the policy for a session that has just been redirected."""
    return Route(
        RECOVERY,
        ranking.ALPHA,
        ranking.MAX_DEFER_TURNS + state.pivot_turn * RECOVERY_RESTART,
    )
