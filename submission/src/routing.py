"""Choosing how to retrieve, from what the conversation looks like so far.

Four routes over two retrievers. A route sets how much the ranking trusts the
customer's words against the popularity prior, how long the slate stays narrow
before it opens, and -- since Phase 6U -- which retrievers run at all: the
lexical index over the resolved category buckets, the dense latent space, or
both, with the discovery route able to reach past the bucket entirely.

That last part is what makes routing a real choice rather than a second set of
constants. While both routes pointed at one retriever, any route-conditional
setting was `alpha` re-tuned to the public target distribution wearing a route
as a disguise, which is exactly what findings 3.30 caught and rejected.

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

# How much the dense track weighs on each route, and how far past the category
# bucket the discovery route may reach for candidates. `None` means "whatever
# `ranking.DENSE_WEIGHT` says", which is the neutral setting: no route
# specialises, so routing still costs nothing until it earns something.
#
# The brief assigns dense retrieval to Browsing, and measurement puts its value
# somewhere else entirely. Both halves ship at neutral and the table is in
# findings 3.35, because what it found is more useful than what it was looking
# for.
PRECISION_DENSE: float | None = None
DISCOVERY_DENSE: float | None = None
DISCOVERY_REACH = 0

# How much the discovery route varies its slate instead of ranking it. `None`
# defers to `ranking.DIVERSITY`, which is zero, so no route specialises.
#
# This is the brief's own dual-track claim taken literally: hold the precision
# route on deferred commitment and let the exploring one spread out. It is the
# only route-conditional setting whose argument does not reduce to `alpha` in a
# disguise, because it changes how the slate is *selected* rather than how the
# pool is scored.
#
# Measured negative and shipped neutral. Every session takes this route on turn
# 1, buying included, because buying opens with one constraint and the precision
# threshold is two -- so gating on the route is a turn-1 gate rather than a
# browsing gate, and `ranking.DIVERSITY_MAX_CONSTRAINTS` is the sharper
# instrument for the same idea (findings 3.43).
DISCOVERY_DIVERSITY: float | None = None


@dataclass(frozen=True)
class Route:
    """How this turn should be retrieved."""

    name: str
    alpha: float
    defer_turns: int
    dense_weight: float | None = None
    reach: int = 0
    diversity: float | None = None


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
        return Route(
            PRECISION, PRECISION_ALPHA, ranking.MAX_DEFER_TURNS,
            PRECISION_DENSE,
        )
    return Route(
        DISCOVERY, DISCOVERY_ALPHA, ranking.MAX_DEFER_TURNS,
        DISCOVERY_DENSE, DISCOVERY_REACH, diversity=DISCOVERY_DIVERSITY,
    )


def _recovery(state: dialogue.SessionState) -> Route:
    """Returns the policy for a session that has just been redirected."""
    return Route(
        RECOVERY,
        ranking.ALPHA,
        ranking.MAX_DEFER_TURNS + state.pivot_turn * RECOVERY_RESTART,
    )
