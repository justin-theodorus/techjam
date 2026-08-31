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
from submission.src import policy as policy_module
from submission.src import ranking

# The route names are the policy names. Retrieval is one of the four decisions
# a policy makes, so it does not get a second vocabulary for the same states.
PRECISION = policy_module.PRECISION
DISCOVERY = policy_module.DISCOVERY
RECOVERY = policy_module.RECOVERY
BOUNDARY = policy_module.BOUNDARY
STAGNATION = policy_module.STAGNATION
COVERAGE = policy_module.COVERAGE

MIN_PRECISION_CONSTRAINTS = policy_module.MIN_PRECISION_CONSTRAINTS

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

# How long each route withholds the slots below its committed head, as a
# deviation from `ranking.MAX_DEFER_TURNS`. `None` means "whatever the module
# says", which is the neutral setting every earlier number was taken against.
#
# This is the one place the policy layer is allowed to reach retrieval, and the
# argument for it is the brief's rather than the sweep's. Deferred commitment
# holds a slate back because showing a product is irreversible; that is a
# *buying* premise. A customer who is still browsing is not asking to be
# converted, and answering "show me options" with one product is the wrong
# reply whatever it does to the reciprocal rank. So discovery opens sooner than
# precision, and the pair is ordered rather than tuned freely: a configuration
# with `DISCOVERY_DEFER >= PRECISION_DEFER` would be narrowing hardest on the
# customer who has told us least, which is the behaviour this exists to rule
# out.
DISCOVERY_DEFER: int | None = 3
PRECISION_DEFER: int | None = 6

# How many products each route commits to while it is still narrowing, as a
# deviation from `ranking.HEAD_SIZE`. `None` defers to the module.
#
# The same argument as the deferral pair, applied where it can actually reach.
# Discovery is a turn-1 route -- 215 of its 219 turns land on turns 1 and 2 and
# none on turn 4 or later -- so a *threshold* that opens the slate later cannot
# separate it from precision, and only the width can.
#
# **Ships off, and this is a negative result rather than an unexplored idea.**
# The premise is sound about shoppers and wrong about this scorer: a browsing
# session in the public set still has exactly one ground-truth target and still
# ends the moment it appears, so breadth buys no credit for the browsing it
# serves and costs the rank the target lands on. Monotonically negative --
# -0.0101 at 2, -0.0230 at 3, -0.0356 at 5, -0.0508 at 10 -- and the loss is
# concentrated exactly where the argument said the gain would be, with browsing
# MRR falling 0.9704 to 0.8942 at a head of two.
#
# Kept live and swept because a scorer that rewarded exploration, or a judge
# reading transcripts rather than reciprocal ranks, would want it back, and
# because the number is more useful recorded than rediscovered.
DISCOVERY_HEAD: int | None = None
PRECISION_HEAD: int | None = None

# Every route-conditional sweep before Phase 6W was taken while the precision
# branch was unreachable on hard sets: a scoped exhaustion put the spent arm
# into `state.refused`, and this module read that as a refusal, so 74% of
# `compound_hard` turns took the boundary branch and precision took 1.6%
# (findings 3.46). The branch now reads `state.declined`, which holds only
# arms the customer actually declined. The verdicts above stand -- all four
# routes ship at identical constants, so no score depended on which one was
# named -- but any *re-measurement* of a route-conditional setting is taken
# over a different population than 3.26, 3.30 and 3.35 were.

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
    head_cap: int | None = None
    dense_weight: float | None = None
    reach: int = 0
    diversity: float | None = None
    policy_scores: tuple[tuple[str, float], ...] = ()
    policy_confidence: float = 0.0
    policy_margin: float = 0.0
    decision_readiness: float = 0.0
    policy_runner_up: str = ""
    policy_hybrid: bool = False


def choose(
    state: dialogue.SessionState,
    policy: str | None = None,
    *,
    decision: policy_module.PolicyDecision | None = None,
    candidate_count: int = 0,
    previous_contenders: int = 0,
) -> Route:
    """Returns the retrieval policy for the turn about to be served.

    The branch is `policy.select`'s, not a second copy of it. Two of the six
    policies name no retrieval of their own -- stagnation and coverage change
    what is asked and how it is worded, not where candidates come from -- so
    they fall through to the shared constants, which is what every other route
    is running at anyway (findings 3.44).

    Args:
        state: The session after the latest message has been folded in.
        policy: The policy already selected for this turn, if the caller has
          one. Absent, it is selected here, so the module stays usable on its
          own.
        decision: The scored policy decision, if the caller already computed
          it for this turn.
        candidate_count: Size of the resolved catalog pool, when known.
        previous_contenders: Previous turn's near-tie count, when known.
    """
    decision = decision or policy_module.decide(
        state, candidate_count, previous_contenders
    )
    name = policy or decision.name
    if name == RECOVERY:
        return _recovery(state, decision)
    if name == BOUNDARY:
        return _route(BOUNDARY, ranking.ALPHA, ranking.MAX_DEFER_TURNS,
                      decision)
    if name == PRECISION:
        return _route(
            PRECISION, PRECISION_ALPHA, _defer(PRECISION_DEFER),
            decision, dense_weight=PRECISION_DENSE,
            head_cap=PRECISION_HEAD,
        )
    if name in (STAGNATION, COVERAGE):
        return _route(name, ranking.ALPHA, ranking.MAX_DEFER_TURNS, decision)
    return _route(
        DISCOVERY, DISCOVERY_ALPHA, _defer(DISCOVERY_DEFER),
        decision, dense_weight=DISCOVERY_DENSE, reach=DISCOVERY_REACH,
        diversity=DISCOVERY_DIVERSITY, head_cap=DISCOVERY_HEAD,
    )


def _defer(value: int | None) -> int:
    """Returns this route's deferral window, or the module's when it has none.

    Read at call time rather than bound as a default, for the same reason
    `dense_weight` is: a constant bound at import cannot be patched by a sweep
    (findings 3.27).
    """
    return ranking.MAX_DEFER_TURNS if value is None else value


def _route(
    name: str,
    alpha: float,
    defer_turns: int,
    decision: policy_module.PolicyDecision,
    dense_weight: float | None = None,
    reach: int = 0,
    diversity: float | None = None,
    head_cap: int | None = None,
) -> Route:
    return Route(
        name,
        alpha,
        defer_turns,
        head_cap,
        dense_weight,
        reach,
        diversity,
        decision.scores,
        decision.confidence,
        decision.margin,
        decision.readiness,
        decision.runner_up,
        decision.hybrid,
    )


def _recovery(
    state: dialogue.SessionState,
    decision: policy_module.PolicyDecision,
) -> Route:
    """Returns the policy for a session that has just been redirected."""
    return _route(
        RECOVERY,
        ranking.ALPHA,
        ranking.MAX_DEFER_TURNS + state.pivot_turn * RECOVERY_RESTART,
        decision,
    )
