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

from techjam.submission.src import dialogue
from techjam.submission.src import ranking

PRECISION = "precision"
DISCOVERY = "discovery"
RECOVERY = "recovery"
BOUNDARY = "boundary"
STAGNATION = "stagnation"
COVERAGE = "coverage"

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

# Policy scores are deliberately small, additive signals rather than calibrated
# probabilities. They answer "which controller should speak next?" from the
# evidence the session already exposes: constraints, refusals, pivots, turn
# pressure and the previous retrieval's uncertainty.
LARGE_POOL = 250
SMALL_POOL = 50
STAGNATION_TURN = 5
COVERAGE_TURN = 8
UNCERTAIN_CONTENDERS = 5

_HARD_ATTRIBUTES = frozenset((
    "material", "color", "size", "style", "use_case", "budget", "brand",
    "category",
))

_POLICY_ORDER = (
    RECOVERY, BOUNDARY, COVERAGE, PRECISION, STAGNATION, DISCOVERY,
)


@dataclass(frozen=True)
class PolicyDecision:
    """The policy scorer's turn-level decision."""

    name: str
    scores: tuple[tuple[str, float], ...]
    confidence: float
    margin: float


@dataclass(frozen=True)
class Route:
    """How this turn should be retrieved."""

    name: str
    alpha: float
    defer_turns: int
    dense_weight: float | None = None
    reach: int = 0
    diversity: float | None = None
    policy_scores: tuple[tuple[str, float], ...] = ()
    policy_confidence: float = 0.0
    policy_margin: float = 0.0


def choose(
    state: dialogue.SessionState,
    candidate_count: int = 0,
    previous_contenders: int = 0,
) -> Route:
    """Returns the retrieval policy for the turn about to be served.

    Args:
        state: The session after the latest message has been folded in.
        candidate_count: Size of the resolved catalog pool, when known.
        previous_contenders: Previous turn's near-tie count, when known.
    """
    decision = decide(state, candidate_count, previous_contenders)
    if decision.name == RECOVERY:
        return _recovery(state, decision)
    if decision.name == BOUNDARY:
        return _route(
            BOUNDARY, ranking.ALPHA, ranking.MAX_DEFER_TURNS, decision
        )
    if decision.name == COVERAGE:
        return _route(COVERAGE, ranking.ALPHA, 0, decision)
    if decision.name == PRECISION:
        return _route(
            PRECISION, PRECISION_ALPHA, ranking.MAX_DEFER_TURNS, decision,
            dense_weight=PRECISION_DENSE,
        )
    if decision.name == STAGNATION:
        return _route(
            STAGNATION, DISCOVERY_ALPHA, ranking.MAX_DEFER_TURNS, decision,
            dense_weight=DISCOVERY_DENSE, reach=DISCOVERY_REACH,
            diversity=DISCOVERY_DIVERSITY,
        )
    return _route(
        DISCOVERY, DISCOVERY_ALPHA, ranking.MAX_DEFER_TURNS, decision,
        dense_weight=DISCOVERY_DENSE, reach=DISCOVERY_REACH,
        diversity=DISCOVERY_DIVERSITY,
    )


def decide(
    state: dialogue.SessionState,
    candidate_count: int = 0,
    previous_contenders: int = 0,
) -> PolicyDecision:
    """Scores every dialogue policy and returns the current winner.

    This is the dynamic context-programming layer in miniature: no policy is
    fixed for a session. The same anonymous user can begin in discovery, move
    into precision, hit a boundary refusal, and later recover after a pivot.
    """
    raw = {
        RECOVERY: _recovery_score(state),
        BOUNDARY: _boundary_score(state),
        COVERAGE: _coverage_score(state),
        PRECISION: _precision_score(state, candidate_count),
        STAGNATION: _stagnation_score(state, previous_contenders),
        DISCOVERY: _discovery_score(state, candidate_count,
                                    previous_contenders),
    }
    scores = tuple(
        (name, round(max(0.0, raw[name]), 3)) for name in _POLICY_ORDER
    )
    ranked_scores = sorted(
        scores, key=lambda item: (item[1], -_POLICY_ORDER.index(item[0])),
        reverse=True,
    )
    top_name, top_score = ranked_scores[0]
    second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
    total = sum(score for _, score in scores)
    confidence = top_score / total if total > 0.0 else 0.0
    return PolicyDecision(
        top_name,
        scores,
        round(confidence, 3),
        round(top_score - second_score, 3),
    )


def _recovery_score(state: dialogue.SessionState) -> float:
    score = 0.0
    if state.pivoted or state.scenario == dialogue.OVERRIDE:
        score += 1.6
    if state.pivot_turn == state.turn and state.turn:
        score += 0.4
    if state.superseded:
        score += 0.3
    return score


def _boundary_score(state: dialogue.SessionState) -> float:
    score = 0.05
    if state.scenario == dialogue.BOUNDARY:
        score += 1.25
    if state.refused:
        score += 1.25
    if state.refused and state.turn >= 5:
        score += 0.2
    if state.pivoted:
        score -= 0.4
    return score


def _coverage_score(state: dialogue.SessionState) -> float:
    score = 0.0
    if state.exhausted:
        score += 1.7
    if state.turn >= COVERAGE_TURN:
        score += 1.0 + min(0.3, _constraint_count(state) * 0.1)
    if len(state.shown) >= ranking.SLATE_SIZE * 3:
        score += 0.2
    if state.pivot_turn == state.turn and state.turn:
        score -= 0.8
    return score


def _precision_score(
    state: dialogue.SessionState, candidate_count: int = 0
) -> float:
    constraints = _constraint_count(state)
    hard = _hard_attribute_count(state)
    score = 0.2
    score += min(0.9, constraints * 0.45)
    score += min(0.4, hard * 0.2)
    if state.scenario == dialogue.BUYING:
        score += 0.25
    if constraints and state.turn >= 2:
        score += 0.15
    if 0 < candidate_count <= SMALL_POOL:
        score += 0.25
    if not constraints:
        score -= 0.5
    if state.refused:
        score -= 0.5
    if state.pivoted:
        score -= 0.8
    if state.exhausted:
        score -= 0.4
    return score


def _stagnation_score(
    state: dialogue.SessionState, previous_contenders: int = 0
) -> float:
    constraints = _constraint_count(state)
    score = 0.0
    if state.turn >= STAGNATION_TURN and constraints <= 1:
        score += 1.1
    if state.turn >= STAGNATION_TURN + 1 and constraints <= 1:
        score += 0.35
    if len(state.shown) >= ranking.SLATE_SIZE * 2 and constraints <= 1:
        score += 0.35
    if previous_contenders > UNCERTAIN_CONTENDERS:
        score += 0.25
    if constraints >= MIN_PRECISION_CONSTRAINTS:
        score -= 0.8
    if state.refused or state.pivoted or state.exhausted:
        score -= 0.9
    return score


def _discovery_score(
    state: dialogue.SessionState,
    candidate_count: int = 0,
    previous_contenders: int = 0,
) -> float:
    constraints = _constraint_count(state)
    score = 0.4
    if constraints == 0:
        score += 0.55
    if constraints < MIN_PRECISION_CONSTRAINTS:
        score += 0.3
    if state.scenario in (dialogue.UNKNOWN, dialogue.EXPLORING):
        score += 0.25
    if state.turn <= 2:
        score += 0.1
    if candidate_count >= LARGE_POOL or not state.pool_keys:
        score += 0.2
    if previous_contenders > UNCERTAIN_CONTENDERS:
        score += 0.2
    if constraints >= MIN_PRECISION_CONSTRAINTS:
        score -= 0.45
    if state.refused:
        score -= 0.55
    if state.pivoted:
        score -= 0.8
    if state.exhausted:
        score -= 0.5
    return score


def _constraint_count(state: dialogue.SessionState) -> int:
    if state.slots:
        return sum(1 for slot in state.slots if not slot.negated)
    return len(state.constraints)


def _hard_attribute_count(state: dialogue.SessionState) -> int:
    if not state.slots:
        return min(_constraint_count(state), MIN_PRECISION_CONSTRAINTS)
    return len({
        slot.attribute for slot in state.slots
        if not slot.negated and slot.attribute in _HARD_ATTRIBUTES
    })


def _route(
    name: str,
    alpha: float,
    defer_turns: int,
    decision: PolicyDecision,
    dense_weight: float | None = None,
    reach: int = 0,
    diversity: float | None = None,
) -> Route:
    return Route(
        name,
        alpha,
        defer_turns,
        dense_weight,
        reach,
        diversity,
        decision.scores,
        decision.confidence,
        decision.margin,
    )


def _recovery(
    state: dialogue.SessionState, decision: PolicyDecision | None = None
) -> Route:
    """Returns the policy for a session that has just been redirected."""
    if decision is None:
        decision = decide(state)
    return _route(
        RECOVERY,
        ranking.ALPHA,
        ranking.MAX_DEFER_TURNS + state.pivot_turn * RECOVERY_RESTART,
        decision,
    )
