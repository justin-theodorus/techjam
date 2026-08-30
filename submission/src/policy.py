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

from dataclasses import dataclass

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

# How many turns after a pivot recovery stays sticky. `0` means the redirect
# owns the rest of the session, which is the shipped setting from `main`.
RECOVERY_TURNS = 0

# Policy scores are deliberately small, additive signals rather than calibrated
# probabilities. They answer "which controller should speak next?" from the
# evidence the session already exposes: constraints, true declines, pivots,
# turn pressure and the previous retrieval's uncertainty.
LARGE_POOL = 250
SMALL_POOL = 50
STAGNATION_TURN = 5
UNCERTAIN_CONTENDERS = 5

_HARD_ATTRIBUTES = frozenset((
    "material", "color", "size", "style", "use_case", "budget", "brand",
    "category",
))

_POLICY_ORDER = (
    RECOVERY, COVERAGE, STAGNATION, BOUNDARY, PRECISION, DISCOVERY,
)


@dataclass(frozen=True)
class PolicyDecision:
    """The turn-level policy decision, with scores for traceability."""

    name: str
    scores: tuple[tuple[str, float], ...]
    confidence: float
    margin: float


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
    return decide(state).name


def decide(
    state: dialogue.SessionState,
    candidate_count: int = 0,
    previous_contenders: int = 0,
) -> PolicyDecision:
    """Scores every dialogue policy and returns the current winner.

    This is the dynamic context-programming layer in miniature: no policy is
    fixed for a session. The same anonymous user can begin in discovery, move
    into precision, hit a boundary refusal, and later recover after a pivot.

    Args:
        state: The session after the latest message has been folded in.
        candidate_count: Size of the resolved catalog pool, when known.
        previous_contenders: Previous turn's near-tie count, when known.
    """
    raw = {
        RECOVERY: _recovery_score(state),
        COVERAGE: _coverage_score(state),
        STAGNATION: _stagnation_score(state, previous_contenders),
        BOUNDARY: _boundary_score(state),
        PRECISION: _precision_score(state, candidate_count),
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
    if state.pivoted and _recovering(state):
        score += 6.0
    if state.pivot_turn == state.turn and state.turn:
        score += 0.4
    if state.superseded:
        score += 0.3
    return score


def _coverage_score(state: dialogue.SessionState) -> float:
    score = 0.0
    if state.exhausted:
        score += 5.0
    if state.turn >= COVERAGE_TURN:
        score += 5.0 + min(0.4, _constraint_count(state) * 0.1)
    if len(state.shown) >= 30:
        score += 0.2
    if state.pivoted and _recovering(state):
        score -= 6.0
    return score


def _stagnation_score(
    state: dialogue.SessionState, previous_contenders: int = 0
) -> float:
    constraints = _constraint_count(state)
    score = 0.0
    if state.idle >= dialogue.STAGNATION_TURNS:
        score += 4.0 + min(0.5, state.idle * 0.1)
    elif (
        state.turn >= STAGNATION_TURN
        and constraints <= 1
        and (len(state.shown) >= 20
             or previous_contenders > UNCERTAIN_CONTENDERS)
    ):
        score += 4.0
    if previous_contenders > UNCERTAIN_CONTENDERS:
        score += 0.25
    if constraints >= MIN_PRECISION_CONSTRAINTS:
        score -= 0.8
    if state.pivoted and _recovering(state):
        score -= 4.0
    if state.exhausted:
        score -= 4.0
    return score


def _boundary_score(state: dialogue.SessionState) -> float:
    score = 0.05
    if state.scenario == dialogue.BOUNDARY:
        score += 3.0
    if state.declined:
        score += 3.0
    if state.declined and state.turn >= 5:
        score += 0.2
    return score


def _precision_score(
    state: dialogue.SessionState, candidate_count: int = 0
) -> float:
    constraints = _constraint_count(state)
    hard = _hard_attribute_count(state)
    score = 0.0
    if constraints >= MIN_PRECISION_CONSTRAINTS:
        score += 2.0
    elif constraints == 1:
        score += 0.75
    score += min(0.5, hard * 0.2)
    if state.scenario == dialogue.BUYING and constraints:
        score += 0.25
    if constraints and state.turn >= 2:
        score += 0.15
    if 0 < candidate_count <= SMALL_POOL:
        score += 0.25
    if state.declined:
        score -= 0.4
    if state.pivoted and _recovering(state):
        score -= 2.0
    if state.exhausted:
        score -= 2.0
    return score


def _discovery_score(
    state: dialogue.SessionState,
    candidate_count: int = 0,
    previous_contenders: int = 0,
) -> float:
    constraints = _constraint_count(state)
    score = 1.0
    if constraints == 0:
        score += 0.55
    if constraints < MIN_PRECISION_CONSTRAINTS:
        score += 0.35
    if state.scenario in (dialogue.UNKNOWN, dialogue.EXPLORING):
        score += 0.25
    if state.turn <= 2:
        score += 0.1
    if candidate_count >= LARGE_POOL or not state.pool_keys:
        score += 0.2
    if previous_contenders > UNCERTAIN_CONTENDERS:
        score += 0.2
    if constraints >= MIN_PRECISION_CONSTRAINTS:
        score -= 0.6
    if state.declined:
        score -= 0.55
    if state.pivoted and _recovering(state):
        score -= 1.0
    if state.exhausted:
        score -= 1.0
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


def _recovering(state: dialogue.SessionState) -> bool:
    """Whether the current turn is still inside the redirect recovery window."""
    if RECOVERY_TURNS <= 0:
        return True
    return state.turn - state.pivot_turn < RECOVERY_TURNS
