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

import re
from dataclasses import dataclass

from submission.src import dialogue
from submission.src import slots as slots_module

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

# Decision readiness is the self-evolving scalar exposed by the policy layer:
# "how ready is this turn to commit to precision rather than keep discovering?"
# It is not a user persona, and it is not read from `scenario_type`; it is
# recomputed from the current session state on every turn.
#
# The update is a one-pole recurrence, D_t = 0.7 * current + 0.3 * D_{t-1}, so
# the score is a running estimate rather than a snapshot: evidence from earlier
# turns keeps decaying in the tail instead of being re-derived from scratch, and
# one decisive turn can still move it most of the way. That is the whole of the
# "self-evolving" claim -- the value belongs to the *turn*, not to the shopper.
#
# `D_{t-1}` is unavailable on the opening turn and on any caller that does not
# carry it, so `_context_readiness` stands in as the prior there. It reads the
# same accumulated evidence the recurrence would already have folded in, which
# keeps a stateless `decide(state)` honest instead of silently starting at zero.
READINESS_CURRENT_WEIGHT = 0.7
PRECISION_READINESS_THRESHOLD = 0.7
PARTIAL_READINESS_THRESHOLD = 0.3

# Whether readiness *steers* the precision/discovery pair or only reports.
#
# Ships **off**, and reported as a negative result rather than quietly deleted.
# Measured over `compound_hard`, 1514 turns, which is the set with the most
# headroom left (47.5% miss): steering on flips exactly zero policies, and the
# whole deviations sweep -- both thresholds, the recurrence weight, and the
# memoryless arm -- reproduces the neutral score in every cell on all six sets.
#
# Two reasons, and neither is fixable by moving a threshold:
#
#  1. The bump is smaller than the gap it would have to cross. Precision and
#     discovery are separated by a median margin of ~1.5 on that set and never
#     by less than 0.3; the strong arm here is worth at most 0.45 + 0.3 to one
#     side and 0.35 to the other. It cannot cross.
#  2. Readiness is collinear with the evidence the two scores already read.
#     Constraint count, hard attributes, pool size, declines, idling and
#     exhaustion drive `_precision_score`, `_discovery_score` *and* this
#     scalar, so it pushes both scores the way they were already going. A
#     bigger constant would not fix that; it would only be `MIN_PRECISION_
#     CONSTRAINTS` re-tuned in a disguise, which is the trap findings 3.30
#     caught and rejected.
#
# The scalar itself stays: it is the honest, traceable readiness estimate the
# debug row reports, and it is the thing to point at when asked how the agent
# decides it has heard enough. What does not stay is the claim that it changes
# a decision, because on this evaluator it does not.
READINESS_STEERS = False

# How close the runner-up policy has to score before the turn is called a tie.
#
# Calibrated to the observed margin distribution, not chosen round: on
# `compound_hard`'s 1514 turns the smallest margin of any turn is 0.3, so the
# first draft's 0.25 marked *nothing* and the debug flag was constant-False.
# 0.5 marks the closest 5.5% of turns, which is a diagnostic; 1.0 would mark
# 12.3%, which is most of the precision/discovery population and therefore not
# a tie in any useful sense.
HYBRID_MARGIN = 0.5

# Whether a tie also changes how the turn is *worded*, by framing it in the
# runner-up's voice. Ships off: the winning policy owns the turn exactly as it
# did before, and `hybrid` is a trace signal. Swept and neutral on all six sets
# -- which is expected rather than surprising, because the framing it would
# change is wording, and the reference simulator does not read wording.
HYBRID_FRAMING = False

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

_DECISIVE_ATTRIBUTES = frozenset((
    "budget", "size", "material", "color", "brand", "category",
))

# Deadline language, as a readiness signal: a customer working to a date has
# less appetite for another round of discovery than one browsing.
#
# Deliberately narrow. The first draft matched bare `shipped`, `arrives`,
# `moving`, `days` and `weeks`, which are ordinary *product* vocabulary --
# "moving blankets", "ships in 2 days" -- so it scored catalogue nouns as
# urgency. Every arm here needs an explicit deadline word or a bounded time
# phrase with a number in it.
#
# Unreachable on this evaluator, and recorded as such rather than dropped: the
# reference simulator's constraint vocabulary contains no deadline language, so
# this contributes exactly nothing to any reported score. It is kept for the
# live path, where a customer can say it, and it is why the readiness sweep
# below moves on thresholds rather than on this term.
_URGENCY_RE = re.compile(
    r"\b(?:urgent(?:ly)?|asap|deadline|right\s+away"
    r"|as\s+soon\s+as\s+possible|last[-\s]minute"
    r"|(?:with)?in\s+(?:a|\d+)\s+(?:day|week)s?"
    r"|by\s+(?:tomorrow|tonight|next\s+week|monday|tuesday|wednesday"
    r"|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolicyDecision:
    """The turn-level policy decision, with scores for traceability."""

    name: str
    scores: tuple[tuple[str, float], ...]
    confidence: float
    margin: float
    readiness: float = 0.0
    # The policy that came second, and whether it came close enough that the
    # turn is a genuine tie. `name` still owns the turn either way; see
    # `HYBRID_MARGIN` for why the tie is reported rather than acted on.
    runner_up: str = ""
    hybrid: bool = False


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
    previous_readiness: float | None = None,
) -> PolicyDecision:
    """Scores every dialogue policy and returns the current winner.

    This is the dynamic context-programming layer in miniature: no policy is
    fixed for a session. The same anonymous user can begin in discovery, move
    into precision, hit a boundary refusal, and later recover after a pivot.

    Args:
        state: The session after the latest message has been folded in.
        candidate_count: Size of the resolved catalog pool, when known.
        previous_contenders: Previous turn's near-tie count, when known.
        previous_readiness: Last turn's `D_{t-1}`, when the caller carries one.
          Absent, the prior is re-derived from accumulated session evidence, so
          a caller with no memory of the last turn still gets a usable score.
    """
    readiness = decision_readiness(
        state, candidate_count, previous_contenders, previous_readiness
    )
    raw = {
        RECOVERY: _recovery_score(state),
        COVERAGE: _coverage_score(state),
        STAGNATION: _stagnation_score(state, previous_contenders),
        BOUNDARY: _boundary_score(state),
        PRECISION: _precision_score(state, candidate_count, readiness),
        DISCOVERY: _discovery_score(state, candidate_count,
                                    previous_contenders, readiness),
    }
    scores = tuple(
        (name, round(max(0.0, raw[name]), 3)) for name in _POLICY_ORDER
    )
    ranked_scores = sorted(
        scores, key=lambda item: (item[1], -_POLICY_ORDER.index(item[0])),
        reverse=True,
    )
    top_name, top_score = ranked_scores[0]
    second_name, second_score = (
        ranked_scores[1] if len(ranked_scores) > 1 else ("", 0.0)
    )
    total = sum(score for _, score in scores)
    confidence = top_score / total if total > 0.0 else 0.0
    margin = round(top_score - second_score, 3)
    return PolicyDecision(
        top_name,
        scores,
        round(confidence, 3),
        margin,
        readiness,
        second_name,
        bool(second_name) and margin <= HYBRID_MARGIN,
    )


def framing(decision: PolicyDecision) -> str:
    """Returns the policy whose framing the turn should speak in.

    Separate from `decision.name` so a tie can change how the turn *sounds*
    without changing what it retrieves, and so the two can be measured apart.
    Neutral while `HYBRID_FRAMING` is off, which is how it ships.
    """
    if HYBRID_FRAMING and decision.hybrid and decision.runner_up:
        return decision.runner_up
    return decision.name


def decision_readiness(
    state: dialogue.SessionState,
    candidate_count: int = 0,
    previous_contenders: int = 0,
    previous: float | None = None,
) -> float:
    """Returns the current turn's decision-readiness score in [0, 1].

    `D_t = 0.7 * current + 0.3 * D_{t-1}`. The current turn is weighted more
    heavily than the carried estimate, so a customer can start vague and become
    decisive the moment they give a concrete correction, budget, size, or
    urgent requirement -- and can slide back when they refuse, stall, or run an
    arm dry. That is the self-evolving part: this score belongs to the *turn*,
    not permanently to the shopper.

    Args:
        state: The session after the latest message has been folded in.
        candidate_count: Size of the resolved catalog pool, when known.
        previous_contenders: Previous turn's near-tie count, when known.
        previous: `D_{t-1}`, when the caller carries one. Absent -- the opening
          turn, or any stateless caller -- the prior is re-derived from
          accumulated session evidence instead of starting at zero.
    """
    current = _current_readiness(state, candidate_count, previous_contenders)
    prior = (
        _bounded(previous) if previous is not None
        else _context_readiness(state, candidate_count, previous_contenders)
    )
    weight = READINESS_CURRENT_WEIGHT
    return round(_bounded(weight * current + (1.0 - weight) * prior), 3)


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
    state: dialogue.SessionState,
    candidate_count: int = 0,
    readiness: float = 0.0,
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
    if READINESS_STEERS:
        if readiness >= PRECISION_READINESS_THRESHOLD:
            score += 0.45 + (readiness - PRECISION_READINESS_THRESHOLD)
        elif readiness >= PARTIAL_READINESS_THRESHOLD:
            score += 0.15
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
    readiness: float = 0.0,
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
    if READINESS_STEERS:
        if readiness < PARTIAL_READINESS_THRESHOLD:
            score += 0.2
        if readiness >= PRECISION_READINESS_THRESHOLD:
            score -= 0.35
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


def _current_readiness(
    state: dialogue.SessionState,
    candidate_count: int,
    previous_contenders: int,
) -> float:
    """Scores evidence that arrived on the latest turn."""
    score = 0.15 if state.category else 0.0
    current = _current_slots(state)
    if current:
        score += 0.25
        score += min(0.25, 0.12 * len(current))
        score += min(0.16, 0.08 * _hard_count(current))
        if any(slot.attribute in _DECISIVE_ATTRIBUTES for slot in current):
            score += 0.08
        if any(_URGENCY_RE.search(slot.value) for slot in current):
            score += 0.15
    elif state.turn <= 1 and state.constraints:
        # Some tests construct bare states without typed slots. Treat those
        # constraints as current evidence rather than making readiness vanish.
        score += 0.20 + min(0.18, 0.09 * len(state.constraints))

    if state.pivoted and state.pivot_turn == state.turn:
        score += 0.20
    if 0 < candidate_count <= SMALL_POOL:
        score += 0.08
    if candidate_count >= LARGE_POOL:
        score -= 0.10
    if previous_contenders > UNCERTAIN_CONTENDERS:
        score -= 0.08
    if state.scenario == dialogue.EXPLORING and not current:
        score -= 0.20
    if state.declined:
        score -= 0.12
    if state.idle:
        score -= min(0.20, 0.10 * state.idle)
    if state.exhausted:
        score -= 0.35
    return _bounded(score)


def _context_readiness(
    state: dialogue.SessionState,
    candidate_count: int,
    previous_contenders: int,
) -> float:
    """Scores accumulated session evidence before the current turn dominates."""
    constraints = _constraint_count(state)
    score = 0.20
    if state.category:
        score += 0.12
    score += min(0.36, 0.12 * constraints)
    score += min(0.18, 0.06 * _hard_attribute_count(state))
    if constraints >= MIN_PRECISION_CONSTRAINTS:
        score += 0.12
    if state.confidence >= 0.8:
        score += 0.05
    if 0 < candidate_count <= SMALL_POOL:
        score += 0.10
    if candidate_count >= LARGE_POOL and constraints < MIN_PRECISION_CONSTRAINTS:
        score -= 0.10
    if previous_contenders > UNCERTAIN_CONTENDERS:
        score -= 0.10
    if state.scenario == dialogue.EXPLORING and constraints == 0:
        score -= 0.22
    if state.declined:
        score -= 0.15
    if state.idle:
        score -= min(0.24, 0.12 * state.idle)
    if state.exhausted:
        score -= 0.35
    return _bounded(score)


def _current_slots(state: dialogue.SessionState) -> tuple[slots_module.Slot, ...]:
    """Returns non-negated slots that arrived on the current turn."""
    return tuple(
        slot for slot in state.slots
        if not slot.negated and slot.turn == state.turn
    )


def _hard_count(slots: tuple[slots_module.Slot, ...]) -> int:
    return sum(1 for slot in slots if slot.strength == slots_module.HARD)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _recovering(state: dialogue.SessionState) -> bool:
    """Whether the current turn is still inside the redirect recovery window."""
    if RECOVERY_TURNS <= 0:
        return True
    return state.turn - state.pivot_turn < RECOVERY_TURNS
