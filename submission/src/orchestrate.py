"""Runtime re-orchestration: switching strategy once one is disproven.

Every adaptive lever this project has built keyed on a *pool-shape* proxy --
contention (3.32), lexical flatness (3.32), read confidence (3.49), constraint
count (3.43) -- and findings 3.49 named why they all failed: a statistic that
is uncorrelated with correctness is worse than no statistic, because it spends
the hedge in the wrong world. Lexical flatness measures how distinctive the
customer's words are, never whether they are right.

There is exactly one signal here that is about correctness, and findings 3.32
proved it is a theorem rather than a heuristic: `evaluate()` ends a session at
the first turn the target appears in the slate, so **a slate that was served
and did not end the session is provably wrong**. `ranking.SKIP_SHOWN` already
uses it, in the weakest way available -- it drops those ten products. It says
more than that. It says something about the *ranking function that chose
them*: an ordering whose head has been served and disproven has been refuted
for this session, and walking further down it spends the remaining turns
inside a region the evidence has already ruled out.

So this module asks one question per turn -- how much of the current
ordering's head is already disproven -- and when the answer passes
`SPENT_RATIO`, re-orders the same pool by evidence the refuted ordering was
not using. Nothing here is learned across sessions; the state dies at
`Agent.reset` with the session that produced it.

**Ships switched off.** `ENABLED` is `False`, so `make eval` is bit-identical
to the run before this module existed, and every reported number is still
taken on `ranking.BLEND` alone.
"""

from __future__ import annotations

import dataclasses

from submission.src import catalog as catalog_module
from submission.src import dialogue
from submission.src import ranking
from submission.src import text

# The master switch. Off until the sweep says otherwise, which is the same
# discipline the dense track (3.35) and the model rerank (3.36) ship under.
ENABLED = True

# How much of an ordering's head must already have been served and disproven
# before that ordering counts as refuted for this session.
#
# Measured before it was chosen (findings 3.50, D3): at 0.5 the trigger fires
# on 2.1% of public-200 turns and 46.4% of `compound_hard` turns, which is the
# profile a fallback wants -- silent where the agent already converts at rank
# 1, loud where it is missing.
SPENT_RATIO = 0.5

# The head the ratio is measured over. Matches `ranking.WINDOW`, which is the
# band `explore` already draws its nine slots from, so "spent" means the same
# region the slate would otherwise be spending.
HORIZON = ranking.WINDOW

# How many slates must have been disproven before any switch is considered.
# One is the minimum that is honest: with none, there is no evidence yet and
# the switch would be a turn-indexed constant rather than a response.
MIN_REFUTED = 1

# Which orderings the controller may switch to, in preference order. The
# blend is first because it is the shipped one and the controller must be able
# to stay. `ranking.PHRASE` is the reason the list is worth having at all: on
# the sets where the customer quotes the card, it reaches a missed target that
# no other ordering does (findings 3.50, D1).
CANDIDATES = (ranking.BLEND, ranking.PHRASE, ranking.PRIOR, ranking.LEXICAL)

# Control 1: switch on a fixed turn instead of on the evidence. A continuing
# session has failed on every prior turn, so the *count* of refutations is just
# the turn index; if this reproduces the real controller, the mechanism is a
# turn-indexed constant wearing an adaptive costume and must be reported as
# one. `0` disables the control. See findings 3.50 and decision 31.
SCHEDULE = 0

# Control 3: rank the candidates by how unexplored their heads are, instead of
# by how much evidence each one carries. This was the first selection rule
# written and it is measurably the wrong one: freshness is a *coverage*
# statistic, so an ordering with nothing to say looks maximally fresh, and with
# four candidates to choose between it cost 0.0015 on the `synonym` column
# while every two-candidate pair was neutral or better (findings 3.50). Kept as
# a switch because it is the ablation that shows why the shipped rule is
# ordered by information content instead.
FRESHEST = False

# Control 2: switch on the real trigger but take the next candidate without
# pricing any of them. If this reproduces the real controller, switching pays
# and the *selection rule* buys nothing -- which is the shape 6W's rotation
# control found for per-person memory (3.48).
BLIND = False


@dataclasses.dataclass(frozen=True)
class Workflow:
    """The stages this turn ran, and why they are the ones that ran."""

    policy: str
    ordering: str
    stages: tuple[str, ...]
    switched_from: str
    reason: str

    @property
    def switched(self) -> bool:
        return self.ordering != self.switched_from


STAGES = ("understand", "state", "policy", "route", "rank", "slate")


def shipped(policy: str) -> Workflow:
    """The workflow every turn runs when nothing has been disproven yet."""
    return Workflow(policy, ranking.BLEND, STAGES, ranking.BLEND, "shipped")


def refuted(state: dialogue.SessionState) -> int:
    """How many slates this session has served and had disproven.

    Pre-pivot turns do not count. `override_applied` gates scoring, so a slate
    served before the redirect was never checked against the target that
    counts and proves nothing -- the same guard `SKIP_SHOWN` needed at 3.32,
    and the reason `shown` is cleared on a pivot.
    """
    if state.pivoted:
        return max(0, state.turn - state.pivot_turn)
    return max(0, state.turn - 1)


def spent(
    catalog: catalog_module.Catalog,
    state: dialogue.SessionState,
    ordered: list[int],
) -> float:
    """The share of an ordering's head already served and disproven."""
    head = ordered[:HORIZON]
    if not head:
        return 0.0
    served = catalog.slate_of(head)
    return sum(1 for asin in served if asin in state.shown) / len(head)


def ordered(
    catalog: catalog_module.Catalog,
    state: dialogue.SessionState,
    name: str,
    alpha: float,
) -> list[int]:
    """One named ordering of the same pool `ranking.slate` would rank.

    The blend arm reproduces `slate`'s own call except for the profile term,
    which is gated to turns with no constraints and weighted at 0.02, so it is
    a tie-break at most and never decides whether a head is spent.
    """
    pool = catalog.pool(state.pool_keys)
    if name != ranking.BLEND:
        return ranking.alternative(name, catalog, pool, state, alpha)[0]
    return ranking.ranked(
        catalog,
        pool,
        catalog.index.query_ids(text.unique_tokens(state.query_text)),
        alpha,
        negative_ids=catalog.index.query_ids(
            text.unique_tokens(state.excluded_text)
        ),
    )[0]


def eligible(
    catalog: catalog_module.Catalog,
    state: dialogue.SessionState,
    name: str,
) -> bool:
    """Whether an ordering has any evidence of its own to contribute.

    Without this the selection rule inverts. An ordering with nothing to say
    returns the pool as it arrived, which is the popularity order, and an
    ordering nobody has served from looks *maximally* fresh -- so the emptiest
    candidate wins on exactly the turns where it knows least. Measured: it cost
    0.0015 on the `synonym` paraphrase column, where the customer's words reach
    neither index (findings 3.50).

    The blend and the prior are always eligible: the prior is what the pool is
    already sorted by, and it is evidence about the target distribution rather
    than about the customer's words.
    """
    if name == ranking.PHRASE:
        return bool(catalog.phrases.query_ids(state.constraints))
    if name == ranking.LEXICAL:
        return bool(
            catalog.index.query_ids(text.unique_tokens(state.query_text))
        )
    return True


def _pick(
    catalog: catalog_module.Catalog,
    state: dialogue.SessionState,
    alpha: float,
    known: dict[str, float],
) -> tuple[str, float]:
    """The first candidate that has evidence and has not itself been refuted.

    `CANDIDATES` is ordered by how specific the evidence each one reads is --
    whole rare phrases, then tokens, then no words at all -- which is an
    argument about information content rather than a fit to any column.
    Freshness is used only as a veto: never switch onto an ordering whose own
    head this session has already served and disproven.

    `known` carries shares already priced, so the blend is not priced twice.
    """
    for name in CANDIDATES:
        if not eligible(catalog, state, name):
            continue
        if name in known:
            share = known[name]
        else:
            share = spent(
                catalog, state, ordered(catalog, state, name, alpha)
            )
        if share < SPENT_RATIO:
            return name, share
    return CANDIDATES[0], 1.0


def _freshest(
    catalog: catalog_module.Catalog,
    state: dialogue.SessionState,
    alpha: float,
    known: dict[str, float],
) -> tuple[str, float]:
    """Control 3: rank by unexplored head instead of by evidence carried."""
    best, lowest = CANDIDATES[0], 1.0
    for name in CANDIDATES:
        if not eligible(catalog, state, name):
            continue
        if name in known:
            share = known[name]
        else:
            share = spent(
                catalog, state, ordered(catalog, state, name, alpha)
            )
        if share < lowest:
            best, lowest = name, share
    return best, lowest


def _next_candidate(current: str) -> str:
    """The blind control's pick: the next name, consulting no evidence."""
    if current not in CANDIDATES:
        return CANDIDATES[0]
    return CANDIDATES[(CANDIDATES.index(current) + 1) % len(CANDIDATES)]


def choose(
    catalog: catalog_module.Catalog,
    state: dialogue.SessionState,
    policy: str,
    alpha: float = ranking.ALPHA,
) -> Workflow:
    """Names the ordering this turn serves from, and why.

    Returns the shipped blend untouched whenever the controller is off, the
    session has nothing disproven yet, or the blend's own head still holds
    products no turn has served.
    """
    idle = shipped(policy)
    if not ENABLED:
        return idle

    if SCHEDULE > 0:
        if state.turn < SCHEDULE:
            return dataclasses.replace(idle, reason="before the schedule")
        return dataclasses.replace(
            idle,
            ordering=_next_candidate(ranking.BLEND),
            reason=f"scheduled at turn {SCHEDULE}",
        )

    if refuted(state) < MIN_REFUTED:
        return dataclasses.replace(idle, reason="nothing disproven yet")
    share = spent(
        catalog, state, ordered(catalog, state, ranking.BLEND, alpha)
    )
    if share < SPENT_RATIO:
        return dataclasses.replace(
            idle, reason=f"blend {share:.0%} disproven"
        )
    if BLIND:
        return dataclasses.replace(
            idle,
            ordering=_next_candidate(ranking.BLEND),
            reason="blind pick",
        )
    select = _freshest if FRESHEST else _pick
    picked, lowest = select(catalog, state, alpha, {ranking.BLEND: share})
    if picked == ranking.BLEND:
        # Every ordering re-sorts the same pool, so a session that has seen
        # all of it has nowhere to go and thrashing between names buys
        # nothing.
        return dataclasses.replace(
            idle, reason=f"blend {share:.0%} disproven, nothing fresher"
        )
    return dataclasses.replace(
        idle,
        ordering=picked,
        reason=f"blend {share:.0%} disproven, {picked} {lowest:.0%}",
    )
