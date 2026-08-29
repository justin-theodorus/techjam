"""Choosing what to ask, by what the answer is expected to be worth.

Every turn serves a slate whether or not it also asks something, so a question
costs nothing and the only question worth asking is the one expected to reveal
the most. This module estimates that for each attribute and takes the argmax.

The estimate has three parts: how often products in this catalog say anything
about an attribute at all, how much the session has already heard about it, and
whether the customer has declined it. The wildcard arm is scored as what it is,
the union of every specific arm, which is why it wins: its match set contains
theirs by construction, so its expected yield can never be lower.

That conclusion is not new. What is new is that it is derived rather than
asserted, so it still holds if a customer discloses differently than this one
does, where a hardcoded arm would not.
"""

from __future__ import annotations

import math

from submission.src import dialogue
from submission.src import slots as slots_module

# The attributes a question can name. `other` is deliberately absent: it is not
# an attribute, it is a request for anything, and it is scored separately.
ARMS = (
    slots_module.MATERIAL,
    slots_module.COLOR,
    slots_module.SIZE,
    slots_module.STYLE,
    slots_module.USE_CASE,
    slots_module.FEATURE,
    slots_module.BUDGET,
    slots_module.BRAND,
    slots_module.CATEGORY,
)

WILDCARD = "other"

# How many things a customer volunteers in one answer. Asking about an
# attribute they have plenty to say about does not drain them any faster.
MAX_DISCLOSURE = 2.0

# How much less a customer is expected to add about something they have already
# described. Not zero: a product has several features and one question rarely
# exhausts them.
REPEAT_DECAY = 0.4

# Whether a question may name a specific attribute instead of the wildcard.
#
# Ships live, and it is the one switch here that costs score: measured at
# -0.0104 mean across eight frozen sets, and the wildcard is served on 0 of
# 444 public turns with it on (findings 3.37). It is on because the wildcard
# arm is a question no shopper asks -- "anything else?" repeated for ten turns
# is the conversational cost the efficiency metric exists to discourage, and
# the brief names structured clarification as a goal in its own right.
#
# Turn it off and the agent reverts to the wildcard exactly, which is what the
# reported 0.9554 measured.
SPECIFIC_ARMS = True

# How many of the live contenders to score an arm against. The pool is
# popularity-ordered, so this is the head of what the customer is most likely
# looking at, not a sample.
POOL_SIZE = 150

# How much an attribute is discounted once the customer has spoken about it.
# Separate from `REPEAT_DECAY` above, which decays the wildcard's own estimate:
# this one decays a specific arm against its rivals, and one mention should not
# retire an attribute the pool is still split on.
ARM_DECAY = 0.35

# How much of the pool a specific arm must reach, as a share of what the
# wildcard would reach, before it is asked instead of the wildcard.
#
# A specific arm can come back empty; the wildcard cannot, because its match set
# is the union of every arm's. So the honest comparison at the moment of asking
# is coverage against that union, and this is the share below which the question
# is more likely to waste the turn than to narrow anything.
#
# A ratio rather than an absolute score, and that is the point. Coverage scales
# with how talkative a bucket is, so an absolute threshold measures the bucket
# rather than the decision -- which is how the first attempt at this died, with a
# per-set optimum spread across the whole swept range (findings 3.37). This one
# is monotonic: 0.0 never falls back and costs 0.0100, 1.0 always does and costs
# nothing, and every value between trades the two off smoothly.
WILDCARD_FALLBACK_RATIO = 0.2


def expected_yield(
    state: dialogue.SessionState, taxonomy: slots_module.Taxonomy
) -> dict[str, float]:
    """Returns the constraints each question is expected to reveal.

    Args:
        state: The session so far, including what has been disclosed and
          declined.
        taxonomy: Supplies how prevalent each attribute is in this catalog.
    """
    if state.exhausted:
        return {arm: 0.0 for arm in (*ARMS, WILDCARD)}

    heard: dict[str, int] = {}
    for slot in state.slots:
        heard[slot.attribute] = heard.get(slot.attribute, 0) + 1

    scores = {}
    for arm in ARMS:
        if arm in state.refused:
            scores[arm] = 0.0
            continue
        seen = heard.get(arm, 0)
        scores[arm] = taxonomy.prevalence(arm) * REPEAT_DECAY ** seen

    # The wildcard matches whatever a specific arm would have matched, so its
    # yield is their union rather than any one of them.
    scores[WILDCARD] = min(MAX_DISCLOSURE, sum(scores[arm] for arm in ARMS))
    return scores


def choose(
    state: dialogue.SessionState,
    taxonomy: slots_module.Taxonomy,
    catalog=None,
) -> str | None:
    """Returns the attribute to ask about, or None when nothing is worth asking.

    Declining to ask is the right answer once the customer has said they are
    out of preferences: the question cannot be answered, and continuing to ask
    is the conversational cost the efficiency metric is meant to discourage.

    Args:
        state: The session so far.
        taxonomy: Supplies how prevalent each attribute is in this catalog.
        catalog: Supplies the live pool. Absent, or with `SPECIFIC_ARMS` off,
          the wildcard answer stands, which is what the 0.9554 headline
          measured.
    """
    if state.exhausted:
        return None
    if SPECIFIC_ARMS and catalog is not None:
        return specific(state, catalog)
    scores = expected_yield(state, taxonomy)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.0 else None


def specific(
    state: dialogue.SessionState, catalog, pool_size: int = POOL_SIZE
) -> str | None:
    """Returns the arm expected to separate the live pool most.

    Scores each attribute by `coverage x spread x decay`: how much of the pool
    still leads with something of that kind, how varied those values are, and
    how much the customer has already said about it. An arm nobody can answer
    is worth nothing however discriminating it would be, and an arm everyone
    answers identically separates nothing however often it is offered
    (findings 3.37).

    Falls back to the wildcard rather than to nothing: a turn that asks
    something unanswerable is worse than one that asks for anything, and only an
    exhausted customer is worth not asking at all (which `choose` handles before
    calling here).

    Args:
        state: The session so far.
        catalog: Supplies the candidate pool and the per-product lead lines.
        pool_size: How many contenders to score against.
    """
    pool = catalog.pool(state.pool_keys)[:pool_size]
    if not pool:
        return WILDCARD

    said = {
        catalog.offer_ids[value.casefold()]
        for value in state.constraints
        if value.casefold() in catalog.offer_ids
    }
    heard: dict[str, int] = {}
    for value in state.constraints:
        arm = catalog.taxonomy.classify_text(value)
        heard[arm] = heard.get(arm, 0) + 1

    coverage: dict[str, float] = {}
    weights: dict[str, dict[int, float]] = {}
    union = 0.0
    for index in pool:
        best: dict[str, float] = {}
        for arm, value_id, weight in catalog.offers[index]:
            if value_id in said:
                continue
            tally = weights.setdefault(arm, {})
            tally[value_id] = tally.get(value_id, 0.0) + weight
            if weight > best.get(arm, 0.0):
                best[arm] = weight
        for arm, weight in best.items():
            coverage[arm] = coverage.get(arm, 0.0) + weight
        if best:
            union += max(best.values())

    scores = {}
    for arm in ARMS:
        if arm in state.refused or not coverage.get(arm):
            continue
        scores[arm] = (
            (coverage[arm] / len(pool))
            * _spread(weights[arm])
            * ARM_DECAY ** heard.get(arm, 0)
        )
    if not scores:
        return WILDCARD
    best_arm = max(scores, key=scores.get)
    if scores[best_arm] <= 0.0:
        return WILDCARD
    if union > 0.0 and coverage[best_arm] / union < WILDCARD_FALLBACK_RATIO:
        return WILDCARD
    return best_arm


def _spread(weights: dict[int, float]) -> float:
    """Returns the entropy of one attribute's values across the pool."""
    total = sum(weights.values())
    if total <= 0.0:
        return 0.0
    return -sum(
        (weight / total) * math.log2(weight / total)
        for weight in weights.values() if weight > 0.0
    )
