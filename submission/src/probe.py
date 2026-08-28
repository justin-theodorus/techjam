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
    state: dialogue.SessionState, taxonomy: slots_module.Taxonomy
) -> str | None:
    """Returns the attribute to ask about, or None when nothing is worth asking.

    Declining to ask is the right answer once the customer has said they are
    out of preferences: the question cannot be answered, and continuing to ask
    is the conversational cost the efficiency metric is meant to discourage.
    """
    scores = expected_yield(state, taxonomy)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.0 else None
