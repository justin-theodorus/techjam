"""The frozen manifest: twenty-two named session sets, seeds fixed.

Data only. Each name says what the set *varies*, never what it might show,
because a set named for its expected result is an argument rather than an
instrument.

**The seeds are frozen.** They were assigned in listing order before any set
was scored and are never searched. The only number ever consulted while shaping
a recipe was the shipped agent's overall rank-1 rate on that recipe, never a
comparison between two agent configurations. A later session consumes this file
read-only; changing a seed silently invalidates every number taken against it.

Every value of every axis appears in at least one set, which
`harness/tests/test_sessions.py` asserts, so adding an axis value without a set
to exercise it fails the suite.
"""

from __future__ import annotations

from harness import session_axes

Recipe = session_axes.Recipe

ALL_BUYING = (("buying", 1),)
ALL_OVERRIDE = (("intent_override", 1),)
ALL_BOUNDARY = (("boundary", 1),)

MANIFEST = (
    # Neutral on every axis: the public set's own recipe, and the far end of
    # the spectrum.
    Recipe("mirror", 1),

    # Axis 1, target selection.
    Recipe("unpopular_targets", 2, weights="uniform"),
    Recipe("sqrt_targets", 3, weights="sqrt"),
    Recipe("crowded_buckets", 4, pool="crowded"),
    Recipe("sparse_buckets", 5, pool="sparse"),
    Recipe("thin_cards", 6, pool="thin"),
    Recipe("twin_cards", 7, pool="twin"),

    # Axis 2, constraint wording.
    Recipe("reworded_constraints", 8, text="synonym"),
    Recipe("abbreviated_constraints", 9, text="abbreviate"),
    Recipe("negated_constraints", 10, text="negate"),
    Recipe("comparative_constraints", 11, text="comparative"),
    Recipe("unstated_constraints", 12, text="implicit"),
    Recipe("typo_constraints", 13, text="typo"),

    # Axis 3, profiles.
    Recipe("wide_profiles", 14, profiles="wide"),
    Recipe("contradictory_profiles", 15, profiles="adversarial"),
    Recipe("blank_profiles", 16, profiles="empty"),

    # Axes 4 and 5, scenario mix and disclosure shape. The pivot sets isolate
    # intent_override because a shifted pivot changes nothing anywhere else.
    Recipe("front_loaded_buying", 17, mix=ALL_BUYING, dialogue="front_loaded"),
    Recipe("silent_customer", 18, dialogue="silent"),
    Recipe("early_pivot", 19, mix=ALL_OVERRIDE, dialogue="early_pivot"),
    Recipe("late_pivot", 20, mix=ALL_OVERRIDE, dialogue="late_pivot"),
    Recipe("unrelated_pivot", 21, mix=ALL_OVERRIDE,
           dialogue="unrelated_pivot"),

    # Every hard lever at once: unpopular targets, in the most crowded buckets,
    # with the thinnest cards, misspelled.
    Recipe("compound_hard", 22, pool="crowded+thin", weights="uniform",
           text="typo"),

    # Axis 6, identity. The only axis that relates rows to each other, and the
    # only instrument on which per-person memory is readable at all: the
    # organizer's own harness never sends the same shopper twice
    # (measurements 3.33).
    #
    # There is deliberately no second set for the scrambled control. Rotating
    # the identities of *these* rows at scoring time holds the targets, their
    # order, the profiles and the number of memory reads exactly fixed, so the
    # control differs in one variable; a separately seeded set would differ in
    # its whole draw as well.
    #
    # Hard on purpose, and by `compound_hard`'s own recipe: at the neutral
    # pool this set converts 78% of sessions at rank 1, which leaves a memory
    # nothing to win and would read as "no effect" for the same reason the
    # public 200 does (3.29). Thin cards are also where a carried memory
    # should matter most, because the customer has least to say.
    #
    # Boundary, and the reason is a fact about the evaluator rather than
    # about this agent: it is the only scenario whose customer ever says "I
    # don't have a preference for X", so it is the only one that leaves a
    # declined dimension for a later visit to remember. Its opener is
    # byte-identical to browsing's, so the carried-preference gate, which
    # fires only on a turn where the customer has said nothing, still applies.
    # A buying opener would close that gate on turn 1 by disclosing
    # `hard_constraints[0]`.
    #
    # One scenario, because `scale_mix` emits group by group and a mixed set
    # would score a second visit before the first. 198 = 66 x 3, so the three
    # visit blocks are equal and a shopper's rows share index parity, which
    # keeps `run.split_samples` from dealing one visit into each half.
    Recipe("returning_shopper", 23, count=198, mix=ALL_BOUNDARY,
           pool="crowded+thin", weights="uniform", shoppers="returning"),
)
