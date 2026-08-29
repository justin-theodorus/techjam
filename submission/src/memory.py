"""What one shopper's earlier visits leave behind for the next one.

The pillar's "long-term user profile" is this module; its short-term half is
`dialogue`, which already distils one session as it runs. The two are different
mechanisms and only this one needs an identity to be about.

**Nothing here can be scored on the organizer's harness, and that is structural
rather than statistical.** `local_evaluator.evaluate()` mints a fresh
`public_{uuid4().hex}` per session and the published contract fixes both
`reset_request` and `user_profile` with `additionalProperties: false`, so the
declared interface carries no shopper identity at all. A store keyed on identity
is therefore written once per session and read never, and no sweep on that
harness argues against it in either direction (findings 3.33). `Agent.remember`
is the seam an identity arrives through when one exists; absent a caller the
whole module is inert, which is what keeps the reported score untouched.

The one shortcut the data invites is a defect: `user_profile` repeats across 75
of 200 public sessions, but those are distinct people whose aggregates coincide,
so keying on the blurb would merge 26 strangers into one fictional customer.
Key on identity or do not key.
"""

from __future__ import annotations

from dataclasses import dataclass

from submission.src import dialogue
from submission.src import slots as slots_module

# The master switch. Ships on, but reads as off everywhere it can be measured,
# because no set the organizer supplies ever names a shopper twice.
ENABLED = True

# Hard caps. `reset()` may not do I/O or grow without bound across 800 private
# sessions, so the store is a fixed-size structure and eviction is oldest-first.
MAX_SHOPPERS = 512
MAX_ENTRIES = 16

# How much a remembered weight survives one further visit, and the floor below
# which it is dropped rather than carried. Geometric decay over roughly four
# visits, so a preference stated once and never repeated stops being asserted.
DECAY = 0.7
MIN_WEIGHT = 0.3

# The four read gates, in increasing order of risk.
#
# A refusal only ever subtracts, so carrying one cannot promote the wrong
# product; a remembered dimension only stops a question being re-asked. Bucket
# affinity reorders a tie the resolver was going to break arbitrarily. Carrying
# a positive constraint is the one that competes with what the customer is
# saying now, which is exactly the shape `ranking.PROFILE_WEIGHT` measured at
# -0.1125 ungated (findings 3.30), so it ships off until swept.
CARRY_REFUSALS = True
CARRY_ARMS = True
CARRY_BUCKETS = True
CARRY_POSITIVES = False


@dataclass(frozen=True)
class Shopper:
    """What several visits by one person have established.

    Every entry is weighted, and a weight is the only thing that decays, so a
    record shrinks by losing confidence rather than by losing history.
    """

    refusals: tuple[tuple[str, str, float], ...] = ()
    arms: tuple[tuple[str, float], ...] = ()
    buckets: tuple[tuple[str, float], ...] = ()
    positives: tuple[tuple[str, str, float], ...] = ()
    visits: int = 0


def _decayed(
    entries: tuple[tuple[str, float], ...]
) -> dict[str, float]:
    """Returns the single-keyed entries one visit older."""
    return {key: weight * DECAY for key, weight in entries}


def _decayed_pairs(
    entries: tuple[tuple[str, str, float], ...]
) -> dict[tuple[str, str], float]:
    """Returns the attribute-and-value entries one visit older."""
    return {
        (attribute, value): weight * DECAY
        for attribute, value, weight in entries
    }


def _bounded(items: dict, key_length: int) -> tuple:
    """Returns the heaviest entries, dropped below the floor and capped.

    Sorted by weight and then by key, because a tie broken by dict order would
    make a record depend on the order two visits happened to arrive in.
    """
    kept = [
        (key, weight) for key, weight in items.items()
        if weight >= MIN_WEIGHT
    ]
    kept.sort(key=lambda entry: (-entry[1], entry[0]))
    if key_length == 1:
        return tuple((key, weight) for key, weight in kept[:MAX_ENTRIES])
    return tuple(
        (*key, weight) for key, weight in kept[:MAX_ENTRIES]
    )


def distil(shopper: Shopper | None, state: dialogue.SessionState) -> Shopper:
    """Folds one visit's end state into the bounded record.

    This is the distillation step the brief names, and it is deterministic on
    purpose: a model summarising several visits into a short profile would
    implement this same signature behind `ranking.Reranker`'s gate, and the
    capability does not depend on one being available.

    Args:
        shopper: What the person's earlier visits established, or `None` on a
          first visit.
        state: The session as it stands after the turn just served.
    """
    base = Shopper() if shopper is None else shopper
    refusals = _decayed_pairs(base.refusals)
    arms = _decayed(base.arms)
    buckets = _decayed(base.buckets)
    positives = _decayed_pairs(base.positives)

    for slot in state.slots:
        target = refusals if slot.negated else positives
        key = (slot.attribute, slot.value)
        target[key] = target.get(key, 0.0) + 1.0
    for arm in state.declined:
        arms[arm] = arms.get(arm, 0.0) + 1.0
    for bucket in state.pool_keys:
        buckets[bucket] = buckets.get(bucket, 0.0) + 1.0

    return Shopper(
        refusals=_bounded(refusals, 2),
        arms=_bounded(arms, 1),
        buckets=_bounded(buckets, 1),
        positives=_bounded(positives, 2),
        visits=base.visits + 1,
    )


def seed(shopper: Shopper | None) -> dialogue.SessionState:
    """Returns the state a session opens on, given what is remembered.

    Carried memory lands in its own fields rather than in `slots` or `refused`.
    Merging it into those would make `constraints` non-empty before the customer
    has said anything, which would open a returning shopper in the `precision`
    policy and withhold the profile from `ranking.personalised` -- two
    behaviours nobody asked for, arriving as side effects of a memory read.
    """
    if shopper is None or not ENABLED:
        return dialogue.SessionState()
    carried = ()
    if CARRY_REFUSALS:
        carried = tuple(
            slots_module.Slot(attribute, value, 0, True)
            for attribute, value, _ in shopper.refusals
        )
    carried_arms = tuple(arm for arm, _ in shopper.arms) if CARRY_ARMS else ()
    carried_positives = ()
    if CARRY_POSITIVES:
        carried_positives = tuple(
            value for _, value, _ in shopper.positives
        )
    return dialogue.SessionState(
        carried=carried,
        carried_arms=carried_arms,
        carried_positives=carried_positives,
    )


def affinity(shopper: Shopper | None) -> dict[str, float]:
    """Returns the buckets this person has shopped, by weight.

    Consumed as a tie-break over an uncertain category read, never as a filter:
    a bucket the customer named outright is not a tie, and `dialogue` latches
    the category at first sight, so this can only ever move turn one.
    """
    if shopper is None or not ENABLED or not CARRY_BUCKETS:
        return {}
    return dict(shopper.buckets)


class Store:
    """Every shopper this agent instance has seen, oldest evicted first.

    An instance attribute rather than a module global, because `evaluate()` is
    handed one agent for a whole run and a store hidden in module state would
    survive between measurements invisibly -- which is how a sweep comes to
    report the first point's memory as the second point's result.
    """

    def __init__(self) -> None:
        self._shoppers: dict[str, Shopper] = {}
        self._current: str | None = None
        # What the current visit started from. Held apart from the accumulating
        # record so that `observe` running once per turn decays exactly once per
        # visit rather than once per turn.
        self._base: Shopper | None = None

    def remember(self, shopper_id: str | None) -> None:
        """Names who the next session belongs to. `None` is anonymous."""
        self._current = shopper_id or None
        if self._current is None:
            self._base = None
            return
        self._base = self._shoppers.get(self._current)

    def forget(self) -> None:
        """Drops every shopper. The measurement path's isolation point."""
        self._shoppers = {}
        self._current = None
        self._base = None

    def recall(self) -> Shopper | None:
        """Returns what the current shopper's earlier visits established."""
        if self._current is None or not ENABLED:
            return None
        return self._base

    def observe(self, state: dialogue.SessionState) -> None:
        """Records the visit as it stands, replacing this visit's entry.

        There is no end-of-session callback anywhere in the evaluator, so the
        record is rewritten from `_base` after every turn and the last turn
        served is what survives.
        """
        if self._current is None or not ENABLED:
            return
        self._shoppers.pop(self._current, None)
        self._shoppers[self._current] = distil(self._base, state)
        while len(self._shoppers) > MAX_SHOPPERS:
            self._shoppers.pop(next(iter(self._shoppers)))
