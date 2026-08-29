"""The per-session dialogue state and how one parsed turn folds into it.

Reading a message is `submission.src.understand`; this module only remembers. Keeping
them apart means the state machine is testable without a catalog, and the
understanding layer can be swapped without touching what a session remembers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from techjam.submission.src import slots as slots_module

UNKNOWN = "unknown"
BUYING = "buying"
EXPLORING = "browsing_or_boundary"
BOUNDARY = "boundary"
OVERRIDE = "intent_override"

# The probe arm that matches any undisclosed constraint. Named here rather than
# imported from `probe` because that module imports this one.
WILDCARD = "other"

# What the customer did on a turn, as opposed to what they said. Drives the
# state fold and, from Phase 6R.3, the retrieval route.
ACT_UNKNOWN = "unknown"
ACT_OPEN = "open"
ACT_DISCLOSE = "disclose"
ACT_RESET = "reset"
ACT_REFUSE = "refuse"
ACT_EXHAUST = "exhaust"
ACT_REJECT = "reject"

# Whether a replacement erases only the attributes it contradicts, or the whole
# constraint list. Both extremes had been measured against each other and total
# erasure won; targeted erasure is the third option neither tested, and it beats
# both. Worth +0.016 overall and it converts every override session, 0.900 to
# 1.000 hit@10 (findings 3.26).
TARGETED_OVERRIDE = True

# Whether "I don't have an additional preference for X" retires only X, or the
# whole session.
#
# The evaluator returns that sentence whenever the arm that was asked has no
# undisclosed match, so it means "nothing more about X" and not "nothing more
# at all". The two readings coincide only while the wildcard is the only arm
# ever asked, because the wildcard matches everything -- which is why this was
# latent rather than wrong for as long as it was. Ask specifically and the
# unscoped reading ends the session's narrowing on the first miss: `probe`
# stops asking and `ranking.head_size` opens the slate (findings 3.37).
#
# Ships live. What it replaces is an inverted default rather than a neutral
# one, and it is worth up to +0.0996 on `compound_hard` once a specific arm is
# ever asked, exactly nothing before that.
SCOPED_EXHAUSTION = True


@dataclass(frozen=True)
class ParsedTurn:
    """What one customer message yields. Absent fields simply do not fire."""

    category: str | None = None
    buckets: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    pivot: bool = False
    scenario_hint: str | None = None
    boundary_refusal: bool = False
    exhausted: bool = False
    # The attribute an exhaustion names, when it names one. The customer says
    # which thing they are out of, so scoping reads that rather than trusting
    # the agent's record of what it asked.
    exhausted_arm: str | None = None
    act: str = ACT_UNKNOWN
    confidence: float = 0.0


@dataclass(frozen=True)
class SessionState:
    """One session's dialogue memory. Bounded, and never mutated in place."""

    category: str | None = None
    buckets: tuple[str, ...] = ()
    scenario: str = UNKNOWN
    constraints: tuple[str, ...] = ()
    superseded: tuple[str, ...] = ()
    pivoted: bool = False
    exhausted: bool = False
    refused: tuple[str, ...] = ()
    slots: tuple[slots_module.Slot, ...] = ()
    turn: int = 0
    pivot_turn: int = 0
    confidence: float = 0.0
    last_slate: tuple[str, ...] = ()
    shown: frozenset[str] = frozenset()

    @property
    def query_text(self) -> str:
        """The accumulated constraint text, as one BM25 query.

        Refusals are held out. Scoring "not polyester" as a positive query
        walks the customer toward the thing they declined: measured at 2.3x
        that material's shelf rate, and up to 5.1x on the rarer ones
        (findings 3.31).
        """
        if not self.slots:
            return " ".join(self.constraints)
        return " ".join(
            slot.value for slot in self.slots if not slot.negated
        )

    @property
    def excluded_text(self) -> str:
        """The refused constraint text, cue stripped, as one BM25 query."""
        return " ".join(
            slots_module.polarity(slot.value)[1]
            for slot in self.slots if slot.negated
        )

    @property
    def pool_keys(self) -> tuple[str, ...]:
        """The buckets to retrieve from.

        Falls back to the primary category, so a state carrying a category and
        nothing else still retrieves from it rather than from nowhere.
        """
        if self.buckets:
            return self.buckets
        return (self.category,) if self.category else ()

    def with_slate(self, slate: tuple[str, ...]) -> SessionState:
        """Returns a copy remembering the slate just served, and every one
        before it.

        The session ends at the first turn the target appears in the slate, so
        a product that was shown and did not end the session is provably not
        the answer. Showing it again cannot convert; the slot is spent either
        way. See `ranking.SKIP_SHOWN`.
        """
        return replace(self, last_slate=slate, shown=self.shown | set(slate))


def update(
    state: SessionState,
    parsed: ParsedTurn,
    asked: str | None = None,
    taxonomy: slots_module.Taxonomy | None = None,
) -> SessionState:
    """Folds one parsed turn in. A pivot erases constraints, never merges.

    The turn counter advances on every fold, including a pivot, because it
    measures how long the session has run rather than how much it has disclosed.

    Args:
        state: The session so far.
        parsed: What the customer's latest message yielded.
        asked: The attribute probed on the previous turn, so that a refusal can
          be attributed to the thing that was actually asked about.
        taxonomy: Types arriving constraints. Absent, the session still tracks
          what was said, just not what it was about.
    """
    turn = state.turn + 1
    arriving = _typed(parsed.constraints, turn, taxonomy)

    if parsed.pivot:
        kept = _survivors(state.slots, arriving)
        held = tuple(slot for slot in state.slots if slot not in kept)
        slots = kept + arriving
        superseded = _merge(state.superseded, tuple(s.value for s in held))
    else:
        slots = _merge_slots(state.slots, arriving)
        superseded = state.superseded
    constraints = tuple(slot.value for slot in slots)

    refused = state.refused
    if parsed.boundary_refusal and asked:
        refused = _merge(refused, (asked,))

    exhausted = state.exhausted or parsed.exhausted
    spent = parsed.exhausted_arm or asked
    if SCOPED_EXHAUSTION and parsed.exhausted and _is_specific(spent):
        # `probe.expected_yield` and `probe.specific` both zero a refused arm,
        # so the session keeps narrowing and simply stops asking about this one.
        exhausted = state.exhausted
        refused = _merge(refused, (spent,))

    return SessionState(
        category=state.category or parsed.category,
        buckets=state.buckets or parsed.buckets,
        scenario=_scenario(state, parsed),
        constraints=constraints,
        superseded=superseded,
        pivoted=state.pivoted or parsed.pivot,
        exhausted=exhausted,
        refused=refused,
        slots=slots,
        turn=turn,
        pivot_turn=turn if parsed.pivot else state.pivot_turn,
        confidence=max(state.confidence, parsed.confidence),
        last_slate=state.last_slate,
        # A pivot un-tests everything already shown. `override_applied` gates
        # scoring, so a pre-pivot impression was never checked against the new
        # target and may well be it (findings 3.4, the 65 wasted pre-pivot
        # hits the health line counts).
        shown=frozenset() if parsed.pivot else state.shown,
    )


def _is_specific(asked: str | None) -> bool:
    """Whether a probe named one attribute rather than asking for anything."""
    return bool(asked) and asked != WILDCARD


def _typed(
    constraints: tuple[str, ...],
    turn: int,
    taxonomy: slots_module.Taxonomy | None,
) -> tuple[slots_module.Slot, ...]:
    """Attaches an attribute and a turn to each arriving constraint."""
    if taxonomy is None:
        return tuple(
            slots_module.Slot(slots_module.DEFAULT, value, turn,
                              slots_module.polarity(value)[0])
            for value in constraints
        )
    return taxonomy.slots(constraints, turn)


def _survivors(
    existing: tuple[slots_module.Slot, ...],
    arriving: tuple[slots_module.Slot, ...],
) -> tuple[slots_module.Slot, ...]:
    """Returns the slots a replacement leaves standing.

    A customer replacing their material preference has said nothing about the
    colour they asked for earlier, so keeping it is the literal reading. Whether
    that beats erasing everything is a measurement, not an argument, which is
    what `TARGETED_OVERRIDE` exists to switch between.
    """
    if not TARGETED_OVERRIDE:
        return ()
    replaced = {slot.attribute for slot in arriving}
    return tuple(
        slot for slot in existing if slot.attribute not in replaced
    )


def _merge_slots(
    existing: tuple[slots_module.Slot, ...],
    arriving: tuple[slots_module.Slot, ...],
) -> tuple[slots_module.Slot, ...]:
    """Appends arriving slots, deduped on value and order-preserving."""
    seen = {slot.value for slot in existing}
    merged = list(existing)
    for slot in arriving:
        if slot.value not in seen:
            seen.add(slot.value)
            merged.append(slot)
    return tuple(merged)


def _merge(
    existing: tuple[str, ...], incoming: tuple[str, ...]
) -> tuple[str, ...]:
    """Appends `incoming` to `existing`, deduped and order-preserving."""
    return tuple(dict.fromkeys((*existing, *incoming)))


def _scenario(state: SessionState, parsed: ParsedTurn) -> str:
    """Returns the scenario label the retrieval route is chosen from."""
    if parsed.pivot or state.pivoted:
        return OVERRIDE
    if parsed.boundary_refusal:
        return BOUNDARY
    if state.scenario != UNKNOWN:
        return state.scenario
    return parsed.scenario_hint or UNKNOWN
