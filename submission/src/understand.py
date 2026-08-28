"""Reading one customer message: what they said, and what they meant by it.

Three layers, each producing the same `dialogue.ParsedTurn`.

1.  A template fast path over the strings the reference simulator emits. Exact,
    free, and validated: a template that matches but yields a category the
    catalog does not know is treated as a miss, not as a result.
2.  Cue patterns for the dialogue act, ordered so the shipped literal is the
    first member of each set. Detects a reset, a refusal, an exhaustion or a
    disclosure without depending on any one phrasing.
3.  Category resolution against the catalog's own bucket names, which never
    reads a template at all.

The layering is the point. Layer 1 is an optimization; layers 2 and 3 are the
system. Before they existed, a message matching no template produced no
category, an empty pool, and a slate of pure global popularity: measured at
TechnicalScore 0.0273 with a clean health line (findings 3.24).
"""

from __future__ import annotations

import re

from submission.src import category as category_module
from submission.src import dialogue
from submission.src import text

LOOKING_PREFIX = "I'm looking for "
EXPLORING_SUFFIX = ", but I'm still exploring."
REQUIREMENT_SEP = ". A key requirement is: "
PIVOT_PREFIX = "Actually, ignore my earlier preference."
PIVOT_SEP = "What I need is: "
DISCLOSURE_PREFIX = "For that, what matters is: "
NO_PREFERENCE_PREFIX = "I don't have a preference for "
NO_ADDITIONAL_PREFIX = "I don't have an additional preference for "

# A message that a template read cleanly. Anything below this came from cues.
EXACT = 1.0
CUED = 0.6

_WHITESPACE_RE = re.compile(r"\s+")

# The customer is replacing an earlier preference rather than adding to it.
# Checked before every other act: a reset sentence usually also contains a need
# phrase, and the reset is the part that changes the state.
_RESET_RE = re.compile(
    r"\b(?:ignore|forget|disregard)\b[^.]*\b(?:earlier|previous|preference"
    r"|what i said|that)\b"
    r"|\bscratch that\b"
    r"|\bchanged? (?:of|my) mind\b"
    r"|\bnever ?mind\b",
    re.IGNORECASE,
)

# The customer has run out of things to say about the attribute just asked.
# Checked before refusal, because the two overlap on "I don't have a...".
_EXHAUST_RE = re.compile(
    r"\bdon'?t have an additional preference\b"
    r"|\bnothing else\b"
    r"|\bno more preferences\b"
    r"|\bno additional\b"
    r"|\bthat'?s all\b",
    re.IGNORECASE,
)

# The customer declines to express a preference on the attribute just asked.
_REFUSE_RE = re.compile(
    r"\bno (?:strong )?(?:preference|feelings?)\b"
    r"|\bdon'?t have an? preference\b"
    r"|\bdoesn'?t matter\b"
    r"|\bdon'?t care\b"
    r"|\buse your judgment\b"
    r"|\bnot fussed\b"
    r"|\bwhatever you\b",
    re.IGNORECASE,
)

# The slate was wrong and the customer is waiting to be asked something.
_REJECT_RE = re.compile(
    r"\bnot quite\b|\bnone of (?:those|these)\b|\bask me\b",
    re.IGNORECASE,
)

# The customer is browsing and has deliberately not named a requirement.
_EXPLORING_RE = re.compile(
    r"\bstill exploring\b"
    r"|\bjust browsing\b"
    r"|\bnothing specific\b"
    r"|\bhaven'?t decided\b"
    r"|\bfiguring out\b"
    r"|\bstill (?:deciding|looking)\b",
    re.IGNORECASE,
)

# Phrases that introduce a requirement. Group 1 is the payload. The rightmost
# match wins, so "I need Shirts. It has to be cotton" yields "cotton" rather
# than the whole sentence.
_CONSTRAINT_RES = (
    re.compile(r"what matters is:?\s*(.+)", re.IGNORECASE),
    re.compile(r"what i care about is:?\s*(.+)", re.IGNORECASE),
    re.compile(r"what i need is:?\s*(.+)", re.IGNORECASE),
    re.compile(r"key requirement is:?\s*(.+)", re.IGNORECASE),
    re.compile(r"\bthe main thing is:?\s*(.+)", re.IGNORECASE),
    re.compile(r"\bit has to be\s+(.+)", re.IGNORECASE),
    re.compile(r"\bit should be\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmust be\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmainly\s+(.+)", re.IGNORECASE),
    re.compile(r"\bideally\s+(.+)", re.IGNORECASE),
    re.compile(r"\bi (?:really |actually )?need\s+(.+)", re.IGNORECASE),
    re.compile(r"\bi like\s+(.+)", re.IGNORECASE),
)

# Segment separators for a cued payload. Bare commas are deliberately excluded:
# real constraint strings contain them ("Machine Wash, Tumble Dry"), and an
# over-split constraint still tokenizes identically for BM25 while losing its
# phrase-index match.
_SEGMENT_RE = re.compile(r";|\band\b", re.IGNORECASE)

# Sentence boundaries, for splitting an opening into what it names and what it
# requires.
_CLAUSE_RE = re.compile(r"(?<=[.!?])\s+")

_TRAILING = " .!?,;:-"


def interpret(
    message: object,
    resolver: category_module.Resolver,
    fast_path: bool = True,
) -> dialogue.ParsedTurn:
    """Returns what one customer message contributes to the session.

    Args:
        message: The raw customer message. A non-string yields an empty turn.
        resolver: Catalog bucket resolver, used both to validate a template's
          category and to recover one when no template matched.
        fast_path: Whether to try the template layer first. Turning it off is
          how the general path is exercised on its own; see `make eval
          --no-fast-path`.
    """
    if not isinstance(message, str):
        return dialogue.ParsedTurn()
    value = _WHITESPACE_RE.sub(" ", message).strip()
    if not value:
        return dialogue.ParsedTurn()

    if fast_path:
        parsed = _from_template(value, resolver)
        if parsed is not None:
            return parsed
    return _from_cues(value, resolver)


def _from_template(
    value: str, resolver: category_module.Resolver
) -> dialogue.ParsedTurn | None:
    """Returns a parse of a known template, or None to defer to the cue layer.

    Deferring rather than guessing is what fixed the punctuation failure mode: a
    message whose prefix matched but whose suffix did not used to yield the rest
    of the sentence as a category, which is worse than yielding nothing because
    the pool is then empty rather than wide.
    """
    if value.startswith(PIVOT_PREFIX):
        _, _, tail = value.partition(PIVOT_SEP)
        return dialogue.ParsedTurn(
            constraints=_split_semicolons(tail.rstrip(".")),
            pivot=True,
            scenario_hint=dialogue.OVERRIDE,
            act=dialogue.ACT_RESET,
            confidence=EXACT,
        )

    if value.startswith(DISCLOSURE_PREFIX):
        body = value[len(DISCLOSURE_PREFIX):]
        return dialogue.ParsedTurn(
            constraints=_split_semicolons(body.rstrip(".")),
            act=dialogue.ACT_DISCLOSE,
            confidence=EXACT,
        )

    if value.startswith(NO_ADDITIONAL_PREFIX):
        return dialogue.ParsedTurn(
            exhausted=True, act=dialogue.ACT_EXHAUST, confidence=EXACT
        )

    if value.startswith(NO_PREFERENCE_PREFIX):
        return dialogue.ParsedTurn(
            boundary_refusal=True,
            scenario_hint=dialogue.BOUNDARY,
            act=dialogue.ACT_REFUSE,
            confidence=EXACT,
        )

    if value.startswith(LOOKING_PREFIX):
        return _opening(value[len(LOOKING_PREFIX):], resolver)
    return None


def _opening(
    body: str, resolver: category_module.Resolver
) -> dialogue.ParsedTurn | None:
    """Parses the three turn-1 templates, which all start `I'm looking for `."""
    if body.endswith(EXPLORING_SUFFIX):
        return _validated(
            body[: -len(EXPLORING_SUFFIX)].strip(),
            (),
            dialogue.EXPLORING,
            resolver,
        )

    if REQUIREMENT_SEP in body:
        head, _, constraint = body.partition(REQUIREMENT_SEP)
        return _validated(
            head.strip(),
            _split_semicolons(constraint.rstrip(".")),
            dialogue.BUYING,
            resolver,
        )

    # The override opening is the residual, `{category}. {old_value}`, with no
    # marker of its own. Either half may hold a period, so take the longest
    # left-hand prefix that is a real bucket.
    split_at = -1
    for position, character in enumerate(body):
        if character == "." and resolver.contains(body[:position]):
            split_at = position
    if split_at < 0:
        return None
    return _validated(
        body[:split_at].strip(),
        _split_semicolons(body[split_at + 1:].strip()),
        dialogue.OVERRIDE,
        resolver,
    )


def _validated(
    key: str,
    constraints: tuple[str, ...],
    hint: str,
    resolver: category_module.Resolver,
) -> dialogue.ParsedTurn | None:
    """Returns an opening parse only when the category is a real bucket."""
    if not resolver.contains(key):
        return None
    return dialogue.ParsedTurn(
        category=key,
        buckets=(key,),
        constraints=constraints,
        scenario_hint=hint,
        act=dialogue.ACT_OPEN,
        confidence=EXACT,
    )


def _from_cues(
    value: str, resolver: category_module.Resolver
) -> dialogue.ParsedTurn:
    """Returns a parse built from dialogue cues and catalog vocabulary."""
    act = _act(value)
    if act == dialogue.ACT_RESET:
        return dialogue.ParsedTurn(
            constraints=_constraints(value),
            pivot=True,
            scenario_hint=dialogue.OVERRIDE,
            act=act,
            confidence=CUED,
        )
    if act == dialogue.ACT_EXHAUST:
        return dialogue.ParsedTurn(
            exhausted=True, act=act, confidence=CUED
        )
    if act == dialogue.ACT_REFUSE:
        return dialogue.ParsedTurn(
            boundary_refusal=True,
            scenario_hint=dialogue.BOUNDARY,
            act=act,
            confidence=CUED,
        )
    if act == dialogue.ACT_REJECT:
        return dialogue.ParsedTurn(act=act, confidence=CUED)

    # Everything else may carry product vocabulary, so it is worth resolving.
    # The three acts above cannot: they are meta-text about the conversation,
    # and feeding "I don't have a preference for material" into a query is pure
    # noise (findings 3.21).
    buckets = resolver.buckets(value)
    constraints = _requirements(value, buckets)
    return dialogue.ParsedTurn(
        category=buckets[0] if buckets else None,
        buckets=buckets,
        constraints=constraints,
        scenario_hint=_hint(value, constraints),
        act=dialogue.ACT_OPEN if buckets else dialogue.ACT_UNKNOWN,
        confidence=CUED if buckets else 0.0,
    )


def _requirements(
    value: str, buckets: tuple[str, ...]
) -> tuple[str, ...]:
    """Returns what the customer requires, as opposed to what they named.

    A customer who says they are still exploring has told us they have no
    requirement yet, and taking one anyway is how the browsing opening used to
    inject noise. Otherwise prefer an explicit cue, and fall back to whatever
    the message says beyond naming its category: an opening like "I want tops.
    color: black" carries a requirement with no cue to introduce it.
    """
    if _EXPLORING_RE.search(value):
        return ()
    cued = _constraints(value)
    if cued:
        return _without_category(cued, buckets)
    return _residue(value, buckets)


def _residue(value: str, buckets: tuple[str, ...]) -> tuple[str, ...]:
    """Returns the clauses that say something the category name does not."""
    if not buckets:
        return ()
    named: set[str] = set()
    for key in buckets:
        named.update(text.tokens(key))

    kept = []
    for clause in _CLAUSE_RE.split(value):
        stripped = clause.strip(_TRAILING)
        tokens = text.tokens(stripped)
        if tokens and not set(tokens) <= named:
            kept.append(stripped)
    return tuple(kept)


def _act(value: str) -> str:
    """Returns the dialogue act, checked in the order the sets overlap."""
    if _RESET_RE.search(value):
        return dialogue.ACT_RESET
    if _EXHAUST_RE.search(value):
        return dialogue.ACT_EXHAUST
    if _REFUSE_RE.search(value):
        return dialogue.ACT_REFUSE
    if _REJECT_RE.search(value):
        return dialogue.ACT_REJECT
    return dialogue.ACT_OPEN


def _hint(value: str, constraints: tuple[str, ...]) -> str:
    """Returns the scenario the opening suggests. Advisory, never decisive."""
    if _EXPLORING_RE.search(value):
        return dialogue.EXPLORING
    return dialogue.BUYING if constraints else dialogue.UNKNOWN


def _constraints(value: str) -> tuple[str, ...]:
    """Returns the requirement text a cue introduces, if any.

    The rightmost cue wins. An opening such as "I need shoes. It has to be
    leather" contains two, and only the later one introduces the requirement.
    """
    best = None
    for pattern in _CONSTRAINT_RES:
        match = pattern.search(value)
        if match and (best is None or match.start(1) > best.start(1)):
            best = match
    if best is None:
        return ()
    return _segment(best.group(1))


def _segment(payload: str) -> tuple[str, ...]:
    """Splits a cued payload into individual constraints."""
    parts = (part.strip(_TRAILING) for part in _SEGMENT_RE.split(payload))
    return tuple(part for part in parts if part)


def _without_category(
    constraints: tuple[str, ...], buckets: tuple[str, ...]
) -> tuple[str, ...]:
    """Drops a constraint that merely echoes the category back.

    A greedy cue can swallow the category ("I want shirts and cotton is what I
    need"), and re-stating the bucket inside the query is measurably worse than
    leaving it out (findings 3.6).
    """
    if not buckets:
        return constraints
    echoes = {key.casefold() for key in buckets}
    return tuple(
        value for value in constraints if value.casefold() not in echoes
    )


def _split_semicolons(body: str) -> tuple[str, ...]:
    """Splits a disclosure payload on semicolons.

    Deliberately loose: the reference simulator strips `;` only from the ends of
    a constraint, so an internal semicolon over-splits. Harmless, because every
    piece ends up as bag-of-words in one query.
    """
    return tuple(part.strip() for part in body.split(";") if part.strip())
