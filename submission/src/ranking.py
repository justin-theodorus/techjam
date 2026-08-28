"""In-bucket ranking: the popularity and BM25 blend, the slate, the padding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from submission.src import catalog as catalog_module
from submission.src import dialogue
from submission.src import text

ALPHA = 0.6
SLATE_SIZE = 10

# How much the anonymised profile is allowed to move a ranking. The profile is
# an aggregate of preference tags and rating habits with no product in it, and
# nine distinct tag sets cover all 200 public sessions, so the honest prior is
# that it carries almost nothing. Wired, swept and shipped at the weight that
# measurement supports rather than dropped on the floor (findings 3.28).
PROFILE_WEIGHT = 0.0

# How many top-ranked products to commit to while the customer still has
# something left to tell us.
#
# An impression is irreversible. A shopping session ends when the customer sees
# the thing they wanted, at whatever position it happened to occupy, so a slot
# spent on a product the ranking is unsure about is spent for good: if it was
# the answer, it converts at that rank and no later turn can improve it. The
# slots below the top pick are therefore worth more held back than shown, and
# the exploration they would have paid for is better spent on inventory no turn
# has reached at all.
#
# This is deferred commitment, not diversity, and the difference was measured
# rather than assumed: maximal marginal relevance over the same pool loses 0.005
# at its best setting and every weight above zero costs (findings 3.27).
HEAD_SIZE = 1

# How far past the slate the set-selection stage may reach, and how much it
# weighs saying something new against saying the most likely thing. `WINDOW`
# bounds the cost: selection is quadratic in it, everything else here is linear.
WINDOW = 40
DIVERSITY = 0.0

# How close to the leader a product must score to still count as in contention,
# and how few contenders it takes before the slate stops holding back. This is
# the pool-overload signal: many contenders means the evidence so far does not
# separate them, and committing ten slots to an ordering that thin wastes them.
CONTENTION_MARGIN = 0.02
#
# Measured at zero and shipped disabled: converging as soon as one product
# leads raises hit@10 to 0.995 but drops MRR 0.909 to 0.781, and converging
# sooner is worse still (findings 3.27). The signal is kept because it is what
# the reply explains itself with, and because a scorer weighting speed more
# heavily would want it back.
CONVERGE_AT = 0

# A session discloses at most four constraints, after which the customer has
# nothing left to add and there is no reason to keep waiting.
FULL_DISCLOSURE = 4

# The customer may never say they are out of preferences. Without this cap a
# session that also never reaches four constraints would narrow forever.
MAX_DEFER_TURNS = 3


@dataclass(frozen=True)
class Served:
    """One turn's slate, with the numbers that explain its shape."""

    indices: list[int]
    scores: list[float]
    head: int
    contenders: int


def slate(
    catalog: catalog_module.Catalog,
    state: dialogue.SessionState,
    size: int = SLATE_SIZE,
    alpha: float = ALPHA,
    defer_turns: int = MAX_DEFER_TURNS,
    profile_ids: frozenset[int] = frozenset(),
) -> Served:
    """Filters to the resolved buckets, ranks inside them, then pads to `size`.

    A confident category read yields one bucket and this is a hard filter. An
    uncertain one yields a few, which widens the pool rather than emptying it.
    """
    query_ids = catalog.index.query_ids(text.unique_tokens(state.query_text))
    ordered, scores = ranked(
        catalog, catalog.pool(state.pool_keys), query_ids, alpha, profile_ids
    )
    contenders = contention(scores)
    head = head_size(state, size, defer_turns, contenders)
    if DIVERSITY > 0.0:
        chosen = diversify(catalog, ordered, scores, head, size, DIVERSITY)
    else:
        chosen = compose(ordered, head, size)
    padded = pad(catalog, chosen, state.category, size)
    final = rerank(catalog, padded, state)

    # Padding draws from outside the ranked pool, so those slots genuinely have
    # no score rather than a low one. Zero is the honest value for them.
    by_index = dict(zip(ordered, scores))
    return Served(
        final,
        [by_index.get(index, 0.0) for index in final],
        head,
        contenders,
    )


def ranked(
    catalog: catalog_module.Catalog,
    pool: tuple[int, ...],
    query_ids: frozenset[int],
    alpha: float = ALPHA,
    profile_ids: frozenset[int] = frozenset(),
) -> tuple[list[int], list[float]]:
    """Returns every document in `pool` best first, with its blended score.

    Args:
        catalog: The frozen catalog.
        pool: Candidate document indices, already popularity-ordered.
        query_ids: Token ids of the accumulated constraints.
        alpha: Weight on the popularity prior relative to BM25.
        profile_ids: Token ids of the customer's standing preference tags. A
          tie-break at most; see `PROFILE_WEIGHT`.
    """
    if not pool:
        return [], []

    prior = catalog.prior
    max_prior = max(prior[index] for index in pool) or 1.0
    if query_ids:
        lexical = [catalog.index.score(index, query_ids) for index in pool]
    else:
        lexical = [0.0] * len(pool)
    max_lexical = max(lexical) or 1.0
    blended = [
        score / max_lexical + alpha * prior[index] / max_prior
        for index, score in zip(pool, lexical)
    ]
    if profile_ids and PROFILE_WEIGHT > 0.0:
        affinity = [catalog.index.score(index, profile_ids) for index in pool]
        ceiling = max(affinity) or 1.0
        blended = [
            value + PROFILE_WEIGHT * match / ceiling
            for value, match in zip(blended, affinity)
        ]

    # A stable sort over a popularity-ordered pool means ties fall back to the
    # prior, so a query that matches nothing still ranks sensibly.
    positions = sorted(range(len(pool)), key=lambda i: -blended[i])
    return [pool[p] for p in positions], [blended[p] for p in positions]


def order(
    catalog: catalog_module.Catalog,
    pool: tuple[int, ...],
    query_ids: frozenset[int],
    alpha: float = ALPHA,
) -> list[int]:
    """Returns every document in `pool`, best first."""
    return ranked(catalog, pool, query_ids, alpha)[0]


def rank(
    catalog: catalog_module.Catalog,
    pool: tuple[int, ...],
    query_ids: frozenset[int],
    limit: int = SLATE_SIZE,
    alpha: float = ALPHA,
) -> list[int]:
    """Returns the best `limit` documents inside `pool`."""
    return order(catalog, pool, query_ids, alpha)[:limit]


def head_size(
    state: dialogue.SessionState,
    size: int = SLATE_SIZE,
    defer_turns: int = MAX_DEFER_TURNS,
    contenders: int = 0,
) -> int:
    """Returns how many top-ranked documents to serve this turn.

    Showing a product is irreversible: the session ends the moment the customer
    sees the one they wanted, at whatever position it happened to occupy. So a
    slate is a commitment, and the slots below the top pick are worth more held
    back than spent, until either the customer has no more to tell us or the
    ranking is confident enough that holding back buys nothing.

    Args:
        state: The session so far.
        size: How many slots the slate has.
        defer_turns: How long to keep holding back if the customer never says
          they are finished. Load-bearing: without it a customer who never runs
          dry would be served one product for ten turns.
        contenders: How many products are still scoring near the leader. Zero
          means the caller did not measure it.
    """
    if (
        state.exhausted
        or len(state.constraints) >= FULL_DISCLOSURE
        or state.turn > defer_turns
    ):
        return size
    if 0 < contenders <= CONVERGE_AT:
        return size
    return min(HEAD_SIZE, size)


def contention(scores: list[float], margin: float = CONTENTION_MARGIN) -> int:
    """Returns how many products are still competing to be the answer.

    A flat top means the evidence gathered so far does not separate them, which
    is the candidate-pool overload a clarifying question exists to resolve. A
    peaked one means the ranking has already decided.
    """
    if not scores:
        return 0
    best = scores[0]
    if best <= 0.0:
        return len(scores)
    floor = best * (1.0 - margin)
    count = 0
    for score in scores:
        if score < floor:
            break
        count += 1
    return count


def compose(ordered: list[int], head: int, size: int) -> list[int]:
    """Commits to the head, then explores past the slate rather than below it.

    The withheld ranks are the ones a later turn is most likely to promote once
    it knows more, which is exactly why they are the wrong ones to spend an
    irreversible slot on now. The exploration slots go to ranks `size` and
    beyond, which no turn would otherwise reach.
    """
    if head >= size:
        return ordered[:size]
    return list(ordered[:head]) + list(ordered[size:size + size - head])


def diversify(
    catalog: catalog_module.Catalog,
    ordered: list[int],
    scores: list[float],
    head: int,
    size: int,
    weight: float,
) -> list[int]:
    """Fills the slate by marginal relevance rather than by rank.

    Commits the top `head` picks, then chooses each remaining slot to maximise
    what it adds: its own score less how much it repeats a product already on
    the slate. Similarity is token overlap, read straight from the BM25
    postings, so no second index exists to disagree with the first.

    At `weight` zero this is the plain top `size`. As it rises the slate stops
    spending slots on restatements of the leader and reaches deeper into the
    pool, which is the same trade `compose` makes by fixed rank arithmetic and
    for a reason that survives the customer changing.
    """
    if head >= size or weight <= 0.0 or len(ordered) <= size:
        return ordered[:size]

    window = ordered[:WINDOW]
    ceiling = max(scores[:len(window)]) or 1.0
    relevance = [score / ceiling for score in scores[:len(window)]]
    tokens = [catalog.index.tokens_of(index) for index in window]

    chosen = list(range(min(head, len(window))))
    overlap = [_similarity(tokens[i], tokens[0]) for i in range(len(window))]
    remaining = set(range(len(window))) - set(chosen)

    while len(chosen) < size and remaining:
        best = max(
            remaining,
            key=lambda i: (1.0 - weight) * relevance[i] - weight * overlap[i],
        )
        chosen.append(best)
        remaining.discard(best)
        for i in remaining:
            overlap[i] = max(overlap[i], _similarity(tokens[i], tokens[best]))
    return [window[i] for i in chosen]


def _similarity(left: frozenset[int], right: frozenset[int]) -> float:
    """Returns how much two products say the same thing, as token overlap."""
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / (len(left) + len(right) - shared)


class Reranker(Protocol):
    """Reorders an already-chosen slate.

    The contract is deliberately narrow: a reranker returns a permutation of
    what it was given, never a different set. Membership fixes coverage and
    timing, so a stage that only permutes can move precision and nothing else,
    which bounds any reranker's downside at zero by construction rather than by
    good behaviour. A model-backed implementation drops in here without any
    other stage needing to know.
    """

    def __call__(
        self,
        catalog: catalog_module.Catalog,
        chosen: list[int],
        state: dialogue.SessionState,
    ) -> list[int]:
        ...


def rerank(
    catalog: catalog_module.Catalog,
    chosen: list[int],
    state: dialogue.SessionState,
) -> list[int]:
    """Reorders a served slate by rare phrase evidence.

    Always returns a permutation of `chosen`, never a different set. A session
    ends on the first turn the answer appears anywhere in the slate, so
    membership fixes coverage and timing while position fixes precision:
    reordering can only move precision, and this stage is therefore free of
    coverage risk (findings 3.23).

    A constraint the index has not seen, or one so common that no candidate is
    distinguished by it, leaves the blend's order untouched.
    """
    phrase_ids = catalog.phrases.query_ids(state.constraints)
    if not phrase_ids:
        return chosen
    evidence = [
        catalog.phrases.evidence(index, phrase_ids) for index in chosen
    ]
    if not any(evidence):
        return chosen
    positions = sorted(range(len(chosen)), key=lambda i: -evidence[i])
    return [chosen[position] for position in positions]


def pad(
    catalog: catalog_module.Catalog,
    chosen: list[int],
    category: str | None,
    size: int,
) -> list[int]:
    """Fills a short slate from the coarser group, then global popularity.

    Buckets smaller than `size` would otherwise emit empty slots, and every
    slot is a free chance at a hit.
    """
    if len(chosen) >= size:
        return chosen[:size]

    filled = list(chosen)
    seen = set(filled)
    for source in (catalog.fallback_pool(category), catalog.popular):
        for index in source:
            if len(filled) >= size:
                return filled
            if index not in seen:
                seen.add(index)
                filled.append(index)
    return filled
