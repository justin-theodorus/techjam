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

# How much the dense track is allowed to move an in-bucket ranking.
#
# The second retriever, and the only component in the project that can match a
# word the catalog never uses. BM25 scores "trousers" at zero against a product
# whose bullets say "pants"; the latent space scores it at 0.91, because the two
# words occupy the same neighbourhood in text the catalog wrote about itself
# (findings 3.35). That is the `synonym` column's exact failure mode, and it is
# the worst column we have.
#
# Ships at zero. The asset is bundled and loaded, and at this weight it
# contributes nothing to any reported number, so every headline stays Tier 0:
# standard library, no third-party package, no network. `make deviations` reads
# it as an eighth component and the writeup reports the column either way.
DENSE_WEIGHT = 0.0

# The dense half of the refusal subtraction, kept separate from the positive
# weight because there is no reason to assume they are equal. Leaving the
# refusal lexical-only while the preference is both would be exactly the kind of
# asymmetry Phase 6T was created to catch, so it is measured rather than
# assumed.
DENSE_NEGATION_WEIGHT = 0.0

# How hard a refused attribute counts against a product. Unlike the five
# switched-off deviations this ships live, because the behaviour it replaces is
# not a neutral default but an inverted one: without it "not polyester" is
# scored as evidence *for* polyester, and the agent served the refused material
# at 2.3x its shelf rate, up to 5.1x on the rarer ones (findings 3.31).
#
# A penalty rather than a filter. Refusal detection is lexical and the catalog
# spells some attribute names negatively, so a false positive here costs a few
# ranks that a later turn recovers, where a filter would make the target
# unreachable for the rest of the session.
NEGATION_WEIGHT = 0.5

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

# Whether a product the customer has already been shown may occupy a slot
# again. Ships live, because the alternative is not a neutral default but a
# provably dead one: a session ends at the first turn the target appears in the
# slate, so anything already shown is not the answer and re-showing it converts
# nothing. Measured before the change, 62.9% of impressions on `thin_cards`
# were repeats and 60 of its 78 misses had the target inside the slot budget
# the session had already spent (findings 3.32).
SKIP_SHOWN = True

# Whether a built model-backed rerank stage may reorder the served ten.
#
# Tier 2, and the second of two gates: `llm.build()` returns nothing at all
# unless `USE_LLM=1`, and this decides whether a stage that was built gets
# consulted. Both are needed, so the scored configuration reads neither a
# credential nor the network.
#
# Ships at zero, and the reason is arithmetic rather than taste. A permutation
# cannot move HitRate or MTTC, but on the public 200 only 20 of 200 sessions
# convert below rank 1, so a *perfect* reordering is worth 0.022 of score while
# a careless one can move all 180 sessions that already convert first
# (findings 3.36).
LLM_RERANK = 0

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
    dense_weight: float | None = None,
    reach: int = 0,
    reranker: Reranker | None = None,
) -> Served:
    """Filters to the resolved buckets, ranks inside them, then pads to `size`.

    A confident category read yields one bucket and this is a hard filter. An
    uncertain one yields a few, which widens the pool rather than emptying it.
    """
    # Resolved here rather than in the signature: a default argument binds at
    # definition, which is how findings 3.27 lost a whole sweep to a constant
    # that could no longer be patched. `None` means "whatever the module says
    # now", which is what makes this switch readable by `make deviations`.
    if dense_weight is None:
        dense_weight = DENSE_WEIGHT
    query_ids = catalog.index.query_ids(text.unique_tokens(state.query_text))
    negative_ids = catalog.index.query_ids(
        text.unique_tokens(state.excluded_text)
    )
    dense_query = encode(
        catalog, state.query_text, dense_weight > 0.0 or reach > 0
    )
    dense_negative = encode(
        catalog, state.excluded_text, DENSE_NEGATION_WEIGHT > 0.0
    )
    pool = widen(catalog, state, dense_query, reach)
    ordered, scores = ranked(
        catalog, pool, query_ids, alpha, profile_ids,
        negative_ids, dense_query, dense_negative, dense_weight,
    )
    contenders = contention(scores)
    ordered, scores = unseen(catalog, ordered, scores, state.shown, size)
    head = head_size(state, size, defer_turns, contenders)
    if DIVERSITY > 0.0:
        chosen = diversify(catalog, ordered, scores, head, size, DIVERSITY)
    else:
        chosen = compose(ordered, head, size)
    padded = pad(catalog, chosen, state.category, size)
    final = reranked(catalog, padded, state, reranker)

    # Padding draws from outside the ranked pool, so those slots genuinely have
    # no score rather than a low one. Zero is the honest value for them.
    by_index = dict(zip(ordered, scores))
    return Served(
        final,
        [by_index.get(index, 0.0) for index in final],
        head,
        contenders,
    )


def widen(
    catalog: catalog_module.Catalog,
    state: dialogue.SessionState,
    dense_query: tuple[float, ...] | None,
    reach: int,
) -> tuple[int, ...]:
    """Returns the candidate pool, optionally reaching past the category.

    This is the retrieval half of the dense track, as opposed to the ranking
    half in `ranked`: it changes *which* products a route considers rather
    than how the chosen ones are weighed, and it is the only thing that makes
    a route a retriever instead of a second set of constants.

    The reach draws from the coarser group the bucket sits inside, which is
    the same pool `pad` already falls back to. The difference is that padding
    picks by popularity alone, blind to everything the customer said, where
    this picks by similarity to what they said.

    The union is re-sorted by the prior because `ranked` leans on a
    popularity-ordered pool for its tie-break: a query that matches nothing
    must still rank sensibly, and appending candidates in similarity order
    would quietly break that.
    """
    pool = catalog.pool(state.pool_keys)
    if reach <= 0 or dense_query is None or catalog.dense is None:
        return pool
    outside = [
        index for index in catalog.fallback_pool(state.category)
        if index not in set(pool)
    ]
    extra = catalog.dense.nearest(dense_query, outside, reach)
    if not extra:
        return pool
    merged = list(pool) + extra
    merged.sort(key=lambda index: -catalog.prior[index])
    return tuple(merged)


def encode(
    catalog: catalog_module.Catalog, value: str, active: bool
) -> tuple[float, ...] | None:
    """Returns `value` as a latent vector, or `None` if the track is inert.

    Encoded once per turn rather than once per document, which is what keeps
    the dense term linear in the pool. Three separate ways of returning
    `None`, and all three have to degrade to the lexical blend unchanged: the
    track is switched off, the asset is absent or bound to a different
    catalog, or the customer used no word the space has ever seen.

    `active` rather than a weight because retrieval and ranking are separate
    uses. A route may reach for candidates densely and still rank them
    lexically, which is the purest form of the two-retriever split and has to
    be measurable on its own.
    """
    if not active or catalog.dense is None or not value:
        return None
    return catalog.dense.encode(text.unique_tokens(value))


def ranked(
    catalog: catalog_module.Catalog,
    pool: tuple[int, ...],
    query_ids: frozenset[int],
    alpha: float = ALPHA,
    profile_ids: frozenset[int] = frozenset(),
    negative_ids: frozenset[int] = frozenset(),
    dense_query: tuple[float, ...] | None = None,
    dense_negative: tuple[float, ...] | None = None,
    dense_weight: float | None = None,
) -> tuple[list[int], list[float]]:
    """Returns every document in `pool` best first, with its blended score.

    Args:
        catalog: The frozen catalog.
        pool: Candidate document indices, already popularity-ordered.
        query_ids: Token ids of the accumulated constraints.
        alpha: Weight on the popularity prior relative to BM25.
        profile_ids: Token ids of the customer's standing preference tags. A
          tie-break at most; see `PROFILE_WEIGHT`.
        negative_ids: Token ids of what the customer has refused. Subtracted;
          see `NEGATION_WEIGHT`.
        dense_query: The constraints as a latent vector, or `None` when the
          track is off or the words are all unknown to it.
        dense_negative: The refusals as a latent vector; see
          `DENSE_NEGATION_WEIGHT`.
        dense_weight: Weight on the dense similarity for this route.
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
    if dense_weight is None:
        dense_weight = DENSE_WEIGHT
    if dense_query is not None and dense_weight > 0.0:
        similarity = [
            catalog.dense.score(index, dense_query) for index in pool
        ]
        ceiling = max(similarity) or 1.0
        blended = [
            value + dense_weight * match / ceiling
            for value, match in zip(blended, similarity)
        ]
    if dense_negative is not None and DENSE_NEGATION_WEIGHT > 0.0:
        refused = [
            catalog.dense.score(index, dense_negative) for index in pool
        ]
        ceiling = max(refused) or 1.0
        blended = [
            value - DENSE_NEGATION_WEIGHT * match / ceiling
            for value, match in zip(blended, refused)
        ]
    if profile_ids and PROFILE_WEIGHT > 0.0:
        affinity = [catalog.index.score(index, profile_ids) for index in pool]
        ceiling = max(affinity) or 1.0
        blended = [
            value + PROFILE_WEIGHT * match / ceiling
            for value, match in zip(blended, affinity)
        ]
    if negative_ids and NEGATION_WEIGHT > 0.0:
        refused = [catalog.index.score(index, negative_ids) for index in pool]
        ceiling = max(refused) or 1.0
        blended = [
            value - NEGATION_WEIGHT * match / ceiling
            for value, match in zip(blended, refused)
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


def unseen(
    catalog: catalog_module.Catalog,
    ordered: list[int],
    scores: list[float],
    shown: frozenset[str],
    size: int,
) -> tuple[list[int], list[float]]:
    """Drops products the customer has already been shown.

    Every slot spent on one is spent on a product the session has already
    rejected by not ending, so the ranking below it moves up and the
    exploration slots reach that much further down the pool.

    Falls back to the unfiltered order when too little survives to fill a
    slate. A short slate emits empty slots, which is a worse trade than a
    repeat, and `pad` would only refill them from popularity anyway.
    """
    if not SKIP_SHOWN or not shown:
        return ordered, scores
    kept = [
        (index, score)
        for index, score in zip(ordered, scores)
        if catalog.asins[index] not in shown
    ]
    if len(kept) < size:
        return ordered, scores
    return [index for index, _ in kept], [score for _, score in kept]


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
    timing, so a stage that only permutes bounds its downside on *those* at
    zero by construction rather than by good behaviour, and can move precision
    alone. Precision is not similarly protected: 180 of 200 public sessions
    already convert at rank 1, so a careless permutation has far more to lose
    there than a good one has to win (findings 3.36). A model-backed
    implementation drops in here without any other stage needing to know.
    """

    def __call__(
        self,
        catalog: catalog_module.Catalog,
        chosen: list[int],
        state: dialogue.SessionState,
    ) -> list[int]:
        ...


def reranked(
    catalog: catalog_module.Catalog,
    chosen: list[int],
    state: dialogue.SessionState,
    reranker: Reranker | None = None,
) -> list[int]:
    """Applies the offline rerank, then the model stage if one is running.

    The two compose rather than compete: the model is handed the phrase
    reranker's order and permutes that. It matters on the failure path, which
    is the common one. A model stage that *replaced* the offline one would
    make a timeout cost the phrase evidence as well as the model's judgement,
    so a turn the model declines would be served worse than the offline agent
    serves it. Composed, the worst a failed call can do is nothing.

    Whether the model runs is read from the module here rather than bound in a
    default argument, for the same reason `dense_weight` is: `make deviations`
    patches the module, and a bound default would no longer be readable.
    """
    ordered = rerank(catalog, chosen, state)
    if LLM_RERANK and reranker is not None:
        return reranker(catalog, ordered, state)
    return ordered


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
