"""In-bucket ranking: the popularity and BM25 blend, the slate, the padding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from submission.src import catalog as catalog_module
from submission.src import dialogue
from submission.src import text

ALPHA = 0.6
SLATE_SIZE = 10

# How much the anonymised profile is allowed to move a ranking, and how much the
# customer must still have left unsaid for it to be consulted at all.
#
# The tags carry no person-specific information: ranking a target's own bucket
# by a *stranger's* tags scores the same as by its own (61.5% against 62.0%),
# and using all nine vocabulary words with no personalisation at all scores
# better than either (66.0%). What they actually measure is how much text a
# listing carries, which is a corrupted proxy for the popularity prior already
# in the blend (findings 3.28).
#
# So the weight is set by what it can displace rather than by what it knows.
# Inside a bucket the popularity-only scores are packed at a median adjacent gap
# of 0.0058, so an additive term of size `w` jumps a product about `w/0.0058`
# places: at 0.05 it re-sorts 92% of slates, at 0.2 it moves the top pick in 35%
# of sessions. At 0.02 the top pick survives 97% of the time and under one slot
# in ten changes.
#
# **Ships live at that weight, and deliberately not as a score claim.** Gated,
# it is +0.0002 in the mean over the 15 readable sets with 6 better and 2 worse,
# which is two hundred times below the 0.04 noise floor: it is measurably not
# harmful rather than measurably good. It ships because the brief asks the agent
# to use the anonymised profile and this is the configuration that does so
# without spending anything, not because it earns score (findings 3.43).
PROFILE_WEIGHT = 0.02

# Ungated, the same weight is monotonically negative and reaches -0.1125 on
# `silent_customer` (findings 3.30), because it perturbs turns that carry query
# evidence and there it displaces the customer's own words. With nothing
# disclosed there is no lexical half to displace and the term competes only with
# the prior. `-1` never gates, which is the configuration 3.30 rejected.
PROFILE_MAX_CONSTRAINTS = 0

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

# How much the customer must still have left unsaid for diversification to be
# worth a slot. `-1` never gates, which is the ungated sweep 3.30 ran; `0`
# diversifies only while the customer has named nothing, which is the browsing
# opening; higher values keep it running further into a session.
#
# The gate exists because 3.30's verdict was taken on one configuration and read
# as a verdict on the mechanism. Diversity won +0.0756 on `thin_cards` and lost
# 0.0542 on `negated_constraints`, and the losing sessions are the ones that had
# text to rank on, which a gate can exclude and the unconditional sweep could
# not. Measured at -1 and shipped there: gating shrinks the loss on every set it
# was expected to rescue without turning one positive (findings 3.43).
DIVERSITY_MAX_CONSTRAINTS = -1

# How undifferentiated the ranking must be before spreading the slate is worth
# a slot. `0.0` never vetoes, which is the shipped setting.
#
# The dialogue gate above asks what the customer has said; this asks what the
# ranking made of it, which is the same question from the other end and the one
# `thin_cards` actually answered. A session can hold four constraints the
# catalog has no words for, and a session can hold none while the popularity
# prior separates the bucket cleanly.
FLATNESS_GATE = 0.0

# How much the *exploration* slots below the committed head are chosen by
# marginal relevance rather than by a fixed rank offset. `0.0` restores
# `compose`'s fixed arithmetic, which is what every number before 3.45 was
# taken against.
#
# This is not `DIVERSITY` with a different number. That one selects over
# `ordered[:WINDOW]`, which spans the ranks deferred commitment deliberately
# withholds: a target sitting at rank 5 converts *later at rank 1* for a
# reciprocal rank of 1.0, and spending a slot on it now converts it at 0.2
# instead. Diversifying the whole head therefore cannibalises the withheld band,
# and measurement says that is the entire cost -- the loss is 100% MRR while
# MTTC *improves* (findings 3.45).
#
# Restricted to `ordered[SLATE_SIZE:WINDOW]` it cannot touch the withheld band
# at all. What it replaces is only the fixed reach to ranks 11-19, with a reach
# that goes deeper when the scores are flat and stays shallow when they are
# peaked. That is 3.30's own mechanism applied to the band where it costs
# nothing.
#
# **Ships live at 0.95, and it is the one change in this project that trades
# reported score for robustness.** It costs 0.0017 on the public 200 and gains
# on every pessimistic bound: size-biased +0.0083, sqrt +0.0059, uniform
# +0.0025, and the worst paraphrase column +0.0054. Across the 15 readable
# frozen sets it is +0.0031 in the mean, 10 better against 4 worse. The public
# set is 88% rank-1 and its rows are not evidence for a ranking change; the risk
# columns are what a differently-drawn private set would look like. This is the
# same trade `ALPHA` already makes by shipping at 0.6 rather than at the public
# optimum of 1.3, which collapses to 0.595 under uniform targets (findings
# 3.20, 3.45).
EXPLORE_DIVERSITY = 0.95

# Whether the slots below the committed head are filled at all.
#
# `head_size` withholds them because "a slate is a commitment"; `compose` then
# spent them anyway on ranks `size`..`2*size`, on the assumption that a rank no
# turn would otherwise reach is free to serve. It is not free. The evaluator
# breaks the session at the *first* turn the target appears anywhere in the ten
# and scores the rank it occupied then, so a target surfaced from the
# exploration band converts at the position the band put it in and the session
# never gets the turn that would have led it. Measured before the change, 13 of
# the 26 sessions converting below rank 1 were exactly this, every one of them
# with `head=1`, together worth 2.49 of reciprocal rank where converting each at
# rank 1 later is worth 13.
#
# The arithmetic is one-sided. A turn costs 0.02 of score through efficiency;
# moving one session from rank 7 to rank 1 is worth 0.257 through MRR, so an
# extra turn pays for itself at a rank gain of 0.067, and the budget is barely
# touched -- MAX_TURNS is 10 and the agent spent 2.16 of it.
#
# What it risked was coverage, weighted 0.5: a session withheld from has to
# convert later or not at all. It does. **Ships off, and it is the largest
# single gain in the project: +0.0111 on the public 200 (0.9473 to 0.9584) with
# hit@10 unmoved at 1.000, MRR 0.9018 to 0.9502 against MTTC 2.16 to 2.33.**
# Browsing, the column we were weakest on, moves 0.868 to 0.958 and buying 0.936
# to 0.965.
#
# It is not a public-set artefact, and every guarded column says so in the same
# direction: dev +0.0172 and held +0.0049; size-biased +0.0099, sqrt +0.0142,
# uniform and worst +0.0184; reworded +0.0081, punctuation +0.0092, filler
# +0.0137, synonym +0.0101. The pessimistic bounds gain *more* than the public
# set does, which is the opposite of what a tuned constant looks like -- the
# withheld band matters most exactly where the ranking is least sure.
#
# `EXPLORE_DIVERSITY` and `EXPLORE_SORT` only decide what goes in these slots,
# so both are dead while this is off. They are kept live because they are what
# the band reverts to if a scorer ever weights speed heavily enough to want it
# back (findings 3.27 measured that agent).
EXPLORE_FILL = False

# Whether the exploration slots are served in score order or in the order
# marginal relevance picked them. Membership decides hit@10 and MTTC and only
# position decides MRR, so this is a pure permutation and cannot cost coverage.
#
# Ships on. Marginal relevance picks in order of what each slot *adds*, which is
# not the order of what each is *worth*: sorting the picks back into score order
# is worth +0.0005 in the mean and wins at every weight measured.
EXPLORE_SORT = True

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
#
# **Re-measured after `EXPLORE_FILL` and moved 3 -> 6.** The cap was fitted when
# the withheld slots were still being spent on ranks 11-19, so letting it lapse
# cost little: the slate the session opened up to was one it had largely been
# serving anyway. Once withholding actually shortens the slate, the lapse is the
# whole commitment -- `head_size` returns the full `size` the moment
# `state.turn > defer_turns`, and at 3 that fired on exactly the sessions with
# the most turns left to convert in. `intent_override` runs 4.2 turns and was
# taking the full slate from turn 4 onward, which is why it was the one column
# that did not move when `EXPLORE_FILL` shipped.
#
# Worth +0.0049 on the public 200 (0.9584 to 0.9633) with hit@10 unmoved at
# 1.000: MRR 0.9497 to 0.9717 against MTTC 2.33 to 2.41. `intent_override` moves
# 0.896 to 0.983 and `boundary` 0.920 to 1.000; no column regresses.
#
# Six rather than five because the surface is a plateau, not a peak: 6, 7, 8 and
# 10 all score 0.9633 and 5 scores 0.9627, so this is the first point that buys
# the whole gain and nothing past it changes a decision. Guarded the same way
# `EXPLORE_FILL` was -- dev +0.0040 and held +0.0059; sqrt +0.0019, uniform and
# worst +0.0021, size-biased flat; reworded +0.0073, punctuation +0.0016, filler
# +0.0031, synonym +0.0052. Held again gains more than dev.
MAX_DEFER_TURNS = 6

# The named orderings a route may serve from. `BLEND` is the shipped one and
# every reported number is taken on it; the rest exist so that a session whose
# blend head has been served and disproven has somewhere to switch to. See
# `submission/src/orchestrate.py` for what decides, and findings 3.50 for what
# each one is worth.
BLEND = "blend"
LEXICAL = "lexical"
PRIOR = "prior"
PHRASE = "phrase"
ORDERINGS = (BLEND, LEXICAL, PRIOR, PHRASE)


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
    diversity: float | None = None,
    reranker: Reranker | None = None,
    ordering: str = BLEND,
    head_cap: int | None = None,
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
    if diversity is None:
        diversity = DIVERSITY
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
    if ordering == BLEND:
        ordered, scores = ranked(
            catalog, pool, query_ids, alpha, personalised(state, profile_ids),
            negative_ids, dense_query, dense_negative, dense_weight,
        )
    else:
        ordered, scores = alternative(ordering, catalog, pool, state, alpha)
    contenders = contention(scores)
    ordered, scores = unseen(catalog, ordered, scores, state.shown, size)
    head = head_size(state, size, defer_turns, contenders, head_cap)
    if diversity > 0.0 and worth_diversifying(state, scores, size):
        chosen = diversify(catalog, ordered, scores, head, size, diversity)
    else:
        chosen = explore(catalog, ordered, scores, head, size)
    # `pad` exists for a bucket too small to fill ten slots, where the empty
    # slots are genuinely free. A slot the head deliberately withheld is not
    # that: refilling it from the coarse group would swap a deep in-bucket
    # candidate for a shallower out-of-bucket one, which is worse on both
    # counts. So withholding shortens the slate rather than redirecting it.
    served_size = head if (not EXPLORE_FILL and head < size) else size
    padded = pad(catalog, chosen, state.category, served_size)
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


def alternative(
    name: str,
    catalog: catalog_module.Catalog,
    pool: tuple[int, ...],
    state: dialogue.SessionState,
    alpha: float = ALPHA,
) -> tuple[list[int], list[float]]:
    """Returns `pool` under a named ordering other than the blend.

    Each one drops a signal the blend leans on, so a session the blend has
    already been proven wrong about is re-sorted by evidence it was not using.
    Refusals stay in all of them: a refusal is something the customer said,
    not a retrieval signal a route chooses between.
    """
    negative_ids = catalog.index.query_ids(
        text.unique_tokens(state.excluded_text)
    )
    if name == PHRASE:
        return phrase_ordered(catalog, pool, state)
    if name == PRIOR:
        return ranked(
            catalog, pool, frozenset(), alpha, negative_ids=negative_ids
        )
    if name == LEXICAL:
        query_ids = catalog.index.query_ids(
            text.unique_tokens(state.query_text)
        )
        return ranked(
            catalog, pool, query_ids, 0.0, negative_ids=negative_ids
        )
    raise ValueError(f"unknown ordering: {name}")


def phrase_ordered(
    catalog: catalog_module.Catalog,
    pool: tuple[int, ...],
    state: dialogue.SessionState,
) -> tuple[list[int], list[float]]:
    """Orders `pool` by rare whole-phrase evidence alone.

    `rerank` applies the same evidence to the served ten, where it can only
    permute what the blend already chose. Over the pool it can *reach* a
    product no turn would otherwise serve, which is the only reason it exists.

    Findings 3.1 measured this route as a session's *primary* retriever and it
    lost roughly 40% of its hit rate under mild rewording. It is here as a
    fallback instead, and with no phrase evidence it returns the pool as it
    arrived, which is the popularity order: it degrades to the prior rather
    than to noise, which is the difference 3.1 could not see.
    """
    phrase_ids = catalog.phrases.query_ids(state.constraints)
    if not phrase_ids:
        return list(pool), [0.0] * len(pool)
    evidence = [
        catalog.phrases.evidence(index, phrase_ids) for index in pool
    ]
    positions = sorted(range(len(pool)), key=lambda i: -evidence[i])
    return [pool[p] for p in positions], [evidence[p] for p in positions]


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
    head_cap: int | None = None,
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
    return min(HEAD_SIZE if head_cap is None else head_cap, size)


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
    if not EXPLORE_FILL:
        return list(ordered[:head])
    return list(ordered[:head]) + list(ordered[size:size + size - head])


def personalised(
    state: dialogue.SessionState, profile_ids: frozenset[int]
) -> frozenset[int]:
    """Returns the profile tags this turn may rank with, empty when gated out.

    Withholding the ids rather than zeroing the weight keeps the gate in one
    place: `ranked` already no-ops on an empty set, so nothing downstream needs
    to know the profile can be switched off per turn.
    """
    if PROFILE_MAX_CONSTRAINTS < 0:
        return profile_ids
    if len(state.constraints) > PROFILE_MAX_CONSTRAINTS:
        return frozenset()
    return profile_ids


def flatness(scores: list[float], depth: int = SLATE_SIZE) -> float:
    """Returns how little the ranking separates the depth it is about to serve.

    The deepest score the slate reaches, over the best one. 1.0 is a ranking
    that cannot tell its own top `depth` apart; a small value is one that has
    already decided.

    Deliberately not a second reading of `contention`, and that distinction is
    why this exists. `contention` counts how many products sit within a fixed
    margin of the leader, and on the blended score that is p50 1 and max 4 on
    every set measured, because the popularity prior separates the head sharply
    even where the text does not. Counting lexical-only flatness instead does
    vary, but its median is 1 whether the target sits at rank 5 or rank 90, so
    it carries no depth information (findings 3.34). A ratio across the served
    depth is neither of those, and it does spread.

    A non-positive leader has no scale to be a ratio of. `ranked` subtracts for
    refused attributes and can push a whole pool below zero, where -2.0 / -1.0
    would read as twice as flat as flat. Those turns report 0.0, maximally
    separated, which is the reading that leaves the shipped slate alone.
    """
    if not scores or scores[0] <= 0.0:
        return 0.0
    return max(0.0, scores[min(depth, len(scores)) - 1] / scores[0])


def worth_diversifying(
    state: dialogue.SessionState, scores: list[float], size: int
) -> bool:
    """Whether this turn's slate is undifferentiated enough to spread.

    Two independent vetoes, both disabled, and neither can switch spreading on:
    a zero weight stays zero whatever they say. Both are read from the module
    here rather than taken as defaulted arguments, for the same reason
    `dense_weight` is resolved inside `slate` (findings 3.27).
    """
    if 0 <= DIVERSITY_MAX_CONSTRAINTS < len(state.constraints):
        return False
    if FLATNESS_GATE <= 0.0:
        return True
    return flatness(scores, size) >= FLATNESS_GATE


def explore(
    catalog: catalog_module.Catalog,
    ordered: list[int],
    scores: list[float],
    head: int,
    size: int,
) -> list[int]:
    """Commits the head, then fills the rest from past the slate.

    The same shape as `compose` and the same withheld band; the only question
    is whether the exploration slots are the fixed ranks `size`..`2*size` or a
    marginal-relevance selection over `size`..`WINDOW`. Falls back to `compose`
    whenever the band is too shallow to choose from, so the arithmetic path
    stays the one every earlier number was taken against.
    """
    if not EXPLORE_FILL or EXPLORE_DIVERSITY <= 0.0 or head >= size:
        return compose(ordered, head, size)
    band = ordered[size:WINDOW]
    slots = size - head
    if len(band) < slots:
        return compose(ordered, head, size)
    picked = diversify(
        catalog, band, scores[size:WINDOW], 0, slots, EXPLORE_DIVERSITY
    )
    if EXPLORE_SORT:
        rank = {index: position for position, index in enumerate(ordered)}
        picked = sorted(picked, key=lambda index: rank[index])
    return list(ordered[:head]) + list(picked)


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
    # Overlap is measured against what has actually been committed. With an
    # empty head there is nothing to repeat yet, so seeding against `window[0]`
    # would penalise the best candidate for resembling itself and hand the
    # first slot to the second-best.
    if chosen:
        overlap = [_similarity(tokens[i], tokens[0])
                   for i in range(len(window))]
    else:
        overlap = [0.0] * len(window)
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
