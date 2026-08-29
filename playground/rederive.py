"""Recovers the ranking intermediates that `ranking.slate` discards.

`ranked` computes a BM25 term, a popularity term and a negation penalty for
every candidate, adds them, and returns only the sum (`ranking.py:375-421`).
`rerank` computes phrase evidence and returns only the permutation
(`ranking.py:738`). `unseen`, `explore` and `pad` each decide something about a
slot and keep none of it. All of that is exactly what a demo needs to show.

Rather than instrument the shipping path, this replays it. Every stage is a
pure function of `(catalog, state)`, so calling them again in the same order
with the same arguments must produce the same slate -- and that identity is
asserted rather than assumed. `stages()` returns `verified: False` the moment
its own reconstruction stops matching what the agent actually served, so the
screen can say the instrument is broken instead of showing a plausible lie.

Nothing here is imported by the agent. It reads `submission.src` and never
writes to it.
"""

from __future__ import annotations

from submission.src import ranking
from submission.src import text

# The blend is a sum of floats in a fixed order, so replaying it reproduces
# the bits exactly. This tolerance is for the reconstruction, which sums the
# same terms in the same order but through a different expression.
SCORE_TOLERANCE = 1e-9

# How far down the re-derived order the UI is willing to describe. The
# exploration band already reaches `ranking.WINDOW`, and a demo has no use for
# a longer tail than the band it is explaining.
DEPTH = ranking.WINDOW


def stages(agent, state, route, served, size: int) -> dict:
    """Replays `ranking.slate` and returns what each stage did.

    Args:
        agent: The live agent, for its catalog, profile ids and reranker.
        state: The session state `ranking.slate` was called with. This is the
          pre-`with_slate` object, which is why the caller has to be
          `Agent._record` and not anything downstream of it.
        route: The route that turn ran under.
        served: What `ranking.slate` returned.
        size: The slate size the turn ran at.
    """
    catalog = agent.catalog
    alpha = route.alpha
    dense_weight = _resolved(route.dense_weight, ranking.DENSE_WEIGHT)
    diversity = _resolved(route.diversity, ranking.DIVERSITY)
    profile_ids = ranking.personalised(state, agent.profile_ids)

    tokens = text.unique_tokens(state.query_text)
    query_ids = catalog.index.query_ids(tokens)
    negative_ids = catalog.index.query_ids(
        text.unique_tokens(state.excluded_text)
    )
    dense_query = ranking.encode(
        catalog, state.query_text, dense_weight > 0.0 or route.reach > 0
    )
    dense_negative = ranking.encode(
        catalog, state.excluded_text, ranking.DENSE_NEGATION_WEIGHT > 0.0
    )

    bucket_pool = catalog.pool(state.pool_keys)
    pool = ranking.widen(catalog, state, dense_query, route.reach)
    ordered, scores = ranking.ranked(
        catalog, pool, query_ids, alpha, profile_ids, negative_ids,
        dense_query, dense_negative, dense_weight,
    )
    contenders = ranking.contention(scores)
    kept, kept_scores = ranking.unseen(
        catalog, ordered, scores, state.shown, size
    )
    head = ranking.head_size(state, size, route.defer_turns, contenders)
    if diversity > 0.0 and ranking.worth_diversifying(state, kept_scores, size):
        chosen = ranking.diversify(
            catalog, kept, kept_scores, head, size, diversity
        )
    else:
        chosen = ranking.explore(catalog, kept, kept_scores, head, size)
    padded = ranking.pad(catalog, chosen, state.category, size)

    # Deliberately not `agent.reranker`: a model stage is a network call with
    # its own sampling, so replaying it would neither be free nor guaranteed to
    # agree. Offline it is `None` and this is the whole of `reranked`; with the
    # model on, the mismatch surfaces as `verified: False`, which is the honest
    # outcome rather than a second call pretending to be the first.
    final = ranking.reranked(catalog, padded, state, None)

    terms = _decompose(
        catalog, pool, query_ids, alpha, profile_ids, negative_ids,
        dense_query, dense_negative, dense_weight,
    )
    at = {index: position for position, index in enumerate(pool)}
    rank_of = {index: position for position, index in enumerate(kept)}

    return {
        "verified": list(final) == list(served.indices),
        "sizes": {
            "catalog": len(catalog.asins),
            "buckets": len(state.pool_keys),
            "bucket_pool": len(bucket_pool),
            "pool": len(pool),
            "reached": len(pool) - len(bucket_pool),
        },
        "query_tokens": _tokens(catalog, tokens, query_ids),
        "head": head,
        "contenders": contenders,
        "alpha": alpha,
        "slots": [
            _slot(catalog, index, position, terms, at, rank_of,
                  head, size, chosen, padded, served)
            for position, index in enumerate(final)
        ],
        "dropped_shown": _dropped(catalog, ordered, kept, state.shown),
        "rerank": _rerank(catalog, padded, final, state),
        "band": _band(catalog, kept, kept_scores, size),
        "goal": _goal(
            agent.index_of, agent.goal, catalog, ordered, scores, terms, at
        ),
    }


def _goal(
    index_of: dict, goal: str | None, catalog, ordered: list[int],
    scores: list[float], terms: dict, at: dict,
) -> dict | None:
    """Returns where a nominated product sits in the full ranking.

    A slate shows the top ten, which says nothing about a product at rank 40
    climbing to rank 12. Following one named product down the whole ordered
    pool is what makes a turn's disclosure visibly worth something, and in a
    replay that product is the session's own target.
    """
    if not goal:
        return None
    missing = {
        "asin": goal, "in_pool": False, "rank": None,
        "of": len(ordered), "score": None, "breakdown": {},
        "in_catalog": goal in index_of,
    }
    index = index_of.get(goal)
    if index is None:
        return missing
    try:
        rank = ordered.index(index)
    except ValueError:
        # In the catalog but outside this turn's candidate pool, which is what
        # a category filter is for. Worth saying plainly: the product cannot
        # be ranked at all until the conversation resolves a bucket holding it.
        return missing
    pooled = at[index]
    return {
        "asin": goal,
        "in_pool": True,
        "in_catalog": True,
        "rank": rank + 1,
        "of": len(ordered),
        "score": scores[rank],
        "breakdown": {
            name: values[pooled]
            for name, values in terms.items() if not name.startswith("_")
        },
    }


def _resolved(value: float | None, fallback: float) -> float:
    """Returns the weight a `None`-carrying route actually ran at.

    A route carrying `None` is deferring to the module, not switching the
    stage off, and `ranking.slate` resolves it the same way. Reading the two
    as one would report a live switch as dead on exactly the turn it mattered.
    """
    return fallback if value is None else value


def _decompose(
    catalog, pool, query_ids, alpha, profile_ids, negative_ids,
    dense_query, dense_negative, dense_weight,
) -> dict:
    """Returns each blend term separately, over the whole pool.

    Mirrors `ranking.ranked`'s arithmetic term for term. That duplication is
    the reason `stages` checks the terms sum back to the score the agent
    served: a divergence here has to be loud, because a score breakdown that
    does not add up is worse than none.
    """
    prior = catalog.prior
    max_prior = max(prior[index] for index in pool) or 1.0
    if query_ids:
        lexical = [catalog.index.score(index, query_ids) for index in pool]
    else:
        lexical = [0.0] * len(pool)
    max_lexical = max(lexical) or 1.0

    terms = {
        "bm25": [value / max_lexical for value in lexical],
        "prior": [alpha * prior[index] / max_prior for index in pool],
    }
    if dense_query is not None and dense_weight > 0.0:
        terms["dense"] = _weighted(
            [catalog.dense.score(index, dense_query) for index in pool],
            dense_weight,
        )
    if dense_negative is not None and ranking.DENSE_NEGATION_WEIGHT > 0.0:
        terms["dense_negation"] = _weighted(
            [catalog.dense.score(index, dense_negative) for index in pool],
            -ranking.DENSE_NEGATION_WEIGHT,
        )
    if profile_ids and ranking.PROFILE_WEIGHT > 0.0:
        terms["profile"] = _weighted(
            [catalog.index.score(index, profile_ids) for index in pool],
            ranking.PROFILE_WEIGHT,
        )
    if negative_ids and ranking.NEGATION_WEIGHT > 0.0:
        terms["negation"] = _weighted(
            [catalog.index.score(index, negative_ids) for index in pool],
            -ranking.NEGATION_WEIGHT,
        )
    terms["_max_lexical"] = max_lexical
    return terms


def _weighted(raw: list[float], weight: float) -> list[float]:
    """Returns one blend term, normalised by its own ceiling as `ranked` does."""
    ceiling = max(raw) or 1.0
    return [weight * value / ceiling for value in raw]


def _tokens(catalog, tokens: list[str], query_ids: frozenset[int]) -> list[dict]:
    """Returns each query word and whether the index has ever seen it.

    A word the catalog does not know contributes nothing and is worth showing
    as such: it is the difference between "the customer said it" and "the
    ranking heard it".
    """
    known = []
    for token in tokens:
        ids = catalog.index.query_ids([token])
        known.append({
            "token": token,
            "known": bool(ids),
            "id": min(ids) if ids else None,
        })
    return known


def _slot(
    catalog, index: int, position: int, terms: dict, at: dict, rank_of: dict,
    head: int, size: int, chosen: list[int], padded: list[int], served,
) -> dict:
    """Returns one served slot: where it came from and what it scored."""
    pooled = at.get(index)
    breakdown = {}
    total = 0.0
    if pooled is not None:
        for name, values in terms.items():
            if name.startswith("_"):
                continue
            breakdown[name] = values[pooled]
            total += values[pooled]

    reported = served.scores[position] if position < len(served.scores) else 0.0
    return {
        "asin": catalog.asins[index],
        "position": position,
        "rank": rank_of.get(index),
        "source": _source(index, position, head, chosen, padded),
        # Whether the blend ever scored this product. `pad` draws from the
        # coarser group, which overlaps the ranked pool, so a padded slot is
        # not the same thing as an unscored one: only a product from outside
        # the pool genuinely has no score rather than a low one.
        "pooled": pooled is not None,
        "score": reported,
        "breakdown": breakdown,
        "reconstructed": total,
        "agrees": pooled is None or abs(total - reported) < SCORE_TOLERANCE,
        "reviews_rank": None,
    }


def _source(
    index: int, position: int, head: int, chosen: list[int], padded: list[int]
) -> str:
    """Names which stage put a product in the slate.

    The three are not decoration. `head` is the commitment, `explore` is the
    band deferred commitment reaches into instead of spending ranks 2-10, and
    `pad` is a slot with no score at all rather than a low one.
    """
    if index not in chosen:
        return "pad"
    if chosen.index(index) < head:
        return "head"
    return "explore"


def _dropped(catalog, ordered: list[int], kept: list[int], shown) -> list[dict]:
    """Returns products `SKIP_SHOWN` removed, best first.

    Empty on turn one and on any turn the filter fell back, which is itself
    worth seeing: `unseen` keeps the unfiltered order when too little survives.
    """
    if not shown or ordered is kept:
        return []
    surviving = set(kept)
    return [
        {"asin": catalog.asins[index], "rank": position}
        for position, index in enumerate(ordered)
        if index not in surviving
    ][:DEPTH]


def _rerank(catalog, padded: list[int], final: list[int], state) -> dict:
    """Returns the phrase evidence and how far it moved each product."""
    phrase_ids = catalog.phrases.query_ids(state.constraints)
    before = {index: position for position, index in enumerate(padded)}
    return {
        "active": bool(phrase_ids) and list(padded) != list(final),
        "phrases": len(phrase_ids),
        "moves": [
            {
                "asin": catalog.asins[index],
                "from": before.get(index),
                "to": position,
                "evidence": catalog.phrases.evidence(index, phrase_ids),
            }
            for position, index in enumerate(final)
        ],
    }


def _band(catalog, kept: list[int], scores: list[float], size: int) -> list[dict]:
    """Returns the withheld ranks, which are the point of deferred commitment.

    Ranks 1..size-1 are the ones a later turn is most likely to promote, so
    spending an irreversible slot on them now is the trade the head size
    exists to refuse. Showing them beside the slate is the only way that
    decision is visible at all.
    """
    return [
        {
            "asin": catalog.asins[index],
            "rank": position,
            "score": scores[position],
        }
        for position, index in enumerate(kept[:size])
    ]
