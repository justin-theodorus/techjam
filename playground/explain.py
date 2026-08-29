"""An agent that also says why, for the screen. Never on the scoring path.

`Agent` publishes a flat `debug` dict of about thirty scalars, because
`harness/trace.py` renders one terminal line per turn and clips every value at
40 characters. That is the right shape for a trace and the wrong shape for a
demo: it carries `pool=329` but not which 329, `head=1` but not what the other
nine slots were spent on, and `alpha=0.6` but not what the blend it weights
actually did to any particular product.

This subclass adds a second, nested channel beside it. It changes nothing the
evaluator can read: `submission/src` is untouched, the overrides call `super()`
first and only then look at what came back, and no organizer path ever
constructs this class.

Two methods, and the choice of those two is the whole design:

  `_record`  runs at the end of `_serve` and is handed the state object
             `ranking.slate` was actually called with -- the one before
             `with_slate` replaces it. Nothing downstream still has it.
  `respond`  is where the probe and the message are decided, after `_serve`
             has returned, so they are only readable from out here.
"""

from __future__ import annotations

from playground import rederive
from submission.src import agent as agent_module
from submission.src import dialogue
from submission.src import policy as policy_module
from submission.src import probe
from submission.src import ranking
from submission.src import response
from submission.src import slots as slots_module
from submission.src import understand

# The two arms `evaluator.local_evaluator.classify_constraint` can never
# return, so a question naming either is answered "I don't have an additional
# preference" whatever the catalog says about it (findings 3.37).
DEAD_ARMS = (slots_module.BRAND, slots_module.CATEGORY)


class ExplainingAgent(agent_module.Agent):
    """The shipped agent, plus a nested account of each turn."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.explain: dict = {}
        self.goal: str | None = None
        self._top_k = ranking.SLATE_SIZE
        self._recorded = False
        # `Catalog` keeps `ids` as a membership set and never needs to go the
        # other way; following one named product down the ranking does.
        self.index_of = {
            asin: index for index, asin in enumerate(self.catalog.asins)
        }

    @property
    def profile_ids(self) -> frozenset[int]:
        """The profile tags this session ranks with, for the replay."""
        return self._profile_ids

    @property
    def state(self) -> dialogue.SessionState:
        """The session state after the last turn was folded in."""
        return self._state

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        """Serves the turn, then explains the half `_record` cannot see."""
        self._top_k = top_k if isinstance(top_k, int) and top_k > 0 else (
            ranking.SLATE_SIZE
        )
        self._recorded = False
        self.explain = {}
        served = super().respond(session_id, user_message, turn, top_k)
        if not self._recorded:
            # `_serve` raised and `_degrade` served the fallback, so there are
            # no stages to describe. Say that rather than leave the panel
            # showing the previous turn's numbers.
            self.explain = {"degraded": True, "debug": dict(self.debug)}
            return served
        self.explain["probe"] = self._probe(served.get("ask_attribute"))
        self.explain["message"] = self._message(served, len(
            served.get("recommendations") or ()
        ))
        self.explain["debug"] = dict(self.debug)
        return served

    def _record(self, state, parsed, route, served, asins) -> None:
        """Publishes the flat trace, then the nested one beside it."""
        super()._record(state, parsed, route, served, asins)
        self._recorded = True
        self.explain = {
            "degraded": False,
            "understand": _understand(parsed),
            "state": _state(state),
            "policy": _policy(state, self._policy),
            "route": _route(route),
            "ranking": rederive.stages(
                self, state, route, served, self._top_k
            ),
        }

    def _probe(self, asked: object) -> dict:
        """Returns the arm table, the choice, and why the choice was made.

        Replays `probe.choose`'s ladder rather than reporting only its answer:
        three of its four exits return `None` or the wildcard for reasons that
        have nothing to do with which arm scores best, and a table shown
        without them would suggest the argmax was consulted when it was not.
        """
        state = self._state
        table = _arm_table(state, self.catalog)
        silent = probe.COVERAGE_SILENCE and state.turn >= probe.FINAL_TURN
        if state.exhausted:
            reason = "customer is out of preferences"
        elif silent:
            reason = f"turn {state.turn} has no turn left to spend an answer"
        elif asked == probe.WILDCARD:
            reason = "no arm clears the wildcard fallback ratio"
        else:
            reason = "best expected disclosure over the live pool"
        return {
            "asked": asked,
            "reason": reason,
            "specific_arms": probe.SPECIFIC_ARMS,
            "fallback_ratio": probe.WILDCARD_FALLBACK_RATIO,
            "table": table,
            "options": list(
                probe.options(state, self.catalog, asked)
                if isinstance(asked, str) else ()
            ),
            "yield": probe.expected_yield(state, self.catalog.taxonomy),
        }

    def _message(self, served: dict, count: int) -> dict:
        """Returns the message split back into the three parts that made it.

        `response.compose` joins them with a space and keeps none of the
        seams, so the only way to label them is to call the same helpers
        again with the same arguments.
        """
        asked = served.get("ask_attribute")
        options = probe.options(self._state, self.catalog, asked) if (
            isinstance(asked, str)
        ) else ()
        return {
            "full": served.get("message"),
            "acknowledge": response._acknowledge(self._state, self._parsed),
            "slate": response._slate(self._contenders, self._head, count),
            "question": response._question(
                self._state, self._parsed, asked, self._policy, options
            ),
        }


def _understand(parsed) -> dict:
    """What one message was read as, before any state was folded in."""
    return {
        "act": parsed.act,
        "confidence": parsed.confidence,
        "exact": parsed.confidence >= understand.EXACT,
        "category": parsed.category,
        "buckets": list(parsed.buckets),
        "constraints": list(parsed.constraints),
        "pivot": parsed.pivot,
        "boundary_refusal": parsed.boundary_refusal,
        "exhausted": parsed.exhausted,
        "exhausted_arm": parsed.exhausted_arm,
        "scenario_hint": parsed.scenario_hint,
    }


def _state(state) -> dict:
    """The accumulated session, with the two refusal kinds kept apart.

    `refused` is a value the customer rejected and `declined` is a dimension
    they would not discuss. Reading one as the other routed 74% of hard-set
    turns to the boundary branch (findings 3.47), so they never share a panel.
    """
    return {
        "turn": state.turn,
        "scenario": state.scenario,
        "category": state.category,
        "buckets": list(state.buckets),
        "slots": [
            {
                "attribute": slot.attribute,
                "value": slot.value,
                "turn": slot.turn,
                "negated": slot.negated,
            }
            for slot in state.slots
        ],
        "constraints": list(state.constraints),
        "superseded": list(state.superseded),
        "declined": list(state.declined),
        "spent_arms": list(state.refused),
        "refused_text": state.excluded_text,
        "query_text": state.query_text,
        "pivoted": state.pivoted,
        "pivot_turn": state.pivot_turn,
        "exhausted": state.exhausted,
        "idle": state.idle,
        "shown": len(state.shown),
        "carried": list(state.carried),
        "carried_arms": list(state.carried_arms),
    }


def _policy(state, chosen: str) -> dict:
    """The six-name ladder, with the rung that fired and the ones not reached.

    The order is the design (`policy.py:49-72`), so the panel shows the order
    rather than only its answer.
    """
    ladder = [
        (policy_module.RECOVERY, "the customer redirected", state.pivoted),
        (
            policy_module.COVERAGE,
            f"nothing left to learn, or turn >= {policy_module.COVERAGE_TURN}",
            bool(state.exhausted or state.turn >= policy_module.COVERAGE_TURN),
        ),
        (
            policy_module.STAGNATION,
            f"{dialogue.STAGNATION_TURNS} answers added nothing",
            state.idle >= dialogue.STAGNATION_TURNS,
        ),
        (
            policy_module.BOUNDARY,
            "boundary scenario, or an attribute was declined",
            bool(state.scenario == dialogue.BOUNDARY or state.declined),
        ),
        (
            policy_module.PRECISION,
            f"at least {policy_module.MIN_PRECISION_CONSTRAINTS} constraints",
            len(state.constraints)
            >= policy_module.MIN_PRECISION_CONSTRAINTS,
        ),
        (policy_module.DISCOVERY, "nothing above applied", True),
    ]

    rungs = []
    decided = False
    for name, test, holds in ladder:
        rungs.append({
            "policy": name,
            "test": test,
            "holds": holds,
            "state": "skipped" if decided else ("fired" if holds else "passed"),
        })
        decided = decided or holds
    return {"chosen": chosen, "ladder": rungs}


def _route(route) -> dict:
    """The retrieval constants this turn ran at.

    Every route ships at the shared constants because each attempt to
    differentiate them measured negative (findings 3.26, 3.30, 3.35, 3.43),
    so the panel marks a value as shared rather than implying it was chosen.
    """
    return {
        "name": route.name,
        "alpha": route.alpha,
        "defer_turns": route.defer_turns,
        "dense_weight": route.dense_weight,
        "reach": route.reach,
        "diversity": route.diversity,
        "specialised": [
            field for field, value in (
                ("dense_weight", route.dense_weight),
                ("diversity", route.diversity),
            ) if value is not None
        ] + (["reach"] if route.reach else []),
    }


def _arm_table(state, catalog) -> list[dict]:
    """Scores every probe arm the way `probe.specific` does.

    Mirrors `probe.py:236-278` term for term: coverage over the live pool,
    spread across the values that pool offers, and decay in what the customer
    has already said. Reproduced rather than read because `specific` keeps
    only its argmax.
    """
    pool = catalog.pool(state.pool_keys)[:probe.POOL_SIZE]
    if not pool:
        return []

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

    rows = []
    for arm in probe.ARMS:
        blocked = None
        if arm in state.declined:
            blocked = "declined to discuss it"
        elif arm in state.refused:
            blocked = "spent: nothing further to say about it"
        elif arm in state.carried_arms:
            blocked = "remembered as declined"
        elif not coverage.get(arm):
            blocked = "nothing in the pool leads with it"
        spread = probe._spread(weights[arm]) if arm in weights else 0.0
        rows.append({
            "arm": arm,
            "dead": arm in DEAD_ARMS,
            "blocked": blocked,
            "coverage": coverage.get(arm, 0.0) / len(pool),
            "spread": spread,
            "heard": heard.get(arm, 0),
            "decay": probe.ARM_DECAY ** heard.get(arm, 0),
            "share": coverage.get(arm, 0.0) / union if union else 0.0,
            "score": 0.0 if blocked else (
                (coverage[arm] / len(pool)) * spread
                * probe.ARM_DECAY ** heard.get(arm, 0)
            ),
        })
    rows.sort(key=lambda row: -row["score"])
    return rows
