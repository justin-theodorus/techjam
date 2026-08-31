"""The conversational shopping agent exported to the evaluator."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from submission.src import catalog as catalog_module
from submission.src import dialogue
from submission.src import llm
from submission.src import memory
from submission.src import orchestrate
from submission.src import outcome_tracker
from submission.src import policy as policy_module
from submission.src import probe
from submission.src import ranking
from submission.src import response
from submission.src import routing
from submission.src import text
from submission.src import understand


class Agent:
    """A shopping agent over a frozen catalog. Stdlib only, no network."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        fast_path: bool = True,
        persona_log_path: str | Path | None = None,
    ) -> None:
        """Builds every index once.

        Args:
            catalog_path: The frozen catalog.
            fast_path: Whether message reading may use its template shortcut.
              Off, the agent runs on cue detection and catalog vocabulary alone,
              which is how that path is verified rather than assumed.
            persona_log_path: Optional development-only JSONL outcome log.
              Omitted during scoring, so the official path performs no writes.
        """
        # Order matters: the reranker decides whether the catalog keeps the
        # product text it needs, and neither is built again afterwards.
        self._reranker = llm.build()
        self.catalog = catalog_module.build(catalog_path, cards=llm.wanted())
        self.debug: dict = {}
        self._fast_path = fast_path
        self._global_slate = tuple(
            self.catalog.slate_of(self.catalog.popular[:ranking.SLATE_SIZE])
        )
        self._session_id: str | None = None
        self._memory = memory.Store()
        self._state = dialogue.SessionState()
        self._parsed = dialogue.ParsedTurn()
        self._contenders = 0
        self._head = 0
        self._asked: str | None = None
        self._policy = policy_module.DISCOVERY
        # Which policy the turn *sounds* like, as opposed to which one it
        # retrieves under. The two differ only on a near-tie, and only while
        # `policy.HYBRID_FRAMING` is on; they are the same name otherwise.
        self._framing = policy_module.DISCOVERY
        self._workflow = orchestrate.shipped(policy_module.DISCOVERY)
        # `D_{t-1}`. `None` means "no previous turn to carry", which is what
        # makes the opening turn fall back to the derived prior rather than to
        # zero. Session-scoped: `reset` clears it, so one shopper's readiness
        # can never carry into a stranger's session.
        self._readiness: float | None = None
        self._profile_ids: frozenset[int] = frozenset()
        self._user_profile: dict = {}
        self._scores: tuple[float, ...] = ()
        self._usage = llm.no_usage()
        self._conversation_history: list[tuple[str, str]] = []
        self._outcome_tracker = outcome_tracker.OutcomeTracker(
            str(persona_log_path) if persona_log_path is not None else None
        )
        self._pending_persona: dict | None = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Starts a session. No I/O and no indexing happen here.

        800 private sessions re-parsing 50k JSONL records would blow any
        timeout, so everything expensive is built in `__init__`.

        Args:
            session_id: Opaque per-session identifier.
            user_profile: The anonymised aggregate. Only `preference_tags`
              carries product vocabulary; the rest is rating habits with no
              product in them.
        """
        self._profile_ids = self._tags_of(user_profile)
        self._user_profile = (
            dict(user_profile) if isinstance(user_profile, dict) else {}
        )
        self._session_id = session_id
        self._state = memory.seed(self._memory.recall())
        self._parsed = dialogue.ParsedTurn()
        self._contenders = 0
        self._head = 0
        self._asked = None
        self._policy = policy_module.DISCOVERY
        self._framing = policy_module.DISCOVERY
        self._workflow = orchestrate.shipped(policy_module.DISCOVERY)
        self._readiness = None
        self._usage = llm.no_usage()
        self.debug = {}
        self._conversation_history = []
        self._pending_persona = None

    def remember(self, shopper_id: str | None) -> None:
        """Names the shopper the next session belongs to.

        The published contract has no field for this: `reset_request` and
        `user_profile` are both closed with `additionalProperties: false` and
        `session_id` is a fresh uuid per session, so an identity can only reach
        the agent through a caller that has one. Nothing on the organizer's
        path does, which is what keeps the reported score untouched.

        Args:
            shopper_id: Who is shopping, or `None` for an anonymous session.
        """
        self._memory.remember(shopper_id)

    def forget(self) -> None:
        """Drops every remembered shopper.

        A measurement re-scores many sets against one agent instance, so the
        isolation between them has to be explicit or the first set's memory is
        reported as the second set's result.
        """
        self._memory.forget()

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        """Returns one turn's message, probe, and slate."""
        try:
            recommendations = self._serve(session_id, user_message, top_k)
            asked = probe.choose(
                self._state, self.catalog.taxonomy, self.catalog,
                self._policy,
            )
            candidate_count = len(self.catalog.pool(self._state.pool_keys))
            self._resolve_pending(user_message, candidate_count)
            persona_match = response.select_persona(
                self._state, user_message, self._conversation_history,
                candidate_count, self._user_profile,
            )
            message = response.compose(
                self._state, self._parsed, self._contenders,
                self._head, len(recommendations), asked,
                self._framing,
                probe.options(self._state, self.catalog, asked),
                self._names(recommendations),
                ranking.SLATE_SIZE,
            )

            self._conversation_history.append((user_message, message))
            self._pending_persona = {
                "turn": turn,
                "match": persona_match,
                "message": message,
                "constraints": list(self._state.constraints),
                "pool": self._contenders,
            }
            self.debug["persona"] = persona_match.persona_type.value
            self.debug["persona_conf"] = round(persona_match.confidence, 2)
        except Exception as error:
            # Deliberate isolation point. A caller that turns an exception into
            # an empty turn makes a crash cost a turn silently, so failure has
            # to degrade here instead of propagating. `message` must survive it
            # too: a non-string discards the recommendations with it.
            recommendations = self._degrade(error)
            asked = probe.WILDCARD
            self._policy = policy_module.DISCOVERY
            self._framing = policy_module.DISCOVERY
            message = response.FALLBACK
            self._scores = (0.0,) * len(recommendations)
        self._usage = self._take_usage()
        self._record_usage()
        self._asked = asked
        return {
            "message": message,
            "ask_attribute": asked,
            "recommendations": self._payload(recommendations),
            "usage": {
                "prompt_tokens": self._usage["prompt_tokens"],
                "completion_tokens": self._usage["completion_tokens"],
            },
        }

    def _resolve_pending(self, user_message: str, candidate_count: int) -> None:
        """Scores the previous persona using this turn's observable response."""
        pending = self._pending_persona
        if pending is None or self._session_id is None:
            return
        self._outcome_tracker.record_turn(
            session_id=self._session_id,
            turn=pending["turn"],
            persona_match=pending["match"],
            user_message=user_message,
            llm_question=pending["message"],
            constraints_before=pending["constraints"],
            constraints_after=list(self._state.constraints),
            user_rating_style=str(
                self._user_profile.get("rating_style", "unknown")
            ),
            products_before=pending["pool"],
            products_after=candidate_count,
        )

    def _take_usage(self) -> dict:
        """Returns the model usage this turn spent, honest zeros without one.

        Taken outside the envelope above, so a turn that degraded still
        reports whatever it had already spent before it did.
        """
        if self._reranker is None:
            return llm.no_usage()
        return self._reranker.take()

    def _record_usage(self) -> None:
        """Publishes the turn's model cost into the trace, flat.

        Skipped entirely without a model, so the offline trace keeps the exact
        key set every earlier run was read with.
        """
        if self._reranker is None:
            return
        self.debug["llm"] = self._reranker.model
        self.debug["calls"] = self._usage["calls"]
        self.debug["failures"] = self._usage["failures"]
        self.debug["tokens"] = (
            self._usage["prompt_tokens"] + self._usage["completion_tokens"]
        )
        self.debug["llm_ms"] = self._usage["milliseconds"]

    def _names(self, asins: tuple[str, ...]) -> tuple[str, ...]:
        """Returns the catalog titles behind a slate, for the reply to name.

        Read-only and reply-only: `ranking` never sees these, so a missing or
        empty title costs the customer a name and costs the score nothing.
        """
        titles = self.catalog.titles
        if not titles:
            return ()
        return tuple(titles.get(asin, "") for asin in asins)

    def _payload(self, asins: tuple[str, ...]) -> list[dict]:
        """Returns the recommendation list, scored where a score exists.

        Deliberately not a `zip`: every slot is a free chance at a hit, and a
        scores list that ever fell short of the slate would silently truncate
        it. Padded slots draw from outside the ranked pool and genuinely have
        no score, so zero is the honest value rather than a low one.
        """
        return [
            {
                "parent_asin": asin,
                "score": round(self._scores[position], 6)
                if position < len(self._scores) else 0.0,
            }
            for position, asin in enumerate(asins)
        ]

    def _serve(
        self, session_id: str, user_message: str, top_k: int
    ) -> tuple[str, ...]:
        """Reads the message, updates state, and ranks a fresh slate."""
        if session_id != self._session_id:
            # A turn for a session nobody opened means the caller lost track
            # of the boundary, so the identity it last named cannot be
            # vouched for either. Drop it alongside the profile: attaching one
            # shopper's memory to a stranger's session is the defect this
            # whole layer is most likely to produce.
            self._memory.remember(None)
            self.reset(session_id, {})
        parsed = understand.interpret(
            user_message, self.catalog.resolver, self._fast_path
        )
        parsed = self._preferred(parsed)
        state = dialogue.update(
            self._state, parsed, self._asked, self.catalog.taxonomy
        )
        size = ranking.SLATE_SIZE
        if isinstance(top_k, int) and top_k > 0:
            size = top_k

        candidate_count = len(self.catalog.pool(state.pool_keys))
        decision = policy_module.decide(
            state, candidate_count, self._contenders, self._readiness
        )
        policy = decision.name
        route = routing.choose(state, policy, decision=decision)
        workflow = orchestrate.choose(self.catalog, state, policy, route.alpha)
        served = ranking.slate(
            self.catalog, state, size,
            alpha=route.alpha,
            defer_turns=route.defer_turns,
            profile_ids=self._profile_ids,
            dense_weight=route.dense_weight,
            reach=route.reach,
            reranker=self._reranker,
            diversity=route.diversity,
            ordering=workflow.ordering,
            head_cap=route.head_cap,
        )
        asins = tuple(self.catalog.slate_of(served.indices))
        self._state = state.with_slate(asins)
        self._memory.observe(self._state)
        self._parsed = parsed
        self._contenders = served.contenders
        self._head = served.head
        self._scores = tuple(served.scores)
        self._policy = policy
        self._framing = policy_module.framing(decision)
        self._workflow = workflow
        # Carried into the next turn's recurrence. Written after the slate is
        # served, so a turn that raises before this point leaves the estimate
        # where it was rather than half-updated.
        self._readiness = decision.readiness
        self._record(state, parsed, route, served, asins)
        return asins

    def _visits(self) -> int:
        """How many earlier visits this shopper's record was built from."""
        recalled = self._memory.recall()
        return recalled.visits if recalled is not None else 0

    def _preferred(self, parsed: dialogue.ParsedTurn) -> dialogue.ParsedTurn:
        """Breaks an uncertain category read toward where this person shops.

        A single bucket is not a tie: `category.Resolver.buckets` returns one
        only when the message stated its name outright, and overriding that
        would be reading memory over the customer. Anything wider is a near-tie
        the resolver was about to break on coverage alone, so reordering it is
        the whole of the read. `dialogue` latches the category at first sight,
        so this can only ever move turn one.
        """
        weights = memory.affinity(self._memory.recall())
        if not weights or len(parsed.buckets) < 2:
            return parsed
        ordered = sorted(
            parsed.buckets, key=lambda key: -weights.get(key, 0.0)
        )
        if tuple(ordered) == parsed.buckets:
            return parsed
        return dataclasses.replace(
            parsed, buckets=tuple(ordered), category=ordered[0]
        )

    def _tags_of(self, user_profile: object) -> frozenset[int]:
        """Returns indexable token ids for the profile's preference tags."""
        if not isinstance(user_profile, dict):
            return frozenset()
        tags = user_profile.get("preference_tags")
        if not isinstance(tags, list):
            return frozenset()
        words = " ".join(str(tag) for tag in tags)
        return self.catalog.index.query_ids(text.unique_tokens(words))

    def _degrade(self, error: Exception) -> tuple[str, ...]:
        """Falls back to the previous slate, then the pool, then popularity.

        The last rung is precomputed in `__init__`, so it cannot itself fail.
        """
        self.debug = {"error": type(error).__name__, "degraded": "global"}
        if len(self._state.last_slate) >= ranking.SLATE_SIZE:
            self.debug["degraded"] = "last_slate"
            return self._state.last_slate
        pool = self.catalog.pool(self._state.pool_keys)
        if len(pool) >= ranking.SLATE_SIZE:
            self.debug["degraded"] = "pool"
            return tuple(self.catalog.slate_of(pool[:ranking.SLATE_SIZE]))
        return self._global_slate

    def _dense_of(self, route: routing.Route) -> float:
        """Returns the dense weight this turn actually ran at.

        A route carries `None` when it does not specialise, so the trace has
        to resolve it the same way `ranking.slate` does or it would report a
        switch as off while it was on.
        """
        if not self.catalog.dense:
            return 0.0
        if route.dense_weight is None:
            return ranking.DENSE_WEIGHT
        return route.dense_weight

    def _diversity_of(self, route: routing.Route) -> float:
        """Returns the diversity weight this turn actually ran at.

        Resolved the same way `_dense_of` is, and for the same reason: a route
        carrying `None` is deferring, not switched off, and a trace that read
        the two as one would report the wrong thing on the turn it mattered.
        """
        if route.diversity is None:
            return ranking.DIVERSITY
        return route.diversity

    def _record(
        self,
        state: dialogue.SessionState,
        parsed: dialogue.ParsedTurn,
        route: routing.Route,
        served: ranking.Served,
        asins: tuple[str, ...],
    ) -> None:
        """Publishes turn diagnostics for the harness trace.

        Flat scalars only: `harness/trace.py` clips each value to 40 characters.
        """
        self.debug = {
            "scenario": state.scenario,
            "policy": self._policy,
            "route": route.name,
            "act": parsed.act,
            "read": round(parsed.confidence, 2),
            "bucket": state.category or "-",
            "buckets": len(state.pool_keys),
            "pool": len(self.catalog.pool(state.pool_keys)),
            "turn": state.turn,
            "constraints": len(state.constraints),
            "slots": "/".join(
                sorted({slot.attribute for slot in state.slots})
            ) or "-",
            "superseded": len(state.superseded),
            "pivot": parsed.pivot,
            "exhausted": state.exhausted,
            "head": served.head,
            "contenders": served.contenders,
            "refused": state.excluded_text or "-",
            "declined": "/".join(state.declined) or "-",
            "idle": state.idle,
            "shown": len(state.shown),
            "visits": self._visits(),
            "carried": len(state.carried) + len(state.carried_arms),
            "alpha": route.alpha,
            "dense": self._dense_of(route),
            "reach": route.reach,
            "variety": self._diversity_of(route),
            "policy_conf": route.policy_confidence,
            "policy_margin": route.policy_margin,
            "readiness": route.decision_readiness,
            "runner_up": route.policy_runner_up or "-",
            "hybrid": route.policy_hybrid,
            "ordering": self._workflow.ordering,
            "switched": self._workflow.reason,
            "refuted": orchestrate.refuted(state),
            "top1": asins[0] if asins else "-",
        }
        for name, score in route.policy_scores:
            self.debug[f"p_{name}"] = score
