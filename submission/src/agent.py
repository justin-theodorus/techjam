"""The conversational shopping agent exported to the evaluator."""

from __future__ import annotations

from pathlib import Path

from techjam.submission.src import catalog as catalog_module
from techjam.submission.src import dialogue
from techjam.submission.src import llm
from techjam.submission.src import probe
from techjam.submission.src import ranking
from techjam.submission.src import response
from techjam.submission.src import routing
from techjam.submission.src import text
from techjam.submission.src import understand


class Agent:
    """A shopping agent over a frozen catalog. Stdlib only, no network."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        fast_path: bool = True,
    ) -> None:
        """Builds every index once.

        Args:
            catalog_path: The frozen catalog.
            fast_path: Whether message reading may use its template shortcut.
              Off, the agent runs on cue detection and catalog vocabulary alone,
              which is how that path is verified rather than assumed.
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
        self._state = dialogue.SessionState()
        self._parsed = dialogue.ParsedTurn()
        self._contenders = 0
        self._head = 0
        self._asked: str | None = None
        self._profile_ids: frozenset[int] = frozenset()
        self._scores: tuple[float, ...] = ()
        self._usage = llm.no_usage()

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
        self._session_id = session_id
        self._state = dialogue.SessionState()
        self._parsed = dialogue.ParsedTurn()
        self._contenders = 0
        self._head = 0
        self._asked = None
        self._usage = llm.no_usage()
        self.debug = {}

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        """Returns one turn's message, probe, and slate."""
        try:
            recommendations = self._serve(session_id, user_message, top_k)
            asked = probe.choose(
                self._state, self.catalog.taxonomy, self.catalog
            )
            message = response.compose(
                self._state, self._parsed, self._contenders,
                self._head, len(recommendations), asked,
            )
        except Exception as error:
            # Deliberate isolation point. A caller that turns an exception into
            # an empty turn makes a crash cost a turn silently, so failure has
            # to degrade here instead of propagating. `message` must survive it
            # too: a non-string discards the recommendations with it.
            recommendations = self._degrade(error)
            asked = probe.WILDCARD
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
            self.reset(session_id, {})
        parsed = understand.interpret(
            user_message, self.catalog.resolver, self._fast_path
        )
        state = dialogue.update(
            self._state, parsed, self._asked, self.catalog.taxonomy
        )
        size = ranking.SLATE_SIZE
        if isinstance(top_k, int) and top_k > 0:
            size = top_k

        route = routing.choose(state)
        served = ranking.slate(
            self.catalog, state, size,
            alpha=route.alpha,
            defer_turns=route.defer_turns,
            profile_ids=self._profile_ids,
            dense_weight=route.dense_weight,
            reach=route.reach,
            reranker=self._reranker,
            diversity=route.diversity,
        )
        asins = tuple(self.catalog.slate_of(served.indices))
        self._state = state.with_slate(asins)
        self._parsed = parsed
        self._contenders = served.contenders
        self._head = served.head
        self._scores = tuple(served.scores)
        self._record(state, parsed, route, served, asins)
        return asins

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
            "shown": len(state.shown),
            "alpha": route.alpha,
            "dense": self._dense_of(route),
            "reach": route.reach,
            "variety": self._diversity_of(route),
            "top1": asins[0] if asins else "-",
        }
