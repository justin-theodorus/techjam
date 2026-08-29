"""Tier 2: the model-backed ranking stage behind `ranking.Reranker`.

The one stage the brief names that the offline agent does not implement.
Pillar I's pipeline base is *multi-route retrieval -> LLM semantic ranking*,
and everything up to that arrow ships; this is the arrow.

Three properties make it safe to have at all, and all three are structural
rather than behavioural:

  - **It permutes.** The stage receives the ten products already chosen and
    returns an ordering of exactly those ten. A session ends at the first turn
    the target appears anywhere in the slate, so membership fixes HitRate and
    MTTC; only the position, and therefore MRR, can move. `_permute` enforces
    this against *any* model output, including no output at all.
  - **Absence is not an error.** No `USE_LLM`, no `anthropic` package, no
    credential, no network, a timeout, a refusal, a malformed reply: every one
    of them returns the slate unchanged. The stage can never cost a session.
  - **It is off unless asked twice.** `build` returns `None` unless the
    environment opts in, and `ranking.LLM_RERANK` decides whether a built
    stage is consulted. The scored offline configuration reads neither the
    key nor the network.

Token counts are accumulated per turn and surface through the agent's `usage`
field, which the evaluator already sums into `reported_token_usage`. Cost is
computed from those counts and the published rates below, so the disclosure is
arithmetic over a measurement rather than a figure typed into a README.
"""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Sequence

from submission.src import catalog as catalog_module
from submission.src import dialogue

# The opt-in. Anything but this exact value leaves Tier 2 unbuilt, so a stray
# `USE_LLM=0` or `USE_LLM=false` fails closed rather than open.
ENV_SWITCH = "USE_LLM"
ENV_ON = "1"

# Ranking ten short product cards against at most four constraints is a small
# reading task, and the disclosure this phase owes is latency and cost. Haiku
# is the model that answers it without making the Feasibility line worse.
MODEL = "claude-haiku-4-5"

# Published rates for the model above, US dollars per million tokens. Used
# only to turn measured counts into the disclosed cost.
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00

# The reply is a permutation of ten indices under a schema, so the ceiling is
# generous rather than tuned. It exists to bound a runaway, not to shape output.
MAX_TOKENS = 512

# A slow turn is worse than an unranked one: the rules warn that timeouts may
# count as a miss, and the offline stage is waiting behind this. Two attempts
# at four seconds bounds a turn's added latency below the point where giving up
# and serving the lexical order is the better trade.
TIMEOUT_S = 4.0
MAX_RETRIES = 1

# How much of a product's own text the prompt quotes. Ten cards at this width
# is roughly a thousand input tokens, which is what the cost line is built on.
MAX_CARD_CHARS = 220

SYSTEM = (
    "You rank products for an online shopper. You are given what the shopper "
    "has told us they want, and a numbered list of candidate products already "
    "retrieved for them.\n\n"
    "Return the same candidate numbers, every one of them exactly once, "
    "reordered so the product most likely to be the one the shopper is "
    "describing comes first. Judge on how well each product's own text "
    "satisfies the stated wants, including wants expressed indirectly. "
    "Treat anything listed as refused as a reason to rank a product lower.\n\n"
    "Do not add, drop, or invent candidate numbers. If the evidence does not "
    "separate two products, keep their existing relative order."
)

# `output_config.format` constrains the reply to this shape, so parsing is a
# `json.loads` rather than a scrape. `_permute` still repairs the result: a
# schema constrains the shape of the answer, never its truthfulness.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["order"],
    "additionalProperties": False,
}


def wanted() -> bool:
    """Whether this run has opted into Tier 2.

    Read before the catalog is built, because the product text the prompt
    quotes is only retained when the answer is yes.
    """
    return os.environ.get(ENV_SWITCH) == ENV_ON


def build(model: str = MODEL) -> LLMReranker | None:
    """Returns the model-backed stage, or `None` if it does not apply here.

    Never raises, and never touches the network. A bundle run without the
    opt-in, without the `anthropic` package, or without a resolvable
    credential must behave exactly as the offline agent does rather than fail
    at construction, so every one of those is a `None`.
    """
    if not wanted():
        return None
    try:
        # Deliberately lazy and deliberately local. The scored configuration
        # never reaches this line, which is what keeps the offline path free
        # of any third-party import.
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(
            timeout=TIMEOUT_S, max_retries=MAX_RETRIES
        )
    except Exception:
        # Isolation point: credential resolution and client construction are
        # the SDK's business, and every way they can fail means the same thing
        # here, which is that Tier 2 is not available on this machine.
        return None
    if not _credentialed(client):
        return None
    return LLMReranker(client, model)


def _credentialed(client: Any) -> bool:
    """Whether the constructed client actually resolved a credential.

    The SDK defers authentication to request time, so a client built with no
    key at all constructs happily and then raises on every call. Left alone
    that is safe but wasteful and misleading: the stage would report several
    hundred failures rather than simply not existing. Checked here so a
    keyless run is Tier 2 being *absent*, which is what the docstring above
    promises and what the offline configuration already is.
    """
    return any(
        getattr(client, field, None)
        for field in ("api_key", "auth_token", "_credentials")
    )


class LLMReranker:
    """The model-backed rerank stage. One call per turn, at most."""

    def __init__(self, client: Any, model: str = MODEL) -> None:
        self.model = model
        self._client = client
        self._reset_counts()

    def __call__(
        self,
        catalog: catalog_module.Catalog,
        chosen: list[int],
        state: dialogue.SessionState,
    ) -> list[int]:
        """Returns a permutation of `chosen`, model-ordered where it can be.

        Silent about its own failures by design. A turn the model cannot help
        with, for any reason, is a turn served in the blend's order, and the
        counters below are what makes that visible in the trace rather than
        invisible in the score.
        """
        if not state.constraints or len(chosen) < 2:
            return chosen
        cards = _cards(catalog, chosen)
        if cards is None:
            return chosen

        started = perf_counter()
        try:
            order = self._ask(_prompt(state, cards))
        except Exception:
            # Isolation point. The recording proxy re-raises, and the agent's
            # envelope would turn one network hiccup into a lost turn, so the
            # degradation has to happen here and be counted.
            self.failures += 1
            order = ()
        self.milliseconds += (perf_counter() - started) * 1000.0
        return _permute(chosen, order)

    def _ask(self, prompt: str) -> tuple[int, ...]:
        """Returns the model's ordering as one-based candidate numbers."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {
                "type": "json_schema",
                "schema": SCHEMA,
            }},
        )
        self._count(response)
        text = next(
            (block.text for block in response.content
             if getattr(block, "type", None) == "text"),
            "",
        )
        order = json.loads(text).get("order")
        if not isinstance(order, list):
            return ()
        return tuple(
            value for value in order
            if isinstance(value, int) and not isinstance(value, bool)
        )

    def _count(self, response: Any) -> None:
        """Accumulates one reply's reported usage.

        `input_tokens` counts only what was not served from cache, so the
        honest prompt total is that plus both cache figures. Reporting the
        uncached count alone would understate what we sent.
        """
        usage = getattr(response, "usage", None)
        cached = _tokens(usage, "cache_read_input_tokens")
        self.calls += 1
        self.prompt_tokens += (
            _tokens(usage, "input_tokens")
            + _tokens(usage, "cache_creation_input_tokens")
            + cached
        )
        self.cached_tokens += cached
        self.completion_tokens += _tokens(usage, "output_tokens")

    def take(self) -> dict:
        """Returns the counts accumulated since the last call, and clears them.

        A fresh dict every time: the harness snapshots the agent's `debug`
        shallowly, so a reused mapping would rewrite every earlier turn.
        """
        counts = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "calls": self.calls,
            "failures": self.failures,
            "milliseconds": round(self.milliseconds, 3),
        }
        self._reset_counts()
        return counts

    def _reset_counts(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.calls = 0
        self.failures = 0
        self.milliseconds = 0.0


def no_usage() -> dict:
    """Returns the counts a turn that ran no model honestly reports."""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "calls": 0,
        "failures": 0,
        "milliseconds": 0.0,
    }


def cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Returns the dollar cost of the counts, at the published rates."""
    return (
        prompt_tokens * PRICE_IN_PER_MTOK / 1_000_000.0
        + completion_tokens * PRICE_OUT_PER_MTOK / 1_000_000.0
    )


def _tokens(usage: Any, field: str) -> int:
    """Returns one usage field as a non-negative integer.

    The evaluator sums only `int` values and silently drops everything else,
    so a count that arrives as `None` or a float has to be made honest here
    rather than reported as a gap.
    """
    value = getattr(usage, field, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _cards(
    catalog: catalog_module.Catalog, chosen: Sequence[int]
) -> list[str] | None:
    """Returns one short description per candidate, or `None` if there is none.

    The catalog retains product text only when Tier 2 asked for it at
    construction, so a stage built after the fact has nothing to quote and
    says so instead of prompting with bare identifiers.
    """
    if not catalog.cards:
        return None
    return [catalog.cards[index][:MAX_CARD_CHARS] for index in chosen]


def _prompt(state: dialogue.SessionState, cards: Sequence[str]) -> str:
    """Composes the turn's user message from state and candidate text."""
    lines = []
    if state.category:
        lines.append(f"Shopping in: {state.category}")
    lines.append("Wants: " + "; ".join(state.constraints))
    if state.excluded_text:
        lines.append(f"Refused: {state.excluded_text}")
    lines.append("")
    lines.append("Candidates:")
    lines.extend(
        f"{number}. {card}" for number, card in enumerate(cards, start=1)
    )
    return "\n".join(lines)


def _permute(chosen: list[int], order: Sequence[int]) -> list[int]:
    """Applies a one-based ordering to `chosen`, repairing whatever is wrong.

    The entire safety argument for the stage lives here. Numbers outside the
    slate, repeats, omissions and an empty reply are all ordinary model
    behaviour, and none of them may change *which* products are served: what
    the model got right is honoured in the order it gave, and everything it
    left out keeps the blend's order behind that. The result is a permutation
    of `chosen` for every possible input.
    """
    taken: set[int] = set()
    ranked = []
    for number in order:
        position = number - 1
        if 0 <= position < len(chosen) and position not in taken:
            taken.add(position)
            ranked.append(chosen[position])
    ranked.extend(
        index for position, index in enumerate(chosen) if position not in taken
    )
    return ranked
