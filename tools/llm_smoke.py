"""One model call, to check Tier 2's request shape before spending on a run.

`make llm` is roughly 2,500 requests. If the model rejects the request shape --
`output_config.format` unsupported, the model id wrong, the key unfunded --
every one of them fails silently into the offline order, because that is what
`llm.LLMReranker` is built to do. This asks once, loudly, first.

    USE_LLM=1 ANTHROPIC_API_KEY=... python3 tools/llm_smoke.py
"""

from __future__ import annotations

import sys
from time import perf_counter

from techjam.submission.src import llm


CARDS = (
    "Merino wool crew sock, cushioned sole, charcoal",
    "Polyester athletic sock, moisture wicking, white",
    "Cotton dress sock, ribbed, navy",
)
WANTS = "Wants: merino wool; charcoal\nRefused: polyester"


def main() -> int:
    stage = llm.build()
    if stage is None:
        print(
            "Tier 2 did not build. Needs USE_LLM=1, `pip install anthropic`, "
            "and a resolvable ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        return 1

    prompt = WANTS + "\n\nCandidates:\n" + "\n".join(
        f"{number}. {card}" for number, card in enumerate(CARDS, start=1)
    )
    started = perf_counter()
    try:
        order = stage._ask(prompt)
    except Exception as error:
        # Deliberate isolation point: this script exists to surface exactly
        # the failures the reranker is designed to swallow.
        print(f"call failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    elapsed_ms = (perf_counter() - started) * 1000.0
    counts = stage.take()
    print(f"model    {stage.model}")
    print(f"order    {order}   (expected the wool sock first)")
    print(f"tokens   {counts['prompt_tokens']} in, "
          f"{counts['completion_tokens']} out, "
          f"{counts['cached_tokens']} from cache")
    dollars = llm.cost(
        counts["prompt_tokens"], counts["completion_tokens"]
    )
    print(f"cost     ${dollars:.6f}")
    print(f"latency  {elapsed_ms:.0f} ms")
    if not order:
        print("\nthe call succeeded but returned no ordering; the request "
              "shape is wrong somewhere.", file=sys.stderr)
        return 1
    # This prompt carries three candidates; a real turn carries ten, so the
    # per-call figure above is a floor rather than a forecast. The stub run
    # measured 325 calls and ~910 input tokens each (findings 3.36).
    print(f"\nprojected `make eval` column: 325 calls, "
          f"~${llm.cost(325 * 910, 325 * 40):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
