# Playground

A local UI for showing what happens between the customer's message and the
slate. It exists for the demo video and for nothing else.

```bash
make playground          # then open http://127.0.0.1:8765
```

Start-up takes a few seconds: it builds the agent and every index once, the
same way the scoring path does, and loads a second copy of the catalog's
display fields because `catalog.build` folds titles into the index and drops
them.

## What it does not do

**It cannot change a reported number.** `submission/src/` has no edits and no
knowledge that this exists. Nothing here is imported by the agent, no organizer
path constructs `ExplainingAgent`, and the whole directory is outside the
submission bundle. Confirm with `git diff --stat submission/` and `make eval`.

It also takes no dependency. `http.server` and a hand-written page, consistent
with `submission/requirements.txt` declaring none.

## The two modes

**Replay a scored session** hands one row of `public_set.jsonl` to the
organizer's own `evaluate()`. The customer, the pivot and the scoring are all
the evaluator's, so the hit turn and rank on screen are its verdict rather than
this tool's opinion of one. Same discipline `harness/run.py` follows.

**Free typing** calls the agent directly with whatever you type. There is no
ground truth, so there is no score, and the header says so. You may nominate a
goal product to watch its rank move as the conversation discloses more.

## How the explanation is produced, and why you can trust it

`Agent` publishes a flat `debug` dict of about thirty scalars, sized for
`harness/trace.py`, which clips every value at 40 characters. It carries
`pool=329` but not which 329, and `alpha=0.6` but not what the blend did to any
particular product. `ranking.ranked` computes a BM25 term, a popularity term
and a negation penalty and returns only their sum; `phrase_promoted` and
`rerank` compute phrase evidence and return only the permutation.

Rather than instrument the shipping path, `playground/rederive.py` **replays**
it: every stage is a pure function of `(catalog, state)`, so calling them again
in the same order with the same arguments must produce the same slate. That
identity is checked, not assumed. Each turn carries a `verified` flag, and the
page shows a red banner instead of a breakdown when it is false.

`playground/tests/test_explain.py` is the standing version of that check: it
asserts the replay reproduces the served slate, that each score breakdown sums
to the score the evaluator saw, and that the reproduced probe table's argmax is
the arm actually asked. It runs as part of `make test`.

The seam that makes this cheap is `Agent._record`, which is handed the state
object `ranking.slate` was called with, before `with_slate` replaces it.
`ExplainingAgent` overrides that and `respond`, and nothing else.

## Known limits

- **There are no product images anywhere in the catalog schema**, so cards are
  text. Review count is shown prominently instead, because it *is* the
  popularity prior the ranking blends.
- **The understanding layer is tuned for the simulator's dialogue.** Free-typed
  prose is often captured as a single `feature` constraint rather than split
  into typed slots, and a negation phrased differently from the simulator's may
  not be read as one. That is the agent's real behaviour off-distribution, and
  the panels show it honestly rather than hiding it. Replay mode is the one to
  use when you want the system at its best.
- With the model rerank on (`USE_LLM=1`), the replay cannot reproduce a
  network call's sampling and reports `verified: false`. That is deliberate:
  the alternative is a second call pretending to be the first.
