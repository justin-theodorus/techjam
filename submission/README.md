# Conversational Shopping Agent

A multi-turn shopping agent that recovers a hidden target product from a
simulated customer conversation over a frozen 50,000-product Amazon catalog.

**Requires network: No. Bundled assets: none. Third-party packages: none.
Token usage: 0. Model cost: $0.00.**

## Results

Scored by the organizer's own `evaluate()` over the 200 public sessions.

| scenario | n | HitRate@10 | MRR | MTTC | score |
|---|---|---|---|---|---|
| buying | 80 | 0.988 | 0.921 | 1.80 | 0.9541 |
| browsing | 80 | 0.988 | 0.887 | 1.94 | 0.9411 |
| intent_override | 30 | 1.000 | 0.906 | 3.77 | 0.9166 |
| boundary | 10 | 1.000 | 1.000 | 2.60 | 0.9680 |
| **overall** | **200** | **0.990** | **0.909** | **2.19** | **0.9440** |
| shipped baseline | 200 | 0.125 | 0.068 | 9.81 | 0.1067 |

`exceptions=0 discarded=0 dropped_slots=0 short_slates=0`.

**The headline is conditional, and the conditions are published rather than
buried.** Sessions are sampled from review records, which makes a product's
chance of being the target proportional to its review count. If that is wrong on
the private set, the popularity prior is the component that swings. Both risks
are gated in CI:

| gate | what it varies | result |
|---|---|---|
| held-out split | which half of the public set | dev 0.9288 / held 0.9592 |
| `make risk` | how targets are drawn | size-biased 0.9209, sqrt 0.8396, uniform **0.7484** |
| `make paraphrase` | how the customer words things | reworded 0.9052, punctuation 0.9345, filler 0.9068, synonym **0.8752** |
| `--no-fast-path` | template matching disabled entirely | **0.9428** |

Read `uniform` as a pessimistic bound rather than a forecast: a uniformly drawn
target is not merely less popular, it is a genuinely harder product with thinner
text.

## Setup

Python 3.10 or later (developed on 3.14). No dependencies.

```bash
gzip -dkc catalog.jsonl.gz > data/catalog.jsonl   # one time, ~19 MB packed
pip install -r submission/requirements.txt        # no-op, stdlib only
```

**One command to score the agent:**

```bash
PYTHONPATH=. python3 -m harness.run --agent submission.agent:Agent
```

No environment variables are required or read.

## Architecture

Nine stages, all in memory, all built once at construction.

```
__init__(catalog_path)          catalog, buckets, BM25 postings, priors,
                                category resolver, slot taxonomy. Once.
reset(session_id, user_profile) allocate state. No I/O, no indexing.
respond(session_id, msg, turn, k)
    interpret(msg)   -> ParsedTurn   template, then cues, then vocabulary
    update(state)    -> SessionState typed slots; targeted override
    choose(state)    -> Route        precision / discovery / recovery / boundary
    ranked(state)    -> pool         category filter, then BM25 + popularity blend
    compose(ordered) -> head + tail  deferred commitment
    pad(chosen)      -> 10 ids       coarser group, then global popularity
    rerank(ten)      -> 10 ids       rare phrase evidence; a permutation
    probe(state)     -> attribute    argmax expected disclosure
```

Everything expensive happens in `__init__`, because 800 private sessions
re-parsing 50,000 JSONL records would blow any timeout.

**Understanding is layered, and the layers are ordered by how much they assume.**
A template fast path runs first because it is exact and free. Beneath it,
dialogue-act cue patterns detect a redirect, a refusal or a disclosure without
depending on any one phrasing. Beneath that, the customer's words are matched
against the catalog's own 1,115 category names, which is what makes the agent
independent of how a sentence is framed. Disabling the fast path entirely costs
0.0012, so nothing rests on it.

**Retrieval is a hard category filter, then a blend.** Filtering to the coarse
category cuts 50,000 products to a median of 182 at recall@100 = 0.990. Inside
it, `bm25(title + features) + 0.6 * log1p(review_count)`, both max-normalised.
Each half insures the other: the prior never reads constraint text so it survives
rewording, and BM25 never reads review counts so it survives a change in how
targets were sampled. Keeping both load-bearing is the robustness argument, and
it is why the weight sits mid-range rather than at either optimum.

**The slate is deferred commitment.** Showing a product is irreversible: the
session ends when the customer sees what they wanted, at whatever position it
occupied. So the agent commits to its single best guess and spends the remaining
slots on inventory no turn has reached, until the customer runs out of things to
say. Worth +0.064 and it raises coverage while doing it.

## Limitations, and things that did not work

Five components were built, measured and **shipped switched off**. Each keeps a
live switch and a test asserting it is off, so they are results rather than
missing features.

| built | measured | shipped |
|---|---|---|
| Route-conditional prior weight | +0.014 on dev, *worse* on held-out, non-monotone surface | neutral |
| Restarting the turn budget after a redirect | +0.003 adversarial, -0.0005 real, inside seed noise | neutral |
| Maximal marginal relevance slate | loses 0.005, and improves the more its relevance term is switched off | off |
| Converging early on a confident ranking | HitRate 0.990 to 0.995, MRR 0.909 to 0.781 | off |
| Profile-weighted ranking | monotonically negative | weight 0.0 |

The MMR result is the informative one. A diversity objective that gets better
the more you ignore relevance is not being rewarded for diversity; it is being
rewarded for reaching deeper into the pool. Redundancy is not what costs here,
irreversibility is.

The profile result is a coverage/signal inversion rather than emptiness. The tags
most customers carry (`fit` in 163 of 200 sessions, `material` 154, `comfort`
144) have a lift near 1.0 on whether their own word appears in the target, and
each matches 15-28% of the catalog, so boosting them promotes a quarter of the
shelf. The tags that do predict (`warmth` 4.0x, `performance` 2.6x) appear in 12
to 26 sessions. Restricting the boost to those cuts the damage but still does not
clear the baseline.

**Known weaknesses.**

- Constraint text is assumed to be drawn from the catalog's own product fields.
  This is inferred from the public set, not guaranteed. The phrase reranker is
  the component most exposed, which is why it runs last and only ever returns a
  permutation of an already-chosen slate: an unmatched constraint yields no
  evidence and the slate is returned untouched, so its downside is bounded at
  zero by construction rather than by good behaviour.
- The probe estimator scores attributes by what *products* declare, while
  disclosure depends on how the *customer* classifies their own preference.
  Those are different functions, and the gap shows: `brand` scores well and
  yields nothing. It does not change the decision, because the wildcard arm
  dominates by construction, but a policy needing to rank specific arms would
  need customer-side evidence this estimator does not have.
- There is no cross-session identity, so no long-term user profile is possible.
  What is implemented is within-session context distillation.
- No dense retrieval and no model-backed reranking. `ranking.Reranker` is the
  seam for one; its contract is that an implementation returns a permutation, so
  a model tier cannot cost coverage. Nothing is built behind it, and no reported
  number depends on one.

## Cost and efficiency disclosure

| measure | value |
|---|---|
| Model | none |
| Prompt / completion tokens | 0 / 0 |
| Estimated API cost | $0.00 |
| Network calls | none |
| Per-turn latency | p50 0.31 ms, p95 1.55 ms, max 2.61 ms over 436 turns |
| One-time index build | 3.1 s |
| Resident memory | 106 MB |
| Bundled assets | none |

Latency and memory come from the same run as the reported score.

## Reproducing every number above

```bash
make eval                                          # 0.9440 and the health line
make baseline                                      # asserts the frozen 0.10671 reference
make split                                         # dev 0.9288 / held 0.9592
make risk                                          # target-distribution surface
make paraphrase                                    # wording surface
make test                                          # 216 tests
python3 -m harness.run --no-fast-path --no-diff    # 0.9428, general path alone
```

`make eval` writes `runs/latest.json` and prints a per-scenario table, a health
line and a diff against the previous run. The health line is the real gate: the
evaluator swallows agent exceptions into an empty turn, so a crash is invisible
in the score and has to be surfaced separately.

## Layout

```
submission/
  agent.py           entry point, exports Agent
  requirements.txt   empty; stdlib only
  README.md          this file
  src/
    agent.py         orchestration, exception envelope, debug dict
    understand.py    message -> ParsedTurn, three layers
    category.py      free text -> catalog category buckets
    dialogue.py      session state machine, targeted override
    slots.py         constraint typing, taxonomy learned from the catalog
    routing.py       retrieval policy per route
    probe.py         expected-disclosure question selection
    ranking.py       the blend, the slate, the rerank seam
    response.py      customer-facing reply, composed from state
    catalog.py       one-pass build: buckets, priors, indexes
    bm25.py          flat token-id postings
    phrases.py       whole-phrase rarity index
    text.py          the single tokenisation path
    tests/           216 tests
harness/             measurement and robustness gates
```
