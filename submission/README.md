# Conversational Shopping Agent

A multi-turn shopping agent that recovers a hidden target product from a
simulated customer conversation over a frozen 50,000-product Amazon catalog.

**Requires network: No. Third-party packages: none. Token usage: 0. Model cost: $0.00.
Bundled assets: one 4.92 MB dense-retrieval index, shipped switched off; no reported
number depends on it. An optional model-backed rerank tier exists behind `USE_LLM=1`
and is off; with it unset the agent imports nothing beyond the standard library.**

## Results

Scored by the organizer's own `evaluate()` over the 200 public sessions.

| scenario | n | HitRate@10 | MRR | MTTC | score |
|---|---|---|---|---|---|
| buying | 80 | 1.000 | 0.965 | 1.96 | 0.9703 |
| browsing | 80 | 1.000 | 0.970 | 2.19 | 0.9674 |
| intent_override | 30 | 1.000 | 0.983 | 4.17 | 0.9317 |
| boundary | 10 | 1.000 | 1.000 | 2.50 | 0.9700 |
| **overall** | **200** | **1.000** | **0.972** | **2.41** | **0.9633** |
| shipped baseline | 200 | 0.125 | 0.068 | 9.81 | 0.1067 |

`exceptions=0 discarded=0 dropped_slots=0 short_slates=430 wasted_pre_pivot_hits=29`.

`short_slates` counts turns that served fewer than ten recommendations, and it
is the deferred-commitment policy rather than a fault: the slate commits to the
products the ranking cannot separate -- one on 429 turns, ten on 52, two on one
-- and opens to the full page once the customer has nothing further to disclose.
**Serving all ten every turn instead scores 0.8977**, so the 0.0656 between the
two is bought by *when* the ranking is revealed rather than by what it finds.
That number is published here rather than left to be discovered.

**The headline is conditional, and the conditions are published rather than
buried.** Sessions are sampled from review records, which makes a product's
chance of being the target proportional to its review count. If that is wrong on
the private set, the popularity prior is the component that swings. Both risks
are gated in CI:

| gate | what it varies | result |
|---|---|---|
| held-out split | which half of the public set | dev **0.9595** / held **0.9672** |
| `make risk` | how targets are drawn | size-biased 0.9629, sqrt 0.9280, uniform **0.9038** |
| `make paraphrase` | how the customer words things | reworded 0.9555, punctuation 0.9567, filler 0.9499, synonym **0.9324** |
| `--no-fast-path` | template matching disabled entirely | **0.9622** |
| `make sessions` | 23 manufactured session sets | 93.5% rank-1 (`front_loaded_buying`) down to 16.5% (`compound_hard`) |
| `make deviations` | every live and switched-off component, on all of the above | ten components, all verdicts hold; health clean, no `unmoved` warning *(last run at an earlier commit -- re-run before quoting)* |
| `make dense` | the Tier 1 dense track, on every gate | loses on 14 of 15 readable sets; a wash on the full battery. **Switched off** *(last run at an earlier commit)* |

Read `uniform` as a pessimistic bound rather than a forecast: a uniformly drawn
target is not merely less popular, it is a genuinely harder product with thinner
text.

## Setup

Python 3.10 or later (developed on 3.14). No dependencies. The bundled dense asset is
read with `array` and `struct`; nothing needs installing to use it, and the agent runs
identically if it is deleted.

```bash
gzip -dkc catalog.jsonl.gz > data/catalog.jsonl   # one time, ~19 MB packed
pip install -r submission/requirements.txt        # no-op, stdlib only
```

**One command to score the agent:**

```bash
PYTHONPATH=. python3 -m harness.run --agent submission.agent:Agent
```

One environment variable is read, and only one: `USE_LLM=1` opts into the model-backed
rerank tier (below). Unset, nothing in the agent reads the environment, the network or a
credential, which is the configuration every number in this file was measured in.

## Architecture

Ten stages, all in memory, all built once at construction.

```
__init__(catalog_path)          catalog, buckets, BM25 postings, priors,
                                category resolver, slot taxonomy. Once.
reset(session_id, user_profile) allocate state. No I/O, no indexing.
respond(session_id, msg, turn, k)
    interpret(msg)   -> ParsedTurn   template, then cues, then vocabulary
    update(state)    -> SessionState typed slots; targeted override
    choose(state)    -> Route        precision / discovery / recovery / boundary
    ranked(state)    -> pool         category filter, then BM25 + popularity
                                    blend, less what was refused
    unseen(ordered)  -> pool         drop what this session already showed
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

**A slot is never spent twice.** The same irreversibility cuts the other way: a
product that was shown and did not end the session is provably not the answer,
so showing it again converts nothing. Before this, 62.9% of impressions on the
hardest session set were repeats, and 60 of its 78 misses had the target inside
the slot budget the session had already spent. A redirect clears the memory,
because a pre-pivot impression was never scored against the new target. Ablating
it is negative on every readable set.

**"Not polyester" is not a request for polyester.** Constraint text used to be
one positive bag of words, so a refusal was scored as evidence for the thing
refused, and the agent served that material at 2 to 5 times its shelf rate. The
cue is now split from the text, held out of the positive query, and subtracted.
The cue set is narrower than English allows -- no bare `no` or `non-` -- because
this catalog spells attribute *names* that way (`Non-Polarized`, `No Closure
closure`) 431 times, three of which reach the public set. **The measurement says
the win is holding the term out, not subtracting it:** the whole mechanism is
worth 0.215 on the set that tests it, the subtraction inside it 0.014.

## What we spent score on

One component ships **live at a measured cost**, and it is the only one. It is
listed here rather than in the limitations below because it is a choice rather
than a shortfall.

### Why the agent does not just ask `"other"`

`customer_reply()` matches the probe like this:

```python
if value not in disclosed and (attribute == "other" or classify_constraint(value) == attribute)
```

`attribute == "other"` short-circuits the type check, so the wildcard's match
set is a strict **superset** of every named attribute's. "Anything else?"
returns any two undisclosed constraints; "what colour?" returns only the ones
that classify as colour, and most cards hold none. No named attribute can
extract more disclosure than the wildcard, ever. It is a dominant strategy by
construction.

So the score-maximising agent asks `"other"` on every turn of every session,
and `probe.SPECIFIC_ARMS = False` does exactly that. Measured across the public
set and the 23 frozen session sets:

| | with specific arms | wildcard only |
|---|---|---|
| public 200 | **0.9633** | 0.9622 |
| mean over 24 sets | 0.8981 | **0.9002** |
| questions that name a real attribute | 10,576 of 14,016 | **0** |
| questions that are "anything else?" | 3,440 | 11,332 |

**The wildcard is worth +0.0020 on average, and we do not take it.**

The cost is not evenly spread, and the sets where it bites are worth naming:
`compound_hard` −0.0242, `thin_cards` −0.0229, `returning_shopper` −0.0177,
`unpopular_targets` −0.0133. It is also not uniformly a cost — specific arms
*win* on 13 of the 24 sets, most clearly on `unrelated_pivot` (+0.0145) and
`silent_customer` (+0.0101), where a pointed question restarts a conversation
that the wildcard lets drift.

### What it buys

Turn the switch off and the agent asks `"anything else?"` **eleven thousand
times and asks nothing else, ever** — the same sentence, up to ten times per
session, to every customer. Turn it on and roughly three questions in four
name something real: material, feature, style, size, colour, each chosen by
entropy over the products still in contention, each offering answer options
drawn from those products.

The brief names *"proactive structured clarification"* and *"adaptive
clarification and question-value estimation"* as goals in their own right, and
a shopper cannot answer "anything else?" any better on the ninth asking than
the first. Two thousandths of a point is what that costs, it is measured on 24
independent session sets rather than asserted, and the switch is one line for
anyone who would rather have the score.

## Limitations, and things that did not work

Five components were built, measured and **shipped switched off**. Each keeps a
live switch and a test asserting it is off, so they are results rather than
missing features.

| built | measured on the public 200 | re-measured on 18 harder sets | shipped |
|---|---|---|---|
| Route-conditional prior weight | +0.014 on dev, *worse* on held-out | +0.002 mean, 8 sets better and 7 worse | neutral |
| Restarting the turn budget after a redirect | +0.003 adversarial, -0.0005 real | **+0.005 mean, 11 better and 1 worse** | neutral |
| Maximal marginal relevance slate | loses 0.005, and improves the more its relevance term is switched off | loses on 13 of 18; the monotonicity does not reproduce | off |
| Converging early on a confident ranking | HitRate 0.990 to 0.995, MRR 0.909 to 0.781 | loses on 15 of 18, including the set with 150 misses | off |
| Profile-weighted ranking | monotonically negative | negative on 18 of 18 at weight 0.4 | weight 0.0 |

Every one of the five was first measured on a set where 176 of 200 sessions
already convert at rank 1, which can detect harm and not benefit. All five were
re-swept against the manufactured session sets, through the organizer's own
evaluator. (That sweep was taken when the manifest held 22 sets running from
88% rank-1 down to 9%; it now holds 23, running 93.5% down to 16.5%, and the
five-component sweep has not been re-taken against them.) **All five decisions survived. Two of
the explanations behind them did not**, and that is the more useful output.

The MMR result is the informative one. A diversity objective that gets better
the more you ignore relevance is not being rewarded for diversity; it is being
rewarded for reaching deeper into the pool. Redundancy is not what costs here,
irreversibility is. On the harder sets the *monotonicity* turns out to be a
property of the public target distribution rather than of the technique -- one
set runs the other way -- but the mechanism holds, and it is confirmed by two
sets that disagree about the two slate policies in exactly the way it predicts.
Where the target's card carries one feature and no price, so the ranking has
almost no text to work with, MMR lifts HitRate from 0.610 to 0.715 at no cost in
MRR, because its reach adapts to how flat the scores are while a fixed rank
offset reaches a fixed distance. It still loses everywhere the text is real.

Two components pay under a named condition and are reported that way rather than
switched on. Restarting the post-redirect budget is worth +0.037 where a
redirect names a category the session did not open on, and improves both
adversarial risk columns, but it regresses the public set, both halves of the
held-out split and the paraphrase column by 0.0004 to 0.0008 each, and the
shipping rule for that phase was fixed in advance and admits no regression.

The profile result is a coverage/signal inversion rather than emptiness. The tags
most customers carry (`fit` in 163 of 200 sessions, `material` 154, `comfort`
144) have a lift near 1.0 on whether their own word appears in the target, and
each matches 15-28% of the catalog, so boosting them promotes a quarter of the
shelf. The tags that do predict (`warmth` 4.0x, `performance` 2.6x) appear in 12
to 26 sessions. Restricting the boost to those cuts the damage but still does not
clear the baseline. The obvious objection is that the ranking simply had no room
left, so it was re-measured where there is plenty: on the set where the customer
discloses nothing at all, the profile is the only evidence beyond the category,
and two thirds of sessions are not at rank 1, the weight is the most harmful it
is anywhere. **When the profile is the only thing left, it is still worse than
nothing.**

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
- **Dense retrieval is built and switched off, which is a result rather than an
  omission.** `submission/src/dense.py` is a 64-dimension latent semantic space
  over the catalog's own `title` + `features`, and `routing.choose()` selects
  between it and the lexical index. It loses on 14 of the 15 readable synthetic
  sets and is a wash on the full battery, because a latent space's leading
  dimensions encode *category* -- exactly the structure the bucket filter has
  already applied -- so it is a coarser retriever than the one already in place,
  not a finer one. The case for building it rested on a fallback ("no query token
  matches, so ranking drops to popularity order") that fires on 0.0% of public
  turns. `make dense` prints the whole table.
- The model-backed rerank tier is built and switched off. `src/llm.py` sits behind
  `ranking.Reranker`, composed **over** the phrase rerank rather than in place of
  it, so a timeout or a refusal serves exactly what the offline agent would have
  served. Two gates keep it off: `USE_LLM=1` decides whether it is built at all,
  and `ranking.LLM_RERANK` whether a built stage is consulted.
  **Measured and switched off** (findings 3.36). `claude-haiku-4-5` over all 200
  sessions, 323 live calls: **0.9333 against the 0.9554 the offline agent scored
  at that commit**, hit@10 and MTTC unchanged to the digit, all of the loss in
  MRR (0.925 to 0.852). The offline path has since risen to 0.9633 and the model
  tier has *not* been re-measured against it, so the honest reading is "it lost
  by 0.022 at the commit where both were taken", not "it loses by 0.030 now." The model is not bad at
  the task -- it fixed 6 of the 20 sessions a permutation could win, which no
  lexical stage reaches. It is the base rate: 180 of 200 sessions already convert
  at rank 1, so a 12.2% error rate on those outweighs a 30% success rate on the
  other 20 by 22 to 1. Cost $0.385 and a p50 of 1,087 ms per turn against
  2.5 ms. Extrapolated to 800 sessions that is ~27 minutes and ~$1.54 for a
  negative return. Not re-taken on the frozen sets, so the claim is "it loses on
  the public 200", not "it loses".

## Cost and efficiency disclosure

| measure | value |
|---|---|
| Model | none on the scored path; `claude-haiku-4-5` behind `USE_LLM=1`, measured at 0.9333 and switched off |
| Prompt / completion tokens | 0 / 0 |
| Estimated API cost | $0.00 |
| Network calls | none |
| Environment read | `USE_LLM` only; unset in every number above |
| Per-turn latency | p50 2.5 ms, p95 8.0 ms, max 20.1 ms over 482 turns |
| Same, with Tier 2 on | p50 1,087 ms, p95 1,393 ms, max 8,604 ms; 326,851/11,628 tokens, $0.3850 |
| One-time index build | ~12 s |
| Resident memory | 242 MB, of which 11.2 MB is the product-title table the reply names its recommendations from |
| Bundled assets | `assets/dense.bin`, 4.92 MB, loaded in 2.6 ms, **weight 0** |

Latency and memory come from the same run as the reported score.

## Reproducing every number above

```bash
make eval                                          # 0.9633 and the health line
make baseline                                      # asserts the frozen 0.10671 reference
make split                                         # dev 0.9595 / held 0.9672
make risk                                          # target-distribution surface
make paraphrase                                    # wording surface
make sessions                                      # the 22 frozen synthetic sets
make deviations                                    # every component with a live switch, re-swept
#                                                  # the wildcard trade, 24 sets:
#                                                  #   probe.SPECIFIC_ARMS False vs True
make dense                                         # the Tier 1 ablation, switched on
make llm                                           # the Tier 2 ablation; needs a key
make test                                          # 335 tests
python3 -m harness.run --no-fast-path --no-diff    # 0.9622, general path alone
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
  assets/
    dense.bin        4.92 MB latent space over the catalog; shipped at weight 0.
                     Rebuild: python3 tools/build_dense.py (needs numpy; offline,
                     deterministic, never imported at scoring time)
  src/
    agent.py         orchestration, exception envelope, debug dict
    understand.py    message -> ParsedTurn, three layers
    category.py      free text -> catalog category buckets
    dialogue.py      session state machine, targeted override
    slots.py         constraint typing, taxonomy learned from the catalog
    routing.py       retrieval policy per route
    probe.py         expected-disclosure question selection
    ranking.py       the blend, the slate, the rerank seam
    dense.py         Tier 1 latent retrieval; switched off
    llm.py           Tier 2 model rerank; switched off, lazily imported
    response.py      customer-facing reply, composed from state
    catalog.py       one-pass build: buckets, priors, indexes
    bm25.py          flat token-id postings
    phrases.py       whole-phrase rarity index
    text.py          the single tokenisation path
    tests/           256 tests
harness/             measurement and robustness gates
```
