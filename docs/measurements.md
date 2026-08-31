# Measurement log

Every claim a code comment in this repository cites. The numbering is stable: a
comment that reads `(measurements 3.31)` refers to section 3.31 below.

**How to read this.** Each entry states one measured result, the numbers behind
it, and the command that reproduces it. Everything was scored through the
organizer's own `evaluate()` in `evaluator/local_evaluator.py`, never through a
re-implementation of the scoring loop, so a trace and the score it belongs to
come from the same run and cannot disagree. Numbers taken at an earlier commit
are labelled as such; a component's verdict is only valid against the
configuration it was taken in (decision 25).

Two instruments recur:

- **The public 200** (`make eval`). The released development set. It is
  *saturated*: the shipped agent converts every session, and 176 of 200 already
  convert at rank 1, so it has 24 sessions of upside against 176 of downside and
  can no longer separate a good idea from a bad one.
- **The frozen sets** (`make sessions`, `make deviations`). 23 synthetic session
  sets manufactured through the same `evaluate()`, spanning 99% rank-1 down to
  16%. Their seeds are frozen and read-only: a set reshaped because of what it
  showed about a feature is worthless as evidence.

---

## 3.3 Catalog phrase statistics

Indexing every normalized `features` bullet and `details` value as a whole
string yields 222,605 distinct phrases, of which **89.8% belong to exactly one
product**. Ranking by rarity-weighted phrase membership with all four
constraints known puts the target at rank 1 in 139 of 200 public sessions.

This is why whole-phrase evidence is the sharpest signal the catalog offers, and
why it must not be the *primary* retrieval route: an exact key does not survive
a synonym substitution. It ships as a promotion stage over an already-chosen
pool (3.58), where an unmatched phrase yields no evidence and the ordering is
returned untouched.

## 3.4 Probe arm payouts

All 800 constraints across the 200 public sessions, classified through
`classify_constraint()`'s own branches:

| attribute | count | share |
|---|---|---|
| feature | 404 | 50.5% |
| material | 302 | 37.8% |
| color | 60 | 7.5% |
| style | 19 | 2.4% |
| size | 11 | 1.4% |
| use_case | 4 | 0.5% |
| budget / category / brand | 0 | 0.0% |

`budget` never fires because `price` is null on 79% of the catalog, and when
present the budget string is appended last and sliced off by
`hard_constraints[:2]` / `soft_preferences[2:4]`. `category` and `brand` are not
branches of `classify_constraint()` at all, so those two arms are dead by
construction.

**Consequence: the ten-armed probe policy is a solved problem, not a research
opportunity.** It is dominated by a one-line rule, and information-gain arm
selection is reported as an ablation rather than a headline.

## 3.6 Filtering by category beats querying by category, by a lot

Same scorer over `title + features + details` in all three rows, all four
constraints known. This is a full-information ceiling, not an achievable score:

| retrieval configuration | hit@10 | MRR | recall@100 |
|---|---|---|---|
| constraints only, no category | 0.510 | 0.437 | 0.680 |
| category appended as query *terms* | 0.555 | 0.443 | 0.800 |
| **hard category filter, constraints scored in-bucket** | **0.830** | **0.647** | **0.990** |

**recall@100 inside the filtered bucket is 0.990.** Retrieval is therefore not
the binding constraint; ranking within a correctly chosen bucket is. The
category filter is the single highest-value line in the system and the least
fragile one: it is structural, catalog-derived and scenario-independent, and it
survives a private set that ships authored intent cards because `evaluate()`
reads the category from the catalog rather than from the card.

Reproduce: `make eval`, then compare against `--no-fast-path`.

## 3.11 Catalog structure

50,000 products. `title` and `categories` are on 100% of rows; `details` 96.7%,
`features` 89.6%, `description` 52.2%, `price` **21.1%**.

`details` spreads across **287 distinct sub-keys** and the frequent ones are
metadata rather than attributes: `Date First Available` 93.8%, `Department`
87.2%, `Item model number` 55.5%. The attribute-shaped keys are rare: `Color`
4.9%, `Brand` 4.7%, `Material` 4.1%, `Style` 3.5%, `Size` 1.9%.

`coarse_category()` collapses 863 raw category strings to **1,115 buckets**, 234
of them singletons. The median bucket holds 8 products and the largest
(`Shirts T-Shirts`) holds 1,354; 54 buckets hold half the catalog. Sessions land
in large buckets far more often than a uniform draw would, which is why the
*session-weighted* median pool is 182.

## 3.12 Variant sprawl is small, and so is intent-card ambiguity

Only exact `parent_asin` equality scores, so near-duplicate listings would be
slate poison. Exact normalized-title collisions across different parents:
**1,444 products, 2.9%**. A looser `store` + title-prefix grouping catches 7.5%
but inspection shows it is mostly same-brand rather than same-product, so 2.9%
is the trustworthy figure.

The question that actually bites is whether two products generate an *identical
intent card*, because then the customer's four constraint strings cannot
distinguish them. 11,175 products (22.4%) share a card with at least one other,
and the degenerate classes are generic (442 products share
`hard=('cotton', 'color: grey')`). But **175 of 200 public targets have a card
that is unique inside their own bucket**, 23 more sit in a same-bucket class of
10 or fewer that a slate can cover outright, and only 2 sit in a larger class.
Expected hit@10 if the exact ambiguity class were always covered: **0.994**.

**Ambiguity does not bound hit@10.** It bounds MRR slightly, since 25 of 200
sessions cannot reliably be ranked first.

## 3.13 Field ablation: `features` carries the signal, `description` actively hurts

BM25 (k1=1.2, b=0.75), oracle query holding all four constraint strings, ranks
pessimistic under ties. Right-hand columns are hard-filtered to the target's
`coarse_category` bucket.

| indexed fields | r@10 unfiltered | r@10 in-bucket | r@100 in-bucket | MRR@10 |
|---|---|---|---|---|
| `title` only | 0.185 | 0.355 | 0.490 | 0.219 |
| `details` only | 0.055 | 0.140 | 0.225 | 0.081 |
| `features` only | 0.525 | 0.805 | 0.985 | 0.640 |
| **`title`+`features`** | **0.535** | **0.830** | 0.980 | **0.650** |
| `title`+`features`+`details` | 0.505 | 0.830 | 0.985 | 0.649 |
| + `description` | 0.450 | 0.785 | 0.970 | 0.580 |
| evaluator `SEARCH_FIELDS` (all six) | 0.440 | 0.785 | 0.970 | 0.580 |
| `title` 3x, all six fields (starter-like) | 0.405 | 0.745 | 0.955 | 0.529 |

1. **`features` is the field.** Alone it beats every combination that dilutes it.
   `title` alone is nearly useless despite being what the starter weights most.
2. **`description` costs 4.5 points of hit@10 and 7 points of MRR.** It is on
   only 52% of products and repeats the features without adding discriminative
   tokens. The evaluator's own `SEARCH_FIELDS` includes it; that list is a
   specification of what the *simulator* reads, not a recommendation.

This is standing decision 7.

## 3.16 A popularity prior is worth more than every constraint in the session

Scored end to end through the real `evaluate()`. Every variant parses the coarse
category from turn 1, hard-filters to that bucket, asks `other` every turn and
emits ten; they differ only in the ranking rule inside the bucket.
`score(a) = bm25(a)/max_bm25 + alpha * log1p(rating_number)/max`.

| variant | hit@10 | MRR | MTTC | score |
|---|---|---|---|---|
| shipped BM25 baseline | 0.125 | 0.068 | 9.81 | 0.1067 |
| `alpha`=0, BM25 only | 0.850 | 0.512 | 3.06 | 0.7374 |
| **popularity only, every constraint ignored** | **0.815** | 0.498 | 3.18 | **0.7133** |
| `alpha`=0.7 | 0.950 | 0.592 | 2.12 | 0.8301 |
| **`alpha`=1.3** | 0.965 | **0.609** | 1.90 | **0.8473** |
| `alpha`=2.0 | 0.955 | 0.599 | 1.97 | 0.8377 |

**An agent that ignores every word the customer says scores 0.7133**, because
the targets are popularity-skewed by arithmetic rather than by curation:
sessions are sampled from review records, so a product's chance of being the
target is proportional to its review count. The median public target sits at the
99th percentile of catalog `rating_number`.

The public optimum at `alpha`=1.3 is **not** what ships. See 3.20.

## 3.17 The blend is nearly immune to rewording

Every session held fixed, only the customer's language varied, scored through
the real `evaluate()`. Cells are hit@10 / TechnicalScore.

| agent | none | truncate | dropout | shuffle | **synonym** |
|---|---|---|---|---|---|
| popularity only | 0.815/0.713 | 0.815/0.713 | 0.815/0.713 | 0.815/0.713 | 0.815/0.713 |
| BM25 only | 0.850/0.737 | 0.850/0.729 | 0.850/0.737 | 0.850/0.737 | 0.900/0.786 |
| **blend `alpha`=1.3** | 0.965/**0.847** | 0.955/0.836 | 0.950/0.834 | 0.965/0.847 | 0.955/**0.832** |

**The blend loses 0.015 of score under synonym substitution.** The prior is
immune by construction because it never reads constraint text, and the category
filter is immune for the reason in 3.6, so the only exposed component is BM25.

**Corrected in place by 3.35:** the BM25-only row *improving* under synonym
substitution was read at the time as a popularity fallback firing when no query
token matches. That fallback fires on **0.0% of public turns and 0.9% under
synonym substitution**, so it is not the mechanism, and the 0.015 here is a
ceiling on BM25's loss rather than a forecast of what a second retriever would
recover.

Reproduce: `make paraphrase`.

## 3.18 Buying does not underperform browsing for a design reason

The scenarios have identical information from turn 3 onward. Mean constraints
known before responding, replaying disclosure with `other` every turn:

| scenario | turn 1 | turn 2 | turn 3 | turn 4 |
|---|---|---|---|---|
| buying | 1.00 | 3.00 | 4.00 | 4.00 |
| browsing | 0.00 | 2.00 | 4.00 | 4.00 |

Buying is weakly *ahead*, never behind, so any deficit is about the targets
rather than the protocol. `intent_card()` inserts the material match at position
0, so `hard_constraints[0]` -- the one string buying discloses for free -- is a
**bare material word in 76% of public sessions**, against 1% for
`hard_constraints[1]`. That constraint leaves a median **45% of the bucket still
matching**; `cotton` matches 1,069 of its bucket's 1,354 products.

## 3.19 A catalog-derived slot schema is not available, and is not needed

Products carrying a usable `details` value per dimension: brand 49.2%, size
22.1%, color 4.9%, material 4.4%, style 3.7%, use_case 1.7%.

A hard filter on a dimension covering 4% of the catalog discards 96% of the pool
on no evidence. The only dense closed vocabularies are the evaluator's own
regexes (`MATERIAL_RE` 57.1% over 9 values, `COLOR_RE` 38.8% over 12), and those
are exactly the two dimensions 3.18 shows are least informative.

**Constraints belong in the BM25 query over `title`+`features`, not in typed
slots backed by hard filters.** Typed state is still right for *dialogue*
bookkeeping -- what was disclosed, what a pivot superseded, what was refused --
and that is what `slots.py` is for.

## 3.20 The two risks are insured by opposite components, and that sets `alpha`

Two different risks, and neither component covers both:

- **Risk A, wording.** The private sessions are reworded or synonym-substituted.
  The popularity prior is immune; the BM25 term is not.
- **Risk B, target sampling.** The private targets are not drawn proportional to
  review count. The BM25 term is immune; the prior is not.

Risk B is directly measurable offline, because `materialize_hidden_fields()`
derives every session from the target's catalog record, so a 200-session set can
be manufactured with any targets at all. Three were built with the same
80/80/30/10 mix: targets drawn proportional to `rating_number` (the documented
rule), to `sqrt(rating_number)` (a weakened skew), and uniformly (adversarial).

**The `alpha`=1.3 that maximises the public 200 collapses to 0.595 against
uniformly-drawn targets.** `ALPHA` therefore ships at **0.6**, mid-range, so
both halves stay load-bearing. This is a deliberate trade of reported score for
robustness, and it is the same trade `EXPLORE_DIVERSITY` makes in 3.45.

**A dense embedding tier only ever improves the BM25 half, so it insures Risk A
and does nothing for Risk B.** Measurement later removed even that: see 3.35.

Reproduce: `make risk`.

## 3.21 The first agent from committed code

Measured 2026-08-27, the first number in this log that comes from committed code
rather than a throwaway probe. **TechnicalScore 0.8220** at `alpha`=0.6.

| scenario | n | hit@10 | MRR | MTTC | score |
|---|---|---|---|---|---|
| **overall** | 200 | **0.945** | **0.572** | **2.11** | **0.8220** |

Health clean, `short_slates=0`, latency p50 0.302 ms over 411 turns, agent-only
RSS 59 MB. The 268 MB peak of a full run is dominated by the evaluator's own 50k
product dicts, not by the agent.

## 3.22 MRR, and the two things that moved it

Measured 2026-08-27. **0.8220 to 0.9160**, from two changes, neither of them the
reranker the phase was scoped around.

The load-bearing one is the slate policy. `evaluate()` ends a session at the
first turn the target appears *anywhere* in the served ten, so **membership
fixes HitRate and MTTC while position fixes MRR**. A wide early slate converts
at whatever rank it happened to reach. Serving the committed head alone until
disclosure finishes, rather than the plain top ten, is worth **+0.064**.

**This section's conclusion was wrong about the reranker and 3.23 corrects it.**

## 3.23 Rarity-weighted phrase evidence, and why 3.22 was wrong to skip it

Measured hours after 3.22. **0.9160 to 0.9279.** 3.22 declined to build this on
two arguments and both were wrong:

1. *The addressable band is near zero.* Inferred from 3.12's 25 twin-capped
   sessions, never measured. The actual gain is **+0.0119**, five times the
   estimate.
2. *Phrase lookup loses ~40% of its hit rate under paraphrase.* True of phrase
   lookup as the **primary retrieval route**. Used as a tie-break inside an
   already-chosen ten, an unmatched constraint yields no evidence and the slate
   is returned untouched: the route degrades to identity, not to noise.

**Standing lesson, and it cost 0.012 to learn: an argument by analogy to a prior
measurement is not a measurement.** This is standing decision 9, and it is the
ancestor of decision 23.

Because a rerank stage returns a *permutation* of the served ten, it cannot
regress coverage or timing by construction.

## 3.24 The paraphrase gate

`harness/paraphrase.py` holds every session fixed -- same targets, same mix, same
disclosure order, same override turn -- and varies only the language. It
monkeypatches the four language producers in `evaluator.local_evaluator`,
reproducing each one's logic exactly and changing only the surface string. The
patched `behavior_for` delegates to the original before rewriting, so `rng` is
consumed in the same order and **the pivot lands on the turn it always would
have**. `harness/tests/test_paraphrase.py` pins all of that.

| column | what changes |
|---|---|
| `clean` | nothing. Must reproduce `make eval` exactly, as a self-check |
| `reworded` | same information, different frame. No shipped literal survives |
| `punctuation` | shipped frames, separators moved |
| `filler` | shipped frames, conversational padding added |
| `synonym` | vocabulary substituted inside the constraint strings |

The gate found a cliff: an agent leaning on template matching scored **0.0273**
on the worst column. Converting those shortcuts into fast paths of general
mechanisms took that to 0.8752, and it now reads 0.9318.

Reproduce: `make paraphrase`.

## 3.26 Typed slots, routing, and a derived probe

Measured 2026-08-28. **0.9279 to 0.9440**, hit@10 0.990, and every column of the
risk surface improves:

| gate | before | after |
|---|---|---|
| `make eval` | 0.9279 | **0.9440** |
| `--no-fast-path` | 0.9279 | 0.9428 |
| `make risk` size-biased / sqrt / uniform | 0.8963 / 0.8153 / 0.7135 | **0.9209 / 0.8396 / 0.7484** |
| `make paraphrase` worst | 0.8686 | **0.8752** |

The dev/held gap widened to 0.030 and it is **not** overfitting: held-out is the
*higher* half, and the shipped configuration was deliberately not tuned on dev.

Route-conditional `alpha` was built here and is a measured negative result. It
ships switched off as `routing.PRECISION_ALPHA` / `routing.DISCOVERY_ALPHA`.

## 3.27 The slate is deferred commitment, not diversity

Measured 2026-08-28. **0.9440, unchanged.** Two alternatives to the slate policy
were built; both lose, and *why* they lose is what renames the mechanism.

Maximal marginal relevance, greedy over the top 40 by
`(1 - w) * relevance - w * max_similarity_to_selected`:

| policy | public | hit@10 | MRR | MTTC |
|---|---|---|---|---|
| **rank offset (shipped)** | **0.9440** | 0.990 | **0.909** | 2.19 |
| MMR `w`=0.3 | 0.8983 | 0.995 | 0.714 | 1.67 |

MMR buys coverage and pays for it in position, at a ratio that loses. The
withheld band is not there to be *diverse*; it is there because showing a
product is irreversible, so the turn should commit only to what the ranking can
actually separate. Early convergence was measured in the same pass and is worse
still. Both ship switched off (`ranking.DIVERSITY`, `ranking.CONVERGE_AT`).

## 3.28 A grounded reply, and the profile finally measured

Measured 2026-08-28. **0.9440, unchanged.** `message` had been four canned
strings rotated on `turn % 4`, referencing no state at all, in a field the
contract calls customer-facing natural language. `submission/src/response.py`
composes it from the session instead: what was understood, what a redirect
replaced, what the slate is, and the question worth asking. Deterministic
template composition over live state, zero tokens, no network.

The anonymised `user_profile` was measured here rather than assumed. Weighting
the ranking by its `preference_tags` is monotonically negative when ungated
(-0.1125 on `silent_customer`), because it perturbs turns that already carry
query text. It ships at `ranking.PROFILE_WEIGHT` = 0.02 behind
`ranking.PROFILE_MAX_CONSTRAINTS` = 0, consulted only on turns where the
customer has disclosed nothing and there is no query text for it to displace.
That is a capability claim, not a score claim (decision 26).

## 3.30 The switched-off components, re-read where they had room to move

Measured 2026-08-28. **All five verdicts survive. Two of the explanations behind
them do not, and that is the phase's output.**

The five verdicts in 3.26-3.28 were taken on the public 200, where 176 of 200
sessions already convert at rank 1, so each had a bounded upside against an
unbounded downside. `harness/deviations.py` re-sweeps the same switches over the
same ranges against the frozen sets, which run from 88% rank-1 down to 9%, with
the public 200 kept as a reference row. It marks a set **saturated** above 85%
rank-1 and refuses to count it as evidence.

The gate consumes the frozen manifest read-only; all four files are checksummed
before and after and are byte-identical.

**One reading from this section has since expired.** `ranking.DIVERSITY` won
+0.0756 on `thin_cards` here. A later component moved that set from 0.5661 to
0.8025 and diversity now *loses* 0.0147 on it. Corrected in 3.43; generalised as
decision 25.

Reproduce: `make deviations`.

## 3.31 The customer says "not", and the agent hears "yes"

A refusal was scored as a preference. `SessionState.query_text` carried
`not polyester` as raw text, joined it into one positive bag of words and scored
it with BM25. `text.STOPWORDS` holds no negation cue and `Bm25Index.score` sums
over matched terms only, so the pipeline was **structurally incapable of a
penalty**.

Measured before the fix, over 11,490 recommendations served across 200 sessions
carrying a refusal, against each material's rate inside the retrieved bucket:

| refused material | served | shelf rate | lift |
|---|---|---|---|
| wool | 5.5% | 1.1% | **5.13x** |
| silk | 23.2% | 5.0% | 4.65x |
| nylon | 22.7% | 5.2% | 4.37x |
| cotton | 30.6% | 10.9% | 2.81x |
| polyester | 35.6% | 15.6% | 2.28x |
| spandex | 45.1% | 21.8% | 2.07x |

`slots.NEGATION` and `ranking.NEGATION_WEIGHT` ship **live**, unlike the
switched-off components above, because what they replace is an inverted default
rather than a neutral one. Their `make deviations` rows therefore read
backwards: a positive delta argues for switching them *off*.

## 3.32 Two thirds of the hard sets' slots went to products already declined

A session ends at the first turn the target appears in the slate. So a product
that was shown and did *not* end the session is **provably not the answer**, and
showing it again converts nothing. This is a theorem about the evaluator, not a
heuristic. The agent had no memory of it.

| set | impressions | distinct | repeat | reachable, still missed |
|---|---|---|---|---|
| public 200 | 4,360 | 3,327 | 23.7% | 2 |
| `twin_cards` | 5,710 | 3,707 | 35.1% | 14 |
| `unpopular_targets` | 7,710 | 3,896 | 49.5% | 23 |
| `thin_cards` | 10,290 | 3,813 | **62.9%** | **60** |
| `negated_constraints` | 11,490 | 3,756 | 67.3% | **73** |
| `compound_hard` | 16,270 | 4,841 | **70.2%** | 58 |

**60 of `thin_cards`'s 78 misses were reachable on slots the agent spent
re-showing declined products.** `ranking.SKIP_SHOWN` ships live. 22 of 22 frozen
sets improved with this and 3.31 together, none regressed, and the `uniform`
risk column went 0.7484 to 0.8560.

A pre-registered *adaptive reach* lever died on measurement in the same pass:
`ranking.contention()` maxes at 4 on every set, so it can never steer anything
that needs a wider range.

## 3.33 Cross-session memory: two mechanisms, and only one is measurable

"Long-term profile" can mean two unrelated things:

| | mechanism | keyed on |
|---|---|---|
| **A. Per-person memory** | one shopper returns; session 5 remembers sessions 1-4 | that shopper's identity |
| **B. Cohort inference** | shoppers who look alike want alike things | a description shared by many |

They are different systems, and B is collaborative filtering wearing memory's
clothes. **A is not measurable through the published API at all**, and the
reason is structural rather than statistical: `evaluate()` issues a fresh
`public_{uuid4().hex}` per session and never sends the same shopper twice, so a
per-person store is written once per session and read never.

**Strengthened by 3.48:** `docs/agent_api_contract.json` closes both
`reset_request` and `user_profile` with `additionalProperties: false`, so there
is **no field an identity could travel in**. That is why identity reaches the
agent through its own `Agent.remember()` and a row-level key the contract does
not govern, supplied by `harness/identity.py` beside the real `evaluate()`. Do
not add a sixth `user_profile` key to carry identity;
`harness/tests/test_sessions.py`'s `ContractTest` exists to catch exactly that.

Reproduce: `make memory`.

## 3.34 The instrument re-read, and the three sets it cost us

`harness/deviations.py` computes its saturation mark from each set's **rank-1
share**, and the negation and skip-shown fixes changed that share on all 22 sets.
Re-run before anything else was measured with the instrument; seeds untouched
and the frozen files checksum identical before and after.

**There are 15 readable sets, not 18.** `mirror`, `blank_profiles` and
`contradictory_profiles` crossed the 85% saturation line. Any aggregate quoted
from 3.30 is taken over a different population than one taken today, and
`contention()` maxing at 4 (3.32) means a ratio across the served slate carries
no depth information.

Reproduce: `make sessions`.

## 3.35 The dense track, built, measured, and switched off

A 64-dimension latent semantic space over the catalog's own `title` + `features`,
built offline by `tools/build_dense.py` and bundled as a **4.92 MB** int8 asset
with per-row fp32 scales. `routing.choose()` selects between two retrievers
rather than two equal constants, which was the phase's done-condition.
`submission/src/dense.py` is standard library only; a missing, truncated or
stale asset returns `None` and every caller degrades to the lexical blend.

**All three of its switches ship at zero**, and the phase's real output is why.

It loses on **14 of the 15 readable sets**, and on the `synonym` paraphrase
column it was built for it is *worse*, 0.9143 to 0.9036. Three independent
wording generators agree, so no argument from the saturated public 200 is
needed.

**The premise failed before the component did.** The "no query token matches, so
ranking falls back to popularity order" fallback that 3.17 and 3.20 built the
case on fires on **0.0% of public turns and 0.9% under synonym substitution**.
Both of those sections are corrected in place above.

Two findings generalise, as decisions 23 and 24: measure the mechanism before
the lever, and the category filter is already the coarse retriever, so a second
one has no room -- dense full-catalog recall@100 is **0.395** against the bucket
filter's **0.990**.

Reproduce: `make dense`.

## 3.36 The model-backed ranking stage, wired and bounded

`submission/src/llm.py` sits behind `ranking.Reranker`, composed **over** the
phrase stage rather than in place of it, so a timeout or a refusal serves exactly
what the offline agent would have served. Two gates keep it off: `USE_LLM=1`
decides whether it is built at all, and `ranking.LLM_RERANK` whether a built
stage is consulted. `anthropic` is imported lazily and only under `USE_LLM=1`.

**The band a permutation controls, measured rather than argued.** Decomposed
over the public 200:

| | sessions | recoverable MRR /200 |
|---|---|---|
| already at rank 1 | 180 | -- |
| **convert below rank 1: the whole addressable set** | **20** | **0.0745** |
| ...holding 2+ constraints, where a semantic ranker has text to read | 11 | **0.0371** |

Measured with `claude-haiku-4-5` over all 200 sessions, 323 live calls:
**0.9333 against the 0.9554 the offline agent scored at that commit**, hit@10
and MTTC unchanged to the digit, all of the loss in MRR (0.925 to 0.852). The
model is not bad at the task -- it fixed 6 of the 20 sessions a permutation could
win, which no lexical stage reaches. It is the base rate: 180 of 200 already
convert at rank 1, so a 12.2% error rate on those outweighs a 30% success rate
on the other 20 by 22 to 1.

Cost $0.385, p50 1,087 ms per turn against 2.5 ms offline. The offline path has
since risen and the model tier has **not** been re-measured against it, so the
honest reading is "it lost by 0.022 at the commit where both were taken".

Reproduce: `make llm` (needs network, credentials and real money).

## 3.37 `other` dominates the probe channel by construction

`local_evaluator.py:174-181` filters a *fixed ordered list* built from the
target's card. `other` passes everything; every specific arm is a strict subset.
Asking about material does not make the customer think harder about material, it
filters a recitation they were going to give anyway.

Measured: `other` on every turn from turn 1 reaches all four constraints by turn
3 or 4 in 200 of 200 public sessions, while any fixed specific arm never
completes in 180 of 200. An **oracle** probe that reads the target's own card
and asks for its rarest undisclosed constraint scores **-0.0006** on public,
**-0.0471** on `thin_cards` and **-0.0505** on `compound_hard`. No real policy
beats an oracle, so that bounds the whole channel.

**Ordering is the only lever, and it is worth about -0.01, not zero.**

## 3.38 The intent-card leak, and the listing prior that replaced it

The 3.37 policy first scored -0.0066 by taking its offer sets from
`ev.intent_card()` and typing constraints with `classify_constraint()`'s own
word lists. Both are evaluator knowledge: `intent_card` is a **hidden field**
that `materialize_hidden_fields()` derives at runtime and the public set never
carries, and shipping code must not import `evaluator/` at all. The
**800/800 classifier agreement reported as validation was the tell** -- perfect
because the rules were transcribed rather than learned.

Rebuilt from catalog data alone the same policy scores -0.0104, so the leak was
worth +0.0038.

**What the leak was providing was a salience prior.** The clean policy's first
cut asked about `size` on **28%** of turns against a true rate of 1.4%: nearly
every product declares a size, so raw coverage rates the arm highly, and entropy
does not correct it because size values genuinely vary. What was missing is how
likely the customer is to *mention* it, which the card's ordering supplied for
free. The transferable substitute is **position within the product's own
listing**, which is why `catalog.py` builds position-weighted lead lines.

## 3.41 The specific-arm probe, implemented

3.37 and 3.38 as shipped code, behind live switches. `slots.TaxonomyBuilder`
learns a second, token-level vocabulary from the same `details` values;
`catalog._lead_lines` builds each product's opening lines typed and weighted by
`POSITION_DECAY ** rank`; `probe.specific()` scores each arm by
`coverage x spread x ARM_DECAY ** heard` over the live contenders.

**Public 200: 0.9554 to 0.9450**, hit@10 still 1.000, health clean, and the
wildcard is served on **0 of 445 turns**. Construction 3.07s to 5.65s; per-turn
latency p50 0.292 ms to 0.483 ms. Neither budget is close to binding.

**With both switches off the agent reproduces 0.9554 exactly**, scenario for
scenario, which is the property that makes it reversible. It ships **on** as a
deliberate trade: the score it costs on a saturated set buys a question channel
that is grounded in the live pool rather than in a fixed order (decision 30).

## 3.43 Conditional diversification, and a correction to 3.30

Measured 2026-08-29. **0.9488, unchanged.** Three switches gate slate
diversification -- route-conditional (`routing.DISCOVERY_DIVERSITY`), only while
the customer has said little (`ranking.DIVERSITY_MAX_CONSTRAINTS`), and on a
flatness test over the score distribution (`ranking.FLATNESS_GATE`). Gating
halves the unconditional loss at every matched weight and **never turns it
positive**; the best mean over the 15 readable sets is -0.0060. All three ship
neutral.

**The phase's real output is a correction.** 3.30's `thin_cards` win of +0.0756,
which motivated the whole line, **no longer reproduces**. The set is unchanged;
a different component moved it from 0.5661 to 0.8025, and diversity now loses
0.0147 on it. Generalised as **decision 25: a component's verdict expires when a
different component ships.**

## 3.44 Every remaining miss is a ranking failure, not a retrieval one

Measured after five separate attempts to differentiate retrieval by intent all
returned null or negative. This is the diagnostic that should have been run
first.

For every session on the hard sets, is the target inside the pool the agent
actually retrieved from?

| set | miss% | target in pool | pool median | misses that were reachable |
|---|---|---|---|---|
| `compound_hard` | 48% | **100%** | 591 | 96 of 96 |
| `thin_cards` | 10% | **100%** | 266 | 20 of 20 |
| `unpopular_targets` | 9% | **100%** | 325 | 18 of 18 |
| `comparative_constraints` | 5% | **100%** | 288 | 10 of 10 |
| `silent_customer` | 4% | **100%** | 283 | 7 of 7 |

**The category filter never fails, on any set, and every single miss is a
ranking failure inside a pool that already contains the answer.** This is
standing decision 27: before differentiating a stage, measure whether that stage
is what is failing.

## 3.45 The exploration band, and why diversity was losing

Measured 2026-08-29. **0.9490 to 0.9473 on the public 200, and up on every risk
column.** It came from decomposing a loss rather than from proposing a
mechanism.

Diversity's cost was **100% MRR while MTTC improved**:

| | hit@10 | MRR | MTTC | score |
|---|---|---|---|---|
| shipped | 1.000 | 0.9109 | 2.215 | 0.9490 |
| `DIVERSITY` 0.5, gated | 1.000 | **0.7394** | **1.720** | 0.9074 |

`0.3 x -0.1715` from MRR against `0.2 x +0.0495` from MTTC. Coverage never
moved. The change was making the agent converge faster while converging worse,
because `diversify` selected over `ordered[:WINDOW]` and so spent slots on ranks
2-10 -- the band deferred commitment deliberately withholds.

`ranking.explore()` restricts marginal relevance to `ordered[SLATE_SIZE:WINDOW]`,
replacing only the *fixed* reach to ranks 11-19 with an adaptive one.
`EXPLORE_DIVERSITY` = 0.95 and `EXPLORE_SORT` = True ship **live**.

This is the one change that knowingly trades reported score for robustness:
public -0.0017 against size-biased +0.0083, sqrt +0.0059, uniform +0.0025 and
worst-paraphrase +0.0054, with the frozen sets at +0.0031 (10 better, 4 worse).
Same trade `ALPHA` makes at 0.6 rather than the public optimum.
**Decision 28: decompose a loss before abandoning the mechanism.**

## 3.46 Phrasing is worth exactly zero to the score, and is built anyway

`evaluator/local_evaluator.py:243` type-checks `message` and never reads it:

```python
if not isinstance(response, dict) or not isinstance(response.get("message"), str):
```

`customer_reply()` branches on the `ask_attribute` enum alone. Scenario framing,
guided alternatives and confirmation framing are therefore all the same string
as far as the score is concerned.

**They are built regardless**, which is decision 30: the technical score sits
inside Technical Execution at 35%, against 65% of judging that reads the
transcript. A decision worth zero to the scorer is not worth zero to the
submission, and the two must be reported separately rather than conflated.

Note also that a **non-string `message` discards the entire response,
recommendations included** -- an agent returning the correct target at slot 1
with no `message` key scores 0.000. That is why the exception envelope in
`agent.py` guarantees a string on every path.

## 3.47 The dialogue policy layer, and the routing branch it repaired

`submission/src/policy.py` names one of six policies per turn -- `recovery`,
`coverage`, `stagnation`, `boundary`, `precision`, `discovery` -- from
`SessionState` alone, and `routing.choose()` reads that instead of deriving its
own. Four modules had each been inferring the same thing about the conversation
from the same state, and one of them was inferring it wrongly.

**The repair:** `SCOPED_EXHAUSTION` put a spent probe arm into `state.refused`,
routing read that as a customer refusal, and **74% of `compound_hard` turns took
the boundary branch while precision took 1.6%**. `SessionState` now carries
`declined` alongside `refused`; boundary is 1.9% and precision 22.5%, **at
exactly zero score change**, because every route already carried identical
constants.

That is **decision 29: a state machine whose branches all carry the same
constants is untested, not verified.** The bug was invisible for five phases of
review precisely because it could not move a number.

## 3.49 What the instrument already covers about rare targets

Every headline number is conditional on targets being drawn proportional to
`rating_number`. Four frozen sets already draw targets that are not popular.
Percentiles are against the catalog's own distribution (median 12, n=50,000):

| set | draw | median RN | catalog pctile | share <=12 reviews |
|---|---|---|---|---|
| public 200 | size-biased (real) | 6,846 | 99.5% | 2% |
| `sqrt_targets` | `sqrt(RN)` | 156 | 85.7% | 11% |
| `unpopular_targets` | uniform | 10 | 45.0% | 54% |
| `compound_hard` | uniform, crowded+thin, typo | 4 | 25.7% | 80% |

The instrument already spans the regime, so a rare-target risk does not need a
new set. What it needs is a mechanism, and 3.50 is why there isn't one.

## 3.50 Six attempts to balance the popular/rare trade, and why none of them does

The prior helps when the target is popular and hurts when it is rare, so is
there a formulation that does both? Six mechanism families were built and scored
through the real `evaluate()`. **None beats the shipped operating point.**

Every number is a paired comparison: the same manufactured rows, the same agent,
one mechanism swapped in by monkeypatch. No repo file changed and the frozen
manifest was read only.

| mechanism | public200 | sqrt | unpopular | rare q1 |
|---|---|---|---|---|
| shipped | 0.9473 | 0.8875 | 0.8427 | 0.7855 |
| sqrt prior | 0.9472 | 0.8871 | 0.8402 | 0.7426 |
| rank prior | 0.9373 | 0.8844 | 0.8201 | 0.6774 |

The families included prior compression, a prior-free band, pool-shape gating
and confidence gating. **All six keyed on a *pool-shape* proxy** -- contention,
flatness, read confidence, constraint count -- and all six were null or negative,
because those statistics measure how distinctive a ranking *looks*, never
whether it is *right*.

The trade is real, and `ALPHA` = 0.6 is already the hedge 3.20 chose for exactly
this reason. **Decision 33: an adaptive mechanism must key on something that is
about correctness**, and the session's own refuted slates are the only such
signal available here -- which is what `submission/src/orchestrate.py` keys on.

## 3.58 Phrase evidence picks the head, not just the served order

Measured 2026-08-31. **Public 0.9633 to 0.9672.**

The defect: the phrase bonus lived in `ranking.rerank`, which runs on the
**served slate**. Once the committed head resolved to one product on 429 of 482
public turns, permuting a one-element list did nothing. The sharpest signal the
catalog offers (3.3: 89.8% of phrases belong to a single product) had stopped
reaching the decision it is best at making.

`ranking.phrase_promoted` re-sorts `ordered[:PHRASE_POOL]` by `phrases.evidence`
*before* `head_size` picks the head. `PHRASE_POOL` = 20 ships **live**, so its
sweep row reads backwards: the 0 point is the deviation.

**It is a ranking gain rather than an exposure trade** -- MRR 0.972 to 0.975
*and* MTTC 2.41 to 2.27, with hit@10 unmoved at 1.000. Diffed session by
session: **24 sessions convert a turn earlier, 2 rank better, zero worse in
either direction.** The target is served on the turn its constraint is first
disclosed rather than one or two turns later, because the head is now chosen by
the evidence that names it.

**Every guarded column agrees and the pessimistic ones gain most**, which is the
signature of a mechanism rather than a fitted constant: uniform 0.9042 to
**0.9139**, sqrt 0.9288 to 0.9345, size-biased 0.9639 to 0.9644, dev 0.9595 to
0.9637, held 0.9672 to **0.9707**. Paraphrase: reworded +0.0016, punctuation
+0.0024, filler +0.0017, synonym **-0.0012** -- the one negative, and the
expected one, since synonym substitution is precisely what breaks an
exact-phrase key. Width is a plateau: 10, 15, 25 and 40 all sit within 0.0005 of
20, so 20 is not a fitted peak.

**A lucky bug, caught and not shipped.** Computing `contention` *after* the
promotion is worth a further **+0.0008** on the public 200 and +0.0000 on every
guarded column. It is not a mechanism: `contention` reads its argument as a
descending score list and `phrase_promoted` returns one sorted by evidence, so
the count is an arbitrary prefix. Confirmed by disabling `HEAD_FROM_CONTENTION`,
under which both orderings score 0.9672 exactly. The promotion is therefore
applied *after* `contention`, and the +0.0008 is declined.

Reproduce: `python3 -m harness.deviations --component phrase_pool`.

---

## Appendix A: phases

Development ran in numbered phases. Comments occasionally name one; this is what
each was.

| phase | what it did |
|---|---|
| **6R**, 6R.3 | Built the paraphrase gate, found the template cliff (3.24), and converted every simulator-specific shortcut into a fast path of a general mechanism. Added typed slots, routing and a derived probe (3.26). |
| **6S-B** | Re-swept every switched-off component against the frozen sets, where they had room to move (3.30). |
| **6T** | Fixed the two behaviours that were wrong rather than untuned: refusals scored as preferences, and slots spent on already-shown products (3.31, 3.32). |
| **6U** | Built the second retriever the brief names, measured it, and switched it off (3.35). 6U.0 re-read the instrument afterwards (3.34). |
| **6V** | Wired and bounded the model-backed rerank tier, and shipped it off (3.36). |
| **6W** | Built per-person memory and the rotation control that refuted most of it (3.48, summarised under 3.33). |
| **6X** | Built the six-policy dialogue layer and repaired the routing branch (3.47). |
| **6Y**, 6Y.0 | Adaptive orchestration keyed on refuted orderings. 6Y.0 is the gate that asked whether a portfolio of orderings exists *before* a controller was written (`python3 -m harness.orderings`). |
| **6Z** | Made the slate short: a withheld slot stays empty instead of being refilled, the committed head is derived from contention, and the deferral budget was refitted. The largest single gain in the project. |

## Appendix B: standing decisions

The numbered decisions a comment may cite.

| # | decision |
|---|---|
| **7** | **Index `title` + `features`. Do not index `description`.** It costs 4.5 points of hit@10 and 7 of MRR (3.13). The evaluator's own `SEARCH_FIELDS` list is a specification of what the simulator reads, not a recommendation. |
| **23** | **Measure the mechanism before the lever.** A component is an answer to a question about how the system fails; check that the failure happens at the claimed rate before tuning the answer. Cost: an entire dense-retrieval phase built on a fallback that fires 0.0% of the time (3.35). |
| **27** | **Before differentiating a stage, measure whether that stage is what is failing.** Five attempts to differentiate retrieval by intent all returned null or negative; one recall measurement showed retrieval was already perfect and every miss was a ranking failure (3.44). |
| **31** | **Build the control that can refute the phase, and report it when it does.** Rotating shopper identities so every shopper reads a stranger's memory preserved the entire efficiency saving, which showed that half of it was a generic prior rather than personalisation (3.48). |
