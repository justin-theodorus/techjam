# SATU: Technical Devpost

**TechJam 2026 — Problem Statement #4: Shopping Copilot**

**Repository:** https://github.com/justin-theodorus/techjam

---

## Technical Summary

SATU is a deterministic, stateful shopping agent for a 50,000-product catalog. It treats recommendation as a sequential decision problem: every exposed product consumes a ranking position, and every clarification question consumes a turn.

The system dynamically controls both:

- **slate width:** how many products to expose now
- **question budget:** whether another clarification turn is useful

SATU runs entirely in memory using the Python standard library. It makes no external calls on the scored path.

| Metric         |           SATU | BM25 baseline |
| -------------- | -------------: | ------------: |
| TechnicalScore |     **0.9672** |        0.1067 |
| HitRate@10     |      **1.000** |         0.125 |
| MRR            |      **0.975** |         0.068 |
| MTTC           |       **2.27** |          9.81 |
| Warm latency   | **~1 ms/turn** |   8.5 ms/turn |
| Runtime cost   |      **$0.00** |         $0.00 |

## System Objective

The evaluator terminates a session as soon as the hidden product appears in the returned slate. This creates an asymmetric cost:

- showing the correct product converts the session
- showing an incorrect product proves it is not the target
- hiding a plausible product preserves it for later reranking
- asking a question is useful only if its answer can change a future slate

SATU therefore optimizes the joint objective rather than HitRate alone:

```text
high target coverage
+ high reciprocal rank
+ low turns to conversion
```

The engineering objective goes beyond the benchmark. We wanted a system that could be implemented in a real shopping product: low-latency, inexpensive, explainable, resilient to missing services, and able to justify both the products it shows and the questions it asks. That is why the scored path uses catalog-derived evidence, bounded session state, deterministic fallbacks, and no external runtime dependency.

### Interpreting Buying and Browsing

We did not assume that “buying” and “browsing” require unrelated retrieval systems. The public sessions show a clear difference in **when information arrives**, but not in the information ultimately available:

| Intent   | Constraints before turn 1 | Turn 2 | Turn 3 | Turn 4 |
| -------- | ------------------------: | -----: | -----: | -----: |
| Buying   |                      1.00 |   3.00 |   4.00 |   4.00 |
| Browsing |                      0.00 |   2.00 |   4.00 |   4.00 |

Buying sessions begin one constraint ahead; both intents contain the same mean number of constraints from turn 3 onward. We therefore interpret intent as a **conversation-policy signal**—how quickly SATU should move from discovery to precision—not as evidence that the catalog needs a different semantic retriever. Five attempts to give each intent separate retrieval weights were neutral or negative, so the shipped retrieval constants remain shared.

The public score supports that interpretation: buying reaches conversion in 1.79 turns and browsing in 2.05, while both achieve 100% HitRate@10 and nearly identical MRR (0.967 versus 0.970). The small timing gap follows the one-turn information lead; it is not evidence of weaker retrieval for browsing.

## Mathematical Formulation

### Evaluation Objective

For `N` sessions, target rank `r_i`, and first-hit turn `t_i`:

```text
HitRate@10 = (1/N) Σ 1[r_i ≤ 10]
MRR        = (1/N) Σ 1/r_i             (misses contribute 0)
MTTC       = (1/N) Σ t_i               (misses are assigned turn 11)
Efficiency = clip((11 − MTTC) / 10, 0, 1)

TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·Efficiency
```

This weighting explains why showing ten products immediately is not optimal: it can improve MTTC while reducing reciprocal rank enough to lower the combined score.

### Lexical and Popularity Ranking

SATU uses BM25 with `k₁ = 0.6` and `b = 0.3`:

```text
IDF(q) = ln(1 + (N − df(q) + 0.5) / (df(q) + 0.5))

BM25(d,Q) = Σ IDF(q) · tf(q,d)·(k₁+1)
                         ─────────────────────────────
                         tf(q,d)+k₁·(1−b+b·|d|/avgdl)
```

The normalized lexical score is blended with the normalized popularity prior:

```text
S(d) = BM25(d,Q) / max BM25 + 0.6 · Popularity(d) / max Popularity
```

Moving from textbook BM25 parameters (`1.2`, `0.75`) to (`0.6`, `0.3`) improved TechnicalScore by 0.030. The shorter catalog documents need less term-frequency saturation and less length normalization.

### Candidate Contention

“Contention” has one exact meaning: the number of top-ranked products whose blended scores are within a relative margin `m` of the leader.

```text
C(m) = |{d : S(d) ≥ S(best)·(1−m)}|
```

The shipped margin is:

```text
m = 0.0005 = 0.05%
```

Example: if the leader scores `1.2000`, a product is still in contention only when its score is at least `1.2000 × 0.9995 = 1.1994`.

This is not model fine-tuning and it is not an arbitrary “candidate confidence” label. It is a measured tolerance applied to the deterministic blended scores. The committed head is:

```text
head = min(10, max(1, C(0.0005)))
```

unless the shopper is exhausted, has fully disclosed their constraints, or has passed the deferral budget, in which case SATU opens the full slate.

The margin was selected from a sweep rather than from the single best-looking point. At the evaluation stage where this sweep was run, every value from `0` through `0.0009` produced the same TechnicalScore, `0.9633`; performance first fell at `0.001`. We chose `0.0005` because it lies in the middle of that stable plateau rather than at its edge. A later phrase-promotion improvement raised the final system from `0.9633` to `0.9672`; it did not change the definition of contention.

| Relative margin |   Meaning | TechnicalScore change from the sweep reference |
| --------------: | --------: | ---------------------------------------------: |
|      `0–0.0009` | `0–0.09%` |               No change; score remained 0.9633 |
|    **`0.0005`** | **0.05%** |                       **Shipped, mid-plateau** |
|         `0.001` |     0.10% |                                        −0.0006 |
|          `0.01` |     1.00% |                                        −0.0020 |
|          `0.02` |     2.00% |                                        −0.0038 |
|          `0.05` |     5.00% |                                        −0.0165 |

At `0.0005`, the public run produced head widths `{1: 429, 2: 1, 10: 52}` across 482 turns. At `0.05`, the width spread to `{1: 228, 2: 99, 3: 36, 4: 20, 5: 3, 6: 2, 9: 2, 10: 35}`. The ranking is therefore usually decisive at the shipped 0.05% tolerance; it is not being forced to one product by a hardcoded constant.

### Question Information Value

For attribute `a`, SATU measures the distribution of its remaining catalog values with Shannon entropy:

```text
p(v|a) = weight(v,a) / Σᵤ weight(u,a)
H(a)   = −Σᵥ p(v|a)·log₂ p(v|a)
```

The question score is:

```text
Q(a) = Coverage(a) · H(a) · 0.35^heard(a)
```

`Coverage(a)` is the weighted share of up to 150 live products that provide a value for the attribute. Entropy rewards a useful split: an attribute has little value if every product gives the same answer. The decay term reduces repeated questions about an attribute already discussed. Refused, remembered-refused, and temporarily avoided attributes are assigned no question score.

SATU asks the highest-scoring specific attribute only if its coverage is at least 20% of the wildcard question's union coverage; otherwise it falls back to an open question. It emits no question when the shopper is fully exhausted or when no later turn can use the answer.

### Decision Readiness

SATU computes a bounded turn-level readiness value:

```text
Dₜ = clip(0.7·Currentₜ + 0.3·Dₜ₋₁, 0, 1)
```

`Currentₜ` increases with a recognized category, new typed constraints, decisive attributes, urgency, a small candidate pool, and a current pivot. It decreases with a large pool, many previous contenders, browsing without a new constraint, refusals, idle turns, and exhaustion.

The 0.7/0.3 split makes new evidence dominant while retaining some conversational continuity. A decisive correction can move readiness quickly, while the 0.3 prior prevents one vague turn from erasing the session's accumulated direction. Values at or above `0.7` are reported as precision-ready; values below `0.3` are discovery-leaning.

For accuracy, readiness is currently a **trace and explanation signal**, not an active ranking weight. Its steering switch ships off because the full sweep changed zero policy decisions: readiness was correlated with evidence already present in the discovery and precision scores. We retain the value because it explains how decision state evolves without claiming an evaluation gain it did not produce.

## Architecture

```text
                    STARTUP
50,000 products
      │
      ├── category buckets
      ├── tokenizer and BM25 statistics
      ├── popularity prior
      ├── attribute vocabulary
      ├── rare-phrase index
      └── optional dense asset (weight 0)
                    │
                    ▼
              IN-MEMORY CATALOG
                    │
                    ▼
                  TURN
message → parse → fold state → select policy → retrieve → rank
                                              │
                                              ├── shape product slate
                                              └── choose or suppress question
                                                        │
                                                        ▼
                                           response + session trace
```

Startup takes approximately 12 seconds and 242 MB. After initialization, a warm turn takes approximately 1 ms and requires no disk or network access.

## Turn Pipeline

| Stage           | Input                                  | Decision or output                                                |
| --------------- | -------------------------------------- | ----------------------------------------------------------------- |
| Understanding   | Shopper message                        | Category, constraints, pivot, refusal, or exhaustion              |
| State fold      | Parsed turn + previous state           | Updated slots, refusals, history, and shown products              |
| Policy          | Current state and candidate counts     | Discovery, precision, recovery, boundary, stagnation, or coverage |
| Retrieval       | Active category and constraints        | Candidate pool                                                    |
| Ranking         | Candidate pool and query signals       | Ordered candidates and confidence structure                       |
| Slate shaping   | Scores, turn, route, and history       | Dynamic number of visible products                                |
| Probe selection | Live candidates and conversation state | One useful question or no question                                |
| Response        | Slate, policy, and probe               | Shopper-facing message and products                               |

Every decision is recomputed from the latest state. SATU does not follow a fixed dialogue tree or assign one permanent persona to a session.

## Conversation State

Each message is converted into typed preference slots:

```text
category = crossbody bag
feature  = imported
material = leather
color    = black
```

The session state also stores:

- current category and candidate buckets
- active constraints and their attributes
- refused or exhausted attributes
- products returned on previous turns
- the last question asked
- consecutive questions that added no information
- pivots and intent overrides
- previous candidate contention and decision readiness

### Targeted Override

When a shopper changes one preference, SATU removes only the slots contradicted by the new statement. For example, “Actually, make it brown” replaces the active color but preserves category, material, and feature constraints.

This avoids both common extremes:

- merging mutually inconsistent preferences
- discarding the entire session after a local correction

Targeted override improved the overall score by 0.016 and raised intent-override HitRate@10 from 0.900 to 1.000.

### Exhaustion and Refusal

SATU distinguishes “nothing more about material” from “I have no more preferences.” Attribute-specific exhaustion retires only that dimension. Full exhaustion stops further probing and opens product coverage.

This distinction prevents one unanswered question from prematurely ending all clarification.

### Cross-Session Memory

Only low-risk signals carry across visits:

- refusals
- category affinity
- attributes that should not be asked again immediately

Historical signals decay by `0.7` per visit. Positive product preferences do not automatically carry because a new visit may represent a different shopping mission.

## Retrieval and Ranking

```text
hard category filter
→ BM25 relevance
→ popularity prior
→ negation penalty
→ rare-phrase promotion
→ optional reranking seam
```

### Category Filtering

The hard filter reduces the full catalog to a median of 182 candidates while retaining the target 99.0% of the time. This makes later ranking faster and more precise than applying all signals globally.

### BM25 and Popularity

BM25 responds to explicit shopper language. The popularity prior provides a stable fallback when the query is sparse or uses different wording from the catalog.

### Negation

Refused attributes are removed from the positive query and penalized separately. The cue list is deliberately narrow: the strings `no` and `non` occur inside 431 valid catalog attributes, so broad substring matching damages retrieval.

### Rare Phrases

Specific phrases found in the catalog can promote candidates after the base rank is computed. This stage only reorders already-valid candidates; an unseen phrase cannot inject an unrelated product.

## Dynamic Product Exposure

SATU treats a slate as a commitment. It does not fill ten positions merely because ten are available. The exact 0.05% contention rule above determines the committed head. Products outside that band remain unshown instead of being used to fill ranks 2–10, preserving them for reranking after new evidence arrives.

Previously shown products are removed. If the evaluator continued after they were shown, they are proven non-targets. Before this rule, repeats represented 62.9% of impressions on the hardest test set.

The final end-to-end ablation, after phrase promotion was added, produced:

| Strategy                  |      Relative margin |      Score |        MRR |     MTTC | Avg. shown |
| ------------------------- | -------------------: | ---------: | ---------: | -------: | ---------: |
| **Shipped dynamic slate** | **`0.0005` = 0.05%** | **0.9672** | **0.9750** |     2.27 |       1.74 |
| Wider contention          |          `0.01` = 1% |     0.9641 |     0.9617 |     2.22 |       1.77 |
| Wider contention          |          `0.05` = 5% |     0.9441 |     0.8825 |     2.03 |       2.20 |
| Wider contention          |         `0.25` = 25% |     0.8984 |     0.7045 |     1.65 |       6.53 |
| Always show ten           |       Not applicable |     0.8946 |     0.6901 | **1.62** |       9.84 |

Always showing ten reaches the target slightly earlier, but frequently buries it. Dynamic exposure gives up a small amount of MTTC to gain substantially more MRR and total score.

## Dynamic Question Control

SATU does not use a fixed questionnaire or fixed number of questions. It emits at most one clarification question per turn and may emit none.

### When SATU Asks

Nine attribute arms compete:

```text
material, color, size, style, use case,
feature, budget, brand, category
```

Each arm is scored against up to 150 live candidates using:

```text
expected value = coverage × value spread × repeat decay
```

- **Coverage** estimates how many remaining products can answer that attribute.
- **Value spread** rewards attributes whose values divide the pool.
- **Repeat decay** discounts dimensions the shopper has already described.

An attribute nobody can answer scores zero. An attribute with the same value across all candidates also provides little separation. Question options are catalog-derived and limited to two or three plausible values.

### When SATU Stops Asking

The question count emerges from the conversation. SATU suppresses a question when:

- the shopper has explicitly run out of preferences
- every useful attribute has been refused or exhausted
- the final protocol turn leaves no future slate that could use the answer
- the policy switches from information gathering to coverage

After two answered questions add no new constraint, stagnation handling forces the next probe onto an untouched dimension. This prevents SATU from repeatedly asking sharper versions of the same unproductive question.

This makes question selection dynamic in both dimensions: **what to ask and whether to ask at all**.

## Dynamic Policy Selection

Six policies compete on every turn:

| Policy     | Trigger                             | Effect                                         |
| ---------- | ----------------------------------- | ---------------------------------------------- |
| Discovery  | Sparse constraints or a broad pool  | Seek a high-value preference                   |
| Precision  | Several decisive constraints        | Narrow and commit carefully                    |
| Recovery   | Shopper pivot or disproven ordering | Replace stale state and rerank                 |
| Boundary   | Shopper refuses a dimension         | Respect the refusal and redirect               |
| Stagnation | Repeated answers add no information | Change the question dimension                  |
| Coverage   | Few useful turns remain             | Stop over-narrowing and expose more candidates |

The policy layer coordinates retrieval, slate shaping, question selection, and response wording. The same shopper can move through several policies within one session.

### Ranking Recovery

SATU treats served failures as evidence. When enough products from an ordering are shown without conversion, it reranks the same candidate pool under an alternative ordering.

We rejected score flatness, read confidence, ranking contention, and raw constraint count as primary triggers because they describe uncertainty but do not prove that the ranking is wrong. A shown product that fails to convert does.

Recovery activates on 2.1% of public turns and 46.4% of turns in the hardest set.

## Evaluation

| Scenario        |       n | HitRate@10 |       MRR |     MTTC |      Score |
| --------------- | ------: | ---------: | --------: | -------: | ---------: |
| Buying          |      80 |      1.000 |     0.967 |     1.79 |     0.9744 |
| Browsing        |      80 |      1.000 |     0.970 |     2.05 |     0.9701 |
| Intent override |      30 |      1.000 |     1.000 |     4.03 |     0.9393 |
| Boundary        |      10 |      1.000 |     1.000 |     2.50 |     0.9700 |
| **Overall**     | **200** |  **1.000** | **0.975** | **2.27** | **0.9672** |
| BM25 baseline   |     200 |      0.125 |     0.068 |     9.81 |     0.1067 |

SATU achieves 9.1× the baseline TechnicalScore and 3.9× fewer turns, with zero exceptions, discarded outputs, or dropped slots.

## Test Strategy

The public set is close to saturated: 176 of 200 sessions already convert at rank 1. We therefore use it as a regression gate and supplement it with 580 automated tests and 23 deterministic hard-session sets.

The hard sets vary:

- product-description thinness
- target popularity
- constraint accumulation
- shopper pivots
- silence and unhelpful answers
- target sampling distribution

Seeds were frozen before individual features were evaluated.

| Gate                       | Result                      |
| -------------------------- | --------------------------- |
| Held-out split             | Dev 0.9637; held-out 0.9707 |
| Size-biased targets        | 0.9644                      |
| Square-root targets        | 0.9345                      |
| Uniform targets            | 0.9139                      |
| Paraphrased sessions       | 0.9318–0.9590               |
| Template matching disabled | 0.9660                      |
| Returning-shopper memory   | 0.70 fewer turns            |

## Measured Trade-Offs

Features remain disabled when they lose under the frozen evaluation gates, even if they make the architecture appear more sophisticated.

| Experiment                      | Measurement                                  | Shipped state |
| ------------------------------- | -------------------------------------------- | ------------- |
| Per-route popularity weights    | +0.014 dev, −0.002 held-out                  | Disabled      |
| Restart turn budget after pivot | Improved one condition, regressed four gates | Disabled      |
| MMR diversity                   | Lost on 13 of 18 hard sets                   | Disabled      |
| Early convergence               | MRR fell from 0.909 to 0.781                 | Disabled      |
| Profile-weighted ranking        | Negative on 18 of 18 sets                    | Disabled      |
| Wider browsing slates           | MRR fell from 0.9704 to 0.8942               | Disabled      |
| Dense retrieval                 | Lost on 14 of 15 readable sets               | Weight 0      |

Each component retains its implementation, switch, measurement, and a test confirming its default state.

### LLM Reranking

An optional `claude-haiku-4-5` reranking tier was tested through 323 live API calls.

| Measure          | LLM tier | Offline ranker |
| ---------------- | -------: | -------------: |
| Score            |   0.9333 |     **0.9554** |
| Latency per turn | 1,087 ms |     **2.5 ms** |
| Cost per run     |   $0.385 |      **$0.00** |

It rescued 6 of 20 difficult sessions but changed enough correct rankings to lower the aggregate score. The tier remains implemented as a safe optional layer; failures return the original slate unchanged. It is disabled for scoring.

### Dense Retrieval

We built a 64-dimensional latent representation of the catalog. It lost on 14 of 15 readable test sets because its strongest dimensions mostly encoded category, which the hard category filter already represents more precisely. The asset is 4.92 MB, ships at weight zero, and can be deleted without changing scored output.

## Stack and Data

| Area                   | Details                                                                                  |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| Scored runtime         | Python standard library only                                                             |
| Core modules           | `math`, `array`, `struct`, `re`, `dataclasses`, `pathlib`, `collections`, `enum`, `json` |
| Development            | VS Code, Claude Code, Git, GitHub, GNU Make                                              |
| Tests                  | Python `unittest`, 580 automated tests                                                   |
| Runtime APIs           | None                                                                                     |
| External training data | None                                                                                     |
| Product catalog        | 50,000 frozen organizer products                                                         |
| Public evaluation      | 200 labelled development sessions                                                        |
| Additional evaluation  | 23 deterministic catalog-derived session sets                                            |

The product catalog is derived from **Amazon Reviews 2023 — Clothing, Shoes and Jewelry** by McAuley Lab, UCSD, and was verified against the published SHA256 checksum. SATU uses no scraping, catalog mutation, or manually labelled training data.

NumPy is used only by a standalone preprocessing script for the disabled dense asset. It is never imported by the scored agent.

## Limitations

- Constraint language must map reasonably well to attributes found in the catalog.
- Information-value scoring estimates product separation, not an individual shopper's willingness to answer.
- Shoppers cannot directly correct an incorrectly assigned attribute label; SATU and organizer labels disagree on 142 of 800 values.
- Persistent shopper identity is unavailable in the provided interface, so cross-session memory cannot be fully evaluated.
- Decision readiness resets between shopping missions to avoid transferring stale confidence.

## Next Steps

- Add a direct correction loop for attribute-label errors.
- Model shopper-specific question answerability alongside product information gain.
- Learn when decision readiness can safely persist across visits.
- Gate LLM reranking on expected benefit rather than enabling it globally.
- Evaluate the in-memory architecture at catalog sizes approaching five million products.

---

## Repository

The implementation, tests, evaluation harness, disabled experiments, and reproducible commands are available at:

**https://github.com/justin-theodorus/techjam**
