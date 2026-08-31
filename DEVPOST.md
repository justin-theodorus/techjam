# SATU: Shopping Assistant That Understands

**TechJam 2026 — Problem Statement #4: Shopping Copilot**

**Repository:** https://github.com/justin-theodorus/techjam

---

## Summary

SATU is a multi-turn shopping agent built around one constraint: **every product shown and every question asked has a cost**.

Instead of immediately returning ten products or forcing a fixed interview, SATU tracks the shopper's preferences, narrows the candidate pool, dynamically chooses how many products to reveal, and decides whether another question is worth asking.

| Result | SATU | BM25 baseline |
| --- | ---: | ---: |
| TechnicalScore | **0.9672** | 0.1067 |
| HitRate@10 | **100%** | 12.5% |
| MRR | **0.975** | 0.068 |
| Mean turns to conversion | **2.27** | 9.81 |
| Warm latency per turn | **~1 ms** | 8.5 ms |
| Runtime API cost | **$0.00** | $0.00 |

Across all 200 public sessions, SATU found the hidden product every time and ranked it first in 97.5% of sessions. The scored system uses no external API calls, tokens, or third-party runtime libraries.

---

## The Problem

Traditional search returns a large result set and makes the shopper do the narrowing. A useful sales assistant behaves differently: it listens, shows a few plausible options, asks a focused question, and adapts when the shopper changes their mind.

The evaluator makes this a sequential decision problem. A session ends when the hidden target appears, so showing a product is irreversible. Returning too many products may find the target sooner, but can bury it at a low rank and waste information that later turns could have used.

SATU balances coverage, precision, and speed.

## How SATU Addresses It

| Shopping challenge | SATU's response | Why it matters |
| --- | --- | --- |
| Different shoppers need different interactions | Classifies buying, browsing, boundary, and intent-override behaviour | A browser can explore sooner, while a focused buyer keeps narrowing |
| Preferences accumulate and change | Stores typed slots and applies targeted erasure when a preference is corrected | Preserves valid context without keeping contradictions |
| Large result sets waste ranking positions | Filters by category, reranks candidates, and reveals only near-tied leaders | Protects precision while retaining strong options for later turns |
| Generic questions add friction | Selects the unanswered attribute with the highest expected information value | Each turn reduces uncertainty efficiently |
| A fixed question count can over-question the shopper | Asks at most one question per turn and stops when no useful answer remains | The conversation uses only the clarification turns it needs |
| Irrelevant products create noise | Keeps weak candidates outside the visible slate rather than filling empty positions | Shoppers see only products SATU can justify showing |
| Repeated recommendations provide no value | Removes previously shown products from future results | If the session continues, a shown product is known not to be the target |
| An early ranking can be wrong | Detects failed rankings and switches to an alternative ordering | Recovery activates when observed results disprove the current strategy |
| Runtime dependencies add cost and risk | Builds catalog indexes once and runs entirely in memory | Delivers ~1 ms warm turns with no external services |

---

## How It Works

At startup, SATU builds in-memory indexes for the frozen 50,000-product catalog. On every turn, it:

1. reads the shopper's message
2. extracts and updates preference slots
3. classifies the shopper and current behaviour
4. filters and ranks candidate products
5. decides how many products to reveal
6. decides whether to ask a question and, if so, selects the most informative one
7. records the outcome for the next turn

### Hybrid Retrieval

```text
category filter
→ BM25 relevance
→ popularity prior
→ negation handling
→ rare-phrase promotion
→ optional reranking seam
```

The category filter reduces 50,000 products to a median of 182 candidates while retaining the correct product 99.0% of the time. The remaining stages combine direct keyword relevance with catalog popularity, explicit refusals, and specific phrases.

### Multi-Turn Preference State

SATU converts each message into structured slots:

```text
category = crossbody bag
feature  = imported
material = leather
color    = black
```

When a shopper changes one preference, targeted erasure removes only the contradicted value. Unrelated constraints remain active. SATU also records refusals, shown products, unanswered attributes, and category affinity.

Cross-session memory retains decayed refusal and category signals at `0.7` per visit. Positive preferences do not automatically carry over because a new visit may represent a different shopping mission.

### Deferred Commitment

SATU does not automatically fill all ten result positions. Products within `0.0005` of the leading score may be shown; other candidates are held at rank 11 or below so later information can promote them.

| Slate strategy | Score | MRR | MTTC | Avg. shown |
| --- | ---: | ---: | ---: | ---: |
| **SATU dynamic slate** | **0.9672** | **0.9750** | 2.27 | 1.74 |
| Tie margin 0.01 | 0.9641 | 0.9617 | 2.22 | 1.77 |
| Tie margin 0.05 | 0.9441 | 0.8825 | 2.03 | 2.20 |
| Tie margin 0.25 | 0.8984 | 0.7045 | 1.65 | 6.53 |
| Show all ten | 0.8946 | 0.6901 | **1.62** | 9.84 |

Wider slates slightly reduce conversion time, but MRR falls much faster. Dynamic slates balance finding the product with ranking it correctly.

### Informative Questions

Nine attributes compete based on how evenly they divide the remaining candidates, how often they occur, and whether the shopper has answered them. Options come from products still under consideration.

Instead of asking “Do you have a material preference?”, SATU can ask “Leather, nylon, or PU leather?”

The number of questions is dynamic rather than preset. SATU asks no more than one focused question per turn, retires attributes the shopper has declined, changes dimension after repeated unhelpful answers, and asks nothing when the shopper has no preferences left or the answer cannot affect a later turn.

### Runtime Recovery

Six behaviours compete on each turn: discovery, precision, recovery, boundary, stagnation, and coverage. Once enough products from one ordering have failed, SATU reranks the pool with an alternative strategy.

The recovery trigger fires on 2.1% of public turns and 46.4% of turns in the hardest test set: mostly inactive when the primary ranking works, but active when observed results show that it does not.

---

## Evaluation

We ran the organizer's evaluator across all 200 public sessions.

| Scenario | Sessions | HitRate@10 | MRR | MTTC | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buying | 80 | 1.000 | 0.967 | 1.79 | 0.9744 |
| Browsing | 80 | 1.000 | 0.970 | 2.05 | 0.9701 |
| Intent override | 30 | 1.000 | 1.000 | 4.03 | 0.9393 |
| Boundary | 10 | 1.000 | 1.000 | 2.50 | 0.9700 |
| **Overall** | **200** | **1.000** | **0.975** | **2.27** | **0.9672** |
| BM25 baseline | 200 | 0.125 | 0.068 | 9.81 | 0.1067 |

This is a **9.1× improvement in TechnicalScore** and **3.9× fewer turns** than the baseline, with zero exceptions, discarded outputs, or dropped slots.

## Robustness and Engineering Decisions

Because 176 of 200 public sessions already convert at rank 1, the public benchmark is better at detecting regressions than improvements on difficult cases. We added 580 automated tests and 23 deterministic hard-session sets covering thin descriptions, low-popularity targets, accumulated constraints, shopper pivots, and silence.

| Robustness gate | Result |
| --- | --- |
| Held-out split | Dev 0.9637; held-out 0.9707 |
| Target distribution | Size-biased 0.9644; square-root 0.9345; uniform 0.9139 |
| Paraphrased intent | 0.9318–0.9590 |
| Template matching disabled | 0.9660 |
| Returning-shopper memory | 0.70 fewer turns |
| Dense retrieval forced on | Lost on 14 of 15 readable sets |

Several complete features remain behind switches because measurement showed that they weakened the scored system.

| Experiment | Evidence | Decision |
| --- | --- | --- |
| Per-route popularity weights | +0.014 dev, −0.002 held-out | Disabled |
| Restart budget after a pivot | +0.037 once, but regressed four gates | Disabled |
| MMR diversity | Lost on 13 of 18 hard sets | Disabled |
| Early convergence | MRR fell from 0.909 to 0.781 | Disabled |
| Profile-weighted ranking | Negative on 18 of 18 sets | Disabled |
| Wider browsing slates | Browsing MRR fell from 0.9704 to 0.8942 | Disabled |
| 64-dimensional dense retrieval | Lost on 14 of 15 readable sets | Weight set to zero |
| LLM reranking | Lower score, slower, and more expensive | Disabled |

Each disabled component keeps its implementation, feature switch, evaluation result, and a test confirming that it remains off.

### LLM Reranking Experiment

We evaluated a `claude-haiku-4-5` reranking tier using 323 live API calls.

| Measure | With LLM | Offline ranking |
| --- | ---: | ---: |
| Score | 0.9333 | **0.9554** |
| Latency per turn | 1,087 ms | **2.5 ms** |
| Cost per run | $0.385 | **$0.00** |

The model rescued 6 of 20 difficult sessions, but disturbed enough already-correct rankings to reduce the overall score. It remains available as a failure-safe optional layer but is disabled on the scored path.

---

## Technology and Data

| Area | What we used |
| --- | --- |
| Runtime | Python standard library only |
| Core modules | `math`, `array`, `struct`, `re`, `dataclasses`, `pathlib`, `collections`, `enum`, `json` |
| Development | VS Code, Claude Code, Git, GitHub, GNU Make, `unittest` |
| Runtime APIs | None |
| External training data | None |
| Product data | Organizer's frozen 50,000-product catalog |
| Evaluation data | 200 public sessions and 23 deterministic hard-session sets |

The catalog is derived from **Amazon Reviews 2023 — Clothing, Shoes and Jewelry** by McAuley Lab, UCSD, and was verified against the published SHA256 checksum. SATU does not scrape data, mutate the catalog, or use a manually labelled training set.

An offline NumPy preprocessing script generated an experimental 4.92 MB, 64-dimensional latent-space asset. The scored agent never imports NumPy, the dense track has weight zero, and deleting the asset does not change scored behaviour.

| Catalog size | Cold start | Memory | Warm turn |
| ---: | ---: | ---: | ---: |
| 50,000 products | 12 s | 242 MB | ~1 ms |

---

## Challenges and Learnings

- **A saturated benchmark can hide weaknesses.** Hard-session sets were necessary to test improvements aimed at the remaining difficult cases.
- **Negation is more subtle than keyword removal.** Bare `no` and `non` occur inside 431 valid catalog attribute names, so SATU uses a narrow cue list and separates refused terms from the positive query.
- **More complex does not mean better.** Dense retrieval, wider slates, diversity ranking, profile weighting, and LLM reranking all looked promising but lost under controlled evaluation.
- **Observed failure is the strongest recovery signal.** A product shown without conversion is better evidence than score flatness or constraint count alone.

## Limitations

- Constraint language must resemble attributes present in the catalog.
- Question selection models how well an attribute divides products, not how likely a particular shopper is to answer it.
- Shoppers cannot directly correct an incorrectly classified attribute label; SATU and organizer labels disagree on 142 of 800 values.
- The interface does not expose persistent shopper identity, limiting evaluation of cross-session memory.
- Decision readiness resets between sessions to prevent confidence leaking between unrelated shopping missions.

## What's Next

- Add a correction loop for misclassified preference labels.
- Model which questions each shopper is most likely to answer.
- Learn when decision confidence can safely persist across visits.
- Trigger LLM reranking only when its expected benefit exceeds the risk of changing a correct ranking.
- Test in-memory indexing at catalog sizes closer to five million products.

---

## Repository

The implementation, evaluation harness, automated tests, experimental components, and reproducible commands are available at:

**https://github.com/justin-theodorus/techjam**
