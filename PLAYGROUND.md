# 🛍️ TikTok Shopping Agent Playground — Quick Start

Two ways to play with the agent:

## Option 1: Interactive Chat (Full Playground)

Requires the full 50,000-product catalog.

### Setup

```bash
cd /Users/catherinekang/Documents/tiktoktechjam2026

# Download catalog (from GitHub Release)
# Visit: https://github.com/[repo]/releases
# Download: catalog.jsonl.gz

# Decompress it
gzip -dkc catalog.jsonl.gz > techjam/data/catalog.jsonl

# Verify (should be ~19 MB)
ls -lh techjam/data/catalog.jsonl
```

### Run It

```bash
cd techjam
PYTHONPATH=.. python playground.py
```

### Example Conversation

```
[Turn 1]
You: I need running shoes for jogging

Agent: I can help you find running shoes! What's your preferred price range?
🔍 Asking about: price_range

📋 Top recommendations (10 shown):
1. Nike Air Zoom Pegasus 39
   ASIN: B09TQRW4L4 | Price: $129.99 | Rating: 4.7⭐ (score: 0.9823)

[Turn 2]
You: Something under $80

Agent: Great! Here are running shoes under $80...
📋 Top recommendations (10 shown):
1. ASICS Gel-Contend 7
   ASIN: B09D7P5J8R | Price: $74.95 | Rating: 4.6⭐ (score: 0.8941)
```

---

## Option 2: Watch a Real Conversation (Demo)

Shows how the agent solved one of the 200 real public sessions.

### Run It

```bash
cd /Users/catherinekang/Documents/tiktoktechjam2026
PYTHONPATH=. python techjam/demo.py
```

### What You'll See

- Real customer message from evaluation data
- How the agent parsed it
- Top 10 product recommendations
- Whether it found the target product

Example output:

```
📊 Session Type: buying
🎯 Target Product: B09ABC123DE
👤 User Profile: {'average_rating': 4.2, 'buying_frequency': 'monthly', ...}

💬 Replaying session...

[Turn 1] You: I need a leather backpack
Agent: I can help you find a leather backpack! Do you prefer a laptop compartment?
📋 Top 10 recommendations:
   1. B09ABC123DE (score: 0.9542) ✓ TARGET!

✅ SUCCESS: Agent found the target at rank 1 on turn 1!
```

---

## Commands

### From root directory (`/Users/catherinekang/Documents/tiktoktechjam2026`)

```bash
# Watch a real session replay
PYTHONPATH=. python techjam/demo.py

# Or enter the techjam folder
cd techjam
PYTHONPATH=.. python playground.py
```

### Inside `techjam/` folder

```bash
# Interactive playground (requires full catalog)
python playground.py

# Or run the full test suite
PYTHONPATH=. python -m harness.run --agent submission.agent:Agent
```

---

## Debug Mode

In the interactive playground, type `debug` during a conversation to see:

- Internal state (what slots are filled)
- Feature importances
- Retrieved product scores
- LLM token usage (if enabled)

---

## Files Created

- `playground.py` — Interactive chat with the agent
- `demo.py` — Replay one real evaluation session
- `PLAYGROUND_SETUP.md` — Detailed setup guide

---

## Next Steps

1. **Get the catalog**: Download `catalog.jsonl.gz` from GitHub Releases
2. **Run the demo**: See a real session replay (no catalog needed)
3. **Try the playground**: Chat freely with the agent
4. **Check the score**: `PYTHONPATH=. python -m harness.run --agent submission.agent:Agent`

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'techjam'` | Run with `PYTHONPATH=.` from parent dir, or `cd techjam` first |
| `catalog.jsonl not found` | Playground still works; features products from public_set instead |
| `No recommendations` | Catalog may be loading but empty; verify file integrity |
| `KeyError: 'parent_asin'` | Using wrong catalog format; must decompress the official catalog.jsonl.gz |

---

## What's Happening Under the Hood

The agent's 10-stage pipeline:

1. **Interpret** — Parse "running shoes under $80" → attributes
2. **Update** — Track state (wants: shoes, budget: <$80)
3. **Choose** — Pick strategy (precision vs. discovery)
4. **Ranked** — BM25 search + popularity blend
5. **Unseen** — Hide products already shown
6. **Compose** — Pick best guess + fill remaining slots
7. **Pad** — Add global popular items
8. **Rerank** — Apply LLM ranking (if `USE_LLM=1`)
9. **Probe** — Decide what to ask next
10. **Respond** — Compose natural message

See [submission/README.md](submission/README.md) for full architecture.
