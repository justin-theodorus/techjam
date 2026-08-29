# Playground Setup

The interactive playground lets you chat with the shopping agent in real-time.

## Prerequisites

### 1. Get the Catalog (Required)

The full 50,000-product catalog is in a GitHub Release. Download and decompress it:

```bash
cd /Users/angelicagonathan/Hackathon/techjam

# Option A: Download from GitHub Release
# (Visit the Release page and download catalog.jsonl.gz)
gzip -dkc catalog.jsonl.gz > data/catalog.jsonl

# Expected result: ~19 MB uncompressed file
ls -lh data/catalog.jsonl
```

If you don't have the compressed file, you can fetch it from the Release page in the repo.

## Running the Playground

```bash
cd /Users/angelicagonathan/Hackathon/techjam
python playground.py
```

### Interactive Session

1. Enter a session ID (or press Enter for default)
2. Chat with the agent naturally
   - "I need running shoes for casual jogging"
   - "Something under $100"
   - "Do you have anything in leather?"
3. Watch real-time recommendations
4. Type `debug` to see the agent's internal reasoning
5. Type `quit` to exit

### What You'll See

- **Agent message**: Natural language response
- **Asking about**: What attribute the agent is clarifying
- **Recommendations**: Top 10 products with titles, prices, ratings, and match scores
- **Token usage**: How many tokens the model (if enabled) used

## Features

- No API keys needed (runs entirely offline)
- Uses BM25 search + popularity ranking
- Handles refusals ("no leather") correctly
- Tracks session state (what you already saw, what you rejected)
- Optional: `USE_LLM=1` enables model-backed reranking

## Example Conversation

```
You: I'm looking for winter boots
Agent: I can help you find winter boots! Do you have a preference for material?
Recommendations:
  1. Premium Waterproof Winter Boot - $89.99 ⭐4.5
  2. Insulated Snow Boot - $75.50 ⭐4.7
  ...

You: I like leather
Agent: Great! Leather boots are durable and classic. Would you prefer something with more cushioning?
Recommendations:
  1. Leather Winter Boot Comfort Flex - $95.00 ⭐4.8
  ...

You: Something waterproof please
Agent: Perfect! Here are waterproof leather options:
Recommendations:
  1. Waterproof Leather Winter Boot - $105.00 ⭐4.9
  ...
```

## Troubleshooting

**"catalog.jsonl not found"**
→ Download and decompress as shown above

**Empty recommendations**
→ The agent is still learning; try a different query or check the agent logs

**"ModuleNotFoundError"**
→ Make sure you're in the techjam directory and PYTHONPATH is set:

```bash
cd techjam
PYTHONPATH=. python playground.py
```
