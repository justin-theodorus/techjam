#!/usr/bin/env python3
"""Interactive playground for the TikTok shopping agent.

Lets you chat with the agent, see recommendations, and understand its reasoning.

Usage:
    cd techjam
    python playground.py
"""

import json
import sys
from pathlib import Path

# Import from techjam.submission so it works from either directory
try:
    from techjam.submission.agent import Agent
except ImportError:
    from submission.agent import Agent


def load_sample_profile() -> dict:
    """Load a sample user profile or create a default."""
    sample = {
        "preference_tags": ["casual", "athletic"],
        "average_rating": 4.2,
        "rating_number": 87,
        "buying_frequency": "monthly",
    }
    return sample


def load_catalog_metadata(catalog_path: str) -> dict:
    """Load product metadata for display. Return asin -> product dict."""
    products = {}
    try:
        with open(catalog_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                asin = row.get("parent_asin", "unknown")
                products[asin] = {
                    "title": row.get("title", "Unknown"),
                    "price": row.get("price", "N/A"),
                    "rating": row.get("average_rating", "N/A"),
                    "categories": row.get("categories", []),
                }
                if len(products) >= 50000:  # Only keep first 50k for memory
                    break
    except FileNotFoundError:
        print(f"Warning: Could not load catalog from {catalog_path}")
        print("Continuing without product details.")
    except json.JSONDecodeError:
        pass  # Skip malformed lines
    return products


def format_product(asin: str, products: dict, score: float = 0.0) -> str:
    """Format a product for display."""
    if asin not in products:
        return f"  • {asin} (score: {score:.4f})"
    p = products[asin]
    title = p["title"][:60] + "..." if len(p["title"]) > 60 else p["title"]
    rating = f"{p['rating']}⭐" if p["rating"] != "N/A" else "No rating"
    price = p["price"]
    return f"  • {title}\n    ASIN: {asin} | Price: {price} | Rating: {rating} (score: {score:.4f})"


def run_playground() -> None:
    """Interactive chat loop with the agent."""
    print("=" * 70)
    print("🛍️  TikTok Shopping Agent Playground")
    print("=" * 70)

    # Initialize
    catalog_path = Path("techjam/data/catalog.jsonl")
    if not catalog_path.exists():
        print(f"⚠️  Full catalog not found at {catalog_path}")
        print("Download and decompress it:")
        print("  gzip -dkc techjam/data/catalog.jsonl.gz > techjam/data/catalog.jsonl")
        print("\nContinuing with empty product index...")
        products = {}

    print("\n📦 Loading agent and catalog...")
    agent = Agent(catalog_path=str(catalog_path))
    products = load_catalog_metadata(str(catalog_path))
    print(f"✓ Agent ready. Catalog has {len(products)} products.\n")

    # Session setup
    print("=" * 70)
    print("📝 Session Setup")
    print("=" * 70)

    session_id = input("Session ID (press Enter for 'playground_demo'): ").strip() or "playground_demo"
    profile = load_sample_profile()
    print(f"Using sample profile: {json.dumps(profile, indent=2)}")

    agent.reset(session_id, profile)
    print(f"\n✓ Session {session_id} started.\n")

    # Chat loop
    print("=" * 70)
    print("💬 Start chatting! (type 'quit' to exit, 'debug' to see internals)")
    print("=" * 70)

    turn = 1
    while turn <= 10:
        print(f"\n[Turn {turn}]")
        user_message = input("You: ").strip()

        if user_message.lower() == "quit":
            print("Goodbye! 👋")
            break

        if not user_message:
            print("(empty message, skipping)")
            continue

        # Get agent response
        response = agent.respond(session_id, user_message, turn, top_k=10)

        # Display results
        print("\n" + "─" * 70)
        print(f"Agent: {response['message']}")
        print("─" * 70)

        if response["ask_attribute"]:
            print(f"🔍 Asking about: {response['ask_attribute']}")

        if response["recommendations"]:
            print(f"\n📋 Top recommendations ({len(response['recommendations'])} shown):")
            for i, rec in enumerate(response["recommendations"], 1):
                asin = rec["parent_asin"]
                score = rec.get("score", 0.0)
                print(f"\n{i}. {format_product(asin, products, score)}")
        else:
            print("(No recommendations)")

        usage = response.get("usage", {})
        if usage.get("prompt_tokens") or usage.get("completion_tokens"):
            tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            print(f"\n⚡ Tokens used: {tokens}")

        # Optional debug info
        if user_message.lower() == "debug":
            print("\n" + "─" * 70)
            print("🔧 Agent Debug Info:")
            print("─" * 70)
            if hasattr(agent, "debug") and agent.debug:
                for key, value in agent.debug.items():
                    print(f"  {key}: {value}")
            else:
                print("  (No debug info available)")
            print("─" * 70)
            continue

        turn += 1

    if turn > 10:
        print("\n✓ Session ended after 10 turns.")
    print("\n" + "=" * 70)
    print("Thanks for playing! 👋")
    print("=" * 70)


if __name__ == "__main__":
    run_playground()
