#!/usr/bin/env python3
"""Quick demo: replay a sample public session to see the agent in action.

This works without the full catalog—it replays one real session from the
public evaluation set so you can see conversations and recommendations live.

Usage:
    cd techjam
    python demo.py
"""

import json
import random
import sys
from pathlib import Path

# Ensure the repo root is importable whether we launch from the project root
# or from inside techjam/.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from techjam.submission.agent import Agent
except ImportError:
    from submission.agent import Agent


def load_public_sessions():
    """Load sample sessions from the public evaluation set."""
    sessions = []
    public_file = Path("techjam/data/public_set.jsonl")
    
    if not public_file.exists():
        print(f"❌ Could not find {public_file}")
        return []
    
    try:
        with open(public_file, encoding="utf-8") as f:
            for line in f:
                sessions.append(json.loads(line))
    except Exception as e:
        print(f"❌ Error loading sessions: {e}")
    
    return sessions


def run_demo():
    """Replay a real session from the public set."""
    print("=" * 70)
    print("🛍️  TikTok Shopping Agent — Live Demo")
    print("=" * 70)

    # Load agent (without catalog if it doesn't exist)
    catalog_path = Path("techjam/data/catalog.jsonl")
    if not catalog_path.exists():
        print("\n⚠️  Note: Full catalog not found.")
        print("Using smaller test dataset. To see full recommendations:")
        print("  gzip -dkc techjam/data/catalog.jsonl.gz > techjam/data/catalog.jsonl")
        catalog_path = Path("techjam/data/public_set.jsonl")
    
    print(f"\n📦 Loading agent...")
    try:
        agent = Agent(catalog_path=str(catalog_path))
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        sys.exit(1)
    print("✓ Agent ready.\n")

    # Load sessions
    sessions = load_public_sessions()
    if not sessions:
        print("❌ Could not load public sessions.")
        sys.exit(1)

    # Pick a random session
    session_data = random.choice(sessions)
    session_id = session_data.get("sample_id", "demo_session")
    user_profile = session_data.get("user_profile", {})
    ground_truth = session_data.get("ground_truth", {})
    turns_data = ground_truth.get("messages", [])
    target_asin = ground_truth.get("target_asin", "unknown")
    scenario = session_data.get("scenario_type", "unknown")

    print("=" * 70)
    print(f"📊 Session Type: {scenario}")
    print(f"🎯 Target Product: {target_asin}")
    print(f"👤 User Profile: {user_profile}")
    print("=" * 70)

    # Reset agent
    agent.reset(session_id, user_profile)

    # Replay each turn
    print(f"\n💬 Replaying session {session_id}...\n")
    
    found = False
    for turn_num, turn_data in enumerate(turns_data, 1):
        user_message = turn_data.get("user_message", "(no message)")
        
        print("─" * 70)
        print(f"[Turn {turn_num}] You: {user_message}")
        print("─" * 70)

        # Get agent response
        try:
            response = agent.respond(
                session_id,
                user_message,
                turn_num,
                top_k=10
            )
        except Exception as e:
            print(f"❌ Agent error: {e}")
            continue

        # Display agent message
        print(f"\n🤖 Agent: {response.get('message', '(no message)')}")

        # Display what it's asking for
        asked = response.get("ask_attribute")
        if asked:
            print(f"   🔍 Asking about: {asked}")

        # Display recommendations
        recommendations = response.get("recommendations", [])
        if recommendations:
            print(f"\n   📋 Top {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                asin = rec.get("parent_asin", "unknown")
                score = rec.get("score", 0.0)
                marker = " ✓ TARGET!" if asin == target_asin else ""
                print(f"      {i}. {asin} (score: {score:.4f}){marker}")
                
                # Check if we found the target
                if asin == target_asin and not found:
                    found = True
                    print(f"\n✅ TARGET FOUND at rank {i} on turn {turn_num}!")

        usage = response.get("usage", {})
        tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if tokens > 0:
            print(f"\n   ⚡ Tokens: {tokens}")

        print()

    # Summary
    print("=" * 70)
    if found:
        print("✅ SUCCESS: Agent found the target!")
    else:
        print("❌ Session ended without finding the target.")
    print("=" * 70)

    # Option to replay or play interactive
    print("\nWant to try the interactive playground?")
    print("  Run: python playground.py")


if __name__ == "__main__":
    run_demo()
