"""Drives the agent two ways, and keeps the scored one honest.

**Replay** hands one row of the public set to the organizer's own `evaluate()`.
Nothing here re-implements the customer, the pivot or the scoring, so the hit
turn and the rank on screen are the evaluator's verdict rather than the
playground's opinion of it. That is the same discipline `harness/run.py`
follows and for the same reason.

**Live** bypasses `evaluate()` entirely and calls the agent directly with
whatever was typed. There is no ground truth in that mode and therefore no
score, which the UI has to say out loud. What it can still show is a nominated
goal product's rank moving as the conversation discloses more, which is the
honest version of the same demonstration.
"""

from __future__ import annotations

from evaluator import local_evaluator
from harness import analysis
from harness import record

# The protocol runs ten turns and `turn` is capped at 10 by the published
# contract, so free typing stops where a scored session would.
MAX_TURNS = local_evaluator.MAX_TURNS

TOP_K = local_evaluator.TOP_K

# Stands in for the aggregate the evaluator would pass. Deliberately the
# shape the contract closes with `additionalProperties: false`, so live mode
# exercises the same reset path a scored session does.
DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort", "durability"],
    "summary": "Prior purchases emphasize fit, comfort, durability.",
}


class Capture:
    """Proxy that keeps each turn's nested explanation.

    `harness.record.RecordingAgent` already snapshots the flat `debug` dict,
    but it takes a shallow copy of one dict and nothing else. Rather than
    change the measurement harness for a demo, this sits underneath it and
    catches the second channel, forwarding `debug` so the recorder above
    still sees what it expects.
    """

    def __init__(self, agent) -> None:
        self.agent = agent
        self.turns: list[dict] = []

    @property
    def debug(self) -> object:
        return getattr(self.agent, "debug", None)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.turns = []
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id, user_message, turn, top_k) -> dict:
        served = self.agent.respond(session_id, user_message, turn, top_k)
        self.turns.append(dict(getattr(self.agent, "explain", {}) or {}))
        return served


def replay(agent, sample: dict, dataset) -> dict:
    """Scores one session through the evaluator and explains every turn.

    Args:
        agent: The `ExplainingAgent`.
        sample: One row of `public_set.jsonl`.
        dataset: The `(catalog_ids, categories, products)` triple.
    """
    catalog_ids, categories, products = dataset
    target = str(sample["ground_truth"]["parent_asin"])
    agent.goal = target
    capture = Capture(agent)
    recorder = record.RecordingAgent(capture)
    result = local_evaluator.evaluate(
        recorder, [sample], catalog_ids, categories, products
    )
    sessions = analysis.analyze(
        recorder.sessions, [sample], result, catalog_ids
    )
    session = sessions[0]
    for turn, explained in zip(session["turns"], capture.turns):
        turn["explain"] = explained
    return {
        "session": session,
        "target": target,
        "intent_card": local_evaluator.intent_card(products[target]),
        "metrics": {
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "technical_score": result["recommended_technical_score"],
        },
    }


class Live:
    """One free-typed session, driven straight at the agent.

    `Agent.respond` never raises -- it degrades internally and always returns
    a valid dict -- so nothing here needs an exception path of its own. What
    it does need is the turn counter, because `turn` is bounded by the
    contract and a demo that quietly ran to turn 30 would be showing the agent
    in a state no scored session can reach.
    """

    def __init__(self, agent, session_id: str) -> None:
        self.agent = agent
        self.session_id = session_id
        self.turn = 0
        self.history: list[dict] = []

    def open(self, profile: dict | None = None, goal: str | None = None) -> dict:
        """Starts the session. Mirrors what `evaluate()` does before turn 1."""
        self.agent.goal = goal
        self.turn = 0
        self.history = []
        self.agent.reset(self.session_id, profile or DEFAULT_PROFILE)
        return {"session_id": self.session_id, "turn": 0, "goal": goal}

    def send(self, message: str) -> dict:
        """Serves one typed turn."""
        if self.turn >= MAX_TURNS:
            return {"exhausted": True, "turn": self.turn}
        self.turn += 1
        served = self.agent.respond(
            self.session_id, message, self.turn, TOP_K
        )
        entry = {
            "turn": self.turn,
            "user_message": message,
            "message": served.get("message"),
            "ask_attribute": served.get("ask_attribute"),
            "recommendations": served.get("recommendations"),
            "explain": dict(self.agent.explain or {}),
            "exhausted": False,
        }
        self.history.append(entry)
        return entry
