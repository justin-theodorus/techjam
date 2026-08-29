"""The identity channel the evaluator's API has no field for.

`local_evaluator.evaluate()` mints a fresh `public_{uuid4().hex}` per session
and the published contract closes both `reset_request` and `user_profile` with
`additionalProperties: false`, so an agent cannot be told who is shopping. The
synthetic rows carry `shopper_id` instead, and this proxy hands it over before
each session opens. It is deliberately the thinnest thing that works and
imports nothing from the rest of the harness, because `sessions` uses it and
the reporting module uses `sessions`.
"""

from __future__ import annotations


class ReturningAgent:
    """Names each row's shopper to the agent before its session opens.

    `evaluate()` walks the rows in order and calls `reset()` exactly once per
    row before any `respond()`, so counting resets against the row list
    recovers which row is being served without the evaluator passing anything
    down. An agent with no `remember` is driven untouched, which is what keeps
    the organizer's own starter agent runnable through this path.
    """

    def __init__(self, agent: object, rows: list[dict]) -> None:
        self.agent = agent
        self._identities = [row.get("shopper_id") for row in rows]
        self._served = 0
        # Every measurement re-scores many configurations against one agent
        # instance, and `deviations.patched` never rebuilds it, so isolation
        # between them has to happen here or cell order decides the result.
        forget = getattr(agent, "forget", None)
        if callable(forget):
            forget()

    @property
    def debug(self) -> object:
        """Forwarded so the recording proxy still sees the agent's own trace."""
        return getattr(self.agent, "debug", None)

    def reset(self, session_id: str, user_profile: dict) -> None:
        remember = getattr(self.agent, "remember", None)
        if callable(remember):
            position = self._served
            identity = (
                self._identities[position]
                if position < len(self._identities) else None
            )
            remember(identity)
        self._served += 1
        self.agent.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        return self.agent.respond(session_id, user_message, turn, top_k)
