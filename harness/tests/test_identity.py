from __future__ import annotations

import unittest

from harness import identity


class Spy:
    """An agent that records what it was told, and nothing else."""

    def __init__(self) -> None:
        self.named: list[str | None] = []
        self.sessions: list[str] = []
        self.forgotten = 0
        self.debug = {"turn": 0}

    def remember(self, shopper_id: str | None) -> None:
        self.named.append(shopper_id)

    def forget(self) -> None:
        self.forgotten += 1

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions.append(session_id)

    def respond(self, session_id, user_message, turn, top_k) -> dict:
        return {"message": user_message}


class Plain:
    """An agent with no memory at all, such as the organizer's own."""

    def __init__(self) -> None:
        self.sessions: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions.append(session_id)

    def respond(self, session_id, user_message, turn, top_k) -> dict:
        return {"message": user_message}


def rows(*identities: str | None) -> list[dict]:
    return [
        {"shopper_id": name} if name is not None else {}
        for name in identities
    ]


class ProxyTest(unittest.TestCase):
    """The identity channel the published contract has no field for."""

    def test_each_row_is_named_before_its_session_opens(self) -> None:
        agent = Spy()
        proxy = identity.ReturningAgent(agent, rows("a", "b", "a"))

        for number in range(3):
            proxy.reset(f"s{number}", {})

        self.assertEqual(agent.named, ["a", "b", "a"])

    def test_a_row_naming_nobody_is_shopped_anonymously(self) -> None:
        agent = Spy()
        proxy = identity.ReturningAgent(agent, rows("a", None))

        proxy.reset("s0", {})
        proxy.reset("s1", {})

        self.assertEqual(agent.named, ["a", None])

    def test_more_sessions_than_rows_fall_back_to_anonymity(self) -> None:
        agent = Spy()
        proxy = identity.ReturningAgent(agent, rows("a"))

        proxy.reset("s0", {})
        proxy.reset("s1", {})

        self.assertEqual(agent.named, ["a", None])

    def test_constructing_the_proxy_clears_the_store(self) -> None:
        """Or a sweep point reads what the point before it wrote."""
        agent = Spy()

        identity.ReturningAgent(agent, rows("a"))

        self.assertEqual(agent.forgotten, 1)

    def test_an_agent_without_memory_is_driven_untouched(self) -> None:
        agent = Plain()
        proxy = identity.ReturningAgent(agent, rows("a"))

        proxy.reset("s0", {})
        response = proxy.respond("s0", "hello", 1, 10)

        self.assertEqual(agent.sessions, ["s0"])
        self.assertEqual(response, {"message": "hello"})

    def test_the_debug_dict_is_forwarded(self) -> None:
        """The recording proxy reads it off whatever it was handed."""
        agent = Spy()
        proxy = identity.ReturningAgent(agent, rows("a"))

        self.assertEqual(proxy.debug, agent.debug)

    def test_a_turn_is_passed_through_unchanged(self) -> None:
        agent = Spy()
        proxy = identity.ReturningAgent(agent, rows("a"))
        proxy.reset("s0", {})

        self.assertEqual(
            proxy.respond("s0", "hello", 2, 10), {"message": "hello"})


if __name__ == "__main__":
    unittest.main()
