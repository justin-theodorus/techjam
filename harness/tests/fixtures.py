"""Tiny in-memory catalog and sample set shared by the harness tests."""

from __future__ import annotations

import json
from pathlib import Path

CATALOG_ROWS = [
    {
        "parent_asin": "A",
        "title": "Blue running shoe",
        "features": ["cotton upper"],
        "details": {"department": "womens"},
        "description": ["walking shoe"],
        "categories": ["Clothing", "Shoes"],
        "store": "Example",
        "price": 49.0,
    },
    {
        "parent_asin": "B",
        "title": "Black winter boot",
        "features": ["leather shell"],
        "details": {"department": "womens"},
        "description": ["winter boot"],
        "categories": ["Clothing", "Boots"],
        "store": "Example",
        "price": 89.0,
    },
]


def write_catalog(root: Path) -> Path:
    path = root / "catalog.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in CATALOG_ROWS), encoding="utf-8")
    return path


def sample(sample_id: str, scenario: str, target: str) -> dict:
    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "user_profile": {"summary": "x"},
        "ground_truth": {"parent_asin": target},
    }


class ConstantAgent:
    """Emits a fixed slate every turn, optionally probing."""

    def __init__(self, slate: list[str], ask_attribute: str | None = "other") -> None:
        self.slate = slate
        self.ask_attribute = ask_attribute

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "here you go",
            "ask_attribute": self.ask_attribute,
            "recommendations": [{"parent_asin": asin} for asin in self.slate],
        }
