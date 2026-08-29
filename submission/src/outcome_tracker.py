"""Track persona outcomes for self-evolution analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from submission.src import persona_classifier


@dataclass
class PersonaTurnRecord:
    """Record of a persona's performance in one turn."""
    session_id: str
    turn: int
    persona_used: str  # PersonaType.value
    persona_confidence: float
    user_message: str
    llm_question: str
    constraints_before: list[str]
    constraints_after: list[str]
    user_rating_style: str
    products_before: int
    products_after: int
    led_to_constraint: bool  # Did user provide new info?
    user_accepted: bool  # Did user seem satisfied?


class OutcomeTracker:
    """Track and store persona performance data."""
    
    def __init__(self, log_path: str | None = None):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_session_records: list[PersonaTurnRecord] = []
    
    def record_turn(
        self,
        session_id: str,
        turn: int,
        persona_match: persona_classifier.PersonaMatch,
        user_message: str,
        llm_question: str,
        constraints_before: list[str],
        constraints_after: list[str],
        user_rating_style: str,
        products_before: int,
        products_after: int,
    ) -> None:
        """Record a turn's persona performance."""
        
        # Determine outcome
        led_to_constraint = set(constraints_after) != set(constraints_before)
        # Simple heuristic: user accepted if products narrowed
        user_accepted = products_after < products_before
        
        record = PersonaTurnRecord(
            session_id=session_id,
            turn=turn,
            persona_used=persona_match.persona_type.value,
            persona_confidence=persona_match.confidence,
            user_message=user_message,
            llm_question=llm_question,
            constraints_before=constraints_before,
            constraints_after=constraints_after,
            user_rating_style=user_rating_style,
            products_before=products_before,
            products_after=products_after,
            led_to_constraint=led_to_constraint,
            user_accepted=user_accepted,
        )
        
        self.current_session_records.append(record)
        self._write_record(record)
    
    def _write_record(self, record: PersonaTurnRecord) -> None:
        """Append record to log file."""
        if self.log_path is None:
            return
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")
    
    def get_statistics(self) -> dict:
        """Aggregate statistics across all logged turns."""
        if self.log_path is None or not self.log_path.exists():
            return {}
        
        records = []
        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if not records:
            return {}
        
        # Aggregate by persona
        persona_stats = {}
        for record in records:
            persona = record["persona_used"]
            if persona not in persona_stats:
                persona_stats[persona] = {
                    "count": 0,
                    "led_to_constraint": 0,
                    "user_accepted": 0,
                    "avg_confidence": 0,
                    "product_reduction": [],
                }
            
            stats = persona_stats[persona]
            stats["count"] += 1
            if record["led_to_constraint"]:
                stats["led_to_constraint"] += 1
            if record["user_accepted"]:
                stats["user_accepted"] += 1
            stats["avg_confidence"] += record["persona_confidence"]
            
            reduction = record["products_before"] - record["products_after"]
            stats["product_reduction"].append(reduction)
        
        # Calculate rates
        for persona, stats in persona_stats.items():
            if stats["count"] > 0:
                stats["constraint_rate"] = stats["led_to_constraint"] / stats["count"]
                stats["acceptance_rate"] = stats["user_accepted"] / stats["count"]
                stats["avg_confidence"] = stats["avg_confidence"] / stats["count"]
                stats["avg_product_reduction"] = sum(stats["product_reduction"]) / len(stats["product_reduction"]) if stats["product_reduction"] else 0
                del stats["product_reduction"]  # Remove raw data
        
        return persona_stats
    
    def find_unmatched_sessions(self, data_path: str = "data/public_set.jsonl") -> list[dict]:
        """
        Analyze logs to find sessions where no persona matched well.
        
        This is input for self-evolution analysis.
        """
        if self.log_path is None or not self.log_path.exists():
            return []
        
        # Load all persona records
        persona_turns = {}
        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    sid = record["session_id"]
                    if sid not in persona_turns:
                        persona_turns[sid] = []
                    persona_turns[sid].append(record)
                except json.JSONDecodeError:
                    continue
        
        # Find sessions with low average confidence or poor outcomes
        unmatched = []
        for session_id, turns in persona_turns.items():
            avg_confidence = sum(t["persona_confidence"] for t in turns) / len(turns)
            constraint_rate = sum(1 for t in turns if t["led_to_constraint"]) / len(turns)
            
            # Session is "unmatched" if confidence is low OR constraint rate is low
            if avg_confidence < 0.65 or constraint_rate < 0.5:
                unmatched.append({
                    "session_id": session_id,
                    "avg_confidence": avg_confidence,
                    "constraint_rate": constraint_rate,
                    "turn_count": len(turns),
                    "turns": turns,
                })
        
        return unmatched
