"""Analyze persona performance for self-evolution."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from submission.src import outcome_tracker


class PersonaAnalyzer:
    """Analyze persona effectiveness across sessions."""
    
    def __init__(self, log_path: str = "submission/src/.persona_logs.jsonl"):
        self.log_path = Path(log_path)
        self.records = self._load_records()
    
    def _load_records(self) -> list[dict]:
        """Load all recorded turns."""
        records = []
        if not self.log_path.exists():
            return records
        
        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return records
    
    def persona_distribution(self) -> dict:
        """Which personas are used most? How effective?"""
        if not self.records:
            return {}
        
        stats = defaultdict(lambda: {
            "count": 0,
            "led_to_constraint": 0,
            "user_accepted": 0,
            "confidences": [],
            "product_reductions": [],
        })
        
        for record in self.records:
            persona = record["persona_used"]
            stats[persona]["count"] += 1
            
            if record["led_to_constraint"]:
                stats[persona]["led_to_constraint"] += 1
            if record["user_accepted"]:
                stats[persona]["user_accepted"] += 1
            
            stats[persona]["confidences"].append(record["persona_confidence"])
            
            reduction = record["products_before"] - record["products_after"]
            stats[persona]["product_reductions"].append(reduction)
        
        # Calculate metrics
        result = {}
        for persona, data in stats.items():
            count = data["count"]
            result[persona] = {
                "frequency": count,
                "constraint_rate": data["led_to_constraint"] / count,
                "acceptance_rate": data["user_accepted"] / count,
                "avg_confidence": sum(data["confidences"]) / count,
                "avg_product_reduction": sum(data["product_reductions"]) / len(data["product_reductions"]) if data["product_reductions"] else 0,
            }
        
        return result
    
    def user_profile_patterns(self) -> dict:
        """How do different user profiles respond to different personas?"""
        if not self.records:
            return {}
        
        patterns = defaultdict(lambda: defaultdict(list))
        
        for record in self.records:
            profile = record["user_rating_style"]
            persona = record["persona_used"]
            
            patterns[profile][persona].append({
                "constraint": record["led_to_constraint"],
                "acceptance": record["user_accepted"],
                "confidence": record["persona_confidence"],
            })
        
        # Calculate averages
        result = {}
        for profile, personas in patterns.items():
            result[profile] = {}
            for persona, turns in personas.items():
                result[profile][persona] = {
                    "count": len(turns),
                    "constraint_rate": sum(t["constraint"] for t in turns) / len(turns),
                    "acceptance_rate": sum(t["acceptance"] for t in turns) / len(turns),
                    "avg_confidence": sum(t["confidence"] for t in turns) / len(turns),
                }
        
        return result
    
    def persona_failures(self, threshold: float = 0.5) -> list[dict]:
        """Sessions where persona had low success rate."""
        if not self.records:
            return []
        
        # Group by session
        sessions = defaultdict(list)
        for record in self.records:
            sessions[record["session_id"]].append(record)
        
        failures = []
        for session_id, turns in sessions.items():
            constraint_rate = sum(t["led_to_constraint"] for t in turns) / len(turns)
            avg_confidence = sum(t["persona_confidence"] for t in turns) / len(turns)
            
            if constraint_rate < threshold or avg_confidence < 0.6:
                failures.append({
                    "session_id": session_id,
                    "turn_count": len(turns),
                    "constraint_rate": constraint_rate,
                    "avg_confidence": avg_confidence,
                    "persona_sequence": [t["persona_used"] for t in turns],
                })
        
        return sorted(failures, key=lambda x: x["constraint_rate"])
    
    def persona_transitions(self) -> dict:
        """How well do predicted next personas match actual personas?"""
        if not self.records:
            return {}
        
        # Track: (current_persona, predicted_next) -> (actual_next, count)
        transitions = defaultdict(lambda: defaultdict(int))
        
        # Group by session
        sessions = defaultdict(list)
        for record in self.records:
            sessions[record["session_id"]].append(record)
        
        for session_id, turns in sessions.items():
            for i in range(len(turns) - 1):
                current = turns[i]
                next_turn = turns[i + 1]
                
                # Note: we don't have predicted_next in the record,
                # but we can still track persona sequences
                current_persona = current["persona_used"]
                next_persona = next_turn["persona_used"]
                
                transitions[current_persona][next_persona] += 1
        
        # Convert to percentages
        result = {}
        for persona, nexts in transitions.items():
            total = sum(nexts.values())
            result[persona] = {
                next_p: count / total
                for next_p, count in nexts.items()
            }
        
        return result
    
    def generate_report(self) -> str:
        """Generate a text report of persona analysis."""
        if not self.records:
            return "No persona records found."
        
        report = ["=" * 60, "PERSONA ANALYSIS REPORT", "=" * 60, ""]
        
        # Distribution
        dist = self.persona_distribution()
        report.append("PERSONA DISTRIBUTION & EFFECTIVENESS")
        report.append("-" * 60)
        report.append(f"{'Persona':<35} {'Freq':<8} {'Constraint':<12} {'Acceptance':<12}")
        report.append("-" * 60)
        for persona in sorted(dist.keys()):
            data = dist[persona]
            report.append(
                f"{persona:<35} {data['frequency']:<8} "
                f"{data['constraint_rate']:.1%}         {data['acceptance_rate']:.1%}"
            )
        report.append("")
        
        # User profile patterns
        patterns = self.user_profile_patterns()
        report.append("EFFECTIVENESS BY USER PROFILE")
        report.append("-" * 60)
        for profile in sorted(patterns.keys()):
            report.append(f"\n{profile.upper()}:")
            for persona, metrics in sorted(patterns[profile].items()):
                report.append(
                    f"  {persona:<30} "
                    f"constraint: {metrics['constraint_rate']:.1%}, "
                    f"acceptance: {metrics['acceptance_rate']:.1%}"
                )
        report.append("")
        
        # Failures
        failures = self.persona_failures()
        report.append(f"SESSIONS WITH LOW PERSONA SUCCESS ({len(failures)} found)")
        report.append("-" * 60)
        for failure in failures[:5]:  # Top 5
            report.append(
                f"Session {failure['session_id']}: "
                f"constraint_rate={failure['constraint_rate']:.1%}, "
                f"personas={failure['persona_sequence']}"
            )
        report.append("")
        
        # Transitions
        trans = self.persona_transitions()
        report.append("PERSONA TRANSITIONS (WHERE SESSIONS GO NEXT)")
        report.append("-" * 60)
        for current in sorted(trans.keys()):
            nexts = trans[current]
            top_next = max(nexts.items(), key=lambda x: x[1])
            report.append(f"{current} → {top_next[0]} ({top_next[1]:.1%})")
        
        return "\n".join(report)
    
    def identify_new_persona_opportunities(self) -> list[dict]:
        """Find sessions that might need a new persona."""
        failures = self.persona_failures(threshold=0.5)
        
        opportunities = []
        for failure in failures[:10]:  # Top 10
            # Analyze what went wrong
            session_turns = [r for r in self.records if r["session_id"] == failure["session_id"]]
            
            # Common issues
            issue = None
            if failure["avg_confidence"] < 0.5:
                issue = "Low confidence in persona match"
            elif failure["constraint_rate"] < 0.3:
                issue = "Personas not extracting constraints"
            else:
                issue = "Mixed or unclear session pattern"
            
            opportunities.append({
                "session_id": failure["session_id"],
                "issue": issue,
                "personas_used": failure["persona_sequence"],
                "constraint_rate": failure["constraint_rate"],
                "turns": len(session_turns),
            })
        
        return opportunities
