"""Persona classification: match turn context to behavior pattern."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from techjam.submission.src import dialogue
from techjam.submission.src import intent_detector


class PersonaType(Enum):
    """Behavior personas representing how to interact with user this turn."""
    EARLY_BUYER_SPECIFIC = "early_buyer_specific"
    MID_BROWSER_VAGUE = "mid_browser_vague"
    MID_BROWSER_REFINED = "mid_browser_refined"
    LATE_CRITICAL_EVALUATOR = "late_critical_evaluator"
    INTENT_OVERRIDE_PIVOT = "intent_override_pivot"
    BOUNDARY_REJECTION = "boundary_rejection"
    CLARIFY_CONTRADICTION = "clarify_contradiction"


@dataclass
class PersonaMatch:
    """Result of persona classification."""
    persona_type: PersonaType
    confidence: float  # 0.0-1.0
    rationale: str
    context_signals: dict  # {signal_name: value}
    recommended_next: PersonaType | None  # Predicted next persona


class PersonaClassifier:
    """Classify which persona pattern best matches current turn."""
    
    def classify(
        self,
        intent: intent_detector.IntentDecision,
        session_state: dialogue.SessionState,
        candidate_count: int = 0,
        user_profile: dict | None = None,
    ) -> PersonaMatch:
        """
        Match current turn to a persona.
        
        Args:
            intent: Detected intent from intent_detector
            session_state: Current session state
        
        Returns:
            PersonaMatch with type, confidence, context
        """
        # Collect context signals
        profile = user_profile if isinstance(user_profile, dict) else {}
        signals = {
            "intent_type": intent.intent_type.value,
            "turn": session_state.turn,
            "constraint_count": len(session_state.constraints),
            "products_remaining": candidate_count,
            "user_rating_style": profile.get("rating_style", "unknown"),
            "specificity": self._assess_specificity(session_state),
        }
        
        # Classify based on intent + turn context
        candidates = []
        
        # Early buyer: turn 1-3, buying intent, has constraints
        if (intent.intent_type == intent_detector.IntentType.BUYING 
            and session_state.turn <= 3
            and session_state.constraints):
            candidates.append(PersonaMatch(
                persona_type=PersonaType.EARLY_BUYER_SPECIFIC,
                confidence=0.85,
                rationale="Early stage buying with specific constraint",
                context_signals=signals,
                recommended_next=PersonaType.MID_BROWSER_REFINED
            ))
        
        # Mid browser vague: browsing intent, turn 3-6, few constraints
        if (intent.intent_type == intent_detector.IntentType.BROWSING
            and session_state.turn in range(3, 7)
            and len(session_state.constraints) <= 1
            and candidate_count > 500):
            candidates.append(PersonaMatch(
                persona_type=PersonaType.MID_BROWSER_VAGUE,
                confidence=0.80,
                rationale="Mid-stage browsing, still vague",
                context_signals=signals,
                recommended_next=PersonaType.MID_BROWSER_REFINED
            ))
        
        # Mid browser refined: has multiple constraints, narrowing down
        if (session_state.turn in range(3, 7)
            and len(session_state.constraints) >= 2
            and (not candidate_count or candidate_count < 1000)):
            candidates.append(PersonaMatch(
                persona_type=PersonaType.MID_BROWSER_REFINED,
                confidence=0.78,
                rationale="Mid-stage with multiple constraints, refining",
                context_signals=signals,
                recommended_next=PersonaType.LATE_CRITICAL_EVALUATOR
            ))
        
        # Late critical evaluator: turn 7-10, critical user, limited time
        if (session_state.turn >= 7
            and profile.get("rating_style") == "critical"):
            candidates.append(PersonaMatch(
                persona_type=PersonaType.LATE_CRITICAL_EVALUATOR,
                confidence=0.82,
                rationale="Late stage with critical user, focus on quality",
                context_signals=signals,
                recommended_next=None
            ))
        
        # Intent override pivot: detected override signals
        if intent.intent_type == intent_detector.IntentType.INTENT_OVERRIDE:
            candidates.append(PersonaMatch(
                persona_type=PersonaType.INTENT_OVERRIDE_PIVOT,
                confidence=0.90,
                rationale="User changing preference mid-session",
                context_signals=signals,
                recommended_next=PersonaType.EARLY_BUYER_SPECIFIC
            ))
        
        # Boundary rejection: boundary intent, rejecting suggestions
        if (intent.intent_type == intent_detector.IntentType.BOUNDARY
            and session_state.turn >= 5):
            candidates.append(PersonaMatch(
                persona_type=PersonaType.BOUNDARY_REJECTION,
                confidence=0.75,
                rationale="User with no preference, late in session",
                context_signals=signals,
                recommended_next=None
            ))
        
        # Clarify contradiction: multiple conflicting constraints
        if self._has_contradiction(session_state.constraints):
            candidates.append(PersonaMatch(
                persona_type=PersonaType.CLARIFY_CONTRADICTION,
                confidence=0.80,
                rationale="Conflicting constraints detected",
                context_signals=signals,
                recommended_next=PersonaType.MID_BROWSER_REFINED
            ))
        
        # Return best match, or default
        if candidates:
            best = max(candidates, key=lambda c: c.confidence)
            return best
        
        # Default: mid browser vague
        return PersonaMatch(
            persona_type=PersonaType.MID_BROWSER_VAGUE,
            confidence=0.5,
            rationale="Default persona (no clear match)",
            context_signals=signals,
            recommended_next=PersonaType.MID_BROWSER_REFINED
        )
    
    def _assess_specificity(self, state: dialogue.SessionState) -> str:
        """Assess how specific the constraints are."""
        if not state.constraints:
            return "no_constraints"
        if len(state.constraints) == 1:
            return "single_constraint"
        if len(state.constraints) >= 3:
            return "detailed"
        return "partial"
    
    def _has_contradiction(self, constraints: tuple[str, ...]) -> bool:
        """Check if constraints conflict (e.g., waterproof + lightweight)."""
        constraint_text = " ".join(constraints).lower()
        
        # Examples of conflicting patterns
        contradictions = [
            ("waterproof", "lightweight"),
            ("formal", "casual"),
            ("minimal", "maximalist"),
            ("premium", "budget"),
        ]
        
        for term1, term2 in contradictions:
            if term1 in constraint_text and term2 in constraint_text:
                return True
        
        return False
