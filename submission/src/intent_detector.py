"""Intent detection: identify session type (buying, browsing, boundary, override)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from techjam.submission.src import dialogue


class IntentType(Enum):
    """Session intent classification."""
    BUYING = "buying"  # Hard constraint disclosed early
    BROWSING = "browsing"  # Vague start, exploratory
    BOUNDARY = "boundary"  # No strong preference, ask directly
    INTENT_OVERRIDE = "intent_override"  # Mid-session preference shift


@dataclass
class IntentDecision:
    """Result of intent classification."""
    intent_type: IntentType
    confidence: float  # 0.0-1.0
    rationale: str
    trigger_signals: list[str]  # Which signals triggered this classification


class IntentDetector:
    """Detect what type of session this is based on user message and state."""
    
    # Signals for each intent type
    BUYING_SIGNALS = [
        "I need", "I want", "I'm looking for", "I require",
        "I'm after", "I'm in the market for", "must have",
        "is it possible", "do you have", "I prefer"
    ]
    
    BROWSING_SIGNALS = [
        "show me", "options", "what do you", "tell me about",
        "something like", "ideas", "suggestions", "browse",
        "casual", "just browsing", "any recommendations"
    ]
    
    BOUNDARY_SIGNALS = [
        "doesn't matter", "no preference", "anything works",
        "i'm flexible", "whatever", "don't care", "no strong",
        "open to", "fine with any", "surprise me"
    ]
    
    OVERRIDE_SIGNALS = [
        "actually", "changed my mind", "no wait", "i changed",
        "scratch that", "forget what i said", "on second thought",
        "i meant", "not what i", "new priority", "different"
    ]
    
    def detect(
        self,
        user_message: str,
        session_state: dialogue.SessionState,
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> IntentDecision:
        """
        Classify the user's intent based on their message and session state.
        
        Args:
            user_message: The user's current message
            session_state: Current session state
            conversation_history: Previous turns (optional)
        
        Returns:
            IntentDecision with type, confidence, rationale
        """
        message_lower = user_message.lower()

        # The shared dialogue parser has already resolved these high-signal
        # acts. Prefer that structured state over re-guessing from prose.
        if session_state.pivot_turn == session_state.turn and session_state.turn:
            return IntentDecision(
                IntentType.INTENT_OVERRIDE, 0.99,
                "Current turn replaced an earlier preference", ["pivot"],
            )
        if session_state.refused and any(
            signal in message_lower for signal in self.BOUNDARY_SIGNALS
        ):
            return IntentDecision(
                IntentType.BOUNDARY, 0.95,
                "Current turn declined a requested attribute", ["refusal"],
            )
        
        # Check for intent override (mid-session pivot)
        if conversation_history and len(conversation_history) >= 2:
            override_decision = self._detect_override(message_lower, conversation_history)
            if override_decision.confidence > 0.7:
                return override_decision
        
        # Check for buying intent (specific, hard constraint)
        buying_decision = self._detect_buying(message_lower, session_state)
        
        # Check for browsing intent (vague, exploratory)
        browsing_decision = self._detect_browsing(message_lower, session_state)
        
        # Check for boundary intent (no preference)
        boundary_decision = self._detect_boundary(message_lower, session_state)
        
        # Return highest confidence
        decisions = [
            buying_decision,
            browsing_decision,
            boundary_decision,
        ]
        return max(decisions, key=lambda d: d.confidence)
    
    def _detect_buying(self, message: str, state: dialogue.SessionState) -> IntentDecision:
        """Detect buying intent: user has a specific need."""
        signals = [s for s in self.BUYING_SIGNALS if s in message]
        confidence = 0.0
        
        # Base score from signal matching
        if signals:
            confidence = 0.6 + (len(signals) * 0.1)  # 0.6-0.9
        
        # Boost if turn <= 3 (early buying is typical buying intent)
        if state.turn <= 3 and signals:
            confidence += 0.1
        
        # Boost if constraint was just added
        if state.turn <= 1 and signals:
            confidence += 0.2
        
        confidence = min(confidence, 0.99)
        
        return IntentDecision(
            intent_type=IntentType.BUYING,
            confidence=confidence,
            rationale=f"Detected buying signals: {', '.join(signals[:3])}",
            trigger_signals=signals
        )
    
    def _detect_browsing(self, message: str, state: dialogue.SessionState) -> IntentDecision:
        """Detect browsing intent: user is exploring, vague."""
        signals = [s for s in self.BROWSING_SIGNALS if s in message]
        confidence = 0.0
        
        # Base score from signal matching
        if signals:
            confidence = 0.5 + (len(signals) * 0.1)  # 0.5-0.8
        
        # Boost if no constraints yet (still exploring)
        if not state.constraints:
            confidence += 0.2
        
        # Boost if multiple turns and still vague
        if state.turn in [2, 3, 4] and len(state.constraints) <= 1:
            confidence += 0.1
        
        confidence = min(confidence, 0.99)
        
        return IntentDecision(
            intent_type=IntentType.BROWSING,
            confidence=confidence,
            rationale=f"Detected browsing signals: {', '.join(signals[:3])}",
            trigger_signals=signals
        )
    
    def _detect_boundary(self, message: str, state: dialogue.SessionState) -> IntentDecision:
        """Detect boundary intent: user has no strong preference."""
        signals = [s for s in self.BOUNDARY_SIGNALS if s in message]
        confidence = 0.0
        
        # Base score from signal matching
        if signals:
            confidence = 0.7 + (len(signals) * 0.1)  # 0.7-0.99
        
        # Boost if late turn and still no constraints
        if state.turn >= 5 and not state.constraints:
            confidence += 0.1
        
        confidence = min(confidence, 0.99)
        
        return IntentDecision(
            intent_type=IntentType.BOUNDARY,
            confidence=confidence,
            rationale=f"Detected boundary signals: {', '.join(signals[:3])}",
            trigger_signals=signals
        )
    
    def _detect_override(
        self,
        message: str,
        conversation_history: list[tuple[str, str]]
    ) -> IntentDecision:
        """Detect intent override: mid-session preference shift."""
        signals = [s for s in self.OVERRIDE_SIGNALS if s in message]
        confidence = 0.0
        
        # Base score from signal matching
        if signals:
            confidence = 0.8 + (len(signals) * 0.05)  # 0.8-0.99
        
        confidence = min(confidence, 0.99)
        
        # If low confidence, return with confidence 0 (not an override)
        if confidence < 0.5:
            return IntentDecision(
                intent_type=IntentType.INTENT_OVERRIDE,
                confidence=0.0,
                rationale="No override signals detected",
                trigger_signals=[]
            )
        
        return IntentDecision(
            intent_type=IntentType.INTENT_OVERRIDE,
            confidence=confidence,
            rationale=f"Detected override signals: {', '.join(signals[:3])}",
            trigger_signals=signals
        )
