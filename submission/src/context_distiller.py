"""Context distillation: summarize session state for LLM templates."""

from __future__ import annotations

from dataclasses import dataclass

from techjam.submission.src import dialogue


@dataclass
class DistilledContext:
    """Compressed session context for LLM generation."""
    session_summary: str  # One-liner about what's happened
    confirmed_constraints: list[str]  # What user has confirmed
    rejected_constraints: list[str]  # What user has rejected
    ambiguous_areas: list[str]  # What's still unclear
    turns_remaining: int  # How many turns left (10 - current)
    user_profile_summary: str  # User's decision-making style
    products_status: str  # How many options remain
    estimated_opportunity: str  # What question would help most


class ContextDistiller:
    """Compress session state into narrative for LLM."""
    
    def distill(
        self,
        session_state: dialogue.SessionState,
        conversation_history: list[tuple[str, str]] | None = None,
        candidate_count: int = 0,
        user_profile: dict | None = None,
        asked: str | None = None,
    ) -> DistilledContext:
        """
        Create a compressed context summary for LLM templates.
        
        Args:
            session_state: Current session state
            conversation_history: Previous turns
        
        Returns:
            DistilledContext with summarized information
        """
        # Analyze constraints
        confirmed = list(session_state.constraints) if session_state.constraints else []
        rejected = self._extract_rejections(conversation_history or [])
        
        # Determine ambiguous areas (attributes not yet touched)
        all_attributes = {"material", "budget", "style", "brand", "size", "weather", "occasion"}
        mentioned = set(self._extract_mentioned_attributes(confirmed + rejected))
        ambiguous = list(all_attributes - mentioned)
        
        # Create narrative
        session_summary = self._create_summary(
            session_state.turn,
            len(confirmed),
            candidate_count
        )
        
        user_summary = self._summarize_user_profile(
            user_profile if isinstance(user_profile, dict) else {}
        )
        products_summary = self._summarize_products(candidate_count)
        opportunity = self._identify_opportunity(
            confirmed, rejected, ambiguous, candidate_count, asked
        )
        
        return DistilledContext(
            session_summary=session_summary,
            confirmed_constraints=confirmed,
            rejected_constraints=rejected,
            ambiguous_areas=ambiguous,
            turns_remaining=10 - session_state.turn,
            user_profile_summary=user_summary,
            products_status=products_summary,
            estimated_opportunity=opportunity,
        )
    
    def _extract_rejections(self, history: list[tuple[str, str]]) -> list[str]:
        """Extract what user has explicitly rejected."""
        rejections = []
        
        for user_msg, _ in history:
            msg_lower = user_msg.lower()
            
            # Look for negation patterns
            if any(pattern in msg_lower for pattern in ["not", "no", "don't", "avoid", "not interested"]):
                # Try to extract what was rejected
                if "not" in msg_lower:
                    # Simple extraction: "not X" → extract X
                    parts = msg_lower.split("not")
                    if len(parts) > 1:
                        rejection = parts[1].strip().split()[0]
                        if rejection and len(rejection) > 2:
                            rejections.append(rejection)
        
        return list(set(rejections))  # Deduplicate
    
    def _extract_mentioned_attributes(self, constraints: list[str]) -> list[str]:
        """Extract which attributes have been mentioned."""
        attributes = []
        constraint_text = " ".join(constraints).lower()
        
        # Map keywords to attributes
        attribute_keywords = {
            "material": ["leather", "fabric", "cotton", "polyester", "wool", "synthetic", "silk", "linen"],
            "budget": ["price", "cost", "expensive", "cheap", "dollar", "under", "above", "range"],
            "style": ["casual", "formal", "modern", "vintage", "minimalist", "classic", "trendy"],
            "size": ["small", "large", "medium", "fit", "length", "width", "oversized", "fitted"],
            "brand": ["brand", "nike", "adidas", "gucci", "designer", "luxury"],
            "weather": ["warm", "cold", "waterproof", "rain", "summer", "winter", "weather"],
            "occasion": ["work", "party", "sport", "gym", "casual", "formal", "outdoor"],
        }
        
        for attr, keywords in attribute_keywords.items():
            if any(keyword in constraint_text for keyword in keywords):
                attributes.append(attr)
        
        return attributes
    
    def _create_summary(self, turn: int, constraints_count: int, products_count: int) -> str:
        """Create one-liner summary of session state."""
        if turn <= 2:
            return f"Early in session, user has {constraints_count} clear preference(s)"
        elif turn <= 5:
            return f"Mid-session, {constraints_count} constraint(s) narrowing {products_count:,} options"
        else:
            return f"Late session (turn {turn}/10), {constraints_count} constraint(s), {products_count:,} options remain"
    
    def _summarize_user_profile(self, profile: dict) -> str:
        """Summarize user's decision-making style."""
        rating_style = profile.get("rating_style", "unknown")
        tags = profile.get("preference_tags", [])
        
        style_desc = {
            "critical": "Critical reviewer, values quality and durability",
            "usually positive": "Positive reviewer, values fit and comfort",
            "mixed": "Mixed reviewer, balanced preferences",
        }
        
        desc = style_desc.get(rating_style, "Unknown preference style")
        if tags:
            desc += f"; cares about {', '.join(tags[:3])}"
        
        return desc
    
    def _summarize_products(self, product_count: int) -> str:
        """Summarize product filtering status."""
        if product_count > 5000:
            return f"Crowded field ({product_count:,} options), need significant narrowing"
        elif product_count > 1000:
            return f"Still {product_count:,} options, more refinement needed"
        elif product_count > 100:
            return f"{product_count:,} focused options, getting close"
        else:
            return f"Well-narrowed ({product_count} options), ready to recommend"
    
    def _identify_opportunity(
        self,
        confirmed: list[str],
        rejected: list[str],
        ambiguous: list[str],
        candidate_count: int,
        asked: str | None,
    ) -> str:
        """Identify what question would help most now."""
        # If many ambiguous areas, pick the highest-variance one
        if asked:
            return f"Ask about {asked} to reduce the remaining uncertainty"

        if ambiguous:
            # Simple heuristic: material, budget, style are usually high-impact
            high_impact = {"material", "budget", "style"}
            for attr in high_impact:
                if attr in ambiguous:
                    return f"Ask about {attr} to narrow options significantly"
            
            return f"Clarify {ambiguous[0]} to refine recommendations"
        
        # If still many products, need more constraints
        if candidate_count > 500:
            return "Multiple constraints needed to narrow the field"
        
        # If few products and few constraints, user might be satisfied
        return "User may be ready for recommendations or final refinement"
