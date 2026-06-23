"""
Corrective Learning (Learning Type 5)

Learn from explicit corrections. When a user says "that's wrong,"
update facts, annotate sources, and prevent the same mistake.

Algorithm: NLI correction detection + fact versioning
Trigger: Real-time (during conversation)
LLM: Yes (cheap - for rephrasing corrected fact)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from logger import log
from observability import audit


# Correction markers in user responses
CORRECTION_MARKERS = [
    "no",
    "wrong",
    "actually",
    "not correct",
    "incorrect",
    "that's not",
    "that isn't",
    "you're wrong",
    "mistake",
    "error",
    "false",
    "inaccurate",
]


@dataclass
class Correction:
    """A detected correction."""
    agent_statement: str
    user_correction: str
    original_fact_id: Optional[str]
    corrected_content: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_statement": self.agent_statement,
            "user_correction": self.user_correction,
            "original_fact_id": self.original_fact_id,
            "corrected_content": self.corrected_content,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CorrectionResult:
    """Result of applying a correction."""
    correction: Correction
    fact_updated: bool
    contrastive_pair_created: bool
    source_annotated: bool
    bandit_updated: bool
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "correction": self.correction.to_dict(),
            "fact_updated": self.fact_updated,
            "contrastive_pair_created": self.contrastive_pair_created,
            "source_annotated": self.source_annotated,
            "bandit_updated": self.bandit_updated,
        }


class CorrectiveLearner:
    """
    Learns from user corrections.
    
    Detects when user corrects the agent, updates facts,
    and creates contrastive pairs to prevent future mistakes.
    
    Attributes:
        nli: NLI model for contradiction detection.
        llm: LLM callable for rephrasing.
    """
    
    def __init__(
        self,
        nli: Optional[Callable[[str, str], Awaitable[dict[str, float]]]] = None,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
        contradiction_threshold: float = 0.5,
    ):
        """
        Initialize corrective learner.
        
        Args:
            nli: NLI model callable (returns {entailment, neutral, contradiction}).
            llm: LLM callable for rephrasing.
            contradiction_threshold: Threshold for contradiction detection.
        """
        self.nli = nli
        self.llm = llm
        self.contradiction_threshold = contradiction_threshold
        self._corrections: list[Correction] = []
    
    def has_correction_markers(self, text: str) -> bool:
        """Check if text contains correction markers."""
        text_lower = text.lower()
        return any(marker in text_lower for marker in CORRECTION_MARKERS)
    
    async def detect_correction(
        self,
        agent_statement: str,
        user_response: str,
    ) -> Optional[Correction]:
        """
        Detect if user is correcting the agent.
        
        Args:
            agent_statement: What the agent said.
            user_response: User's response.
            
        Returns:
            Correction if detected, None otherwise.
        """
        # Quick check for correction markers
        if not self.has_correction_markers(user_response):
            return None
        
        # Use NLI for contradiction detection
        contradiction_score = 0.0
        
        if self.nli:
            scores = await self.nli(agent_statement, user_response)
            contradiction_score = scores.get("contradiction", 0.0)
        else:
            # Fallback: marker-based detection
            contradiction_score = 0.6 if self.has_correction_markers(user_response) else 0.0
        
        if contradiction_score < self.contradiction_threshold:
            return None
        
        # Extract corrected content
        if self.llm:
            corrected = await self._extract_correction(agent_statement, user_response)
        else:
            corrected = user_response
        
        correction = Correction(
            agent_statement=agent_statement,
            user_correction=user_response,
            original_fact_id=None,  # To be filled by caller
            corrected_content=corrected,
            confidence=contradiction_score,
        )
        
        self._corrections.append(correction)
        
        audit.log_raw(
            "learning",
            "correction_detected",
            "mml",
            "completed",
            confidence=contradiction_score,
        )
        
        return correction
    
    async def _extract_correction(
        self,
        agent_statement: str,
        user_response: str,
    ) -> str:
        """Extract the corrected fact from user response."""
        prompt = f"""
The agent said: "{agent_statement}"
The user corrected: "{user_response}"

Write the corrected fact as a single, clear statement.
Combine the original information with the user's correction.
"""
        
        return await self.llm(prompt)
    
    async def apply_correction(
        self,
        correction: Correction,
        update_fact: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        create_contrastive: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        annotate_source: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        update_bandit: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ) -> CorrectionResult:
        """
        Apply a correction to the memory system.
        
        Args:
            correction: The correction to apply.
            update_fact: Function to update SFM fact.
            create_contrastive: Function to create contrastive pair.
            annotate_source: Function to annotate LFM source.
            update_bandit: Function to update bandit with negative reward.
            
        Returns:
            CorrectionResult with actions taken.
        """
        result = CorrectionResult(
            correction=correction,
            fact_updated=False,
            contrastive_pair_created=False,
            source_annotated=False,
            bandit_updated=False,
        )
        
        # Update fact in SFM
        if update_fact and correction.original_fact_id:
            result.fact_updated = await update_fact(
                correction.original_fact_id,
                correction.corrected_content,
            )
        
        # Create contrastive pair
        if create_contrastive:
            contrastive = f"{correction.agent_statement} ≠ {correction.corrected_content}"
            result.contrastive_pair_created = await create_contrastive(
                correction.agent_statement,
                contrastive,
            )
        
        # Annotate source
        if annotate_source and correction.original_fact_id:
            annotation = f"Corrected on {correction.timestamp.isoformat()}: {correction.user_correction}"
            result.source_annotated = await annotate_source(
                correction.original_fact_id,
                annotation,
            )
        
        # Update bandit with negative reward
        if update_bandit and correction.original_fact_id:
            await update_bandit(correction.original_fact_id, -0.5)
            result.bandit_updated = True
        
        audit.log_raw(
            "learning",
            "correction_applied",
            "mml",
            "completed",
            fact_updated=result.fact_updated,
            contrastive_created=result.contrastive_pair_created,
        )
        
        return result
    
    def get_recent_corrections(self, limit: int = 10) -> list[Correction]:
        """Get recent corrections."""
        return self._corrections[-limit:]
    
    def get_correction_stats(self) -> dict[str, Any]:
        """Get correction statistics."""
        if not self._corrections:
            return {"total": 0, "avg_confidence": 0.0}
        
        return {
            "total": len(self._corrections),
            "avg_confidence": sum(c.confidence for c in self._corrections) / len(self._corrections),
            "recent": len([c for c in self._corrections if (
                datetime.now(timezone.utc) - c.timestamp
            ).days < 7]),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_corrective_learner: Optional[CorrectiveLearner] = None


def get_corrective_learner() -> CorrectiveLearner:
    """Get the singleton CorrectiveLearner instance."""
    global _corrective_learner
    if _corrective_learner is None:
        _corrective_learner = CorrectiveLearner()
    return _corrective_learner
