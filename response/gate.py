"""
Gate

Fast, cheap filter for trivial interactions. Prevents expensive
Deep Pipeline calls for greetings, chitchat, or simple trivia.

Binary classifier: "gate" (handle here) or "deep" (forward to pipeline).
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class GateDecision(Enum):
    """Gate classification result."""
    GATE = "gate"  # Handle at gate level
    DEEP = "deep"  # Forward to deep pipeline


class MessageType(Enum):
    """Type of message detected by gate."""
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    THANKS = "thanks"
    FAREWELL = "farewell"
    TRIVIA = "trivia"
    DOMAIN = "domain"  # Requires deep pipeline


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GateResult:
    """Result of gate classification."""
    decision: GateDecision
    message_type: MessageType
    response: Optional[str] = None  # Pre-generated response for gate-level
    confidence: float = 1.0
    processing_time_ms: float = 0.0
    
    @property
    def is_gate(self) -> bool:
        return self.decision == GateDecision.GATE
    
    @property
    def is_deep(self) -> bool:
        return self.decision == GateDecision.DEEP
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "message_type": self.message_type.value,
            "response": self.response,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Gate
# ══════════════════════════════════════════════════════════════════════════════

class Gate:
    """
    Fast filter for trivial interactions.
    
    Provides:
    - Rule-based classification (fast, no LLM)
    - Template responses for greetings/chitchat
    - Escalation detection for rejected responses
    
    Attributes:
        agent_name: Name of the agent for responses.
        domain_terms: Domain-specific terms that indicate deep pipeline.
    """
    
    # Pattern lists
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|howdy|greetings|yo|sup)[\s!.,?]*$",
        r"^good\s+(morning|afternoon|evening|day)[\s!.,?]*$",
        r"^what'?s?\s+up[\s!.,?]*$",
    ]
    
    CHITCHAT_PATTERNS = [
        r"^how\s+are\s+you",
        r"^how'?s?\s+it\s+going",
        r"^what\s+are\s+you\s+doing",
        r"^are\s+you\s+(there|busy|available)",
        r"^nice\s+to\s+(meet|talk)",
    ]
    
    THANKS_PATTERNS = [
        r"^(thanks?|thank\s+you|thx|ty)[\s!.,?]*",
        r"^(appreciate|grateful)",
        r"^(great|awesome|perfect|excellent)[\s!.,?]*$",
    ]
    
    FAREWELL_PATTERNS = [
        r"^(bye|goodbye|see\s+you|later|cya|ttyl)[\s!.,?]*$",
        r"^(good\s+night|have\s+a\s+good)[\s!.,?]*",
        r"^(take\s+care|talk\s+soon)[\s!.,?]*",
    ]
    
    ESCALATION_PATTERNS = [
        r"that'?s?\s+not\s+what\s+i\s+(meant|asked)",
        r"i\s+need\s+more\s+(detail|info|help)",
        r"can\s+you\s+(explain|elaborate|help)",
        r"(no|nope|wrong|incorrect)",
        r"(actually|but|however)",
    ]
    
    # Response templates
    GREETING_RESPONSES = [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! Ready to assist you.",
    ]
    
    CHITCHAT_RESPONSES = [
        "I'm doing well, thanks for asking! How can I help you?",
        "All good here! What would you like to work on?",
        "Ready and waiting! What's on your mind?",
    ]
    
    THANKS_RESPONSES = [
        "You're welcome! Let me know if you need anything else.",
        "Happy to help! Anything else?",
        "Glad I could assist!",
    ]
    
    FAREWELL_RESPONSES = [
        "Goodbye! Have a great day!",
        "See you later! Take care!",
        "Bye for now! Come back anytime.",
    ]
    
    def __init__(
        self,
        agent_name: str = "CogniCore",
        domain_terms: Optional[list[str]] = None,
    ):
        """
        Initialize Gate.
        
        Args:
            agent_name: Name of the agent.
            domain_terms: Terms that indicate domain-specific queries.
        """
        self.agent_name = agent_name
        self.domain_terms = domain_terms or []
        
        # Compile patterns
        self._greeting_re = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self._chitchat_re = [re.compile(p, re.IGNORECASE) for p in self.CHITCHAT_PATTERNS]
        self._thanks_re = [re.compile(p, re.IGNORECASE) for p in self.THANKS_PATTERNS]
        self._farewell_re = [re.compile(p, re.IGNORECASE) for p in self.FAREWELL_PATTERNS]
        self._escalation_re = [re.compile(p, re.IGNORECASE) for p in self.ESCALATION_PATTERNS]
    
    def classify(self, message: str) -> GateResult:
        """
        Classify a message as gate-level or deep pipeline.
        
        Args:
            message: User message.
            
        Returns:
            GateResult with decision and optional response.
        """
        import time
        import random
        
        start_time = time.perf_counter()
        
        text = message.strip()
        lower = text.lower()
        
        # Check for domain terms first - always deep
        if self._has_domain_terms(lower):
            return GateResult(
                decision=GateDecision.DEEP,
                message_type=MessageType.DOMAIN,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Check greeting patterns
        if self._matches_any(text, self._greeting_re):
            return GateResult(
                decision=GateDecision.GATE,
                message_type=MessageType.GREETING,
                response=random.choice(self.GREETING_RESPONSES),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Check chitchat patterns
        if self._matches_any(text, self._chitchat_re):
            return GateResult(
                decision=GateDecision.GATE,
                message_type=MessageType.CHITCHAT,
                response=random.choice(self.CHITCHAT_RESPONSES),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Check thanks patterns
        if self._matches_any(text, self._thanks_re):
            return GateResult(
                decision=GateDecision.GATE,
                message_type=MessageType.THANKS,
                response=random.choice(self.THANKS_RESPONSES),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Check farewell patterns
        if self._matches_any(text, self._farewell_re):
            return GateResult(
                decision=GateDecision.GATE,
                message_type=MessageType.FAREWELL,
                response=random.choice(self.FAREWELL_RESPONSES),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Default to deep pipeline
        return GateResult(
            decision=GateDecision.DEEP,
            message_type=MessageType.DOMAIN,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    def is_escalation(self, message: str) -> bool:
        """
        Check if message is an escalation (user rejected gate response).
        
        Args:
            message: User message.
            
        Returns:
            True if escalation detected.
        """
        return self._matches_any(message, self._escalation_re)
    
    def _has_domain_terms(self, text: str) -> bool:
        """Check if text contains domain-specific terms."""
        return any(term.lower() in text for term in self.domain_terms)
    
    def _matches_any(self, text: str, patterns: list[re.Pattern]) -> bool:
        """Check if text matches any pattern."""
        return any(p.search(text) for p in patterns)


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_gate_instance: Optional[Gate] = None


def get_gate() -> Gate:
    """Get the singleton Gate instance."""
    global _gate_instance
    if _gate_instance is None:
        # Load domain terms from config
        domain_terms = list(config.domains) + list(config.entity_types)
        _gate_instance = Gate(domain_terms=domain_terms)
    return _gate_instance
