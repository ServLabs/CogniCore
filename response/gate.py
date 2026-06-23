"""
Gate

Fast, cheap filter for trivial interactions. Prevents expensive
Deep Pipeline calls for greetings, chitchat, or simple trivia.

Uses AI (cheap model) for classification and response generation.
Binary decision: "gate" (handle here) or "deep" (forward to pipeline).
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from connectors import genai
from prompts import prompts


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
    AI-powered filter for trivial interactions.

    Uses genai.ask() to classify messages and generate
    responses for non-domain interactions (greetings, chitchat, etc.).
    If genai is unreachable, falls back to sending everything
    to the deep pipeline.
    """

    async def classify(self, message: str) -> GateResult:
        """
        Classify a message as gate-level or deep pipeline.

        Args:
            message: User message.

        Returns:
            GateResult with decision and optional response.
        """
        start_time = time.perf_counter()
        messages = prompts.get_messages("response/gate.md", message=message)

        try:
            raw = await genai.ask({
                "model": "cheap",
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })

            return self._parse_response(raw, start_time)

        except Exception:
            return self._deep_fallback(start_time)

    def _parse_response(self, raw: str, start_time: float) -> GateResult:
        """Parse the LLM JSON response into a GateResult."""
        try:
            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0].strip()

            data = json.loads(text)

            decision_str = data.get("decision", "deep")
            type_str = data.get("message_type", "domain")
            response_text = data.get("response")

            decision = GateDecision(decision_str) if decision_str in ("gate", "deep") else GateDecision.DEEP
            message_type = MessageType(type_str) if type_str in MessageType.__members__.values() or type_str in [e.value for e in MessageType] else MessageType.DOMAIN

            return GateResult(
                decision=decision,
                message_type=message_type,
                response=response_text if decision == GateDecision.GATE else None,
                confidence=0.9,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return self._deep_fallback(start_time)

    def _deep_fallback(self, start_time: float) -> GateResult:
        """Fallback: send to deep pipeline."""
        return GateResult(
            decision=GateDecision.DEEP,
            message_type=MessageType.DOMAIN,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_gate_instance: Optional[Gate] = None


def get_gate() -> Gate:
    """Get the singleton Gate instance."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = Gate()
    return _gate_instance
