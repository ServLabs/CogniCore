"""
Thinking Subsystem

Break down the problem, identify what's being asked, plan the approach.
Produces a structured ThoughtPlan for the Deep Pipeline.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from connectors import genai
from prompts import prompts


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class QueryComplexity(Enum):
    """Complexity level of a query."""
    SIMPLE_LOOKUP = "simple_lookup"      # Direct fact retrieval
    ANALYSIS = "analysis"                 # Requires computation/reasoning
    MULTI_STEP = "multi_step"            # Multiple sequential steps
    CREATIVE = "creative"                 # Novel generation required


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThoughtPlan:
    """
    Structured plan for answering a query.
    
    Produced by the Thinking subsystem, consumed by Decision and other subsystems.
    """
    understanding: str  # What is the user actually asking?
    complexity: QueryComplexity
    steps: list[str] = field(default_factory=list)  # Planned steps to solve
    memory_needed: list[str] = field(default_factory=list)  # Memory types to query
    skills_needed: list[str] = field(default_factory=list)  # Skills to invoke
    parallelizable: list[list[str]] = field(default_factory=list)  # Groups that can run in parallel
    confidence: float = 0.8  # How confident the plan is (0-1)
    requires_creativity: bool = False
    requires_prediction: bool = False
    new_insight: Optional[str] = None  # Knowledge extracted from user message (for staging)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "understanding": self.understanding,
            "complexity": self.complexity.value,
            "steps": self.steps,
            "memory_needed": self.memory_needed,
            "skills_needed": self.skills_needed,
            "parallelizable": self.parallelizable,
            "confidence": self.confidence,
            "requires_creativity": self.requires_creativity,
            "requires_prediction": self.requires_prediction,
        }
    
    @classmethod
    def simple_lookup(cls, understanding: str) -> "ThoughtPlan":
        """Create a simple lookup plan."""
        return cls(
            understanding=understanding,
            complexity=QueryComplexity.SIMPLE_LOOKUP,
            steps=["Retrieve relevant facts from memory"],
            memory_needed=["sfm", "lfm"],
            confidence=0.9,
        )
    
    @classmethod
    def analysis(cls, understanding: str, steps: list[str]) -> "ThoughtPlan":
        """Create an analysis plan."""
        return cls(
            understanding=understanding,
            complexity=QueryComplexity.ANALYSIS,
            steps=steps,
            memory_needed=["sfm", "lfm", "am"],
            confidence=0.8,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Thinker
# ══════════════════════════════════════════════════════════════════════════════

class Thinker:
    """
    AI-powered thinking subsystem for query decomposition and planning.

    Uses genai.ask() to analyze queries and produce structured ThoughtPlans.
    If genai is unreachable, falls back to a simple lookup plan.
    """

    async def think(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> ThoughtPlan:
        """
        Analyze a query and produce a thought plan.

        Args:
            query: User query.
            context: Optional conversation context.

        Returns:
            ThoughtPlan for the query.
        """
        context_block = f"Conversation context:\n{context}" if context else ""
        messages = prompts.get_messages("response/thinking.md", message=query, context_block=context_block)

        try:
            raw = await genai.ask({
                "model": "cheap",
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })
            return self._parse_response(raw, query)
        except Exception:
            return self._fallback(query)

    def _parse_response(self, raw: str, query: str) -> ThoughtPlan:
        """Parse the genai JSON response into a ThoughtPlan."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0].strip()

            data = json.loads(text)

            complexity_str = data.get("complexity", "simple_lookup")
            try:
                complexity = QueryComplexity(complexity_str)
            except ValueError:
                complexity = QueryComplexity.SIMPLE_LOOKUP

            return ThoughtPlan(
                understanding=data.get("understanding", query[:100]),
                complexity=complexity,
                steps=data.get("steps", []),
                memory_needed=data.get("memory_needed", ["sfm"]),
                skills_needed=data.get("skills_needed", []),
                parallelizable=data.get("parallelizable", []),
                confidence=float(data.get("confidence", 0.8)),
                requires_creativity=bool(data.get("requires_creativity", False)),
                requires_prediction=bool(data.get("requires_prediction", False)),
                new_insight=data.get("new_insight"),
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return self._fallback(query)

    def _fallback(self, query: str) -> ThoughtPlan:
        """Fallback: simple lookup plan when genai is unreachable."""
        return ThoughtPlan.simple_lookup(query[:100])


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_thinker_instance: Optional[Thinker] = None


def get_thinker() -> Thinker:
    """Get the singleton Thinker instance."""
    global _thinker_instance
    if _thinker_instance is None:
        _thinker_instance = Thinker()
    return _thinker_instance
