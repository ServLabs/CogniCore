"""
Response Synthesis

Final step — combine all outputs (recall, skill results, creativity, prediction)
into a coherent response.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from config import config
from response.thinking import ThoughtPlan
from response.decision import ExecutionPlan


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SynthesisInput:
    """Input for response synthesis."""
    query: str
    thought_plan: ThoughtPlan
    execution_plan: ExecutionPlan
    recall_context: str = ""
    skill_results: list[dict[str, Any]] = field(default_factory=list)
    creativity_output: Optional[str] = None
    prediction_output: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "thought_plan": self.thought_plan.to_dict(),
            "execution_plan": self.execution_plan.to_dict(),
            "recall_context": self.recall_context,
            "skill_results": self.skill_results,
            "creativity_output": self.creativity_output,
            "prediction_output": self.prediction_output,
        }


@dataclass
class SynthesisOutput:
    """Output from response synthesis."""
    response: str
    confidence: float = 0.8
    sources_used: list[str] = field(default_factory=list)
    follow_up_suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "confidence": self.confidence,
            "sources_used": self.sources_used,
            "follow_up_suggestions": self.follow_up_suggestions,
            "metadata": self.metadata,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Synthesizer
# ══════════════════════════════════════════════════════════════════════════════

class Synthesizer:
    """
    Response synthesis subsystem.
    
    Provides:
    - Combining multiple outputs into coherent response
    - Formatting and structuring
    - Source attribution
    - Follow-up suggestions
    
    Attributes:
        llm: LLM callable for synthesis.
    """
    
    def __init__(
        self,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        """
        Initialize Synthesizer.
        
        Args:
            llm: LLM callable for synthesis.
        """
        self.llm = llm
    
    async def synthesize(
        self,
        synthesis_input: SynthesisInput,
    ) -> SynthesisOutput:
        """
        Synthesize a response from all inputs.
        
        Args:
            synthesis_input: All inputs for synthesis.
            
        Returns:
            SynthesisOutput with final response.
        """
        # Collect all content
        content_parts = []
        sources = []
        
        # Add recall context
        if synthesis_input.recall_context:
            content_parts.append(f"Context:\n{synthesis_input.recall_context}")
            sources.append("memory")
        
        # Add skill results
        for result in synthesis_input.skill_results:
            if result.get("output"):
                content_parts.append(f"Result:\n{result['output']}")
                sources.append(result.get("skill_name", "skill"))
        
        # Add creativity output
        if synthesis_input.creativity_output:
            content_parts.append(f"Insights:\n{synthesis_input.creativity_output}")
            sources.append("creativity")
        
        # Add prediction output
        if synthesis_input.prediction_output:
            content_parts.append(f"Forecast:\n{synthesis_input.prediction_output}")
            sources.append("prediction")
        
        # Synthesize with LLM if available
        if self.llm and content_parts:
            response = await self._llm_synthesize(
                query=synthesis_input.query,
                content_parts=content_parts,
                thought_plan=synthesis_input.thought_plan,
            )
        else:
            # Simple concatenation fallback
            response = self._simple_synthesize(
                query=synthesis_input.query,
                content_parts=content_parts,
            )
        
        # Generate follow-up suggestions
        follow_ups = self._generate_follow_ups(synthesis_input)
        
        return SynthesisOutput(
            response=response,
            confidence=synthesis_input.thought_plan.confidence,
            sources_used=sources,
            follow_up_suggestions=follow_ups,
            metadata={
                "complexity": synthesis_input.thought_plan.complexity.value,
                "strategy": synthesis_input.execution_plan.strategy.value,
            },
        )
    
    async def _llm_synthesize(
        self,
        query: str,
        content_parts: list[str],
        thought_plan: ThoughtPlan,
    ) -> str:
        """Use LLM to synthesize response."""
        if not self.llm:
            return self._simple_synthesize(query, content_parts)
        
        prompt = f"""Synthesize a response to the user's query using the provided information.

Query: {query}

Understanding: {thought_plan.understanding}

Available Information:
{chr(10).join(content_parts)}

Instructions:
- Be concise and direct
- Use the provided information to answer the query
- If information is incomplete, acknowledge it
- Maintain a helpful, professional tone
"""
        
        return await self.llm(prompt)
    
    def _simple_synthesize(
        self,
        query: str,
        content_parts: list[str],
    ) -> str:
        """Simple synthesis without LLM."""
        if not content_parts:
            return "I don't have enough information to answer that question."
        
        # Join content with formatting
        return "\n\n".join(content_parts)
    
    def _generate_follow_ups(
        self,
        synthesis_input: SynthesisInput,
    ) -> list[str]:
        """Generate follow-up suggestions."""
        suggestions = []
        
        # Based on complexity
        if synthesis_input.thought_plan.complexity.value == "analysis":
            suggestions.append("Would you like me to dive deeper into any specific aspect?")
        
        # Based on creativity
        if synthesis_input.creativity_output:
            suggestions.append("Would you like to explore any of these alternatives?")
        
        # Based on prediction
        if synthesis_input.prediction_output:
            suggestions.append("Would you like to see different scenarios?")
        
        return suggestions[:3]  # Max 3 suggestions


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_synthesizer_instance: Optional[Synthesizer] = None


def get_synthesizer() -> Synthesizer:
    """Get the singleton Synthesizer instance."""
    global _synthesizer_instance
    if _synthesizer_instance is None:
        _synthesizer_instance = Synthesizer()
    return _synthesizer_instance
