"""
Response Synthesis

Final step — combine all outputs (recall, skill results, creativity, prediction)
into a coherent response via AI.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from connectors import genai
from prompts import prompts
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
    AI-powered response synthesis.

    Uses genai.ask() to combine all pipeline outputs into a coherent response.
    Falls back to a simple concatenation if genai is unreachable.
    """

    async def synthesize(self, synthesis_input: SynthesisInput) -> SynthesisOutput:
        """
        Synthesize a response from all inputs.

        Args:
            synthesis_input: All inputs for synthesis.

        Returns:
            SynthesisOutput with final response.
        """
        information = self._build_information(synthesis_input)
        messages = prompts.get_messages(
            "response/synthesis.md",
            query=synthesis_input.query,
            understanding=synthesis_input.thought_plan.understanding,
            complexity=synthesis_input.thought_plan.complexity.value,
            strategy=synthesis_input.execution_plan.strategy.value,
            information=information,
        )

        try:
            raw = await genai.ask({
                "model": "default",
                "messages": messages,
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            })
            return self._parse_response(raw, synthesis_input)
        except Exception:
            return self._fallback(synthesis_input, information)

    def _build_information(self, synthesis_input: SynthesisInput) -> str:
        """Collect all available information into a single block."""
        parts = []

        if synthesis_input.recall_context:
            parts.append(f"[Memory]\n{synthesis_input.recall_context}")

        for result in synthesis_input.skill_results:
            if result.get("output"):
                name = result.get("skill") or result.get("task", "task")
                parts.append(f"[Task: {name}]\n{result['output']}")

        if synthesis_input.creativity_output:
            parts.append(f"[Creative Insights]\n{synthesis_input.creativity_output}")

        if synthesis_input.prediction_output:
            parts.append(f"[Prediction]\n{synthesis_input.prediction_output}")

        return "\n\n".join(parts) if parts else "(No information available)"

    def _parse_response(
        self, raw: str, synthesis_input: SynthesisInput
    ) -> SynthesisOutput:
        """Parse genai JSON response into SynthesisOutput."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0].strip()

            data = json.loads(text)

            return SynthesisOutput(
                response=data.get("response", ""),
                confidence=float(data.get("confidence", 0.8)),
                sources_used=data.get("sources_used", []),
                follow_up_suggestions=data.get("follow_up_suggestions", [])[:3],
                metadata={
                    "complexity": synthesis_input.thought_plan.complexity.value,
                    "strategy": synthesis_input.execution_plan.strategy.value,
                },
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            information = self._build_information(synthesis_input)
            return self._fallback(synthesis_input, information)

    def _fallback(
        self, synthesis_input: SynthesisInput, information: str
    ) -> SynthesisOutput:
        """Fallback when genai is unreachable."""
        if information == "(No information available)":
            response = "I don't have enough information to answer that question."
        else:
            response = information

        return SynthesisOutput(
            response=response,
            confidence=0.5,
            metadata={
                "complexity": synthesis_input.thought_plan.complexity.value,
                "strategy": synthesis_input.execution_plan.strategy.value,
                "fallback": True,
            },
        )


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
