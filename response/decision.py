"""
Decision Making Subsystem

AI-powered decision engine. Based on the thought plan and recalled context,
decides whether to answer directly, ask for clarification, or execute tasks.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from connectors import genai
from prompts import prompts
from response.thinking import ThoughtPlan


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class ExecutionStrategy(Enum):
    """Strategy for executing the query."""
    DIRECT_ANSWER = "direct_answer"      # Answer from memory, no skills
    CLARIFY = "clarify"                  # Ask user for clarification
    EXECUTE = "execute"                  # Run tasks via sub-agents
    TOOL_USE = "tool_use"                # ReAct loop with tools


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskSpec:
    """A task for a sub-agent to execute."""
    task: str
    skill: Optional[str] = None
    timeout_seconds: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "skill": self.skill,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ExecutionPlan:
    """
    Plan for executing a query.

    Produced by DecisionMaker, consumed by Pipeline.
    """
    strategy: ExecutionStrategy
    reasoning: str = ""
    questions: list[str] = field(default_factory=list)  # For CLARIFY
    tasks: list[TaskSpec] = field(default_factory=list)  # For EXECUTE
    parallel: bool = False  # Whether tasks can run in parallel
    model_tier: str = "cheap"  # "cheap" | "default" | "expensive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "reasoning": self.reasoning,
            "questions": self.questions,
            "tasks": [t.to_dict() for t in self.tasks],
            "parallel": self.parallel,
            "model_tier": self.model_tier,
        }

    @classmethod
    def direct_answer(cls, reasoning: str = "Context is sufficient") -> "ExecutionPlan":
        """Create a direct answer plan."""
        return cls(strategy=ExecutionStrategy.DIRECT_ANSWER, reasoning=reasoning)

    @classmethod
    def clarify(cls, questions: list[str], reasoning: str = "") -> "ExecutionPlan":
        """Create a clarification plan."""
        return cls(
            strategy=ExecutionStrategy.CLARIFY,
            reasoning=reasoning,
            questions=questions,
        )

    @classmethod
    def execute(
        cls, tasks: list["TaskSpec"], parallel: bool = False, model_tier: str = "default",
        reasoning: str = "",
    ) -> "ExecutionPlan":
        """Create an execution plan with tasks."""
        return cls(
            strategy=ExecutionStrategy.EXECUTE,
            reasoning=reasoning,
            tasks=tasks,
            parallel=parallel,
            model_tier=model_tier,
        )
    
    @classmethod
    def tool_use(cls, reasoning: str = "Query requires external data or computation") -> "ExecutionPlan":
        """Create a tool-use plan (ReAct loop)."""
        return cls(strategy=ExecutionStrategy.TOOL_USE, reasoning=reasoning)


# ══════════════════════════════════════════════════════════════════════════════
# Decision Maker
# ══════════════════════════════════════════════════════════════════════════════

class DecisionMaker:
    """
    AI-powered decision engine.

    Uses genai.ask() to decide whether to answer directly, clarify, or execute.
    Falls back to direct_answer if genai is unreachable.
    """

    async def decide(
        self,
        query: str,
        thought_plan: ThoughtPlan,
        recall_context: str = "",
        clarification_history: str = "",
    ) -> ExecutionPlan:
        """
        Decide on execution strategy.

        Args:
            query: User query.
            thought_plan: Plan from Thinking subsystem.
            recall_context: Context from Memory Recall.
            clarification_history: Previous clarification Q&A (for multi-round).

        Returns:
            ExecutionPlan with strategy and details.
        """
        messages = prompts.get_messages(
            "response/decision.md",
            query=query,
            understanding=thought_plan.understanding,
            complexity=thought_plan.complexity.value,
            steps=", ".join(thought_plan.steps),
            skills_needed=", ".join(thought_plan.skills_needed) or "none",
            recall_context=recall_context or "(no context recalled)",
            clarification_history=clarification_history,
        )

        try:
            raw = await genai.ask({
                "model": "cheap",
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })
            return self._parse_response(raw)
        except Exception:
            return ExecutionPlan.direct_answer("Fallback — genai unreachable")

    def _parse_response(self, raw: str) -> ExecutionPlan:
        """Parse genai JSON response into an ExecutionPlan."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0].strip()

            data = json.loads(text)

            action = data.get("action", "direct_answer")
            reasoning = data.get("reasoning", "")
            model_tier = data.get("model_tier", "cheap")

            if action == "clarify":
                questions = data.get("questions", [])[:3]
                if not questions:
                    return ExecutionPlan.direct_answer("No questions generated")
                return ExecutionPlan.clarify(questions, reasoning)

            if action == "execute":
                raw_tasks = data.get("tasks", [])
                tasks = [
                    TaskSpec(
                        task=t.get("task", ""),
                        skill=t.get("skill"),
                        timeout_seconds=int(t.get("timeout_seconds", 60)),
                    )
                    for t in raw_tasks
                    if t.get("task")
                ]
                if not tasks:
                    return ExecutionPlan.direct_answer("No tasks generated")
                parallel = bool(data.get("parallel", False))
                return ExecutionPlan.execute(tasks, parallel, model_tier, reasoning)
            
            if action == "tool_use":
                return ExecutionPlan.tool_use(reasoning)

            return ExecutionPlan.direct_answer(reasoning)

        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return ExecutionPlan.direct_answer("Fallback — parse error")


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_decision_maker_instance: Optional[DecisionMaker] = None


def get_decision_maker() -> DecisionMaker:
    """Get the singleton DecisionMaker instance."""
    global _decision_maker_instance
    if _decision_maker_instance is None:
        _decision_maker_instance = DecisionMaker()
    return _decision_maker_instance
