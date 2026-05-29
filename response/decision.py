"""
Decision Making Subsystem

Based on the thought plan and recalled context, decide the execution strategy.
Determines which skills to execute, whether to spawn sub-agents, etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from config import config
from response.thinking import ThoughtPlan, QueryComplexity


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class ExecutionStrategy(Enum):
    """Strategy for executing the query."""
    DIRECT_ANSWER = "direct_answer"      # Answer from memory, no skills
    SKILL_EXECUTION = "skill_execution"  # Run pre-built skills
    MULTI_AGENT = "multi_agent"          # Spawn sub-agents for parallel work
    CODE_GEN = "code_gen"                # Generate code on-the-fly


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SkillCall:
    """A skill invocation."""
    skill_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Lower = higher priority
    timeout_seconds: int = 30
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "parameters": self.parameters,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class SubAgentSpec:
    """Specification for a sub-agent."""
    task: str
    context: str
    timeout_seconds: int = 60
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "context": self.context,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ExecutionPlan:
    """
    Plan for executing a query.
    
    Produced by DecisionMaker, consumed by Skills and Synthesis.
    """
    strategy: ExecutionStrategy
    skill_calls: list[SkillCall] = field(default_factory=list)
    parallel_groups: list[list[SkillCall]] = field(default_factory=list)
    needs_code_gen: bool = False
    needs_creativity: bool = False
    needs_prediction: bool = False
    sub_agents: list[SubAgentSpec] = field(default_factory=list)
    model_tier: str = "cheap"  # "expensive" | "cheap"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "skill_calls": [s.to_dict() for s in self.skill_calls],
            "parallel_groups": [[s.to_dict() for s in g] for g in self.parallel_groups],
            "needs_code_gen": self.needs_code_gen,
            "needs_creativity": self.needs_creativity,
            "needs_prediction": self.needs_prediction,
            "sub_agents": [a.to_dict() for a in self.sub_agents],
            "model_tier": self.model_tier,
        }
    
    @classmethod
    def direct_answer(cls) -> "ExecutionPlan":
        """Create a direct answer plan (no skills)."""
        return cls(strategy=ExecutionStrategy.DIRECT_ANSWER)
    
    @classmethod
    def with_skills(cls, skills: list[SkillCall]) -> "ExecutionPlan":
        """Create a skill execution plan."""
        return cls(
            strategy=ExecutionStrategy.SKILL_EXECUTION,
            skill_calls=skills,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Decision Maker
# ══════════════════════════════════════════════════════════════════════════════

class DecisionMaker:
    """
    Decision making subsystem.
    
    Provides:
    - Execution strategy selection
    - Skill call planning
    - Sub-agent spawning decisions
    - Model tier selection
    
    Attributes:
        llm: LLM callable for complex decisions.
    """
    
    def __init__(
        self,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        """
        Initialize DecisionMaker.
        
        Args:
            llm: LLM callable for complex decisions.
        """
        self.llm = llm
    
    async def decide(
        self,
        query: str,
        thought_plan: ThoughtPlan,
        recall_context: Optional[str] = None,
    ) -> ExecutionPlan:
        """
        Decide on execution strategy.
        
        Args:
            query: User query.
            thought_plan: Plan from Thinking subsystem.
            recall_context: Context from Memory Recall.
            
        Returns:
            ExecutionPlan for execution.
        """
        # Simple lookup → direct answer
        if thought_plan.complexity == QueryComplexity.SIMPLE_LOOKUP:
            return ExecutionPlan.direct_answer()
        
        # Determine strategy based on complexity and needs
        strategy = self._select_strategy(thought_plan)
        
        # Build skill calls
        skill_calls = self._build_skill_calls(thought_plan)
        
        # Determine parallelization
        parallel_groups = self._identify_parallel_groups(skill_calls)
        
        # Determine sub-agents
        sub_agents = self._plan_sub_agents(thought_plan) if strategy == ExecutionStrategy.MULTI_AGENT else []
        
        # Determine model tier
        model_tier = self._select_model_tier(thought_plan)
        
        return ExecutionPlan(
            strategy=strategy,
            skill_calls=skill_calls,
            parallel_groups=parallel_groups,
            needs_code_gen=strategy == ExecutionStrategy.CODE_GEN,
            needs_creativity=thought_plan.requires_creativity,
            needs_prediction=thought_plan.requires_prediction,
            sub_agents=sub_agents,
            model_tier=model_tier,
        )
    
    def _select_strategy(self, plan: ThoughtPlan) -> ExecutionStrategy:
        """Select execution strategy based on thought plan."""
        # Multi-step with parallelizable work → multi-agent
        if plan.complexity == QueryComplexity.MULTI_STEP and len(plan.parallelizable) > 1:
            return ExecutionStrategy.MULTI_AGENT
        
        # Has skills → skill execution
        if plan.skills_needed:
            return ExecutionStrategy.SKILL_EXECUTION
        
        # Creative/complex without pre-built skills → code gen
        if plan.complexity == QueryComplexity.CREATIVE and not plan.skills_needed:
            return ExecutionStrategy.CODE_GEN
        
        # Default to direct answer
        return ExecutionStrategy.DIRECT_ANSWER
    
    def _build_skill_calls(self, plan: ThoughtPlan) -> list[SkillCall]:
        """Build skill calls from thought plan."""
        calls = []
        
        for i, skill_name in enumerate(plan.skills_needed):
            calls.append(SkillCall(
                skill_name=skill_name,
                parameters={},  # Would be populated based on query analysis
                priority=i + 1,
            ))
        
        return calls
    
    def _identify_parallel_groups(
        self,
        skill_calls: list[SkillCall],
    ) -> list[list[SkillCall]]:
        """Identify groups of skills that can run in parallel."""
        if len(skill_calls) <= 1:
            return []
        
        # Simple heuristic: same priority = can parallelize
        groups: dict[int, list[SkillCall]] = {}
        for call in skill_calls:
            if call.priority not in groups:
                groups[call.priority] = []
            groups[call.priority].append(call)
        
        # Return groups with more than one skill
        return [g for g in groups.values() if len(g) > 1]
    
    def _plan_sub_agents(self, plan: ThoughtPlan) -> list[SubAgentSpec]:
        """Plan sub-agents for parallel work."""
        agents = []
        
        for group in plan.parallelizable:
            if len(group) > 0:
                agents.append(SubAgentSpec(
                    task=f"Execute: {', '.join(group)}",
                    context="",
                ))
        
        return agents
    
    def _select_model_tier(self, plan: ThoughtPlan) -> str:
        """Select model tier based on complexity."""
        # Creative and multi-step need expensive model
        if plan.complexity in (QueryComplexity.CREATIVE, QueryComplexity.MULTI_STEP):
            return "expensive"
        
        # Analysis might need expensive
        if plan.complexity == QueryComplexity.ANALYSIS and plan.confidence < 0.7:
            return "expensive"
        
        return "cheap"


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
