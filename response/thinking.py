"""
Thinking Subsystem

Break down the problem, identify what's being asked, plan the approach.
Produces a structured ThoughtPlan for the Deep Pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from config import config


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
    Thinking subsystem for query decomposition and planning.
    
    Provides:
    - Query understanding and classification
    - Step planning
    - Memory and skill identification
    - Parallelization opportunities
    
    Attributes:
        llm: LLM callable for complex reasoning.
    """
    
    # Keywords that indicate different complexity levels
    ANALYSIS_KEYWORDS = [
        "analyze", "compare", "trend", "pattern", "breakdown",
        "summary", "overview", "explain", "why", "how",
    ]
    
    MULTI_STEP_KEYWORDS = [
        "and then", "after that", "first", "next", "finally",
        "step by step", "process", "workflow",
    ]
    
    CREATIVE_KEYWORDS = [
        "suggest", "recommend", "ideas", "alternatives",
        "what if", "imagine", "create", "design",
    ]
    
    PREDICTION_KEYWORDS = [
        "predict", "forecast", "estimate", "project",
        "will", "future", "expect", "likely",
    ]
    
    def __init__(
        self,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        """
        Initialize Thinker.
        
        Args:
            llm: LLM callable for complex reasoning.
        """
        self.llm = llm
    
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
        lower = query.lower()
        
        # Determine complexity
        complexity = self._classify_complexity(lower)
        
        # Determine what's needed
        requires_creativity = any(kw in lower for kw in self.CREATIVE_KEYWORDS)
        requires_prediction = any(kw in lower for kw in self.PREDICTION_KEYWORDS)
        
        # Determine memory types needed
        memory_needed = self._identify_memory_needs(lower, complexity)
        
        # Determine skills needed
        skills_needed = self._identify_skill_needs(lower)
        
        # Generate steps
        if self.llm and complexity in (QueryComplexity.MULTI_STEP, QueryComplexity.CREATIVE):
            # Use LLM for complex planning
            plan = await self._llm_plan(query, context)
            plan.requires_creativity = requires_creativity
            plan.requires_prediction = requires_prediction
            return plan
        else:
            # Rule-based planning
            steps = self._generate_steps(query, complexity)
            
            return ThoughtPlan(
                understanding=self._extract_understanding(query),
                complexity=complexity,
                steps=steps,
                memory_needed=memory_needed,
                skills_needed=skills_needed,
                requires_creativity=requires_creativity,
                requires_prediction=requires_prediction,
            )
    
    def _classify_complexity(self, text: str) -> QueryComplexity:
        """Classify query complexity."""
        if any(kw in text for kw in self.CREATIVE_KEYWORDS):
            return QueryComplexity.CREATIVE
        
        if any(kw in text for kw in self.MULTI_STEP_KEYWORDS):
            return QueryComplexity.MULTI_STEP
        
        if any(kw in text for kw in self.ANALYSIS_KEYWORDS):
            return QueryComplexity.ANALYSIS
        
        return QueryComplexity.SIMPLE_LOOKUP
    
    def _identify_memory_needs(
        self,
        text: str,
        complexity: QueryComplexity,
    ) -> list[str]:
        """Identify which memory types are needed."""
        needs = ["sfm"]  # Always check SFM first
        
        if complexity != QueryComplexity.SIMPLE_LOOKUP:
            needs.append("lfm")  # Deep knowledge for analysis
        
        # Check for relationship/graph queries
        relationship_keywords = ["related", "connected", "caused", "affects", "between"]
        if any(kw in text for kw in relationship_keywords):
            needs.append("am")
        
        # Check for procedure/how-to queries
        procedure_keywords = ["how to", "steps to", "process", "procedure", "workflow"]
        if any(kw in text for kw in procedure_keywords):
            needs.append("mm")
        
        return needs
    
    def _identify_skill_needs(self, text: str) -> list[str]:
        """Identify which skills might be needed."""
        skills = []
        
        # Data queries
        data_keywords = ["data", "query", "fetch", "get", "show", "list"]
        if any(kw in text for kw in data_keywords):
            skills.append("data_query")
        
        # Calculations
        calc_keywords = ["calculate", "compute", "sum", "average", "total"]
        if any(kw in text for kw in calc_keywords):
            skills.append("calculation")
        
        # Visualization
        viz_keywords = ["chart", "graph", "plot", "visualize", "show me"]
        if any(kw in text for kw in viz_keywords):
            skills.append("visualization")
        
        return skills
    
    def _generate_steps(
        self,
        query: str,
        complexity: QueryComplexity,
    ) -> list[str]:
        """Generate execution steps based on complexity."""
        if complexity == QueryComplexity.SIMPLE_LOOKUP:
            return [
                "Search memory for relevant facts",
                "Return direct answer",
            ]
        
        if complexity == QueryComplexity.ANALYSIS:
            return [
                "Retrieve relevant data from memory",
                "Analyze and synthesize information",
                "Generate insights",
                "Formulate response",
            ]
        
        if complexity == QueryComplexity.MULTI_STEP:
            return [
                "Break down into sub-tasks",
                "Execute each sub-task sequentially",
                "Combine results",
                "Formulate comprehensive response",
            ]
        
        if complexity == QueryComplexity.CREATIVE:
            return [
                "Understand the creative goal",
                "Retrieve relevant context",
                "Generate creative options",
                "Evaluate and refine",
                "Present recommendations",
            ]
        
        return ["Process query", "Generate response"]
    
    def _extract_understanding(self, query: str) -> str:
        """Extract a concise understanding of the query."""
        # Simple extraction - first sentence or up to 100 chars
        understanding = query.split(".")[0].strip()
        if len(understanding) > 100:
            understanding = understanding[:97] + "..."
        return understanding
    
    async def _llm_plan(
        self,
        query: str,
        context: Optional[str],
    ) -> ThoughtPlan:
        """Use LLM for complex planning."""
        if not self.llm:
            return ThoughtPlan.simple_lookup(query)
        
        prompt = f"""Analyze this query and create an execution plan.

Query: {query}
{f"Context: {context}" if context else ""}

Respond with:
1. Understanding: What is the user asking? (1 sentence)
2. Complexity: simple_lookup | analysis | multi_step | creative
3. Steps: List of steps to solve this (numbered)
4. Memory needed: Which memory types? (sfm, lfm, am, mm)
5. Skills needed: What capabilities? (data_query, calculation, visualization)
"""
        
        response = await self.llm(prompt)
        
        # Parse response (simplified - would use structured output in production)
        return ThoughtPlan(
            understanding=query[:100],
            complexity=QueryComplexity.ANALYSIS,
            steps=["Analyze query", "Retrieve context", "Generate response"],
            memory_needed=["sfm", "lfm"],
        )


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
