"""
Response Layer

The agent's processing pipeline — receives user input, routes it,
reasons over it, executes skills, and produces a response.

This package provides:
- Gate: Fast filter for trivial interactions
- Deep Pipeline: Full reasoning for complex queries
- Thinking: Query decomposition and planning
- Decision: Execution strategy selection
- Skills: Script execution and DB queries
- Synthesis: Response generation
"""

from response.gate import (
    Gate,
    GateResult,
    get_gate,
)
from response.pipeline import (
    DeepPipeline,
    PipelineResult,
    get_pipeline,
    process_message,
)
from response.thinking import (
    ThoughtPlan,
    Thinker,
    get_thinker,
)
from response.decision import (
    ExecutionPlan,
    SkillCall,
    DecisionMaker,
    get_decision_maker,
)
from response.synthesis import (
    Synthesizer,
    get_synthesizer,
)

__all__ = [
    # Gate
    "Gate",
    "GateResult",
    "get_gate",
    # Pipeline
    "DeepPipeline",
    "PipelineResult",
    "get_pipeline",
    "process_message",
    # Thinking
    "ThoughtPlan",
    "Thinker",
    "get_thinker",
    # Decision
    "ExecutionPlan",
    "SkillCall",
    "DecisionMaker",
    "get_decision_maker",
    # Synthesis
    "Synthesizer",
    "get_synthesizer",
]
