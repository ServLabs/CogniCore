"""
Deep Pipeline

Full reasoning pipeline for all non-trivial questions.
Orchestrates 6 subsystems to produce high-quality responses.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from core import config
from response.gate import Gate, GateResult, GateDecision, get_gate
from response.thinking import Thinker, ThoughtPlan, get_thinker
from response.decision import DecisionMaker, ExecutionPlan, ExecutionStrategy, get_decision_maker
from response.synthesis import Synthesizer, SynthesisInput, SynthesisOutput, get_synthesizer


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Result from the response pipeline."""
    response: str
    gate_result: Optional[GateResult] = None
    thought_plan: Optional[ThoughtPlan] = None
    execution_plan: Optional[ExecutionPlan] = None
    synthesis_output: Optional[SynthesisOutput] = None
    processing_time_ms: float = 0.0
    used_deep_pipeline: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "gate_result": self.gate_result.to_dict() if self.gate_result else None,
            "thought_plan": self.thought_plan.to_dict() if self.thought_plan else None,
            "execution_plan": self.execution_plan.to_dict() if self.execution_plan else None,
            "synthesis_output": self.synthesis_output.to_dict() if self.synthesis_output else None,
            "processing_time_ms": self.processing_time_ms,
            "used_deep_pipeline": self.used_deep_pipeline,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Deep Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class DeepPipeline:
    """
    Full reasoning pipeline for complex queries.
    
    Orchestrates:
    1. Thinking - Query decomposition and planning
    2. Memory Recall - Context retrieval via MML
    3. Decision - Execution strategy selection
    4. Skill Execution - Run scripts, queries, APIs
    5. Creativity - Novel insights and alternatives
    6. Prediction - Forecasting and risk assessment
    7. Synthesis - Combine into final response
    
    Attributes:
        gate: Gate for trivial query filtering.
        thinker: Thinking subsystem.
        decision_maker: Decision making subsystem.
        synthesizer: Response synthesis subsystem.
        llm: LLM callable for reasoning.
    """
    
    def __init__(
        self,
        gate: Optional[Gate] = None,
        thinker: Optional[Thinker] = None,
        decision_maker: Optional[DecisionMaker] = None,
        synthesizer: Optional[Synthesizer] = None,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        """
        Initialize Deep Pipeline.
        
        Args:
            gate: Gate for filtering.
            thinker: Thinking subsystem.
            decision_maker: Decision making subsystem.
            synthesizer: Response synthesis.
            llm: LLM callable.
        """
        self.gate = gate or get_gate()
        self.thinker = thinker or get_thinker()
        self.decision_maker = decision_maker or get_decision_maker()
        self.synthesizer = synthesizer or get_synthesizer()
        self.llm = llm
    
    async def process(
        self,
        message: str,
        user_id: Optional[str] = None,
        convo_id: Optional[str] = None,
        context: Optional[str] = None,
        force_deep: bool = False,
    ) -> PipelineResult:
        """
        Process a user message through the pipeline.
        
        Args:
            message: User message.
            user_id: User ID for context.
            convo_id: Conversation ID for context.
            context: Optional conversation context.
            force_deep: Force deep pipeline (skip gate).
            
        Returns:
            PipelineResult with response.
        """
        start_time = time.perf_counter()
        
        # Step 0: Gate check (unless forced deep)
        if not force_deep:
            gate_result = self.gate.classify(message)
            
            if gate_result.is_gate:
                # Handle at gate level
                return PipelineResult(
                    response=gate_result.response or "Hello!",
                    gate_result=gate_result,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                    used_deep_pipeline=False,
                )
        else:
            gate_result = None
        
        # Step 1: Thinking - Understand and plan
        thought_plan = await self.thinker.think(message, context)
        
        # Step 2: Memory Recall - Get context from MML
        recall_context = await self._recall_context(message, thought_plan, user_id, convo_id)
        
        # Step 3: Decision - Choose execution strategy
        execution_plan = await self.decision_maker.decide(
            query=message,
            thought_plan=thought_plan,
            recall_context=recall_context,
        )
        
        # Step 4: Skill Execution (if needed)
        skill_results = []
        if execution_plan.strategy == ExecutionStrategy.SKILL_EXECUTION:
            skill_results = await self._execute_skills(execution_plan)
        
        # Step 5: Creativity (if needed)
        creativity_output = None
        if execution_plan.needs_creativity:
            creativity_output = await self._generate_creativity(message, recall_context)
        
        # Step 6: Prediction (if needed)
        prediction_output = None
        if execution_plan.needs_prediction:
            prediction_output = await self._generate_prediction(message, recall_context)
        
        # Step 7: Synthesis - Combine into response
        synthesis_input = SynthesisInput(
            query=message,
            thought_plan=thought_plan,
            execution_plan=execution_plan,
            recall_context=recall_context,
            skill_results=skill_results,
            creativity_output=creativity_output,
            prediction_output=prediction_output,
        )
        
        synthesis_output = await self.synthesizer.synthesize(synthesis_input)
        
        # Record metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        await self._record_metrics(elapsed_ms, thought_plan, execution_plan)
        
        return PipelineResult(
            response=synthesis_output.response,
            gate_result=gate_result,
            thought_plan=thought_plan,
            execution_plan=execution_plan,
            synthesis_output=synthesis_output,
            processing_time_ms=elapsed_ms,
            used_deep_pipeline=True,
        )
    
    async def _recall_context(
        self,
        query: str,
        thought_plan: ThoughtPlan,
        user_id: Optional[str],
        convo_id: Optional[str],
    ) -> str:
        """Recall context from Memory Management Layer."""
        from memory.management import recall
        
        result = await recall(
            query=query,
            user_id=user_id,
            convo_id=convo_id,
            top_k=10,
        )
        
        return result.to_context() if result.has_results else ""
    
    async def _execute_skills(
        self,
        execution_plan: ExecutionPlan,
    ) -> list[dict[str, Any]]:
        """Execute skills from the execution plan."""
        results = []
        
        for skill_call in execution_plan.skill_calls:
            # TODO: Implement actual skill execution via MM procedures
            # For now, placeholder
            results.append({
                "skill_name": skill_call.skill_name,
                "output": f"[Skill {skill_call.skill_name} executed]",
                "success": True,
            })
        
        return results
    
    async def _generate_creativity(
        self,
        query: str,
        context: str,
    ) -> Optional[str]:
        """Generate creative insights."""
        if not self.llm:
            return None
        
        prompt = f"""Generate creative insights and alternatives for this query.

Query: {query}
Context: {context}

Provide:
1. Novel perspectives
2. Alternative approaches
3. Unexpected connections
"""
        
        return await self.llm(prompt)
    
    async def _generate_prediction(
        self,
        query: str,
        context: str,
    ) -> Optional[str]:
        """Generate predictions and forecasts."""
        if not self.llm:
            return None
        
        prompt = f"""Generate predictions and forecasts for this query.

Query: {query}
Context: {context}

Provide:
1. Expected outcomes
2. Risk assessment
3. Confidence levels
"""
        
        return await self.llm(prompt)
    
    async def _record_metrics(
        self,
        elapsed_ms: float,
        thought_plan: ThoughtPlan,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Record pipeline metrics to analytics."""
        from observability import record_latency, record_count
        
        await record_latency("response", "pipeline", elapsed_ms)
        await record_count("response", f"complexity_{thought_plan.complexity.value}")
        await record_count("response", f"strategy_{execution_plan.strategy.value}")


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_pipeline_instance: Optional[DeepPipeline] = None


def get_pipeline() -> DeepPipeline:
    """Get the singleton DeepPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DeepPipeline()
    return _pipeline_instance


# ── Convenience Function ──

async def process_message(
    message: str,
    user_id: Optional[str] = None,
    convo_id: Optional[str] = None,
) -> PipelineResult:
    """Process a user message through the response pipeline."""
    return await get_pipeline().process(message, user_id, convo_id)
