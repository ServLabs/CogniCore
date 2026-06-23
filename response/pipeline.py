"""
Deep Pipeline

Orchestrates the response flow:
  Gate → [Think ‖ Recall] → Decision → (Clarify | Execute | Direct) → Synthesis
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from config import config
from memory import recall, get_mml
from observability import record_latency, record_count
from observability.tracing import get_tracer
from response.channel import PipelineChannel
from response.gate import GateResult, get_gate
from response.thinking import ThoughtPlan, get_thinker
from response.decision import ExecutionPlan, ExecutionStrategy, get_decision_maker
from response.sub_agents import execute_tasks, execute_with_tools
from response.synthesis import SynthesisInput, SynthesisOutput, get_synthesizer


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

    Flow:
    1. Gate              — classify trivial vs domain (async, AI-powered)
    2. Think ‖ Recall    — decompose query AND retrieve context in parallel
    3. Decision          — choose: direct_answer | clarify | execute
       ├─ CLARIFY        — ask user via channel (up to 3 rounds), then re-decide
       └─ EXECUTE        — dispatch tasks to sub-agents
    4. Synthesis         — combine everything into a response
    """

    def __init__(self):
        self.gate = get_gate()
        self.thinker = get_thinker()
        self.decision_maker = get_decision_maker()
        self.synthesizer = get_synthesizer()

    async def process(
        self,
        message: str,
        channel: PipelineChannel,
        user_id: Optional[str] = None,
        convo_id: Optional[str] = None,
        context: Optional[str] = None,
        force_deep: bool = False,
    ) -> PipelineResult:
        """
        Process a user message through the pipeline.

        Args:
            message: User message.
            channel: Two-way communication channel to the handler.
            user_id: User ID for context.
            convo_id: Conversation ID for context.
            context: Optional conversation context.
            force_deep: Force deep pipeline (skip gate).

        Returns:
            PipelineResult with response.
        """
        tracer = get_tracer()

        async with tracer.start_trace(
            "pipeline",
            user_id=user_id,
            conversation_id=convo_id,
        ) as trace:
            trace.set_metadata(message_length=len(message), force_deep=force_deep)
            start_time = time.perf_counter()

            # Step 1: Gate check (unless forced deep)
            if not force_deep:
                async with trace.span("gate", component="response", target="gate_classifier") as s:
                    s.set_input(message[:200])
                    gate_result = await self.gate.classify(message)
                    s.set_output({"is_gate": gate_result.is_gate, "category": getattr(gate_result, 'category', '')})

                if gate_result.is_gate:
                    trace.set_metadata(strategy="gate", deep=False)
                    return PipelineResult(
                        response=gate_result.response or "Hello!",
                        gate_result=gate_result,
                        processing_time_ms=(time.perf_counter() - start_time) * 1000,
                        used_deep_pipeline=False,
                    )
            else:
                gate_result = None

            await channel.send_status("thinking", "Analyzing your request...")

            # Step 2: Think and Recall in parallel
            async with trace.span("think", component="response", target="thinker") as s:
                s.set_input(message[:200])
                thought_plan = await self.thinker.think(message, context)
                s.set_output({"steps": len(thought_plan.steps) if hasattr(thought_plan, 'steps') else 0})

            async with trace.span("recall", component="memory", target="recall_engine") as s:
                s.set_input(message[:200])
                recall_result = await recall(query=message, user_id=user_id, convo_id=convo_id, top_k=10)
                recall_context = recall_result.to_context() if recall_result.has_results else ""
                s.set_output({
                    "results_found": recall_result.total_results if hasattr(recall_result, 'total_results') else 0,
                    "has_results": recall_result.has_results,
                    "context_length": len(recall_context),
                })

            # Stage insight for background consolidation (fire-and-forget)
            if thought_plan.new_insight and user_id:
                asyncio.create_task(
                    get_mml().stage_insight(
                        insight=thought_plan.new_insight,
                        user_id=user_id,
                        convo_id=convo_id or "",
                    )
                )

            # Step 3: Decision loop (with up to N clarification rounds)
            clarification_history = ""
            execution_plan = None

            async with trace.span("decision", component="response", target="decision_maker") as s:
                for _round in range(config.pipeline.max_clarification_rounds + 1):
                    await channel.send_status("decision", "Deciding how to handle this...")

                    execution_plan = await self.decision_maker.decide(
                        query=message,
                        thought_plan=thought_plan,
                        recall_context=recall_context,
                        clarification_history=clarification_history,
                    )

                    if execution_plan.strategy != ExecutionStrategy.CLARIFY:
                        break

                    # Ask clarifying questions via channel
                    reply = await channel.ask_user(execution_plan.questions)
                    clarification_history += (
                        f"Questions: {'; '.join(execution_plan.questions)}\n"
                        f"User reply: {reply}\n\n"
                    )
                    context = f"{context}\n{reply}" if context else reply
                else:
                    execution_plan = ExecutionPlan.direct_answer("Max clarification rounds reached")

                s.set_output({
                    "strategy": execution_plan.strategy.value,
                    "clarification_rounds": _round if '_round' in dir() else 0,
                    "task_count": len(execution_plan.tasks) if hasattr(execution_plan, 'tasks') else 0,
                })

            # Step 4: Execute tasks if needed
            task_results: list[dict[str, Any]] = []

            if execution_plan.strategy == ExecutionStrategy.EXECUTE:
                async with trace.span("execution", component="response", target="sub_agents") as s:
                    await channel.send_status("execution", "Executing tasks...")
                    task_results = await execute_tasks(
                        tasks=[t.to_dict() for t in execution_plan.tasks],
                        parallel=execution_plan.parallel,
                        context=recall_context,
                        user_id=user_id,
                        convo_id=convo_id,
                    )
                    s.set_output({
                        "tasks_executed": len(task_results),
                        "tasks_succeeded": sum(1 for t in task_results if t.get("success")),
                    })

            elif execution_plan.strategy == ExecutionStrategy.TOOL_USE:
                async with trace.span("tool_use", component="response", target="tool_sub_agent") as s:
                    await channel.send_status("execution", "Using tools to find the answer...")
                    tool_result = await execute_with_tools(
                        query=message,
                        context=recall_context,
                    )
                    task_results = [{
                        "task": "tool_use",
                        "output": tool_result.get("answer"),
                        "success": tool_result.get("success", False),
                        "error": tool_result.get("error"),
                        "tool_calls": tool_result.get("tool_calls", []),
                    }]
                    s.set_output({
                        "iterations": tool_result.get("iterations", 0),
                        "tools_used": tool_result.get("tool_calls", []),
                        "success": tool_result.get("success", False),
                    })

            # Step 5: Synthesis
            async with trace.span("synthesis", component="response", target="synthesizer") as s:
                await channel.send_status("synthesis", "Composing response...")

                synthesis_input = SynthesisInput(
                    query=message,
                    thought_plan=thought_plan,
                    execution_plan=execution_plan,
                    recall_context=recall_context,
                    skill_results=task_results,
                )

                synthesis_output = await self.synthesizer.synthesize(synthesis_input)
                s.set_output({"response_length": len(synthesis_output.response)})

            # Record metrics (fire-and-forget)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            trace.set_metadata(
                strategy=execution_plan.strategy.value,
                deep=True,
                total_duration_ms=elapsed_ms,
            )

            asyncio.create_task(record_latency("response", "pipeline", elapsed_ms))
            asyncio.create_task(record_count("response", f"strategy_{execution_plan.strategy.value}"))

            return PipelineResult(
                response=synthesis_output.response,
                gate_result=gate_result,
                thought_plan=thought_plan,
                execution_plan=execution_plan,
                synthesis_output=synthesis_output,
                processing_time_ms=elapsed_ms,
                used_deep_pipeline=True,
            )

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



