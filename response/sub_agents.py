"""
Sub-Agent Executor

Two sub-agent types:
1. SubAgent — Autonomous task execution with memory recall + reasoning.
2. ToolSubAgent — ReAct-style tool-use loop for answering queries that
   require external data, computation, or multi-step reasoning.
"""

import asyncio
import json
import time
from typing import Any, Optional

from connectors import genai, get_registry
from config import config
from logger import log
from memory import recall
from observability import audit, record_latency, record_count
from prompts import prompts


# ══════════════════════════════════════════════════════════════════════════════
# Sub-Agent
# ══════════════════════════════════════════════════════════════════════════════

class SubAgent:
    """
    A single autonomous execution unit.

    Given a task description and optional skill hint, the sub-agent:
    1. Calls genai to plan its approach
    2. Executes actions (recall, skill, reason)
    3. Returns a result dict
    """

    def __init__(
        self,
        task: str,
        skill: Optional[str] = None,
        context: str = "",
        user_id: Optional[str] = None,
        convo_id: Optional[str] = None,
        timeout_seconds: int = 0,
    ):
        self.task = task
        self.skill = skill
        self.context = context
        self.user_id = user_id
        self.convo_id = convo_id
        self.timeout = timeout_seconds or config.pipeline.sub_agent_timeout_seconds

    async def run(self) -> dict[str, Any]:
        """
        Execute the task autonomously.

        Returns:
            Result dict with task, output, success, error keys.
        """
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(self._execute(), timeout=self.timeout)
            elapsed_ms = (time.perf_counter() - start) * 1000
            asyncio.create_task(record_latency("sub_agent", "execution", elapsed_ms, skill=self.skill or "none"))
            asyncio.create_task(record_count("sub_agent", "success_count"))
            return result
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            asyncio.create_task(record_latency("sub_agent", "execution", elapsed_ms, skill=self.skill or "none"))
            asyncio.create_task(record_count("sub_agent", "timeout_count"))
            return {
                "task": self.task,
                "output": None,
                "success": False,
                "error": f"Task timed out after {self.timeout}s",
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            asyncio.create_task(record_latency("sub_agent", "execution", elapsed_ms, skill=self.skill or "none"))
            asyncio.create_task(record_count("sub_agent", "error_count"))
            return {
                "task": self.task,
                "output": None,
                "success": False,
                "error": str(e),
            }

    async def _execute(self) -> dict[str, Any]:
        """Core execution: call genai, process actions, return result."""
        messages = prompts.get_messages(
            "response/sub_agent.md",
            task=self.task,
            skill=self.skill or "none",
            context=self.context or "(no additional context)",
        )

        raw = await genai.ask({
            "model": config.pipeline.default_model_tier,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        })

        data = self._parse(raw)

        # Execute actions if the agent planned any recall steps
        actions = data.get("actions", [])
        recall_results = []

        for action in actions:
            if action.get("type") == "recall" and action.get("query"):
                recall_result = await recall(
                    query=action["query"],
                    user_id=self.user_id,
                    convo_id=self.convo_id,
                    top_k=5,
                )
                if recall_result.has_results:
                    recall_results.append(recall_result.to_context())

        # If recall returned new info, do a second genai pass with enriched context
        if recall_results:
            enriched_context = f"{self.context}\n\nRecalled:\n" + "\n".join(recall_results)
            messages = prompts.get_messages(
                "response/sub_agent.md",
                task=self.task,
                skill=self.skill or "none",
                context=enriched_context,
            )
            raw = await genai.ask({
                "model": config.pipeline.default_model_tier,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            })
            data = self._parse(raw)

        return {
            "task": self.task,
            "skill": self.skill,
            "output": data.get("result", ""),
            "success": data.get("success", True),
            "error": data.get("error"),
        }

    def _parse(self, raw: str) -> dict[str, Any]:
        """Parse genai JSON response."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)


# ══════════════════════════════════════════════════════════════════════════════
# Tool Sub-Agent (ReAct Loop)
# ══════════════════════════════════════════════════════════════════════════════

class ToolSubAgent:
    """
    ReAct-style tool-use sub-agent.
    
    Loop: Think → Act (call tool) → Observe → repeat until answer found.
    
    Uses the connector registry to discover available tools at runtime
    and executes them via registry.execute_tool().
    """
    
    MAX_ITERATIONS = 5
    
    def __init__(
        self,
        query: str,
        context: str = "",
        max_iterations: int = 0,
        timeout_seconds: int = 0,
    ):
        self.query = query
        self.context = context
        self.max_iterations = max_iterations or self.MAX_ITERATIONS
        self.timeout = timeout_seconds or config.pipeline.sub_agent_timeout_seconds
        self._observations: list[dict[str, Any]] = []
    
    async def run(self) -> dict[str, Any]:
        """
        Execute the ReAct loop.
        
        Returns:
            Dict with: query, answer, success, iterations, tool_calls, error.
        """
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(self._loop(), timeout=self.timeout)
            elapsed_ms = (time.perf_counter() - start) * 1000
            asyncio.create_task(record_latency("tool_agent", "loop", elapsed_ms))
            asyncio.create_task(record_count("tool_agent", "success_count" if result.get("success") else "partial_count"))
            return result
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            asyncio.create_task(record_latency("tool_agent", "loop", elapsed_ms))
            asyncio.create_task(record_count("tool_agent", "timeout_count"))
            return {
                "query": self.query,
                "answer": self._best_partial_answer(),
                "success": False,
                "iterations": len(self._observations),
                "tool_calls": [o["tool"] for o in self._observations],
                "error": f"Timed out after {self.timeout}s",
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            asyncio.create_task(record_latency("tool_agent", "loop", elapsed_ms))
            asyncio.create_task(record_count("tool_agent", "error_count"))
            log.error("ToolSubAgent failed: %s", e, exc_info=True)
            return {
                "query": self.query,
                "answer": None,
                "success": False,
                "iterations": len(self._observations),
                "tool_calls": [o["tool"] for o in self._observations],
                "error": str(e),
            }
    
    async def _loop(self) -> dict[str, Any]:
        """Core ReAct loop."""
        registry = get_registry()
        tool_schemas = registry.get_tool_schemas()
        
        # Format tool schemas for prompt
        tools_text = json.dumps(tool_schemas, indent=2)
        
        for iteration in range(self.max_iterations):
            # Build observations context
            obs_text = json.dumps(self._observations, indent=2) if self._observations else "None yet"
            
            messages = prompts.get_messages(
                "response/tool_use.md",
                tools=tools_text,
                max_iterations=str(self.max_iterations),
                query=self.query,
                context=self.context or "(no additional context)",
                observations=obs_text,
            )
            
            raw = await genai.ask({
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })
            
            step = self._parse_step(raw)
            
            # Check if agent returned a final answer
            if "answer" in step:
                audit.log_raw(
                    "agent", "tool_use", "sub_agent", "completed",
                    details={
                        "iterations": iteration + 1,
                        "tools_used": [o["tool"] for o in self._observations],
                    },
                )
                return {
                    "query": self.query,
                    "answer": step["answer"],
                    "success": not step.get("incomplete", False),
                    "iterations": iteration + 1,
                    "tool_calls": [o["tool"] for o in self._observations],
                    "error": None,
                }
            
            # Execute the tool action
            action = step.get("action", {})
            tool_name = action.get("tool", "")
            tool_params = action.get("params", {})
            
            log.debug(
                "ToolSubAgent iteration %d: calling %s with %s",
                iteration + 1, tool_name, tool_params,
            )
            
            start = time.monotonic()
            try:
                tool_result = await registry.execute_tool(tool_name, tool_params)
            except KeyError:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            except Exception as e:
                tool_result = {"error": f"Tool execution failed: {e}"}
            elapsed_ms = (time.monotonic() - start) * 1000
            
            # Record observation
            self._observations.append({
                "iteration": iteration + 1,
                "thought": step.get("thought", ""),
                "tool": tool_name,
                "params": tool_params,
                "result": tool_result,
                "duration_ms": round(elapsed_ms, 1),
            })
            
            audit.log_raw(
                "agent", "tool_call", "sub_agent", "completed",
                target=tool_name,
                duration_ms=elapsed_ms,
                details={"params": tool_params, "success": "error" not in tool_result},
            )
        
        # Exhausted iterations — ask for final answer with all observations
        return {
            "query": self.query,
            "answer": self._best_partial_answer(),
            "success": False,
            "iterations": self.max_iterations,
            "tool_calls": [o["tool"] for o in self._observations],
            "error": f"Max iterations ({self.max_iterations}) reached",
        }
    
    def _parse_step(self, raw: str) -> dict[str, Any]:
        """Parse a single ReAct step from genai response."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If parsing fails, treat the raw text as a final answer
            return {"answer": text, "thought": "Failed to parse JSON, returning raw"}
    
    def _best_partial_answer(self) -> Optional[str]:
        """Extract the best partial answer from observations so far."""
        if not self._observations:
            return None
        # Return the last successful tool result as context
        for obs in reversed(self._observations):
            result = obs.get("result", {})
            if "result" in result:
                return str(result["result"])
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Tool Use Entry Point
# ══════════════════════════════════════════════════════════════════════════════

async def execute_with_tools(
    query: str,
    context: str = "",
    max_iterations: int = 0,
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    """
    Execute a query using the tool sub-agent.
    
    This is the main entry point for tool-assisted problem solving.
    Called by the response pipeline when the thinking layer determines
    tools are needed.
    
    Args:
        query: The user's question or task requiring tools.
        context: Additional context (memory recall results, conversation).
        max_iterations: Max ReAct loop iterations (0 = default).
        timeout_seconds: Timeout (0 = default from config).
        
    Returns:
        Dict with answer, success, iterations, tool_calls.
    """
    agent = ToolSubAgent(
        query=query,
        context=context,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
    )
    return await agent.run()


# ══════════════════════════════════════════════════════════════════════════════
# Executor
# ══════════════════════════════════════════════════════════════════════════════

async def execute_tasks(
    tasks: list[dict[str, Any]],
    parallel: bool = False,
    context: str = "",
    user_id: Optional[str] = None,
    convo_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Execute a list of tasks via sub-agents.

    Args:
        tasks: List of TaskSpec-like dicts with task, skill, timeout_seconds.
        parallel: Whether to run tasks concurrently.
        context: Shared context for all sub-agents.
        user_id: User ID for memory access.
        convo_id: Conversation ID for memory access.

    Returns:
        List of result dicts.
    """
    agents = [
        SubAgent(
            task=t.get("task", ""),
            skill=t.get("skill"),
            context=context,
            user_id=user_id,
            convo_id=convo_id,
            timeout_seconds=t.get("timeout_seconds", 0),
        )
        for t in tasks
    ]

    if parallel:
        results = await asyncio.gather(
            *(agent.run() for agent in agents),
            return_exceptions=True,
        )
        return [
            r if isinstance(r, dict) else {"task": a.task, "output": None, "success": False, "error": str(r)}
            for a, r in zip(agents, results)
        ]
    else:
        results = []
        for agent in agents:
            results.append(await agent.run())
        return results