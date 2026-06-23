"""
Response Layer

The agent's processing pipeline — receives user input, routes it,
reasons over it, executes tasks via sub-agents, and produces a response.

Internal modules (not re-exported):
- gate.py        — AI-powered trivial message filter (greetings, chitchat)
- thinking.py    — Query decomposition and planning via AI
- decision.py    — AI-powered strategy selection (direct | clarify | execute | tool_use)
- sub_agents.py  — Autonomous task executors + ReAct tool-use agent
- synthesis.py   — AI-powered response composition
- channel.py     — Two-way communication protocol (clarification support)

pipeline.py orchestrates the above:
  Gate → [Think ‖ Recall] → Decision → (Clarify | Execute | ToolUse | Direct) → Synthesis

Downstream dependencies (what response/ calls):
- core.config                          — pipeline settings
- connectors.internal.genai             — AI calls (internal brain)
- connectors.registry                  — tool discovery and execution
- memory.management.recall             — context retrieval
- observability.record_latency/count   — pipeline metrics
- prompts (prompts/response/)          — all AI prompts
"""

from response.pipeline import get_pipeline
from response.sub_agents import execute_with_tools
from response.channel import PipelineChannel, StreamChannel, NullChannel

__all__ = [
    "get_pipeline",
    "execute_with_tools",
    "PipelineChannel",
    "StreamChannel",
    "NullChannel",
]
