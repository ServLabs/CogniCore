"""
AI Connectors

Tool-capable AI connectors:
- LLM: OpenAI, Anthropic, local endpoints (exposed as generate_text tool)
"""

from connectors.ai.llm import LLMConnector

__all__ = [
    "LLMConnector",
]
