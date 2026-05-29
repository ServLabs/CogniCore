"""
AI Connectors

Interface to AI models:
- LLM: OpenAI, Anthropic, local endpoints
- Embedder: sentence-transformers, OpenAI embeddings
- NLI: Natural Language Inference (contradiction detection)
- Reranker: Cross-encoder reranking
"""

from connectors.ai.llm import LLMConnector
from connectors.ai.embedder import EmbeddingConnector
from connectors.ai.nli import NLIConnector

__all__ = [
    "LLMConnector",
    "EmbeddingConnector",
    "NLIConnector",
]
