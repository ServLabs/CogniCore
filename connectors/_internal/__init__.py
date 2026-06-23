"""  
Internal Connectors

Non-tool infrastructure used by the agent internally.
These are NOT exposed as runtime tools — they ARE the agent's brain.

- genai: Unified LLM gateway (governor-gated, audited)
- embedder: Embedding model connector (local + API)
- nli: Natural Language Inference (contradiction detection)
"""

from connectors._internal.genai import genai, GenAI
from connectors._internal.embedder import EmbeddingConnector
from connectors._internal.nli import NLIConnector

__all__ = [
    "genai",
    "GenAI",
    "EmbeddingConnector",
    "NLIConnector",
]
