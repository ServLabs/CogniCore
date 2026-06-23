"""
Helpers

Shared utilities used across CogniCore:
- errors: Error handling and retry policies
- ingestion: Data ingestion pipeline (staging → MML → LFM/SFM)
"""

from helpers.errors import (
    RetryPolicy,
    RETRY_POLICIES,
    FALLBACK_CHAINS,
    AgentError,
    ErrorCategory,
    FallbackExecutor,
    RateLimitError,
    with_retry,
)
from helpers.ingestion import (
    IngestionResult,
    IngestionService,
    get_ingestion_service,
    chunk_text,
    csv_to_text,
    json_to_text,
)

__all__ = [
    # Errors
    "RetryPolicy",
    "RETRY_POLICIES",
    "FALLBACK_CHAINS",
    "AgentError",
    "ErrorCategory",
    "FallbackExecutor",
    "RateLimitError",
    "with_retry",
    # Ingestion
    "IngestionResult",
    "IngestionService",
    "get_ingestion_service",
    "chunk_text",
    "csv_to_text",
    "json_to_text",
]
