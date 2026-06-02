"""
Core Module

Foundational utilities used across all CogniCore modules:
- config: Root configuration singleton
- logger: Standard Python logging with rotation
- redact: PII redaction utilities
- audit: Universal audit logging
- errors: Error handling and retry policies
- migrations: Schema versioning
"""

from core.config import config, AgentConfig, AgentIdentity
from core.logger import log, setup_logger
from core.redact import redact, redact_dict, Redactor
from core.audit import AuditEvent, AuditLogger, audit, audited
from core.errors import (
    RetryPolicy,
    RETRY_POLICIES,
    FALLBACK_CHAINS,
    AgentError,
    ErrorCategory,
    FallbackExecutor,
    with_retry,
)
from core.migrations import migrate, get_current_version, MIGRATIONS
from core.faiss_migrations import (
    check_index_compatibility,
    rebuild_index,
    get_index_metadata,
    save_index_metadata,
)

__all__ = [
    # Config
    "config",
    "AgentConfig",
    "AgentIdentity",
    # Logger
    "log",
    "setup_logger",
    # Redact
    "redact",
    "redact_dict",
    "Redactor",
    # Audit
    "AuditEvent",
    "AuditLogger",
    "audit",
    "audited",
    # Errors
    "RetryPolicy",
    "RETRY_POLICIES",
    "FALLBACK_CHAINS",
    "AgentError",
    "ErrorCategory",
    "FallbackExecutor",
    "with_retry",
    # Migrations
    "migrate",
    "get_current_version",
    "MIGRATIONS",
    "check_index_compatibility",
    "rebuild_index",
    "get_index_metadata",
    "save_index_metadata",
]
