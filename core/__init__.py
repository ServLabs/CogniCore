"""
Core Module

Foundational utilities used across all CogniCore modules:
- config: Root configuration singleton
- logger: Standard Python logging with rotation
- redact: PII redaction utilities
"""

from core.config import config, AgentConfig, AgentIdentity
from core.logger import log, setup_logger
from core.redact import redact, redact_dict, Redactor

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
]
