"""
Logger

Standard Python logging for developers and ops.
DEBUG, INFO, WARNING, ERROR with rotating file handler.

The three observability layers:
- Logger: DEBUG/INFO/WARN/ERROR → developers, ops → console + logs/agent.log
- Stream Events: Pipeline steps → end users → WebSocket "Behind the scenes"
- Audit: Every action with status/duration → compliance, analytics → audit/{date}.jsonl
"""

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# PII Redaction Filter
# ══════════════════════════════════════════════════════════════════════════════

# Patterns: (name, compiled_regex, replacement)
_REDACTION_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    ("phone", re.compile(r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    ("credit_card", re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CARD]"),
    ("api_key", re.compile(r"\b(sk|pk|api|key|token|secret)[-_]?[a-zA-Z0-9]{20,}\b", re.IGNORECASE), "[API_KEY]"),
    ("ip_address", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP]"),
    ("bearer_token", re.compile(r"Bearer\s+[a-zA-Z0-9._\-]+", re.IGNORECASE), "Bearer [REDACTED]"),
]


class PIIRedactionFilter(logging.Filter):
    """
    Logging filter that redacts PII patterns from log messages.
    
    Applied before any handler emits the record, so all outputs
    (console, file, external) are clean.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact PII from the message and args, always allow the record through."""
        if record.msg and isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._redact(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._redact(str(a)) if isinstance(a, str) else a for a in record.args)
        
        return True

    def _redact(self, text: str) -> str:
        """Apply all redaction patterns to text."""
        for _name, pattern, replacement in _REDACTION_PATTERNS:
            text = pattern.sub(replacement, text)
        return text


def setup_logger() -> logging.Logger:
    """
    Create the agent-wide logger.
    
    Called once at import time.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(config.logging.name)
    logger.setLevel(config.logging.level.upper())
    
    # Prevent duplicate handlers on reimport
    if logger.handlers:
        return logger
    
    fmt = logging.Formatter(
        config.logging.format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    if config.logging.log_to_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

    if config.logging.redact_pii:
        logger.addFilter(PIIRedactionFilter())
    
    # Rotating file handler
    log_dir = config.paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_dir / "agent.log",
        maxBytes=config.logging.max_file_size_mb * 1_000_000,
        backupCount=config.logging.backup_count,
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

log = setup_logger()
