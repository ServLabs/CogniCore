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
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import config


def setup_logger() -> logging.Logger:
    """
    Create the agent-wide logger.
    
    Called once at import time.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("cognicore")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    
    # Prevent duplicate handlers on reimport
    if logger.handlers:
        return logger
    
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s.%(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    if config.logging.log_to_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)
    
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


# ══════════════════════════════════════════════════════════════════════════════
# Child Loggers
# ══════════════════════════════════════════════════════════════════════════════

def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger for module-specific logging.
    
    Args:
        name: Logger name (e.g., "recall", "cen", "mml").
        
    Returns:
        Child logger instance.
    """
    return log.getChild(name)
