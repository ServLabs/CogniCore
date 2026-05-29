"""
Audit Layer

Universal logging fabric. Every action in the system is recorded.
Append-only, queryable, zero-friction.

Audit is not a feature of any module; it is the air that every module breathes.
"""

from audit.audit import (
    AuditEvent,
    AuditLogger,
    audit,
    audited,
)

__all__ = [
    "AuditEvent",
    "AuditLogger",
    "audit",
    "audited",
]
