"""
PII Redaction

Scrub sensitive data before it hits logs, audit, or stream events.
Regex-based pattern matching for SSN, credit cards, phone numbers, etc.

Applied at three points:
1. Logger — log.info(redact(message))
2. Audit — audit.log() passes details through redaction
3. Stream Events — stream.emit() redacts text before sending
"""

import re
from typing import Any

from config import config


class Redactor:
    """
    PII redaction engine.
    
    Regex-based pattern matching for common PII types.
    """
    
    # Standard PII patterns
    PATTERNS = {
        "ssn": {
            "regex": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "label": "[SSN]",
            "enabled_key": "redact_ssn",
        },
        "credit_card": {
            "regex": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
            "label": "[CREDIT_CARD]",
            "enabled_key": "redact_credit_card",
        },
        "phone": {
            "regex": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "label": "[PHONE]",
            "enabled_key": "redact_phone",
        },
        "email": {
            "regex": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "label": "[EMAIL]",
            "enabled_key": "redact_email",
        },
        "account_number": {
            "regex": re.compile(r"\b(?:acct?|account)[#:\s]*\d{6,12}\b", re.IGNORECASE),
            "label": "[ACCOUNT]",
            "enabled_key": "redact_account_numbers",
        },
    }
    
    def __init__(self):
        """Initialize redactor with active patterns from config."""
        self._active_patterns = []
        
        if config.redaction.enabled:
            for name, pattern in self.PATTERNS.items():
                enabled_key = pattern["enabled_key"]
                if getattr(config.redaction, enabled_key, True):
                    self._active_patterns.append(pattern)
    
    def redact(self, text: str) -> str:
        """
        Redact all PII from text.
        
        Args:
            text: Input text.
            
        Returns:
            Cleaned string with PII replaced.
        """
        if not config.redaction.enabled or not text:
            return text
        
        for pattern in self._active_patterns:
            text = pattern["regex"].sub(pattern["label"], text)
        
        return text
    
    def redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Deep-redact all string values in a dict.
        
        Args:
            data: Input dictionary.
            
        Returns:
            Dictionary with all string values redacted.
        """
        if not config.redaction.enabled:
            return data
        
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, str):
                cleaned[key] = self.redact(value)
            elif isinstance(value, dict):
                cleaned[key] = self.redact_dict(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    self.redact(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                cleaned[key] = value
        
        return cleaned
    
    def is_enabled(self) -> bool:
        """Check if redaction is enabled."""
        return config.redaction.enabled


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_redactor = Redactor()

redact = _redactor.redact
redact_dict = _redactor.redact_dict
