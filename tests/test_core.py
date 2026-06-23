"""
Tests for core module.
"""

import pytest


class TestConfig:
    """Tests for configuration."""
    
    def test_config_loads(self):
        """Config singleton loads without error."""
        from config import config
        assert config is not None
    
    def test_config_has_paths(self):
        """Config has path settings."""
        from config import config
        assert config.paths is not None
        assert config.paths.data_dir is not None
    
    def test_config_has_api_settings(self):
        """Config has API settings."""
        from config import config
        assert config.api.ws_port > 0
        assert config.api.rest_port > 0


class TestLogger:
    """Tests for logger."""
    
    def test_logger_exists(self):
        """Logger is available."""
        from logger import log
        assert log is not None
    
    def test_logger_can_log(self):
        """Logger can write messages."""
        from logger import log
        log.info("Test message")


class TestRedact:
    """Tests for PII redaction."""
    
    def test_redact_function_exists(self):
        """Redact function is available."""
        # redact removed
        assert callable(redact)
    
    def test_redact_email(self):
        """Redacts email addresses."""
        # redact removed
        text = "Contact me at user@example.com"
        result = redact(text)
        assert "user@example.com" not in result
        assert "[EMAIL]" in result


class TestAudit:
    """Tests for audit logging."""
    
    def test_audit_exists(self):
        """Audit logger is available."""
        from observability import audit
        assert audit is not None
    
    def test_audit_can_log(self):
        """Audit can log events."""
        from observability import audit
        audit.log_raw("test", "test_action", "test_actor", "completed")


class TestErrors:
    """Tests for error handling."""
    
    def test_retry_policy(self):
        """RetryPolicy works."""
        from helpers import RetryPolicy
        
        policy = RetryPolicy(max_retries=3, base_delay_seconds=1.0)
        
        assert policy.delay_for_attempt(0) == 1.0
        assert policy.delay_for_attempt(1) == 2.0
        assert policy.is_retryable("timeout")
    
    def test_agent_error(self):
        """AgentError works."""
        from helpers import AgentError, ErrorCategory
        
        error = AgentError(
            category=ErrorCategory.CONNECTOR,
            message="Test error",
        )
        assert error.category == "connector"
