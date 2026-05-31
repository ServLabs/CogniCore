"""
Tests for core module (config, logger, redact).
"""

import pytest


class TestConfig:
    """Tests for configuration."""
    
    def test_config_loads(self):
        """Config singleton loads without error."""
        from core import config
        assert config is not None
    
    def test_config_has_paths(self):
        """Config has path settings."""
        from core import config
        assert config.paths is not None
        assert config.paths.data_dir is not None
    
    def test_config_has_api_settings(self):
        """Config has API settings."""
        from core import config
        assert config.api.ws_port > 0
        assert config.api.rest_port > 0


class TestLogger:
    """Tests for logger."""
    
    def test_logger_exists(self):
        """Logger is available."""
        from core import log
        assert log is not None
    
    def test_logger_can_log(self):
        """Logger can write messages."""
        from core import log
        log.info("Test message")  # Should not raise


class TestRedact:
    """Tests for PII redaction."""
    
    def test_redact_function_exists(self):
        """Redact function is available."""
        from core import redact
        assert callable(redact)
    
    def test_redact_email(self):
        """Redacts email addresses."""
        from core import redact
        text = "Contact me at user@example.com"
        result = redact(text)
        assert "user@example.com" not in result
        assert "[EMAIL]" in result
    
    def test_redact_preserves_normal_text(self):
        """Normal text is preserved."""
        from core import redact
        text = "Hello world"
        result = redact(text)
        assert result == text
