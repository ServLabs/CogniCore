"""
Error Handling

Global error handling philosophy. How the agent recovers from failures
without crashing, losing user context, or returning garbage.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from logger import log


# ══════════════════════════════════════════════════════════════════════════════
# Retry Policy
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetryPolicy:
    """
    Retry configuration for a connector or operation type.
    
    Uses exponential backoff with cap.
    """
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    retryable_errors: frozenset[str] = field(default_factory=lambda: frozenset({
        "timeout",
        "rate_limited",
        "server_error",
        "connection_reset",
        "connection_refused",
        "temporary_failure",
    }))
    
    def delay_for_attempt(self, attempt: int) -> float:
        """
        Calculate delay for a retry attempt.
        
        Uses exponential backoff with cap.
        
        Args:
            attempt: Attempt number (0-indexed).
            
        Returns:
            Delay in seconds.
        """
        delay = self.base_delay_seconds * (self.exponential_base ** attempt)
        return min(delay, self.max_delay_seconds)
    
    def is_retryable(self, error_type: str) -> bool:
        """Check if an error type is retryable."""
        return error_type.lower() in self.retryable_errors


# Default policies per connector type
RETRY_POLICIES = {
    "llm": RetryPolicy(max_retries=3, base_delay_seconds=1.0),
    "data": RetryPolicy(max_retries=2, base_delay_seconds=2.0),
    "sandbox": RetryPolicy(max_retries=1, base_delay_seconds=0),  # fail fast
    "memory": RetryPolicy(max_retries=2, base_delay_seconds=0.5),
}


# ══════════════════════════════════════════════════════════════════════════════
# Fallback Chains
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_CHAINS = {
    "llm_call": [
        "retry_same_model",          # retry with backoff
        "switch_to_cheap_model",     # gpt-4o → gpt-4o-mini
        "use_cached_response",       # if similar query was answered before
        "return_partial_response",   # return what we have so far
        "return_error_to_user",      # "I'm unable to process this right now"
    ],
    "data_query": [
        "retry_query",               # retry with backoff
        "use_cached_data",           # last successful query result
        "try_alternative_connector", # Snowflake down → try Databricks
        "return_partial_response",   # answer with whatever data we have
        "return_error_to_user",
    ],
    "memory_recall": [
        "retry_recall",              # retry
        "skip_failed_memory_type",   # SFM failed → still use LFM, AM results
        "broaden_search",            # relax similarity threshold
        "return_without_memory",     # LLM answers from general knowledge
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# Error Categories
# ══════════════════════════════════════════════════════════════════════════════

class ErrorCategory:
    """Error category constants."""
    CONNECTOR = "connector"
    MEMORY = "memory"
    SANDBOX = "sandbox"
    BUDGET = "budget"
    AUTH = "auth"
    UNKNOWN = "unknown"


@dataclass
class AgentError:
    """Structured error with category and recovery info."""
    category: str
    message: str
    recoverable: bool = True
    suggestion: Optional[str] = None
    original_error: Optional[Exception] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "recoverable": self.recoverable,
            "suggestion": self.suggestion,
        }


class RateLimitError(Exception):
    """Raised when Governor rate or budget limits are exceeded."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Retry Decorator
# ══════════════════════════════════════════════════════════════════════════════

def with_retry(
    policy_name: str = "llm",
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator that adds retry logic to an async function.
    
    Args:
        policy_name: Name of retry policy to use.
        on_retry: Optional callback(attempt, error) called before each retry.
        
    Returns:
        Decorated function with retry logic.
    """
    policy = RETRY_POLICIES.get(policy_name, RetryPolicy())
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(policy.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_type = _classify_error(e)
                    
                    if attempt < policy.max_retries and policy.is_retryable(error_type):
                        delay = policy.delay_for_attempt(attempt)
                        log.warning(
                            f"Retry {attempt + 1}/{policy.max_retries} for {func.__name__}: "
                            f"{error_type} - waiting {delay:.1f}s"
                        )
                        
                        if on_retry:
                            on_retry(attempt, e)
                        
                        await asyncio.sleep(delay)
                    else:
                        raise
            
            raise last_error
        
        return wrapper
    return decorator


def _classify_error(error: Exception) -> str:
    """Classify an exception into an error type."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    if "timeout" in error_str or "timeout" in error_type:
        return "timeout"
    if "rate" in error_str or "429" in error_str:
        return "rate_limited"
    if "500" in error_str or "502" in error_str or "503" in error_str:
        return "server_error"
    if "connection" in error_str:
        return "connection_reset"
    
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Fallback Executor
# ══════════════════════════════════════════════════════════════════════════════

class FallbackExecutor:
    """
    Execute a chain of fallback strategies.
    
    Tries each strategy in order until one succeeds.
    """
    
    def __init__(self, chain_name: str):
        """
        Initialize fallback executor.
        
        Args:
            chain_name: Name of fallback chain to use.
        """
        self.chain = FALLBACK_CHAINS.get(chain_name, [])
        self.strategies: dict[str, Callable] = {}
    
    def register(self, name: str, strategy: Callable) -> None:
        """
        Register a fallback strategy.
        
        Args:
            name: Strategy name (must match chain entry).
            strategy: Async callable that implements the strategy.
        """
        self.strategies[name] = strategy
    
    async def execute(self, context: dict[str, Any]) -> Any:
        """
        Execute fallback chain.
        
        Args:
            context: Context dict passed to each strategy.
            
        Returns:
            Result from first successful strategy.
            
        Raises:
            AgentError: If all strategies fail.
        """
        errors = []
        
        for strategy_name in self.chain:
            strategy = self.strategies.get(strategy_name)
            
            if not strategy:
                log.debug(f"Fallback strategy not registered: {strategy_name}")
                continue
            
            try:
                log.debug(f"Trying fallback strategy: {strategy_name}")
                result = await strategy(context)
                log.info(f"Fallback succeeded: {strategy_name}")
                return result
            except Exception as e:
                log.warning(f"Fallback failed: {strategy_name} - {e}")
                errors.append((strategy_name, str(e)))
        
        raise AgentError(
            category=ErrorCategory.UNKNOWN,
            message="All fallback strategies failed",
            recoverable=False,
            suggestion="Please try again later",
        )
