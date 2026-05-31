"""
Governor

Resource limits and safety controls for the agent.
Enforces budgets, rate limits, and safety boundaries.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from core import config


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceLimits:
    """Resource limits for the agent."""
    # LLM limits
    max_llm_calls_per_hour: int = 1000
    max_llm_cost_per_day_usd: float = 50.0
    
    # Request limits
    max_concurrent_requests: int = 5
    max_requests_per_minute: int = 60
    
    # Memory limits
    max_redis_memory_mb: int = 512
    max_faiss_vectors: int = 1_000_000
    
    # Sub-agent limits
    max_sub_agents: int = 10
    max_sub_agent_depth: int = 3  # No sub-agent spawning sub-agents beyond this
    
    # Timeout limits
    request_timeout_seconds: int = 60
    skill_timeout_seconds: int = 30
    sub_agent_timeout_seconds: int = 120
    
    # Section 15: Ingestion limits
    ingestion_requests_per_minute: int = 10
    ingestion_requests_per_hour: int = 100
    ingestion_max_text_size_kb: int = 1024      # 1MB max per text
    ingestion_max_file_size_mb: int = 10        # 10MB max per file
    ingestion_max_facts_per_request: int = 100
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "max_llm_calls_per_hour": self.max_llm_calls_per_hour,
            "max_llm_cost_per_day_usd": self.max_llm_cost_per_day_usd,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_redis_memory_mb": self.max_redis_memory_mb,
            "max_faiss_vectors": self.max_faiss_vectors,
            "max_sub_agents": self.max_sub_agents,
            "max_sub_agent_depth": self.max_sub_agent_depth,
            "request_timeout_seconds": self.request_timeout_seconds,
            "skill_timeout_seconds": self.skill_timeout_seconds,
            "sub_agent_timeout_seconds": self.sub_agent_timeout_seconds,
        }


@dataclass
class UsageStats:
    """Current usage statistics."""
    llm_calls_this_hour: int = 0
    llm_cost_today_usd: float = 0.0
    requests_this_minute: int = 0
    hour_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    day_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    minute_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def reset_if_needed(self) -> None:
        """Reset counters if time windows have passed."""
        now = datetime.now(timezone.utc)
        
        # Reset hourly counter
        if (now - self.hour_start).total_seconds() >= 3600:
            self.llm_calls_this_hour = 0
            self.hour_start = now
        
        # Reset daily counter
        if (now - self.day_start).total_seconds() >= 86400:
            self.llm_cost_today_usd = 0.0
            self.day_start = now
        
        # Reset minute counter
        if (now - self.minute_start).total_seconds() >= 60:
            self.requests_this_minute = 0
            self.minute_start = now


# ══════════════════════════════════════════════════════════════════════════════
# Governor
# ══════════════════════════════════════════════════════════════════════════════

class Governor:
    """
    Resource governor for the agent.
    
    Provides:
    - Rate limiting
    - Budget enforcement
    - Safety boundaries
    - Usage tracking
    
    Attributes:
        limits: Resource limits.
        usage: Current usage stats.
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """
        Initialize Governor.
        
        Args:
            limits: Resource limits (uses defaults if None).
        """
        self.limits = limits or ResourceLimits()
        self.usage = UsageStats()
    
    def can_make_llm_call(self, estimated_cost_usd: float = 0.01) -> bool:
        """
        Check if an LLM call is allowed.
        
        Args:
            estimated_cost_usd: Estimated cost of the call.
            
        Returns:
            True if allowed.
        """
        self.usage.reset_if_needed()
        
        # Check hourly limit
        if self.usage.llm_calls_this_hour >= self.limits.max_llm_calls_per_hour:
            return False
        
        # Check daily cost limit
        if self.usage.llm_cost_today_usd + estimated_cost_usd > self.limits.max_llm_cost_per_day_usd:
            return False
        
        return True
    
    def record_llm_call(self, actual_cost_usd: float) -> None:
        """
        Record an LLM call.
        
        Args:
            actual_cost_usd: Actual cost of the call.
        """
        self.usage.reset_if_needed()
        self.usage.llm_calls_this_hour += 1
        self.usage.llm_cost_today_usd += actual_cost_usd
    
    def can_accept_request(self) -> bool:
        """
        Check if a new request can be accepted.
        
        Returns:
            True if allowed.
        """
        self.usage.reset_if_needed()
        return self.usage.requests_this_minute < self.limits.max_requests_per_minute
    
    def record_request(self) -> None:
        """Record a request."""
        self.usage.reset_if_needed()
        self.usage.requests_this_minute += 1
    
    def get_remaining_llm_budget(self) -> dict[str, Any]:
        """
        Get remaining LLM budget.
        
        Returns:
            Dict with remaining calls and cost.
        """
        self.usage.reset_if_needed()
        
        return {
            "calls_remaining_this_hour": max(0, self.limits.max_llm_calls_per_hour - self.usage.llm_calls_this_hour),
            "cost_remaining_today_usd": max(0, self.limits.max_llm_cost_per_day_usd - self.usage.llm_cost_today_usd),
            "hour_resets_at": (self.usage.hour_start + timedelta(hours=1)).isoformat(),
            "day_resets_at": (self.usage.day_start + timedelta(days=1)).isoformat(),
        }
    
    def get_status(self) -> dict[str, Any]:
        """
        Get governor status.
        
        Returns:
            Dict with limits and usage.
        """
        self.usage.reset_if_needed()
        
        return {
            "limits": self.limits.to_dict(),
            "usage": {
                "llm_calls_this_hour": self.usage.llm_calls_this_hour,
                "llm_cost_today_usd": self.usage.llm_cost_today_usd,
                "requests_this_minute": self.usage.requests_this_minute,
            },
            "remaining": self.get_remaining_llm_budget(),
        }
    
    def is_under_pressure(self) -> bool:
        """
        Check if system is under resource pressure.
        
        Returns:
            True if any resource is > 80% utilized.
        """
        self.usage.reset_if_needed()
        
        # Check LLM calls
        if self.usage.llm_calls_this_hour > self.limits.max_llm_calls_per_hour * 0.8:
            return True
        
        # Check LLM cost
        if self.usage.llm_cost_today_usd > self.limits.max_llm_cost_per_day_usd * 0.8:
            return True
        
        # Check request rate
        if self.usage.requests_this_minute > self.limits.max_requests_per_minute * 0.8:
            return True
        
        return False
    
    def get_pressure_type(self) -> Optional[str]:
        """
        Get the type of resource pressure if any.
        
        Returns:
            Pressure type or None.
        """
        self.usage.reset_if_needed()
        
        if self.usage.llm_calls_this_hour > self.limits.max_llm_calls_per_hour * 0.8:
            return "llm_calls"
        
        if self.usage.llm_cost_today_usd > self.limits.max_llm_cost_per_day_usd * 0.8:
            return "llm_budget"
        
        if self.usage.requests_this_minute > self.limits.max_requests_per_minute * 0.8:
            return "request_rate"
        
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_governor_instance: Optional[Governor] = None


def get_governor() -> Governor:
    """Get the singleton Governor instance."""
    global _governor_instance
    if _governor_instance is None:
        _governor_instance = Governor()
    return _governor_instance


# ══════════════════════════════════════════════════════════════════════════════
# Section 15: Ingestion Rate Limiting
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IngestionVerdict:
    """Result of ingestion authorization check."""
    allowed: bool
    reason: str = ""
    wait_seconds: float = 0.0
    suggestion: str = ""


class IngestionRateLimiter:
    """
    Rate limiter specifically for ingestion API.
    
    Enforces:
    - Requests per minute/hour
    - Max text/file size
    - Max facts per request
    """
    
    def __init__(self, limits: ResourceLimits):
        """
        Initialize rate limiter.
        
        Args:
            limits: Resource limits.
        """
        self.limits = limits
        self._timestamps: list[datetime] = []
    
    def _prune_timestamps(self) -> None:
        """Remove timestamps older than 1 hour."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)
        self._timestamps = [t for t in self._timestamps if t > cutoff]
    
    def check_rate_limit(self) -> IngestionVerdict:
        """
        Check if ingestion is rate limited.
        
        Returns:
            IngestionVerdict with allowed status.
        """
        now = datetime.now(timezone.utc)
        self._prune_timestamps()
        
        # Check per-minute limit
        minute_ago = now - timedelta(minutes=1)
        calls_last_minute = sum(1 for t in self._timestamps if t > minute_ago)
        
        if calls_last_minute >= self.limits.ingestion_requests_per_minute:
            oldest_in_minute = min(t for t in self._timestamps if t > minute_ago)
            wait = 60 - (now - oldest_in_minute).total_seconds()
            
            return IngestionVerdict(
                allowed=False,
                reason="rate_limited",
                wait_seconds=max(0, wait),
                suggestion=f"Rate limit: {calls_last_minute}/{self.limits.ingestion_requests_per_minute} per minute",
            )
        
        # Check per-hour limit
        calls_last_hour = len(self._timestamps)
        
        if calls_last_hour >= self.limits.ingestion_requests_per_hour:
            oldest = min(self._timestamps)
            wait = 3600 - (now - oldest).total_seconds()
            
            return IngestionVerdict(
                allowed=False,
                reason="rate_limited",
                wait_seconds=max(0, wait),
                suggestion=f"Hourly limit: {calls_last_hour}/{self.limits.ingestion_requests_per_hour} per hour",
            )
        
        return IngestionVerdict(allowed=True)
    
    def check_text_size(self, size_bytes: int) -> IngestionVerdict:
        """
        Check if text size is within limits.
        
        Args:
            size_bytes: Size in bytes.
            
        Returns:
            IngestionVerdict.
        """
        size_kb = size_bytes / 1024
        
        if size_kb > self.limits.ingestion_max_text_size_kb:
            return IngestionVerdict(
                allowed=False,
                reason="size_exceeded",
                suggestion=f"Text too large: {size_kb:.1f}KB > {self.limits.ingestion_max_text_size_kb}KB. Split into smaller chunks.",
            )
        
        return IngestionVerdict(allowed=True)
    
    def check_file_size(self, size_bytes: int) -> IngestionVerdict:
        """
        Check if file size is within limits.
        
        Args:
            size_bytes: Size in bytes.
            
        Returns:
            IngestionVerdict.
        """
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > self.limits.ingestion_max_file_size_mb:
            return IngestionVerdict(
                allowed=False,
                reason="size_exceeded",
                suggestion=f"File too large: {size_mb:.1f}MB > {self.limits.ingestion_max_file_size_mb}MB",
            )
        
        return IngestionVerdict(allowed=True)
    
    def check_facts_count(self, count: int) -> IngestionVerdict:
        """
        Check if facts count is within limits.
        
        Args:
            count: Number of facts.
            
        Returns:
            IngestionVerdict.
        """
        if count > self.limits.ingestion_max_facts_per_request:
            return IngestionVerdict(
                allowed=False,
                reason="count_exceeded",
                suggestion=f"Too many facts: {count} > {self.limits.ingestion_max_facts_per_request}. Split into batches.",
            )
        
        return IngestionVerdict(allowed=True)
    
    def authorize(
        self,
        size_bytes: int = 0,
        facts_count: int = 0,
        is_file: bool = False,
    ) -> IngestionVerdict:
        """
        Full authorization check for ingestion.
        
        Args:
            size_bytes: Size of content.
            facts_count: Number of facts (if applicable).
            is_file: Whether this is a file upload.
            
        Returns:
            IngestionVerdict.
        """
        # Check rate limit first
        verdict = self.check_rate_limit()
        if not verdict.allowed:
            return verdict
        
        # Check size
        if is_file:
            verdict = self.check_file_size(size_bytes)
        else:
            verdict = self.check_text_size(size_bytes)
        
        if not verdict.allowed:
            return verdict
        
        # Check facts count
        if facts_count > 0:
            verdict = self.check_facts_count(facts_count)
            if not verdict.allowed:
                return verdict
        
        return IngestionVerdict(allowed=True)
    
    def record_request(self) -> None:
        """Record an ingestion request."""
        self._timestamps.append(datetime.now(timezone.utc))


_ingestion_limiter: Optional[IngestionRateLimiter] = None


def get_ingestion_limiter() -> IngestionRateLimiter:
    """Get the singleton IngestionRateLimiter."""
    global _ingestion_limiter
    if _ingestion_limiter is None:
        _ingestion_limiter = IngestionRateLimiter(get_governor().limits)
    return _ingestion_limiter
