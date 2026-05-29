"""
Governor

Resource limits and safety controls for the agent.
Enforces budgets, rate limits, and safety boundaries.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from config import config


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
