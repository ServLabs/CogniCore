"""
Autobiographical Memory (ABM)

The agent's frozen identity — who it is, what it can do, what it cannot do.
Loaded once at boot, never modified at runtime.

This module provides:
- Immutable identity dataclass
- Identity prompt generation for LLM injection
- Identity serialization for system prompts
"""

from dataclasses import dataclass
from typing import Optional

from config import config, AgentIdentity


@dataclass(frozen=True)
class AutobiographicalMemory:
    """
    Immutable agent identity.
    
    This is a frozen dataclass - any attempt to modify attributes after
    creation will raise FrozenInstanceError. This prevents identity drift
    from conversations, prompt injection, or memory poisoning.
    
    The identity is:
    - Loaded once at startup from config
    - Injected into every LLM system prompt
    - Never modified at runtime
    """
    name: str
    role: str
    domain: str
    capabilities: tuple[str, ...]
    constraints: tuple[str, ...]
    personality_traits: tuple[str, ...]
    version: str
    
    @classmethod
    def from_config(cls, identity: Optional[AgentIdentity] = None) -> "AutobiographicalMemory":
        """
        Create ABM instance from configuration.
        
        Args:
            identity: Optional AgentIdentity config. Uses global config if not provided.
            
        Returns:
            Frozen AutobiographicalMemory instance.
        """
        if identity is None:
            identity = config.identity
            
        return cls(
            name=identity.name,
            role=identity.role,
            domain=identity.domain,
            capabilities=identity.capabilities,
            constraints=identity.constraints,
            personality_traits=identity.personality_traits,
            version=identity.version,
        )
    
    def to_system_prompt(self) -> str:
        """
        Generate the identity portion of the system prompt.
        
        This is injected into every LLM call to establish the agent's
        identity, capabilities, and constraints.
        
        Returns:
            Formatted identity prompt string.
        """
        capabilities_list = "\n".join(f"  - {cap}" for cap in self.capabilities)
        constraints_list = "\n".join(f"  - {con}" for con in self.constraints)
        traits_list = ", ".join(self.personality_traits)
        
        return f"""You are {self.name}, a {self.role}.

## Your Capabilities
You can:
{capabilities_list}

## Your Constraints
You cannot and must not:
{constraints_list}

## Your Personality
You are: {traits_list}

## Domain Expertise
Your primary domain is: {self.domain}

Always stay within your defined capabilities and constraints. If asked to do something outside your capabilities, politely explain what you can and cannot do.
"""
    
    def to_compact_prompt(self) -> str:
        """
        Generate a compact identity prompt for token-constrained contexts.
        
        Returns:
            Shortened identity prompt string.
        """
        caps = ", ".join(self.capabilities[:5])  # Top 5 capabilities
        cons = ", ".join(self.constraints[:3])   # Top 3 constraints
        
        return f"""You are {self.name}, a {self.role} specializing in {self.domain}.
Capabilities: {caps}
Constraints: {cons}
Personality: {", ".join(self.personality_traits)}
"""
    
    def to_dict(self) -> dict:
        """
        Serialize identity to dictionary.
        
        Returns:
            Dictionary representation of identity.
        """
        return {
            "name": self.name,
            "role": self.role,
            "domain": self.domain,
            "capabilities": list(self.capabilities),
            "constraints": list(self.constraints),
            "personality_traits": list(self.personality_traits),
            "version": self.version,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name} v{self.version} ({self.role})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"AutobiographicalMemory(name={self.name!r}, role={self.role!r}, "
            f"domain={self.domain!r}, version={self.version!r})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ══════════════════════════════════════════════════════════════════════════════

# The agent's identity - loaded once, used everywhere
_abm_instance: Optional[AutobiographicalMemory] = None


def get_identity() -> AutobiographicalMemory:
    """
    Get the singleton ABM instance.
    
    Creates the instance on first call, returns cached instance thereafter.
    This ensures identity is consistent across all components.
    
    Returns:
        The agent's AutobiographicalMemory instance.
    """
    global _abm_instance
    if _abm_instance is None:
        _abm_instance = AutobiographicalMemory.from_config()
    return _abm_instance


def get_system_prompt() -> str:
    """
    Convenience function to get the identity system prompt.
    
    Returns:
        The identity portion of the system prompt.
    """
    return get_identity().to_system_prompt()


def get_compact_prompt() -> str:
    """
    Convenience function to get the compact identity prompt.
    
    Returns:
        Shortened identity prompt for token-constrained contexts.
    """
    return get_identity().to_compact_prompt()
