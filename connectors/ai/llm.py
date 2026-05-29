"""
LLM Connector

Unified LLM interface. Supports OpenAI, Anthropic, and local endpoints.
Provider-agnostic — swapping is a config change, not a code change.
"""

from typing import Any, Optional

from config import config
from connectors.base import BaseConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class LLMConnector(BaseConnector):
    """
    Unified LLM interface.
    
    Supports:
    - OpenAI (GPT-4o, GPT-4o-mini)
    - Anthropic (Claude)
    - Local endpoints (OpenAI-compatible API)
    
    Provider selection via config.llm.provider.
    """
    
    def __init__(self):
        """Initialize LLM connector."""
        self._client = None
    
    def is_configured(self) -> bool:
        """Check if LLM is configured."""
        return config.llm.api_key is not None or config.llm.provider == "local"
    
    async def connect(self) -> None:
        """Initialize LLM client based on provider."""
        match config.llm.provider:
            case "openai":
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=config.llm.api_key,
                    base_url=config.llm.api_base,
                )
            
            case "anthropic":
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=config.llm.api_key)
            
            case "local":
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key="not-needed",
                    base_url=config.llm.api_base,
                )
    
    async def disconnect(self) -> None:
        """Close LLM client."""
        self._client = None
    
    async def health_check(self) -> bool:
        """Check if LLM is responsive."""
        if not self._client:
            return False
        
        try:
            await self.generate("ping", max_tokens=5)
            return True
        except Exception:
            return False
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Provider-agnostic interface.
        
        Args:
            prompt: User prompt.
            system: System prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            model: Model to use (overrides config).
            
        Returns:
            Generated text.
        """
        model = model or config.llm.model
        max_tokens = max_tokens or config.llm.max_tokens
        temperature = temperature if temperature is not None else config.llm.temperature
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        if config.llm.provider == "anthropic":
            resp = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        else:
            # OpenAI / local (OpenAI-compatible)
            resp = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
    
    async def generate_cheap(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> str:
        """
        Generate using the cheaper model.
        
        Use for budget pressure, gate, simple tasks.
        
        Args:
            prompt: User prompt.
            system: System prompt.
            
        Returns:
            Generated text.
        """
        return await self.generate(
            prompt,
            system=system,
            model=config.llm.cheap_model,
        )
    
    async def generate_with_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate from a list of messages.
        
        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            model: Model to use.
            
        Returns:
            Generated text.
        """
        model = model or config.llm.model
        max_tokens = max_tokens or config.llm.max_tokens
        temperature = temperature if temperature is not None else config.llm.temperature
        
        if config.llm.provider == "anthropic":
            # Extract system message if present
            system = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    user_messages.append(msg)
            
            resp = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=user_messages,
            )
            return resp.content[0].text
        else:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        return ConnectorInfo(
            name="llm",
            connector_type=ConnectorType.AI,
            status=ConnectorStatus.CONNECTED if self._client else ConnectorStatus.DISCONNECTED,
            metadata={
                "provider": config.llm.provider,
                "model": config.llm.model,
            },
        )
