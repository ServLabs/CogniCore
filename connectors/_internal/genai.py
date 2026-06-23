import asyncio
import time
from typing import Any, Optional

from config import config
from logger import log
from helpers import RateLimitError
from observability import audit, record_latency, record_count, record_metric




class GenAI:
    """
    Unified LLM gateway — every LLM call in CogniCore goes through here.

    Provides:
    - Governor rate/budget checks before every call
    - Audit logging with latency and token estimates
    - Delegates to LLMConnector from the connector registry
    """

    async def ask(
        self,
        payload: dict[str, Any],
        *,
        cheap: bool = False,
        model: Optional[str] = None,
    ) -> str:
        """
        Send a payload with prompt to the AI model and return the response.

        Args:
            payload: Must contain 'messages' (list[dict]) or 'prompt' (str).
            cheap: Use the cheap model tier.
            model: Explicit model override.

        Returns:
            Generated text.
        """
        from control import get_governor
        from connectors.registry import get_registry

        governor = get_governor()

        if not governor.can_make_llm_call():
            audit.log_raw("connector", "llm_call", "genai", "denied", error="rate/budget limit")
            raise RateLimitError("LLM rate or budget limit exceeded")

        governor.record_request()

        start = time.monotonic()
        try:
            response = await self._call(payload, cheap=cheap, model=model)
            elapsed_ms = (time.monotonic() - start) * 1000

            # Estimate cost based on model tier (rates from config)
            if cheap:
                cost_rate = config.llm.cost_per_1k_chars_cheap
            elif model and model == config.llm.model:
                cost_rate = config.llm.cost_per_1k_chars_expensive
            else:
                cost_rate = config.llm.cost_per_1k_chars_default
            estimated_cost = (len(response) / 1000) * cost_rate
            governor.record_llm_call(actual_cost_usd=estimated_cost)

            audit.log_raw(
                "connector", "llm_call", "genai", "completed",
                target=model or (config.llm.cheap_model if cheap else config.llm.model),
                duration_ms=elapsed_ms,
                details={"chars": len(response)},
            )

            # Record to DuckDB analytics (fire-and-forget)
            resolved_model = model or (config.llm.cheap_model if cheap else config.llm.model)
            asyncio.create_task(record_latency("llm", "call", elapsed_ms, model=resolved_model))
            asyncio.create_task(record_count("llm", "call_count", model=resolved_model))
            asyncio.create_task(record_metric("llm", "response_chars", float(len(response)), {"model": resolved_model}))
            asyncio.create_task(record_metric("llm", "estimated_cost_usd", estimated_cost, {"model": resolved_model}))

            return response

        except RateLimitError:
            raise
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            audit.log_raw(
                "connector", "llm_call", "genai", "failed",
                duration_ms=elapsed_ms, error=str(e),
            )
            log.error("GenAI call failed: %s", e, exc_info=True)
            raise

    async def _call(
        self,
        payload: dict[str, Any],
        *,
        cheap: bool = False,
        model: Optional[str] = None,
    ) -> str:
        """Route to LLMConnector from the registry."""
        from connectors.registry import get_registry

        registry = get_registry()
        llm = registry.get_required("llm")

        messages = payload.get("messages")
        if messages:
            return await llm.generate_with_messages(
                messages=messages,
                model=model or (config.llm.cheap_model if cheap else None),
                max_tokens=payload.get("max_tokens"),
                temperature=payload.get("temperature"),
            )

        prompt = payload.get("prompt", "")
        system = payload.get("system")
        if cheap:
            return await llm.generate_cheap(prompt, system=system)
        return await llm.generate(
            prompt,
            system=system,
            model=model,
            max_tokens=payload.get("max_tokens"),
            temperature=payload.get("temperature"),
        )


genai = GenAI()