"""
Central Executive Network (CEN)

The global orchestrator — the main() of the agent. Coordinates Memory,
Response, Analytics, and Scheduling as one system.

Consumes priority signals from Salience Network, dispatches work,
manages resources.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from config import config
from control.salience import SalienceNetwork, AgentEvent, get_salience_network


# ══════════════════════════════════════════════════════════════════════════════
# Resource Manager
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceManager:
    """
    Tracks and manages agent resources.
    
    Attributes:
        max_requests: Maximum concurrent user requests.
        max_sub_agents: Maximum concurrent sub-agents.
    """
    max_requests: int = 5
    max_sub_agents: int = 10
    active_requests: int = 0
    active_sub_agents: int = 0
    memory_pressure: float = 0.0  # 0-1
    cpu_load: float = 0.0  # 0-1
    
    def can_accept_request(self) -> bool:
        """Check if we can accept a new user request."""
        return self.active_requests < self.max_requests
    
    def can_run_task(self) -> bool:
        """Check if we can run a background task."""
        # Reserve 1 slot for users
        return self.active_requests < self.max_requests - 1
    
    def can_spawn_sub_agent(self) -> bool:
        """Check if we can spawn a sub-agent."""
        return self.active_sub_agents < self.max_sub_agents
    
    def acquire_request(self) -> None:
        """Acquire a request slot."""
        self.active_requests += 1
    
    def release_request(self) -> None:
        """Release a request slot."""
        self.active_requests = max(0, self.active_requests - 1)
    
    def acquire_sub_agent(self) -> None:
        """Acquire a sub-agent slot."""
        self.active_sub_agents += 1
    
    def release_sub_agent(self) -> None:
        """Release a sub-agent slot."""
        self.active_sub_agents = max(0, self.active_sub_agents - 1)
    
    def reduce_max_concurrent(self, by: int = 1) -> None:
        """Reduce max concurrent requests under pressure."""
        self.max_requests = max(1, self.max_requests - by)
    
    def restore_max_concurrent(self, to: int = 5) -> None:
        """Restore max concurrent requests."""
        self.max_requests = to
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "max_sub_agents": self.max_sub_agents,
            "active_requests": self.active_requests,
            "active_sub_agents": self.active_sub_agents,
            "memory_pressure": self.memory_pressure,
            "cpu_load": self.cpu_load,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Central Executive Network
# ══════════════════════════════════════════════════════════════════════════════

class CentralExecutiveNetwork:
    """
    Global orchestrator for the agent.
    
    Provides:
    - Main event loop
    - Resource management
    - Mode switching (active/idle)
    - Event routing to appropriate systems
    
    Attributes:
        salience: Salience network for priority events.
        resources: Resource manager.
    """
    
    def __init__(
        self,
        salience: Optional[SalienceNetwork] = None,
    ):
        """
        Initialize CEN.
        
        Args:
            salience: Salience network instance.
        """
        self.salience = salience or get_salience_network()
        self.resources = ResourceManager()
        self.mode = "active"  # "active" | "idle"
        
        # Event handlers (registered by other systems)
        self._handlers: dict[str, Callable[[AgentEvent], Awaitable[Any]]] = {}
        
        # Running state
        self._running = False
        self._tasks: list[asyncio.Task] = []
    
    def register_handler(
        self,
        event_type: str,
        handler: Callable[[AgentEvent], Awaitable[Any]],
    ) -> None:
        """
        Register a handler for an event type.
        
        Args:
            event_type: Type of event to handle.
            handler: Async handler function.
        """
        self._handlers[event_type] = handler
    
    async def start(self) -> None:
        """Start the CEN event loop."""
        if self._running:
            return
        
        self._running = True
        
        # Start background monitors
        self._tasks.append(asyncio.create_task(self._mode_monitor()))
        self._tasks.append(asyncio.create_task(self._event_loop()))
        
        # Record startup
        from analytics import record_count
        await record_count("cen", "startup")
    
    async def stop(self) -> None:
        """Stop the CEN event loop."""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
    
    async def _event_loop(self) -> None:
        """Main event loop — the brain of the agent."""
        while self._running:
            try:
                # Get next event with timeout
                event = await asyncio.wait_for(
                    self.salience.get_next(),
                    timeout=1.0,
                )
                await self._handle_event(event)
            except asyncio.TimeoutError:
                # No events - continue
                continue
            except asyncio.CancelledError:
                break
    
    async def _handle_event(self, event: AgentEvent) -> None:
        """Route event to the appropriate handler."""
        # Log to analytics
        from analytics import record_count
        await record_count("cen", "event_received", dimensions={"type": event.type})
        
        # Check for registered handler
        handler = self._handlers.get(event.type)
        if handler:
            await self._dispatch_to_handler(event, handler)
            return
        
        # Default handling based on event type
        match event.type:
            case "user_message":
                await self._handle_user_message(event)
            
            case "user_frustrated":
                await self._handle_frustrated_user(event)
            
            case "scheduled_task_due":
                await self._handle_scheduled_task(event)
            
            case "sub_agent_complete":
                await self._handle_sub_agent_complete(event)
            
            case "conflict_detected":
                await self._handle_conflict(event)
            
            case "health_alert":
                await self._handle_health_alert(event)
            
            case "resource_pressure":
                await self._handle_resource_pressure(event)
            
            case "mode_switch_to_idle":
                await self._switch_to_idle()
            
            case "mode_switch_to_active":
                await self._switch_to_active()
            
            case _:
                # Unknown event type - log and skip
                from analytics import record_count
                await record_count("cen", "unknown_event", dimensions={"type": event.type})
    
    async def _dispatch_to_handler(
        self,
        event: AgentEvent,
        handler: Callable[[AgentEvent], Awaitable[Any]],
    ) -> None:
        """Dispatch event to registered handler with resource management."""
        if not self.resources.can_accept_request():
            await self._queue_with_backpressure(event)
            return
        
        self.resources.acquire_request()
        try:
            await handler(event)
        finally:
            self.resources.release_request()
    
    async def _handle_user_message(self, event: AgentEvent) -> None:
        """Handle a user message."""
        if not self.resources.can_accept_request():
            await self._queue_with_backpressure(event)
            return
        
        self.resources.acquire_request()
        try:
            # Route to response layer
            from response import process_message
            
            result = await process_message(
                message=event.payload.get("message", ""),
                user_id=event.payload.get("user_id"),
                convo_id=event.payload.get("convo_id"),
            )
            
            # Deliver response (would be via WebSocket in production)
            await self._deliver_response(event, result)
            
        finally:
            self.resources.release_request()
    
    async def _handle_frustrated_user(self, event: AgentEvent) -> None:
        """Handle frustrated user - escalate response."""
        # Pause low-priority background tasks
        from memory.management import get_mml
        mml = get_mml()
        # mml.pause_background(priority_below="P1")  # TODO: Implement
        
        # Re-process with escalation
        self.resources.acquire_request()
        try:
            from response import process_message
            
            result = await process_message(
                message=event.payload.get("message", ""),
                user_id=event.payload.get("user_id"),
                convo_id=event.payload.get("convo_id"),
            )
            
            await self._deliver_response(event, result)
            
        finally:
            self.resources.release_request()
    
    async def _handle_scheduled_task(self, event: AgentEvent) -> None:
        """Handle a scheduled task."""
        if not self.resources.can_run_task():
            # Defer task
            await self._defer_task(event, delay_seconds=30)
            return
        
        # Execute via PM
        from memory.types.pm import get_prospective_memory
        pm = get_prospective_memory()
        
        task_id = event.payload.get("task_id")
        if task_id:
            await pm.execute_task(task_id)
    
    async def _handle_sub_agent_complete(self, event: AgentEvent) -> None:
        """Handle sub-agent completion."""
        self.resources.release_sub_agent()
        
        # Route result back to response layer
        # TODO: Implement sub-agent result handling
    
    async def _handle_conflict(self, event: AgentEvent) -> None:
        """Handle detected memory conflict."""
        from memory.management import get_mml
        mml = get_mml()
        # mml.queue_conflict(event.payload)  # TODO: Implement
    
    async def _handle_health_alert(self, event: AgentEvent) -> None:
        """Handle system health alert."""
        alert_type = event.payload.get("alert_type")
        
        if alert_type == "memory_corruption":
            # Trigger integrity check
            from memory.management import get_mml
            mml = get_mml()
            await mml.check_integrity()
        
        # Log alert
        from analytics import record_count
        await record_count("cen", "health_alert", dimensions={"type": alert_type})
    
    async def _handle_resource_pressure(self, event: AgentEvent) -> None:
        """Handle resource pressure."""
        resource = event.payload.get("resource")
        
        if resource == "llm_budget":
            # Switch to cheaper model
            # TODO: Implement model switching
            pass
        
        elif resource == "memory":
            # Trigger emergency cleanup
            from memory.management import get_mml
            mml = get_mml()
            # await mml.trigger_emergency_cleanup()  # TODO: Implement
        
        elif resource == "cpu":
            # Reduce max concurrent
            self.resources.reduce_max_concurrent(by=2)
    
    async def _switch_to_idle(self) -> None:
        """Switch to idle mode - activate DMN."""
        if self.mode == "idle":
            return
        
        self.mode = "idle"
        
        from control.dmn import get_dmn
        dmn = get_dmn()
        asyncio.create_task(dmn.activate())
        
        from analytics import record_count
        await record_count("cen", "mode_switch", dimensions={"to": "idle"})
    
    async def _switch_to_active(self) -> None:
        """Switch to active mode - deactivate DMN."""
        if self.mode == "active":
            return
        
        self.mode = "active"
        
        from control.dmn import get_dmn
        dmn = get_dmn()
        dmn.deactivate()
        
        from analytics import record_count
        await record_count("cen", "mode_switch", dimensions={"to": "active"})
    
    async def _mode_monitor(self) -> None:
        """Monitor for mode switches."""
        while self._running:
            await asyncio.sleep(60)  # Check every minute
            
            mode_signal = self.salience.check_idle()
            if mode_signal == "switch_to_idle" and self.mode == "active":
                await self._switch_to_idle()
    
    async def _deliver_response(self, event: AgentEvent, result: Any) -> None:
        """Deliver response to user."""
        # In production, this would send via WebSocket
        # For now, just log
        from analytics import record_count
        await record_count("cen", "response_delivered")
    
    async def _queue_with_backpressure(self, event: AgentEvent) -> None:
        """Queue event with backpressure response."""
        # Re-queue with lower priority
        await self.salience.ingest_event(event)
        
        # TODO: Send "processing" message to user
    
    async def _defer_task(self, event: AgentEvent, delay_seconds: int) -> None:
        """Defer a task for later execution."""
        # Re-queue after delay
        await asyncio.sleep(delay_seconds)
        await self.salience.ingest_event(event)


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_cen_instance: Optional[CentralExecutiveNetwork] = None


def get_cen() -> CentralExecutiveNetwork:
    """Get the singleton CEN instance."""
    global _cen_instance
    if _cen_instance is None:
        _cen_instance = CentralExecutiveNetwork()
    return _cen_instance
