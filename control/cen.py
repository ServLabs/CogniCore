"""
Central Executive Network (CEN)

The global orchestrator — the main() of the agent. Consumes priority
events from Salience Network via a fixed-size worker pool, dispatches
work to the response pipeline and other subsystems.

Worker pool model:
- N workers (configurable, default 30) pull from Salience's priority queue.
- Each worker blocks on pipeline.process() until completion.
- Pool size = max concurrent pipeline executions.
- When all workers are busy, new events wait in queue and callers
  receive a "queued" status via their channel.
"""

import asyncio
from typing import Optional

from control.dmn import get_dmn
from control.events import (
    AgentMode,
    BaseEvent,
    UserMessageEvent,
    UserFrustratedEvent,
    ScheduledTaskEvent,
    SubAgentCompleteEvent,
    HealthAlertEvent,
    ResourcePressureEvent,
    ConflictDetectedEvent,
    ModeSwitchEvent,
)
from control.governor import get_governor
from control.salience import SalienceNetwork, get_salience_network
from config import config
from logger import log
from memory import add_assistant_message, get_mml, get_prospective_memory
from observability import record_count, record_latency
from response import get_pipeline, NullChannel


# ══════════════════════════════════════════════════════════════════════════════
# Central Executive Network
# ══════════════════════════════════════════════════════════════════════════════

class CentralExecutiveNetwork:
    """
    Global orchestrator with a fixed-size async worker pool.

    Workers pull events from the Salience priority queue and block
    on the handler until completion. Pool size controls max concurrency.
    """

    def __init__(
        self,
        salience: Optional[SalienceNetwork] = None,
        pool_size: Optional[int] = None,
    ):
        self.salience = salience or get_salience_network()
        self.pool_size = pool_size or config.control.worker_pool_size
        self.mode: AgentMode = AgentMode.ACTIVE

        # Running state
        self._running = False
        self._worker_tasks: list[asyncio.Task] = []
        self._monitor_task: Optional[asyncio.Task] = None

    # ── Lifecycle ──

    async def start(self) -> None:
        """Start the worker pool and mode monitor."""
        if self._running:
            return

        self._running = True

        # Spawn fixed worker pool
        for i in range(self.pool_size):
            task = asyncio.create_task(self._worker(i), name=f"cen-worker-{i}")
            self._worker_tasks.append(task)

        # Mode monitor (idle detection)
        self._monitor_task = asyncio.create_task(
            self._mode_monitor(), name="cen-mode-monitor",
        )

        await record_count("cen", "startup", dimensions={"pool_size": self.pool_size})
        log.info(f"CEN started with {self.pool_size} workers")

    async def stop(self) -> None:
        """Stop all workers and the mode monitor gracefully."""
        self._running = False

        all_tasks = self._worker_tasks.copy()
        if self._monitor_task:
            all_tasks.append(self._monitor_task)

        for task in all_tasks:
            task.cancel()

        await asyncio.gather(*all_tasks, return_exceptions=True)

        self._worker_tasks.clear()
        self._monitor_task = None

        log.info("CEN stopped")

    # ── Worker ──

    async def _worker(self, worker_id: int) -> None:
        """
        A single worker loop — pulls the next event and handles it.

        Blocks on the handler, so pool size = max concurrent handlers.
        """
        while self._running:
            try:
                event = await self.salience.get_next()
                await self._handle(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"CEN worker-{worker_id} error: {e}", exc_info=True)

    # ── Event Routing ──

    async def _handle(self, event: BaseEvent) -> None:
        """Route a typed event to the appropriate handler."""
        await record_count("cen", "event_received", dimensions={"type": event.type})

        match event:
            case UserMessageEvent():
                await self._handle_user_message(event)

            case UserFrustratedEvent():
                await self._handle_user_message(event)

            case ScheduledTaskEvent():
                await self._handle_scheduled_task(event)

            case SubAgentCompleteEvent():
                await self._handle_sub_agent_complete(event)

            case HealthAlertEvent():
                await self._handle_health_alert(event)

            case ResourcePressureEvent():
                await self._handle_resource_pressure(event)

            case ConflictDetectedEvent():
                await self._handle_conflict(event)

            case ModeSwitchEvent():
                await self._handle_mode_switch(event)

            case _:
                await record_count("cen", "unknown_event", dimensions={"type": event.type})

    # ── Handlers ──

    async def _handle_user_message(
        self,
        event: UserMessageEvent | UserFrustratedEvent,
    ) -> None:
        """Route a user message through the response pipeline."""
        import time
        _start = time.perf_counter()
        
        governor = get_governor()
        if not governor.can_accept_request():
            # Stream "queued" status if channel is available
            if event.channel:
                await event.channel.send_status("queued", "Your request is queued — please wait.")
            # Re-queue the event so another worker picks it up after delay
            await asyncio.sleep(1)
            await self.salience.ingest(event)
            return

        governor.record_request()

        try:
            channel = event.channel or NullChannel()

            result = await get_pipeline().process(
                message=event.message,
                channel=channel,
                user_id=event.user_id,
                convo_id=event.convo_id,
            )

            # Emit response and done events through the channel's stream
            if hasattr(channel, 'stream') and channel.stream is not None:
                if result.response:
                    await add_assistant_message(event.user_id, event.convo_id, result.response)

                channel.stream.emit_response_chunk(result.response)
                channel.stream.emit_done({
                    "processing_time_ms": result.processing_time_ms,
                    "used_deep_pipeline": result.used_deep_pipeline,
                })

            elapsed_ms = (time.perf_counter() - _start) * 1000
            await record_latency("cen", "handle_message", elapsed_ms)
            await record_count("cen", "response_delivered")

        except Exception as e:
            log.error(f"CEN pipeline error: {e}", exc_info=True)
            if event.channel is not None and hasattr(event.channel, 'stream'):
                event.channel.stream.emit_error(str(e), recoverable=False)
                event.channel.stream.emit_done({"error": str(e)})

    async def _handle_scheduled_task(self, event: ScheduledTaskEvent) -> None:
        """Execute a scheduled task via Prospective Memory."""
        try:
            pm = get_prospective_memory()
            if event.task_id:
                await pm.execute_task(event.task_id)

            await record_count("cen", "scheduled_task_executed", dimensions={"task_type": event.task_type})
        except Exception as e:
            log.error(f"CEN scheduled task error: {e}", exc_info=True)

    async def _handle_sub_agent_complete(self, event: SubAgentCompleteEvent) -> None:
        """Handle a sub-agent completion notification."""
        await record_count("cen", "sub_agent_complete", dimensions={"task": event.task_name})

    async def _handle_health_alert(self, event: HealthAlertEvent) -> None:
        """Handle a system health alert."""
        if event.alert_type == "memory_corruption":
            await get_mml().check_integrity()

        await record_count("cen", "health_alert", dimensions={"type": event.alert_type})

    async def _handle_resource_pressure(self, event: ResourcePressureEvent) -> None:
        """Handle resource pressure (budget, memory, CPU)."""
        await record_count("cen", "resource_pressure", dimensions={"resource": event.resource})

    async def _handle_conflict(self, event: ConflictDetectedEvent) -> None:
        """Handle a detected memory conflict."""
        await record_count("cen", "conflict_detected")

    async def _handle_mode_switch(self, event: ModeSwitchEvent) -> None:
        """Switch agent mode (active ↔ idle)."""
        target = event.target_mode

        if target == self.mode:
            return

        self.mode = target

        if target == AgentMode.IDLE:
            asyncio.create_task(get_dmn().activate())
        else:
            get_dmn().deactivate()

        await record_count("cen", "mode_switch", dimensions={"to": target.value})
        log.info(f"CEN mode → {target.value}")

    # ── Mode Monitor ──

    async def _mode_monitor(self) -> None:
        """Periodically check for idle-mode transition."""
        interval = config.control.mode_check_interval_seconds
        while self._running:
            try:
                await asyncio.sleep(interval)
                idle_mode = self.salience.check_idle()
                if idle_mode == AgentMode.IDLE and self.mode == AgentMode.ACTIVE:
                    await self._handle_mode_switch(ModeSwitchEvent(target_mode=AgentMode.IDLE))
            except asyncio.CancelledError:
                break


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_cen_instance: Optional[CentralExecutiveNetwork] = None


def get_cen() -> CentralExecutiveNetwork:
    global _cen_instance
    if _cen_instance is None:
        _cen_instance = CentralExecutiveNetwork()
    return _cen_instance
