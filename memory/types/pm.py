"""
Prospective Memory (PM)

The agent's future-oriented task memory — what it needs to do and when.
Analogous to a human's ability to remember to do things at specific times.

This module provides:
- Schedule and ExecutionLog dataclasses
- ProspectiveMemory manager for CRUD operations
- Event-driven scheduler with asyncio
- Cron expression support for recurring tasks
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class ScheduleStatus(Enum):
    """Status of a scheduled task."""
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"      # Claimed by a pod, currently executing
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"  # For one-time tasks that finished


class ExecutionStatus(Enum):
    """Status of a task execution."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"
    RUNNING = "RUNNING"


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Schedule:
    """
    A scheduled task definition (intent).
    
    Represents what should happen and when. Recurring tasks use cron_expr,
    one-time tasks leave it as None.
    """
    id: str
    task_name: str
    task_type: str
    payload: dict[str, Any]
    scheduled_at: datetime
    priority: int = 0
    status: ScheduleStatus = ScheduleStatus.SCHEDULED
    cron_expr: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def create(
        cls,
        task_name: str,
        task_type: str,
        scheduled_at: datetime,
        payload: Optional[dict[str, Any]] = None,
        priority: int = 0,
        cron_expr: Optional[str] = None,
    ) -> "Schedule":
        """
        Factory method to create a new schedule.
        
        Args:
            task_name: Human-readable task label.
            task_type: Category (e.g., "data_pull", "report", "alert").
            scheduled_at: When to execute (will be rounded to 15-min).
            payload: Task parameters/arguments.
            priority: Execution priority (lower = higher priority).
            cron_expr: Cron expression for recurring tasks.
            
        Returns:
            New Schedule instance with generated ID.
        """
        return cls(
            id=str(uuid.uuid4()),
            task_name=task_name,
            task_type=task_type,
            payload=payload or {},
            scheduled_at=cls._round_to_15min(scheduled_at),
            priority=priority,
            cron_expr=cron_expr,
        )
    
    @staticmethod
    def _round_to_15min(dt: datetime) -> datetime:
        """Round datetime to nearest 15-minute interval."""
        minutes = dt.minute
        rounded_minutes = (minutes // 15) * 15
        return dt.replace(minute=rounded_minutes, second=0, microsecond=0)
    
    @property
    def is_recurring(self) -> bool:
        """Check if this is a recurring task."""
        return self.cron_expr is not None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": self.id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "payload": json.dumps(self.payload),
            "scheduled_at": self.scheduled_at.isoformat(),
            "priority": self.priority,
            "status": self.status.value,
            "cron_expr": self.cron_expr,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Schedule":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            task_name=data["task_name"],
            task_type=data["task_type"],
            payload=json.loads(data["payload"]) if isinstance(data["payload"], str) else data["payload"],
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]),
            priority=data["priority"],
            status=ScheduleStatus(data["status"]),
            cron_expr=data.get("cron_expr"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class ExecutionLog:
    """
    A task execution record (reality).
    
    Represents what actually happened when a scheduled task ran.
    """
    id: str
    schedule_id: str
    scheduled_at: datetime
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.RUNNING
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    
    @classmethod
    def start(cls, schedule: Schedule) -> "ExecutionLog":
        """
        Factory method to create a new execution log when task starts.
        
        Args:
            schedule: The schedule being executed.
            
        Returns:
            New ExecutionLog in RUNNING status.
        """
        return cls(
            id=str(uuid.uuid4()),
            schedule_id=schedule.id,
            scheduled_at=schedule.scheduled_at,
            started_at=datetime.now(timezone.utc),
        )
    
    def complete(self, result: Optional[dict[str, Any]] = None) -> None:
        """Mark execution as successful."""
        self.completed_at = datetime.now(timezone.utc)
        self.status = ExecutionStatus.SUCCESS
        self.result = result
        self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
    
    def fail(self, error: str) -> None:
        """Mark execution as failed."""
        self.completed_at = datetime.now(timezone.utc)
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
    
    def timeout(self) -> None:
        """Mark execution as timed out."""
        self.completed_at = datetime.now(timezone.utc)
        self.status = ExecutionStatus.TIMEOUT
        self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": self.id,
            "schedule_id": self.schedule_id,
            "scheduled_at": self.scheduled_at.isoformat(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "result": json.dumps(self.result) if self.result else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Prospective Memory Manager
# ══════════════════════════════════════════════════════════════════════════════

class ProspectiveMemory:
    """
    Manager for scheduled tasks and their execution.
    
    Provides CRUD operations for schedules and an event-driven scheduler
    that executes tasks at their scheduled times.
    
    Attributes:
        db_path: Path to SQLite database.
        schedules: In-memory cache of active schedules.
        executor: Callback function to execute tasks.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        executor: Optional[Callable[[Schedule], Awaitable[dict[str, Any]]]] = None,
    ):
        """
        Initialize Prospective Memory.
        
        Args:
            db_path: Path to SQLite database. Uses config default if not provided.
            executor: Async function to execute tasks. Receives Schedule, returns result dict.
        """
        self.db_path = db_path or str(config.paths.hot_db)
        self.executor = executor
        
        # In-memory state
        self._schedules: dict[str, Schedule] = {}
        self._wake_event = asyncio.Event()
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Database connection (lazy init)
        self._db_initialized = False
    
    # ── Database Operations ──
    
    async def _init_db(self) -> None:
        """Initialize database tables if they don't exist."""
        if self._db_initialized:
            return
            
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            # Schedules table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS schedules (
                    id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    payload TEXT,
                    scheduled_at TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'SCHEDULED',
                    cron_expr TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Execution log table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS execution_log (
                    id TEXT PRIMARY KEY,
                    schedule_id TEXT NOT NULL,
                    scheduled_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    duration_ms INTEGER,
                    FOREIGN KEY (schedule_id) REFERENCES schedules(id)
                )
            """)
            
            # Indexes for common queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_schedules_status_time 
                ON schedules(status, scheduled_at)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_schedule 
                ON execution_log(schedule_id)
            """)
            
            await db.commit()
        
        self._db_initialized = True
    
    async def _load_schedules(self) -> None:
        """Load active schedules from database into memory."""
        await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM schedules WHERE status = ?",
                (ScheduleStatus.SCHEDULED.value,)
            ) as cursor:
                async for row in cursor:
                    schedule = Schedule.from_dict(dict(row))
                    self._schedules[schedule.id] = schedule
    
    async def _save_schedule(self, schedule: Schedule) -> None:
        """Save a schedule to database."""
        await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            data = schedule.to_dict()
            await db.execute("""
                INSERT OR REPLACE INTO schedules 
                (id, task_name, task_type, payload, scheduled_at, priority, status, cron_expr, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"], data["task_name"], data["task_type"], data["payload"],
                data["scheduled_at"], data["priority"], data["status"],
                data["cron_expr"], data["created_at"], data["updated_at"]
            ))
            await db.commit()
    
    async def _save_execution_log(self, log: ExecutionLog) -> None:
        """Save an execution log to database."""
        await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            data = log.to_dict()
            await db.execute("""
                INSERT OR REPLACE INTO execution_log
                (id, schedule_id, scheduled_at, started_at, completed_at, status, result, error, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"], data["schedule_id"], data["scheduled_at"],
                data["started_at"], data["completed_at"], data["status"],
                data["result"], data["error"], data["duration_ms"]
            ))
            await db.commit()
    
    # ── Public API ──
    
    async def schedule(
        self,
        task_name: str,
        task_type: str,
        scheduled_at: datetime,
        payload: Optional[dict[str, Any]] = None,
        priority: int = 0,
        cron_expr: Optional[str] = None,
    ) -> Schedule:
        """
        Create and register a new scheduled task.
        
        Args:
            task_name: Human-readable task label.
            task_type: Category (e.g., "data_pull", "report", "alert").
            scheduled_at: When to execute.
            payload: Task parameters/arguments.
            priority: Execution priority (lower = higher priority).
            cron_expr: Cron expression for recurring tasks.
            
        Returns:
            The created Schedule.
        """
        schedule = Schedule.create(
            task_name=task_name,
            task_type=task_type,
            scheduled_at=scheduled_at,
            payload=payload,
            priority=priority,
            cron_expr=cron_expr,
        )
        
        # Save to DB and memory
        await self._save_schedule(schedule)
        self._schedules[schedule.id] = schedule
        
        # Wake scheduler to re-evaluate next task
        self._wake_event.set()
        
        return schedule
    
    async def cancel(self, schedule_id: str) -> bool:
        """
        Cancel a scheduled task.
        
        Args:
            schedule_id: ID of the schedule to cancel.
            
        Returns:
            True if cancelled, False if not found.
        """
        if schedule_id not in self._schedules:
            return False
        
        schedule = self._schedules[schedule_id]
        schedule.status = ScheduleStatus.CANCELLED
        schedule.updated_at = datetime.now(timezone.utc)
        
        await self._save_schedule(schedule)
        del self._schedules[schedule_id]
        
        return True
    
    async def pause(self, schedule_id: str) -> bool:
        """Pause a scheduled task."""
        if schedule_id not in self._schedules:
            return False
        
        schedule = self._schedules[schedule_id]
        schedule.status = ScheduleStatus.PAUSED
        schedule.updated_at = datetime.now(timezone.utc)
        
        await self._save_schedule(schedule)
        del self._schedules[schedule_id]
        
        return True
    
    async def resume(self, schedule_id: str) -> bool:
        """Resume a paused task."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM schedules WHERE id = ? AND status = ?",
                (schedule_id, ScheduleStatus.PAUSED.value)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return False
                
                schedule = Schedule.from_dict(dict(row))
                schedule.status = ScheduleStatus.SCHEDULED
                schedule.updated_at = datetime.now(timezone.utc)
                
                await self._save_schedule(schedule)
                self._schedules[schedule.id] = schedule
                self._wake_event.set()
                
                return True
    
    async def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Get a schedule by ID."""
        return self._schedules.get(schedule_id)
    
    async def list_schedules(
        self,
        status: Optional[ScheduleStatus] = None,
        task_type: Optional[str] = None,
    ) -> list[Schedule]:
        """
        List schedules with optional filtering.
        
        Args:
            status: Filter by status.
            task_type: Filter by task type.
            
        Returns:
            List of matching schedules.
        """
        schedules = list(self._schedules.values())
        
        if status:
            schedules = [s for s in schedules if s.status == status]
        if task_type:
            schedules = [s for s in schedules if s.task_type == task_type]
        
        return sorted(schedules, key=lambda s: (s.priority, s.scheduled_at))
    
    # ── Scheduler ──
    
    def _get_next_due(self) -> Optional[Schedule]:
        """Get the next task due for execution."""
        now = datetime.now(timezone.utc)
        due_schedules = [
            s for s in self._schedules.values()
            if s.status == ScheduleStatus.SCHEDULED and s.scheduled_at <= now
        ]
        
        if not due_schedules:
            return None
        
        # Sort by priority (lower first), then by scheduled time
        return min(due_schedules, key=lambda s: (s.priority, s.scheduled_at))
    
    def _get_next_scheduled(self) -> Optional[Schedule]:
        """Get the next scheduled task (may be in future)."""
        scheduled = [
            s for s in self._schedules.values()
            if s.status == ScheduleStatus.SCHEDULED
        ]
        
        if not scheduled:
            return None
        
        return min(scheduled, key=lambda s: s.scheduled_at)
    
    async def _claim_task(self, schedule_id: str) -> bool:
        """
        Attempt to claim a task for execution (optimistic locking).
        
        Uses atomic UPDATE to ensure only one pod can claim a task.
        This prevents duplicate execution in multi-instance deployments.
        
        Args:
            schedule_id: ID of the schedule to claim.
            
        Returns:
            True if claimed successfully, False if another instance got it.
        """
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE schedules 
                SET status = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    ScheduleStatus.RUNNING.value,
                    datetime.now(timezone.utc).isoformat(),
                    schedule_id,
                    ScheduleStatus.SCHEDULED.value,
                )
            )
            await db.commit()
            
            # If rows_affected == 1, we claimed it. If 0, someone else did.
            return cursor.rowcount == 1
    
    async def _release_task(self, schedule: Schedule, next_status: ScheduleStatus) -> None:
        """
        Release a task after execution, setting its next status.
        
        Args:
            schedule: The schedule that was executed.
            next_status: Status to set (SCHEDULED for recurring, COMPLETED for one-time).
        """
        schedule.status = next_status
        schedule.updated_at = datetime.now(timezone.utc)
        await self._save_schedule(schedule)
        
        # Update in-memory cache
        if next_status == ScheduleStatus.SCHEDULED:
            self._schedules[schedule.id] = schedule
        elif schedule.id in self._schedules:
            del self._schedules[schedule.id]
    
    async def _execute_task(self, schedule: Schedule) -> None:
        """
        Execute a single task with claim-before-execute pattern.
        
        Only executes if this instance successfully claims the task.
        Prevents duplicate execution across multiple pods.
        """
        # Try to claim the task (atomic operation)
        claimed = await self._claim_task(schedule.id)
        if not claimed:
            # Another instance already claimed it - skip
            # Remove from our in-memory cache since it's being handled elsewhere
            if schedule.id in self._schedules:
                del self._schedules[schedule.id]
            return
        
        # We claimed it - execute
        log = ExecutionLog.start(schedule)
        
        try:
            if self.executor:
                result = await asyncio.wait_for(
                    self.executor(schedule),
                    timeout=300  # 5 minute timeout
                )
                log.complete(result)
            else:
                # No executor configured - just mark as complete
                log.complete({"message": "No executor configured"})
                
        except asyncio.TimeoutError:
            log.timeout()
        except Exception as e:
            log.fail(str(e))
        
        # Save execution log
        await self._save_execution_log(log)
        
        # Handle recurring vs one-time
        if schedule.is_recurring:
            # Compute next occurrence and release back to SCHEDULED
            next_time = self._compute_next_cron(schedule.cron_expr, schedule.scheduled_at)
            if next_time:
                schedule.scheduled_at = next_time
                await self._release_task(schedule, ScheduleStatus.SCHEDULED)
            else:
                # No next occurrence - mark completed
                await self._release_task(schedule, ScheduleStatus.COMPLETED)
        else:
            # One-time task - mark as completed
            await self._release_task(schedule, ScheduleStatus.COMPLETED)
    
    def _compute_next_cron(self, cron_expr: str, after: datetime) -> Optional[datetime]:
        """
        Compute next occurrence from cron expression.
        
        Args:
            cron_expr: Cron expression string.
            after: Compute next occurrence after this time.
            
        Returns:
            Next occurrence datetime, or None if invalid.
        """
        try:
            from croniter import croniter
            cron = croniter(cron_expr, after)
            return cron.get_next(datetime)
        except Exception:
            # If croniter not installed or invalid expression
            # Fall back to simple daily recurrence
            return after + timedelta(days=1)
    
    async def _recover_stuck_tasks(self) -> None:
        """
        Recover tasks stuck in RUNNING status (from crashed pods).
        
        If a task has been RUNNING for more than 10 minutes, assume the
        pod that claimed it crashed and reset it to SCHEDULED.
        """
        import aiosqlite
        
        stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=10)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Find and reset stuck tasks
            cursor = await db.execute(
                """
                UPDATE schedules 
                SET status = ?, updated_at = ?
                WHERE status = ? AND updated_at < ?
                """,
                (
                    ScheduleStatus.SCHEDULED.value,
                    datetime.now(timezone.utc).isoformat(),
                    ScheduleStatus.RUNNING.value,
                    stale_threshold.isoformat(),
                )
            )
            await db.commit()
            
            if cursor.rowcount > 0:
                # Reload schedules to pick up recovered tasks
                await self._load_schedules()
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop - runs until stopped."""
        # Recover any stuck tasks from crashed pods
        await self._recover_stuck_tasks()
        await self._load_schedules()
        
        while self._running:
            # Check for due tasks
            while True:
                due = self._get_next_due()
                if not due:
                    break
                await self._execute_task(due)
            
            # Calculate sleep time until next task
            next_scheduled = self._get_next_scheduled()
            if next_scheduled:
                now = datetime.now(timezone.utc)
                delta = (next_scheduled.scheduled_at - now).total_seconds()
                sleep_time = max(0, delta)
            else:
                sleep_time = 3600  # No tasks - sleep for an hour
            
            # Wait for either: sleep time elapsed OR new task notification
            self._wake_event.clear()
            try:
                await asyncio.wait_for(
                    self._wake_event.wait(),
                    timeout=sleep_time
                )
            except asyncio.TimeoutError:
                pass  # Normal - sleep time elapsed
    
    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        self._wake_event.set()
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
    
    def notify_new_task(self) -> None:
        """Wake the scheduler to re-evaluate next task."""
        self._wake_event.set()


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_pm_instance: Optional[ProspectiveMemory] = None


def get_prospective_memory() -> ProspectiveMemory:
    """
    Get the singleton ProspectiveMemory instance.
    
    Returns:
        The ProspectiveMemory instance.
    """
    global _pm_instance
    if _pm_instance is None:
        _pm_instance = ProspectiveMemory()
    return _pm_instance


# ── Convenience Functions ──

async def schedule_task(
    task_name: str,
    task_type: str,
    scheduled_at: datetime,
    payload: Optional[dict[str, Any]] = None,
    priority: int = 0,
    cron_expr: Optional[str] = None,
) -> Schedule:
    """Convenience: schedule a task using the singleton."""
    pm = get_prospective_memory()
    return await pm.schedule(
        task_name=task_name,
        task_type=task_type,
        scheduled_at=scheduled_at,
        payload=payload,
        priority=priority,
        cron_expr=cron_expr,
    )


async def cancel_task(schedule_id: str) -> bool:
    """Convenience: cancel a task using the singleton."""
    return await get_prospective_memory().cancel(schedule_id)
