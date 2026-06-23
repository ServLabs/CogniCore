"""
Cron & Timezone Handling (Section 16)

Standard library cron parsing with UTC normalization.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo


class CronParser:
    """
    Simple cron expression parser using standard library.
    
    Supports: minute hour day month weekday
    Examples: 
    - "0 3 * * *" (3 AM daily)
    - "*/15 * * * *" (every 15 min)
    - "0 9 * * 1-5" (9 AM weekdays)
    """
    
    FIELDS = ["minute", "hour", "day", "month", "weekday"]
    
    def __init__(self, expr: str, tz: str = "UTC"):
        """
        Initialize cron parser.
        
        Args:
            expr: Cron expression (5 fields).
            tz: Timezone name (e.g., "America/New_York").
        """
        self.expr = expr
        self.tz = ZoneInfo(tz)
        self.parts = self._parse(expr)
    
    def _parse(self, expr: str) -> dict[str, set[int]]:
        """
        Parse cron expression into field sets.
        
        Args:
            expr: Cron expression.
            
        Returns:
            Dict mapping field names to valid values.
        """
        parts = expr.strip().split()
        
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expr}")
        
        return {
            field: self._parse_field(part, field)
            for field, part in zip(self.FIELDS, parts)
        }
    
    def _parse_field(self, part: str, field: str) -> set[int]:
        """
        Parse a single cron field into a set of valid values.
        
        Args:
            part: Field value (e.g., "*", "*/5", "1-5", "1,3,5").
            field: Field name.
            
        Returns:
            Set of valid integer values.
        """
        ranges = {
            "minute": (0, 59),
            "hour": (0, 23),
            "day": (1, 31),
            "month": (1, 12),
            "weekday": (0, 6),
        }
        min_val, max_val = ranges[field]
        
        # Wildcard
        if part == "*":
            return set(range(min_val, max_val + 1))
        
        # Step (*/n)
        if part.startswith("*/"):
            step = int(part[2:])
            return set(range(min_val, max_val + 1, step))
        
        # Range (a-b)
        if "-" in part and "," not in part:
            start, end = map(int, part.split("-"))
            return set(range(start, end + 1))
        
        # List (a,b,c)
        if "," in part:
            values = set()
            for item in part.split(","):
                if "-" in item:
                    start, end = map(int, item.split("-"))
                    values.update(range(start, end + 1))
                else:
                    values.add(int(item))
            return values
        
        # Single value
        return {int(part)}
    
    def next_run(self, after: Optional[datetime] = None) -> datetime:
        """
        Calculate next run time after given datetime.
        
        Args:
            after: Start time (default: now).
            
        Returns:
            Next run time in UTC.
        """
        if after is None:
            after = datetime.now(timezone.utc)
        
        # Convert to target timezone for matching
        local = after.astimezone(self.tz)
        
        # Start from next minute
        candidate = local.replace(second=0, microsecond=0)
        candidate = candidate + timedelta(minutes=1)
        
        # Find next matching time (max 1 year search)
        for _ in range(525600):  # minutes in a year
            if self._matches(candidate):
                # Convert back to UTC
                return candidate.astimezone(timezone.utc)
            candidate += timedelta(minutes=1)
        
        raise ValueError(f"No valid run time found for {self.expr}")
    
    def _matches(self, dt: datetime) -> bool:
        """
        Check if datetime matches cron expression.
        
        Args:
            dt: Datetime to check.
            
        Returns:
            True if matches.
        """
        return (
            dt.minute in self.parts["minute"] and
            dt.hour in self.parts["hour"] and
            dt.day in self.parts["day"] and
            dt.month in self.parts["month"] and
            dt.weekday() in self.parts["weekday"]
        )
    
    def next_runs(
        self,
        count: int = 5,
        after: Optional[datetime] = None,
    ) -> list[datetime]:
        """
        Calculate next N run times.
        
        Args:
            count: Number of run times to calculate.
            after: Start time (default: now).
            
        Returns:
            List of next run times in UTC.
        """
        runs = []
        current = after
        
        for _ in range(count):
            next_time = self.next_run(current)
            runs.append(next_time)
            current = next_time
        
        return runs


class ScheduleRecovery:
    """
    Missed job recovery for Prospective Memory.
    """
    
    def __init__(self, pm):
        """
        Initialize recovery handler.
        
        Args:
            pm: Prospective Memory instance.
        """
        self.pm = pm
    
    async def recover_missed_jobs(self, max_age_hours: int = 24) -> int:
        """
        Check for and execute missed scheduled jobs.
        
        Called daily by DMN or on startup.
        
        Args:
            max_age_hours: Max age of jobs to recover.
            
        Returns:
            Number of jobs recovered.
        """
        from observability import audit
        
        # Find jobs that were scheduled but not executed
        missed = await self._find_missed_jobs()
        
        audit.log_raw(
            "pm",
            "missed_job_check",
            "scheduler",
            "completed",
            details={"missed_count": len(missed)},
        )
        
        recovered = 0
        
        for job in missed:
            # Check if job is still relevant (not too old)
            scheduled_at = datetime.fromisoformat(job["scheduled_at"])
            age_hours = (datetime.now(timezone.utc) - scheduled_at).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                # Too old, mark as skipped
                await self._mark_skipped(job["id"], reason="too_old")
                continue
            
            # Execute the missed job
            audit.log_raw(
                "pm",
                "missed_job_execute",
                "scheduler",
                "started",
                target=job["task_name"],
            )
            
            try:
                await self.pm.execute_task(job)
                recovered += 1
                
                audit.log_raw(
                    "pm",
                    "missed_job_execute",
                    "scheduler",
                    "completed",
                    target=job["task_name"],
                )
            except Exception as e:
                audit.log_raw(
                    "pm",
                    "missed_job_execute",
                    "scheduler",
                    "failed",
                    target=job["task_name"],
                    error=str(e),
                )
        
        return recovered
    
    async def _find_missed_jobs(self) -> list[dict]:
        """Find jobs that were scheduled but not executed."""
        # Would query PM database
        return []
    
    async def _mark_skipped(self, job_id: str, reason: str) -> None:
        """Mark a job as skipped."""
        # Would update PM database
        pass
