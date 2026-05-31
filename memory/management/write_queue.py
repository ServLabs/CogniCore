"""
Batched Memory Writes (Section 19)

Reduce I/O amplification with write batching.
Flushes on batch size, time interval, or explicit flush.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from audit import audit
from core import log


@dataclass
class WriteOperation:
    """A single write operation to be batched."""
    memory_type: str           # "sfm", "lfm", "am", "mm", "meta"
    operation: str             # "insert", "update", "delete"
    data: dict[str, Any]
    callback: Optional[Callable] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WriteQueue:
    """
    Batches memory writes to reduce I/O operations.
    
    Flushes on:
    - Batch size reached (default: 50)
    - Time elapsed (default: 5 seconds)
    - Explicit flush call
    """
    
    def __init__(
        self,
        mml=None,
        batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
    ):
        """
        Initialize write queue.
        
        Args:
            mml: Memory management layer.
            batch_size: Max operations before auto-flush.
            flush_interval_seconds: Max seconds before auto-flush.
        """
        self.mml = mml
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self._queue: list[WriteOperation] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the background flush task."""
        if self._running:
            return
        
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        log.info("Write queue started")
    
    async def stop(self) -> None:
        """Stop and flush remaining writes."""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        await self.flush()
        log.info("Write queue stopped")
    
    async def enqueue(self, op: WriteOperation) -> None:
        """
        Add a write operation to the queue.
        
        Args:
            op: Write operation to enqueue.
        """
        async with self._lock:
            self._queue.append(op)
            
            if len(self._queue) >= self.batch_size:
                await self._flush_batch()
    
    async def enqueue_write(
        self,
        memory_type: str,
        data: dict[str, Any],
        operation: str = "insert",
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Convenience method to enqueue a write.
        
        Args:
            memory_type: Target memory type.
            data: Data to write.
            operation: Operation type.
            callback: Optional callback after write.
        """
        op = WriteOperation(
            memory_type=memory_type,
            operation=operation,
            data=data,
            callback=callback,
        )
        await self.enqueue(op)
    
    async def flush(self) -> int:
        """
        Force flush all pending writes.
        
        Returns:
            Number of operations flushed.
        """
        async with self._lock:
            return await self._flush_batch()
    
    async def _flush_loop(self) -> None:
        """Background task that flushes periodically."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            async with self._lock:
                if self._queue:
                    await self._flush_batch()
    
    async def _flush_batch(self) -> int:
        """
        Flush current batch to storage.
        
        Returns:
            Number of operations flushed.
        """
        if not self._queue:
            return 0
        
        batch = self._queue.copy()
        self._queue.clear()
        
        # Group by memory type for efficient bulk operations
        by_type: dict[str, list[WriteOperation]] = {}
        for op in batch:
            by_type.setdefault(op.memory_type, []).append(op)
        
        # Execute bulk writes
        for mem_type, ops in by_type.items():
            await self._bulk_write(mem_type, ops)
        
        # Call callbacks
        for op in batch:
            if op.callback:
                try:
                    if asyncio.iscoroutinefunction(op.callback):
                        await op.callback()
                    else:
                        op.callback()
                except Exception as e:
                    log.warning(f"Write callback failed: {e}")
        
        return len(batch)
    
    async def _bulk_write(
        self,
        mem_type: str,
        ops: list[WriteOperation],
    ) -> None:
        """
        Execute bulk write for a memory type.
        
        Args:
            mem_type: Memory type.
            ops: Operations to execute.
        """
        audit.log_raw(
            "memory",
            "bulk_write",
            "write_queue",
            "started",
            target=mem_type,
            details={"count": len(ops)},
        )
        
        try:
            if self.mml is None:
                log.warning(f"MML not set, skipping bulk write for {mem_type}")
                return
            
            # Group by operation type
            inserts = [op.data for op in ops if op.operation == "insert"]
            updates = [op.data for op in ops if op.operation == "update"]
            deletes = [op.data for op in ops if op.operation == "delete"]
            
            # Execute bulk operations
            if inserts:
                if mem_type == "sfm":
                    await self.mml.sfm_bulk_write(inserts)
                elif mem_type == "meta":
                    await self.mml.meta_bulk_register(inserts)
                # Add other memory types as needed
            
            audit.log_raw(
                "memory",
                "bulk_write",
                "write_queue",
                "completed",
                target=mem_type,
                details={"count": len(ops)},
            )
        
        except Exception as e:
            audit.log_raw(
                "memory",
                "bulk_write",
                "write_queue",
                "failed",
                target=mem_type,
                error=str(e),
            )
            log.error(f"Bulk write failed for {mem_type}: {e}")
    
    @property
    def pending_count(self) -> int:
        """Number of pending operations."""
        return len(self._queue)
    
    def get_status(self) -> dict[str, Any]:
        """Get queue status."""
        return {
            "running": self._running,
            "pending": len(self._queue),
            "batch_size": self.batch_size,
            "flush_interval": self.flush_interval,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_write_queue: Optional[WriteQueue] = None


def get_write_queue(mml=None) -> WriteQueue:
    """Get the singleton WriteQueue instance."""
    global _write_queue
    if _write_queue is None:
        _write_queue = WriteQueue(mml=mml)
    return _write_queue
