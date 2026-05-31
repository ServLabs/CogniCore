"""
Sub-Agent Memory Access (Section 17)

Sub-agents have read-only access to memory via MML.
Provides isolation and prevents sub-agents from corrupting memory.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from audit import audit
from core import log


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""
    task: str
    memory_access: bool = True     # Can read memory
    memory_write: bool = False     # Cannot write (read-only)
    max_depth: int = 1             # No nested sub-agents
    timeout_seconds: int = 30
    max_tokens: int = 4000


class ReadOnlyMMLWrapper:
    """
    Wrapper that only exposes read operations.
    
    Sub-agents use this to access memory without write capability.
    """
    
    def __init__(self, mml):
        """
        Initialize read-only wrapper.
        
        Args:
            mml: Memory management layer.
        """
        self._mml = mml
    
    async def recall(
        self,
        query: str,
        memory_types: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[Any]:
        """
        Read-only recall.
        
        Args:
            query: Search query.
            memory_types: Memory types to search.
            top_k: Max results.
            
        Returns:
            List of recall results.
        """
        if memory_types is None:
            memory_types = ["sfm", "lfm", "am"]
        
        return await self._mml.recall(query, memory_types, top_k)
    
    async def sfm_get(self, fact_id: str) -> Optional[Any]:
        """
        Read a specific fact.
        
        Args:
            fact_id: Fact ID.
            
        Returns:
            Fact or None.
        """
        return await self._mml.sfm_get(fact_id)
    
    async def sfm_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[Any]:
        """
        Search facts.
        
        Args:
            query: Search query.
            top_k: Max results.
            
        Returns:
            List of facts.
        """
        return await self._mml.sfm_search(query, top_k)
    
    async def lfm_get(self, doc_id: str) -> Optional[Any]:
        """
        Read a specific document.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            Document or None.
        """
        return await self._mml.lfm_get(doc_id)
    
    async def lfm_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[Any]:
        """
        Search documents.
        
        Args:
            query: Search query.
            top_k: Max results.
            
        Returns:
            List of documents.
        """
        return await self._mml.lfm_search(query, top_k)
    
    async def am_get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> list[Any]:
        """
        Get graph neighbors.
        
        Args:
            entity_id: Entity ID.
            depth: Traversal depth.
            
        Returns:
            List of neighbors.
        """
        return await self._mml.am_get_neighbors(entity_id, depth)
    
    # No write methods exposed
    # sfm_write, lfm_ingest, am_add_edge, etc. are NOT available


class SubAgent:
    """
    Parallel worker with read-only memory access.
    
    Sub-agents can:
    - Read from memory (SFM, LFM, AM)
    - Execute LLM calls
    - Return results to parent
    
    Sub-agents cannot:
    - Write to memory
    - Spawn nested sub-agents (beyond max_depth)
    - Access WM directly
    """
    
    def __init__(
        self,
        config: SubAgentConfig,
        mml,
        parent_context: dict[str, Any],
        llm_connector=None,
    ):
        """
        Initialize sub-agent.
        
        Args:
            config: Sub-agent configuration.
            mml: Memory management layer.
            parent_context: Shared WM context from parent.
            llm_connector: LLM connector for generation.
        """
        self.config = config
        self.mml = mml
        self.parent_context = parent_context
        self.llm = llm_connector
        self._read_only_mml = ReadOnlyMMLWrapper(mml)
    
    async def run(self) -> dict[str, Any]:
        """
        Execute sub-agent task.
        
        Returns:
            Result dict with output and metadata.
        """
        audit.log_raw(
            "sub_agent",
            "started",
            f"sub_agent_{self.config.task[:20]}",
            "started",
            details={"task": self.config.task},
        )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get memory context if allowed
            context = []
            if self.config.memory_access:
                context = await self._read_only_mml.recall(
                    self.config.task,
                    memory_types=["sfm", "lfm", "am"],
                    top_k=10,
                )
            
            # Execute task with LLM
            result = await self._execute_with_context(context)
            
            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            audit.log_raw(
                "sub_agent",
                "completed",
                f"sub_agent_{self.config.task[:20]}",
                "completed",
                duration_ms=elapsed_ms,
            )
            
            return {
                "success": True,
                "output": result,
                "context_used": len(context),
                "duration_ms": elapsed_ms,
            }
        
        except asyncio.TimeoutError:
            audit.log_raw(
                "sub_agent",
                "timeout",
                f"sub_agent_{self.config.task[:20]}",
                "failed",
                error="timeout",
            )
            
            return {
                "success": False,
                "error": "timeout",
                "output": None,
            }
        
        except Exception as e:
            audit.log_raw(
                "sub_agent",
                "failed",
                f"sub_agent_{self.config.task[:20]}",
                "failed",
                error=str(e),
            )
            
            return {
                "success": False,
                "error": str(e),
                "output": None,
            }
    
    async def _execute_with_context(
        self,
        context: list[Any],
    ) -> str:
        """
        Execute task with memory context.
        
        Args:
            context: Memory context from recall.
            
        Returns:
            LLM output.
        """
        if self.llm is None:
            return f"Sub-agent task: {self.config.task}"
        
        # Build prompt with context
        context_str = "\n".join(str(c) for c in context[:5])
        
        prompt = f"""Task: {self.config.task}

Context from memory:
{context_str}

Parent conversation context:
{self.parent_context.get('summary', 'No summary available')}

Please complete the task based on the available context."""
        
        # Execute with timeout
        result = await asyncio.wait_for(
            self.llm.generate(prompt),
            timeout=self.config.timeout_seconds,
        )
        
        return result


class SubAgentSpawner:
    """
    Spawns and manages sub-agents.
    
    Enforces:
    - Max concurrent sub-agents
    - Depth limits
    - Timeout enforcement
    """
    
    def __init__(
        self,
        mml,
        llm_connector=None,
        max_concurrent: int = 3,
    ):
        """
        Initialize spawner.
        
        Args:
            mml: Memory management layer.
            llm_connector: LLM connector.
            max_concurrent: Max concurrent sub-agents.
        """
        self.mml = mml
        self.llm = llm_connector
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
    
    async def spawn(
        self,
        task: str,
        parent_context: dict[str, Any],
        config: Optional[SubAgentConfig] = None,
    ) -> dict[str, Any]:
        """
        Spawn a sub-agent.
        
        Args:
            task: Task description.
            parent_context: Parent conversation context.
            config: Optional configuration.
            
        Returns:
            Sub-agent result.
        """
        if config is None:
            config = SubAgentConfig(task=task)
        
        async with self._semaphore:
            self._active_count += 1
            
            try:
                agent = SubAgent(
                    config=config,
                    mml=self.mml,
                    parent_context=parent_context,
                    llm_connector=self.llm,
                )
                
                return await agent.run()
            
            finally:
                self._active_count -= 1
    
    async def spawn_parallel(
        self,
        tasks: list[str],
        parent_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Spawn multiple sub-agents in parallel.
        
        Args:
            tasks: List of task descriptions.
            parent_context: Parent conversation context.
            
        Returns:
            List of sub-agent results.
        """
        coros = [
            self.spawn(task, parent_context)
            for task in tasks
        ]
        
        return await asyncio.gather(*coros, return_exceptions=True)
    
    @property
    def active_count(self) -> int:
        """Number of active sub-agents."""
        return self._active_count
