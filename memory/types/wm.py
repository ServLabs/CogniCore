"""
Working Memory (WM)

Active conversation context — the agent's "RAM" for ongoing interactions.
Stores current conversations with tiered storage (Redis hot + local cold).

This module provides:
- Message and Conversation dataclasses
- WorkingMemory manager with Redis + local file fallback
- Token-aware windowing with rolling summarization
- Full audit trail via JSONL files
"""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class MessageRole(Enum):
    """Role of a message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """
    A single message in a conversation.
    
    Attributes:
        role: Who sent the message (user/assistant/system).
        content: The message text.
        timestamp: When the message was sent.
        token_count: Estimated token count for this message.
        metadata: Optional additional data (tool calls, etc.).
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Estimate token count if not provided."""
        if self.token_count == 0:
            # Rough estimate: ~4 chars per token
            self.token_count = len(self.content) // 4 + 1
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Deserialize from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
        )
    
    def to_llm_format(self) -> dict[str, str]:
        """Convert to LLM API format."""
        return {
            "role": self.role.value,
            "content": self.content,
        }


@dataclass
class ConversationSummary:
    """
    Rolling summary of a conversation.
    
    Replaces older messages to preserve context without token bloat.
    """
    content: str
    message_count: int  # How many messages were summarized
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = len(self.content) // 4 + 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSummary":
        return cls(
            content=data["content"],
            message_count=data["message_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
            token_count=data.get("token_count", 0),
        )


@dataclass
class Conversation:
    """
    A conversation with a user.
    
    Contains recent messages and optional summary of older context.
    """
    user_id: str
    conversation_id: str
    messages: list[Message] = field(default_factory=list)
    summary: Optional[ConversationSummary] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def total_tokens(self) -> int:
        """Total token count of messages + summary."""
        msg_tokens = sum(m.token_count for m in self.messages)
        summary_tokens = self.summary.token_count if self.summary else 0
        return msg_tokens + summary_tokens
    
    def to_llm_messages(self) -> list[dict[str, str]]:
        """
        Convert to LLM API format.
        
        Returns summary (if exists) as system message, followed by recent messages.
        """
        result = []
        
        if self.summary:
            result.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{self.summary.content}",
            })
        
        for msg in self.messages:
            result.append(msg.to_llm_format())
        
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Working Memory Manager
# ══════════════════════════════════════════════════════════════════════════════

class WorkingMemory:
    """
    Manager for active conversation context.
    
    Provides tiered storage with Redis (hot) and local files (cold).
    Handles token-aware windowing and rolling summarization.
    
    Attributes:
        redis_client: Redis connection (optional, falls back to local).
        data_dir: Directory for local file storage.
        max_tokens: Maximum tokens before summarization triggers.
        summarizer: Async function to generate summaries.
    """
    
    # Redis key prefixes
    MESSAGES_KEY = "wm:{uid}:{cid}:messages"
    SUMMARY_KEY = "wm:{uid}:{cid}:summary"
    TTL_SECONDS = 24 * 60 * 60  # 24 hours
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        data_dir: Optional[Path] = None,
        max_tokens: int = 4000,
        summarizer: Optional[Callable[[str, list[Message]], Awaitable[str]]] = None,
    ):
        """
        Initialize Working Memory.
        
        Args:
            redis_client: Redis async client. If None, uses local-only mode.
            data_dir: Directory for local storage. Uses config default if not provided.
            max_tokens: Token threshold before summarization.
            summarizer: Async function(previous_summary, messages) -> new_summary.
        """
        self.redis = redis_client
        self.data_dir = data_dir or config.paths.jsonl_dir / "conversations"
        self.max_tokens = max_tokens
        self.summarizer = summarizer
        
        # In-memory cache for active conversations
        self._conversations: dict[str, Conversation] = {}
        
        # Fallback mode tracking
        self._redis_available = redis_client is not None
        self._last_redis_check = datetime.now(timezone.utc)
    
    # ── Key Generation ──
    
    def _messages_key(self, user_id: str, convo_id: str) -> str:
        return self.MESSAGES_KEY.format(uid=user_id, cid=convo_id)
    
    def _summary_key(self, user_id: str, convo_id: str) -> str:
        return self.SUMMARY_KEY.format(uid=user_id, cid=convo_id)
    
    def _cache_key(self, user_id: str, convo_id: str) -> str:
        return f"{user_id}:{convo_id}"
    
    # ── Local File Operations ──
    
    def _convo_dir(self, user_id: str, convo_id: str) -> Path:
        """Get directory for a conversation's local files."""
        path = self.data_dir / user_id / convo_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _messages_file(self, user_id: str, convo_id: str) -> Path:
        return self._convo_dir(user_id, convo_id) / "messages.jsonl"
    
    def _summary_file(self, user_id: str, convo_id: str) -> Path:
        return self._convo_dir(user_id, convo_id) / "summary.json"
    
    async def _append_to_jsonl(self, user_id: str, convo_id: str, message: Message) -> None:
        """Append a message to the local JSONL file (audit trail)."""
        file_path = self._messages_file(user_id, convo_id)
        line = json.dumps(message.to_dict()) + "\n"
        
        # Use async file I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: file_path.open("a").write(line))
    
    async def _save_summary_local(self, user_id: str, convo_id: str, summary: ConversationSummary) -> None:
        """Save summary to local JSON file."""
        file_path = self._summary_file(user_id, convo_id)
        content = json.dumps(summary.to_dict(), indent=2)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: file_path.write_text(content))
    
    async def _load_summary_local(self, user_id: str, convo_id: str) -> Optional[ConversationSummary]:
        """Load summary from local JSON file."""
        file_path = self._summary_file(user_id, convo_id)
        
        if not file_path.exists():
            return None
        
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, file_path.read_text)
        data = json.loads(content)
        return ConversationSummary.from_dict(data)
    
    async def _load_recent_messages_local(
        self, user_id: str, convo_id: str, limit: int = 50
    ) -> list[Message]:
        """Load recent messages from local JSONL file."""
        file_path = self._messages_file(user_id, convo_id)
        
        if not file_path.exists():
            return []
        
        loop = asyncio.get_event_loop()
        lines = await loop.run_in_executor(None, lambda: file_path.read_text().strip().split("\n"))
        
        # Get last N lines
        recent_lines = lines[-limit:] if len(lines) > limit else lines
        
        messages = []
        for line in recent_lines:
            if line:
                data = json.loads(line)
                messages.append(Message.from_dict(data))
        
        return messages
    
    # ── Redis Operations ──
    
    async def _redis_available_check(self) -> bool:
        """Check if Redis is available, with rate limiting."""
        if not self.redis:
            return False
        
        # Don't check too frequently
        now = datetime.now(timezone.utc)
        if (now - self._last_redis_check).total_seconds() < config.redis.fallback_reconnect_interval:
            return self._redis_available
        
        self._last_redis_check = now
        
        try:
            await self.redis.ping()
            self._redis_available = True
        except Exception:
            self._redis_available = False
        
        return self._redis_available
    
    async def _push_message_redis(self, user_id: str, convo_id: str, message: Message) -> None:
        """Push a message to Redis list."""
        if not await self._redis_available_check():
            return
        
        key = self._messages_key(user_id, convo_id)
        try:
            await self.redis.rpush(key, json.dumps(message.to_dict()))
            await self.redis.expire(key, self.TTL_SECONDS)
        except Exception:
            self._redis_available = False
    
    async def _get_messages_redis(self, user_id: str, convo_id: str) -> list[Message]:
        """Get all messages from Redis list."""
        if not await self._redis_available_check():
            return []
        
        key = self._messages_key(user_id, convo_id)
        try:
            data = await self.redis.lrange(key, 0, -1)
            await self.redis.expire(key, self.TTL_SECONDS)  # Reset TTL on read
            return [Message.from_dict(json.loads(item)) for item in data]
        except Exception:
            self._redis_available = False
            return []
    
    async def _trim_messages_redis(self, user_id: str, convo_id: str, keep_count: int) -> None:
        """Trim Redis list to keep only the most recent messages."""
        if not await self._redis_available_check():
            return
        
        key = self._messages_key(user_id, convo_id)
        try:
            # Keep only the last `keep_count` messages
            await self.redis.ltrim(key, -keep_count, -1)
        except Exception:
            self._redis_available = False
    
    async def _save_summary_redis(self, user_id: str, convo_id: str, summary: ConversationSummary) -> None:
        """Save summary to Redis."""
        if not await self._redis_available_check():
            return
        
        key = self._summary_key(user_id, convo_id)
        try:
            await self.redis.set(key, json.dumps(summary.to_dict()), ex=self.TTL_SECONDS)
        except Exception:
            self._redis_available = False
    
    async def _get_summary_redis(self, user_id: str, convo_id: str) -> Optional[ConversationSummary]:
        """Get summary from Redis."""
        if not await self._redis_available_check():
            return None
        
        key = self._summary_key(user_id, convo_id)
        try:
            data = await self.redis.get(key)
            if data:
                await self.redis.expire(key, self.TTL_SECONDS)  # Reset TTL
                return ConversationSummary.from_dict(json.loads(data))
        except Exception:
            self._redis_available = False
        
        return None
    
    # ── Public API ──
    
    async def add_message(
        self,
        user_id: str,
        convo_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to a conversation.
        
        Persists to both Redis (hot) and local file (cold).
        Triggers summarization if token threshold exceeded.
        
        Args:
            user_id: User identifier.
            convo_id: Conversation identifier.
            role: Message role (user/assistant/system).
            content: Message content.
            metadata: Optional additional data.
            
        Returns:
            The created Message.
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        
        # Always persist to local file (audit trail)
        await self._append_to_jsonl(user_id, convo_id, message)
        
        # Push to Redis
        await self._push_message_redis(user_id, convo_id, message)
        
        # Update in-memory cache
        cache_key = self._cache_key(user_id, convo_id)
        if cache_key not in self._conversations:
            self._conversations[cache_key] = Conversation(
                user_id=user_id,
                conversation_id=convo_id,
            )
        
        convo = self._conversations[cache_key]
        convo.messages.append(message)
        convo.updated_at = datetime.now(timezone.utc)
        
        # Check if summarization needed
        if convo.total_tokens > self.max_tokens:
            await self._summarize(convo)
        
        return message
    
    async def _summarize(self, convo: Conversation) -> None:
        """
        Perform rolling summarization.
        
        Takes oldest messages + previous summary → new summary.
        Trims oldest messages from Redis and in-memory.
        """
        if not self.summarizer:
            # No summarizer configured - just trim oldest messages
            keep_count = len(convo.messages) // 2
            convo.messages = convo.messages[-keep_count:]
            await self._trim_messages_redis(convo.user_id, convo.conversation_id, keep_count)
            return
        
        # Determine how many messages to summarize
        target_tokens = self.max_tokens // 2
        tokens_to_remove = convo.total_tokens - target_tokens
        
        messages_to_summarize = []
        tokens_removed = 0
        
        for msg in convo.messages:
            if tokens_removed >= tokens_to_remove:
                break
            messages_to_summarize.append(msg)
            tokens_removed += msg.token_count
        
        if not messages_to_summarize:
            return
        
        # Generate new summary
        previous_summary = convo.summary.content if convo.summary else ""
        new_summary_text = await self.summarizer(previous_summary, messages_to_summarize)
        
        new_summary = ConversationSummary(
            content=new_summary_text,
            message_count=(convo.summary.message_count if convo.summary else 0) + len(messages_to_summarize),
        )
        
        # Update conversation
        convo.summary = new_summary
        convo.messages = convo.messages[len(messages_to_summarize):]
        
        # Persist summary to both Redis and local
        await self._save_summary_redis(convo.user_id, convo.conversation_id, new_summary)
        await self._save_summary_local(convo.user_id, convo.conversation_id, new_summary)
        
        # Trim Redis messages
        await self._trim_messages_redis(
            convo.user_id, convo.conversation_id, len(convo.messages)
        )
    
    async def get_conversation(self, user_id: str, convo_id: str) -> Conversation:
        """
        Get a conversation, loading from storage if needed.
        
        Tries Redis first, falls back to local files.
        
        Args:
            user_id: User identifier.
            convo_id: Conversation identifier.
            
        Returns:
            The Conversation object.
        """
        cache_key = self._cache_key(user_id, convo_id)
        
        # Check in-memory cache
        if cache_key in self._conversations:
            return self._conversations[cache_key]
        
        # Try Redis
        messages = await self._get_messages_redis(user_id, convo_id)
        summary = await self._get_summary_redis(user_id, convo_id)
        
        # If Redis miss, load from local
        if not messages and not summary:
            messages = await self._load_recent_messages_local(user_id, convo_id)
            summary = await self._load_summary_local(user_id, convo_id)
            
            # Repopulate Redis if we loaded from local
            if messages and self._redis_available:
                for msg in messages:
                    await self._push_message_redis(user_id, convo_id, msg)
                if summary:
                    await self._save_summary_redis(user_id, convo_id, summary)
        
        convo = Conversation(
            user_id=user_id,
            conversation_id=convo_id,
            messages=messages,
            summary=summary,
        )
        
        self._conversations[cache_key] = convo
        return convo
    
    async def get_context(self, user_id: str, convo_id: str) -> list[dict[str, str]]:
        """
        Get conversation context in LLM-ready format.
        
        Returns summary (if exists) + recent messages.
        
        Args:
            user_id: User identifier.
            convo_id: Conversation identifier.
            
        Returns:
            List of message dicts for LLM API.
        """
        convo = await self.get_conversation(user_id, convo_id)
        return convo.to_llm_messages()
    
    async def clear_conversation(self, user_id: str, convo_id: str) -> None:
        """
        Clear a conversation from memory (keeps local audit trail).
        
        Args:
            user_id: User identifier.
            convo_id: Conversation identifier.
        """
        cache_key = self._cache_key(user_id, convo_id)
        
        # Remove from in-memory cache
        if cache_key in self._conversations:
            del self._conversations[cache_key]
        
        # Remove from Redis
        if await self._redis_available_check():
            try:
                await self.redis.delete(
                    self._messages_key(user_id, convo_id),
                    self._summary_key(user_id, convo_id),
                )
            except Exception:
                pass
        
        # Note: Local files are NOT deleted (audit trail)
    
    async def list_conversations(self, user_id: str) -> list[str]:
        """
        List all conversation IDs for a user.
        
        Args:
            user_id: User identifier.
            
        Returns:
            List of conversation IDs.
        """
        user_dir = self.data_dir / user_id
        if not user_dir.exists():
            return []
        
        return [d.name for d in user_dir.iterdir() if d.is_dir()]


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_wm_instance: Optional[WorkingMemory] = None


def get_working_memory() -> WorkingMemory:
    """
    Get the singleton WorkingMemory instance.
    
    Returns:
        The WorkingMemory instance.
    """
    global _wm_instance
    if _wm_instance is None:
        _wm_instance = WorkingMemory()
    return _wm_instance


# ── Convenience Functions ──

async def add_user_message(user_id: str, convo_id: str, content: str) -> Message:
    """Add a user message to a conversation."""
    wm = get_working_memory()
    return await wm.add_message(user_id, convo_id, MessageRole.USER, content)


async def add_assistant_message(user_id: str, convo_id: str, content: str) -> Message:
    """Add an assistant message to a conversation."""
    wm = get_working_memory()
    return await wm.add_message(user_id, convo_id, MessageRole.ASSISTANT, content)


async def get_context(user_id: str, convo_id: str) -> list[dict[str, str]]:
    """Get conversation context for LLM."""
    return await get_working_memory().get_context(user_id, convo_id)
