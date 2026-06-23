"""
Emotional Memory (EM)

Tracks user sentiment and communication preferences per interaction.
Not empathy simulation — a professional calibration signal for self-correction.

This module provides:
- SessionSentiment for real-time sentiment tracking
- UserPreferences for learned communication style
- EmotionalMemory manager with SQLite persistence

Note: Sentiment values and preference options are loaded from config.py
to keep the platform domain-agnostic. No hardcoded enums.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SentimentEvent:
    """
    A single sentiment detection event.
    
    Recorded when sentiment is detected or shifts.
    Sentiment values come from config.domain_config.sentiments.
    """
    id: str
    user_id: str
    convo_id: str
    sentiment: str  # From config.domain_config.sentiments
    confidence: float  # 0.0 - 1.0
    trigger: str  # What caused this sentiment
    context_snippet: str  # Brief excerpt of context
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def create(
        cls,
        user_id: str,
        convo_id: str,
        sentiment: str,
        confidence: float,
        trigger: str,
        context_snippet: str,
    ) -> "SentimentEvent":
        """Factory method to create a new sentiment event."""
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            convo_id=convo_id,
            sentiment=sentiment,
            confidence=confidence,
            trigger=trigger,
            context_snippet=context_snippet[:500],  # Truncate long snippets
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "convo_id": self.convo_id,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "trigger": self.trigger,
            "context_snippet": self.context_snippet,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SessionSentiment:
    """
    Real-time sentiment state for an active session.
    
    Kept in memory for immediate response calibration.
    Sentiment values come from config.domain_config.sentiments.
    """
    user_id: str
    convo_id: str
    current_sentiment: str = ""  # From config, defaults in __post_init__
    confidence: float = 0.5
    sentiment_history: list[str] = field(default_factory=list)
    message_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Set default sentiment from config if not provided."""
        if not self.current_sentiment:
            self.current_sentiment = config.domain_config.default_sentiment
    
    def update(self, sentiment: str, confidence: float) -> bool:
        """
        Update sentiment state.
        
        Args:
            sentiment: New sentiment classification.
            confidence: Model confidence in classification.
            
        Returns:
            True if sentiment shifted (triggers flush), False otherwise.
        """
        shifted = sentiment != self.current_sentiment
        
        if shifted:
            self.sentiment_history.append(self.current_sentiment)
        
        self.current_sentiment = sentiment
        self.confidence = confidence
        self.message_count += 1
        self.last_updated = datetime.now(timezone.utc)
        
        return shifted
    
    @property
    def has_been_frustrated(self) -> bool:
        """Check if user was frustrated at any point in session."""
        frustrated = "frustrated"  # Convention: first sentiment is negative
        return (
            self.current_sentiment == frustrated
            or frustrated in self.sentiment_history
        )


@dataclass
class UserPreferences:
    """
    Learned user communication preferences.
    
    One row per user, updated over time.
    Preference values come from config.domain_config.
    """
    user_id: str
    response_length: str = ""  # From config.domain_config.response_lengths
    formality: str = ""  # From config.domain_config.formality_levels
    detail_level: str = ""  # From config.domain_config.detail_levels
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Set defaults from config if not provided."""
        if not self.response_length:
            self.response_length = config.domain_config.default_response_length
        if not self.formality:
            self.formality = config.domain_config.default_formality
        if not self.detail_level:
            self.detail_level = config.domain_config.default_detail_level
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "response_length": self.response_length,
            "formality": self.formality,
            "detail_level": self.detail_level,
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserPreferences":
        return cls(
            user_id=data["user_id"],
            response_length=data["response_length"],
            formality=data["formality"],
            detail_level=data["detail_level"],
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
    
    def to_prompt_hints(self) -> str:
        """Generate hints for LLM based on preferences."""
        hints = []
        
        # Response length hints
        if self.response_length == "short":
            hints.append("Keep responses concise and to the point.")
        elif self.response_length == "detailed":
            hints.append("Provide comprehensive, detailed responses.")
        
        # Formality hints
        if self.formality == "casual":
            hints.append("Use a casual, friendly tone.")
        elif self.formality == "formal":
            hints.append("Use formal, professional language.")
        
        # Detail level hints
        if self.detail_level == "summary":
            hints.append("Focus on high-level summaries.")
        elif self.detail_level == "deep_dive":
            hints.append("Include in-depth analysis and details.")
        
        return " ".join(hints) if hints else ""


# ══════════════════════════════════════════════════════════════════════════════
# Emotional Memory Manager
# ══════════════════════════════════════════════════════════════════════════════

class EmotionalMemory:
    """
    Manager for user sentiment and preferences.
    
    Provides real-time sentiment tracking with SQLite persistence.
    Uses same database as Prospective Memory for easy JOINs.
    
    Attributes:
        db_path: Path to SQLite database.
        flush_threshold: Messages between automatic flushes.
    """
    
    FLUSH_THRESHOLD = 10  # Flush every N messages as fallback
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Emotional Memory.
        
        Args:
            db_path: Path to SQLite database. Uses config default if not provided.
        """
        self.db_path = db_path or str(config.paths.hot_db)
        
        # In-memory state for active sessions
        self._sessions: dict[str, SessionSentiment] = {}
        self._preferences: dict[str, UserPreferences] = {}
        
        # Pending events to flush
        self._pending_events: list[SentimentEvent] = []
        
        self._db_initialized = False
    
    def _session_key(self, user_id: str, convo_id: str) -> str:
        return f"{user_id}:{convo_id}"
    
    # ── Database Operations ──
    
    async def _init_db(self) -> None:
        """Initialize database tables if they don't exist."""
        if self._db_initialized:
            return
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            # Sentiment log table (event stream)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_sentiment_log (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    convo_id TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    trigger TEXT,
                    context_snippet TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # User preferences table (state)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    response_length TEXT DEFAULT 'medium',
                    formality TEXT DEFAULT 'professional',
                    detail_level TEXT DEFAULT 'standard',
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Indexes
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_user_time 
                ON user_sentiment_log(user_id, timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_convo 
                ON user_sentiment_log(convo_id)
            """)
            
            await db.commit()
        
        self._db_initialized = True
    
    async def _flush_events(self) -> None:
        """Flush pending sentiment events to database."""
        if not self._pending_events:
            return
        
        await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            for event in self._pending_events:
                await db.execute("""
                    INSERT INTO user_sentiment_log 
                    (id, user_id, convo_id, sentiment, confidence, trigger, context_snippet, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.id, event.user_id, event.convo_id,
                    event.sentiment, event.confidence,
                    event.trigger, event.context_snippet,
                    event.timestamp.isoformat(),
                ))
            await db.commit()
        
        self._pending_events.clear()
    
    async def _save_preferences(self, prefs: UserPreferences) -> None:
        """Save user preferences to database."""
        await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO user_preferences 
                (user_id, response_length, formality, detail_level, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                prefs.user_id,
                prefs.response_length,
                prefs.formality,
                prefs.detail_level,
                prefs.updated_at.isoformat(),
            ))
            await db.commit()
    
    async def _load_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Load user preferences from database."""
        await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return UserPreferences.from_dict(dict(row))
        
        return None
    
    # ── Public API ──
    
    async def record_sentiment(
        self,
        user_id: str,
        convo_id: str,
        sentiment: str,
        confidence: float,
        trigger: str,
        context_snippet: str = "",
    ) -> bool:
        """
        Record a sentiment observation.
        
        Updates in-memory state and queues event for persistence.
        Flushes immediately on sentiment shift or threshold.
        
        Args:
            user_id: User identifier.
            convo_id: Conversation identifier.
            sentiment: Detected sentiment.
            confidence: Model confidence (0.0-1.0).
            trigger: What caused this sentiment.
            context_snippet: Brief context excerpt.
            
        Returns:
            True if sentiment shifted, False otherwise.
        """
        session_key = self._session_key(user_id, convo_id)
        
        # Get or create session state
        if session_key not in self._sessions:
            self._sessions[session_key] = SessionSentiment(
                user_id=user_id,
                convo_id=convo_id,
            )
        
        session = self._sessions[session_key]
        shifted = session.update(sentiment, confidence)
        
        # Create event
        event = SentimentEvent.create(
            user_id=user_id,
            convo_id=convo_id,
            sentiment=sentiment,
            confidence=confidence,
            trigger=trigger,
            context_snippet=context_snippet,
        )
        self._pending_events.append(event)
        
        # Flush on shift or threshold
        if shifted or session.message_count % self.FLUSH_THRESHOLD == 0:
            await self._flush_events()
        
        return shifted
    
    async def get_session_sentiment(
        self, user_id: str, convo_id: str
    ) -> Optional[SessionSentiment]:
        """
        Get current session sentiment state.
        
        Args:
            user_id: User identifier.
            convo_id: Conversation identifier.
            
        Returns:
            SessionSentiment if session exists, None otherwise.
        """
        session_key = self._session_key(user_id, convo_id)
        return self._sessions.get(session_key)
    
    async def get_preferences(self, user_id: str) -> UserPreferences:
        """
        Get user preferences, loading from DB if needed.
        
        Args:
            user_id: User identifier.
            
        Returns:
            UserPreferences (defaults if not found).
        """
        # Check cache
        if user_id in self._preferences:
            return self._preferences[user_id]
        
        # Load from DB
        prefs = await self._load_preferences(user_id)
        if not prefs:
            prefs = UserPreferences(user_id=user_id)
        
        self._preferences[user_id] = prefs
        return prefs
    
    async def update_preferences(
        self,
        user_id: str,
        response_length: Optional[str] = None,
        formality: Optional[str] = None,
        detail_level: Optional[str] = None,
    ) -> UserPreferences:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier.
            response_length: Preferred response length.
            formality: Preferred formality level.
            detail_level: Preferred detail level.
            
        Returns:
            Updated UserPreferences.
        """
        prefs = await self.get_preferences(user_id)
        
        if response_length:
            prefs.response_length = response_length
        if formality:
            prefs.formality = formality
        if detail_level:
            prefs.detail_level = detail_level
        
        prefs.updated_at = datetime.now(timezone.utc)
        
        await self._save_preferences(prefs)
        self._preferences[user_id] = prefs
        
        return prefs
    
    async def end_session(self, user_id: str, convo_id: str) -> None:
        """
        End a session, flushing all pending data.
        
        Args:
            user_id: User identifier.
            convo_id: Conversation identifier.
        """
        await self._flush_events()
        
        session_key = self._session_key(user_id, convo_id)
        if session_key in self._sessions:
            del self._sessions[session_key]
    
    async def get_sentiment_history(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[SentimentEvent]:
        """
        Get sentiment history for a user.
        
        Args:
            user_id: User identifier.
            limit: Maximum events to return.
            
        Returns:
            List of SentimentEvent, most recent first.
        """
        await self._init_db()
        
        import aiosqlite
        
        events = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM user_sentiment_log 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (user_id, limit)
            ) as cursor:
                async for row in cursor:
                    data = dict(row)
                    events.append(SentimentEvent(
                        id=data["id"],
                        user_id=data["user_id"],
                        convo_id=data["convo_id"],
                        sentiment=data["sentiment"],
                        confidence=data["confidence"],
                        trigger=data["trigger"] or "",
                        context_snippet=data["context_snippet"] or "",
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                    ))
        
        return events
    
    def get_calibration_hints(self, session: SessionSentiment, prefs: UserPreferences) -> str:
        """
        Generate calibration hints for LLM based on sentiment and preferences.
        
        Args:
            session: Current session sentiment.
            prefs: User preferences.
            
        Returns:
            Hints string for system prompt.
        """
        hints = []
        
        # Sentiment-based hints
        if session.current_sentiment == "frustrated":
            hints.append("User seems frustrated. Be extra clear, acknowledge any issues, and focus on solutions.")
        elif session.current_sentiment == "satisfied":
            hints.append("User is satisfied. Maintain current approach.")
        
        # Recovery hint
        if session.has_been_frustrated and session.current_sentiment != "frustrated":
            hints.append("User was previously frustrated but has recovered. Continue carefully.")
        
        # Preference hints
        pref_hints = prefs.to_prompt_hints()
        if pref_hints:
            hints.append(pref_hints)
        
        return " ".join(hints)


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_em_instance: Optional[EmotionalMemory] = None


def get_emotional_memory() -> EmotionalMemory:
    """
    Get the singleton EmotionalMemory instance.
    
    Returns:
        The EmotionalMemory instance.
    """
    global _em_instance
    if _em_instance is None:
        _em_instance = EmotionalMemory()
    return _em_instance


# ── Convenience Functions ──

async def record_sentiment(
    user_id: str,
    convo_id: str,
    sentiment: str,
    confidence: float,
    trigger: str,
    context_snippet: str = "",
) -> bool:
    """Record a sentiment observation."""
    em = get_emotional_memory()
    return await em.record_sentiment(
        user_id, convo_id, sentiment, confidence, trigger, context_snippet
    )


async def get_preferences(user_id: str) -> UserPreferences:
    """Get user preferences."""
    return await get_emotional_memory().get_preferences(user_id)


async def get_calibration_hints(user_id: str, convo_id: str) -> str:
    """Get calibration hints for current session."""
    em = get_emotional_memory()
    session = await em.get_session_sentiment(user_id, convo_id)
    prefs = await em.get_preferences(user_id)
    
    if not session:
        return prefs.to_prompt_hints()
    
    return em.get_calibration_hints(session, prefs)
