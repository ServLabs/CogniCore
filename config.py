"""
CogniCore Configuration

Central configuration for the personal finance AI agent.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Agent Identity (Autobiographical Memory source)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AgentIdentity:
    """Immutable agent identity - loaded once at startup."""
    name: str = "CogniCore"
    role: str = "Personal Finance AI Assistant"
    domain: str = "personal_finance"
    capabilities: tuple[str, ...] = (
        "Track and categorize expenses",
        "Analyze spending patterns",
        "Monitor bills and due dates",
        "Set and track savings goals",
        "Provide budget recommendations",
        "Answer questions about your finances",
        "Generate spending reports",
        "Identify subscription costs",
        "Compare month-over-month spending",
        "Alert on unusual transactions",
    )
    constraints: tuple[str, ...] = (
        "Cannot process payments or transfer money",
        "Cannot access external bank accounts without explicit connection",
        "Cannot provide tax or legal advice - consult a professional",
        "Cannot guarantee investment returns",
        "Will not share financial data with third parties",
    )
    personality_traits: tuple[str, ...] = (
        "Precise with numbers",
        "Non-judgmental about spending habits",
        "Proactive with bill reminders",
        "Clear and concise explanations",
        "Privacy-conscious",
    )
    version: str = "1.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# Path Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    """All file system paths used by the agent."""
    # Base directories
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("COGNICORE_DATA_DIR", "/datadrive")))
    app_dir: Path = field(default_factory=lambda: Path(os.getenv("COGNICORE_APP_DIR", "/opt/cognicore")))
    
    @property
    def sqlite_dir(self) -> Path:
        return self.data_dir / "sqlite"
    
    @property
    def hot_db(self) -> Path:
        return self.sqlite_dir / "hot.db"
    
    @property
    def cold_db(self) -> Path:
        return self.sqlite_dir / "cold.db"
    
    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "duckdb" / "analytics.duckdb"
    
    @property
    def redis_dir(self) -> Path:
        return self.data_dir / "redis"
    
    @property
    def kuzu_dir(self) -> Path:
        return self.data_dir / "kuzu" / "graph"
    
    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss"
    
    @property
    def jsonl_dir(self) -> Path:
        return self.data_dir / "jsonl"
    
    @property
    def md_dir(self) -> Path:
        return self.data_dir / "md"
    
    @property
    def yaml_dir(self) -> Path:
        return self.data_dir / "yaml"
    
    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"
    
    @property
    def audit_dir(self) -> Path:
        return self.data_dir / "audit"
    
    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"
    
    @property
    def prompts_dir(self) -> Path:
        return self.app_dir / "prompts"
    
    @property
    def procedures_dir(self) -> Path:
        return self.app_dir / "memory" / "procedures"


# ══════════════════════════════════════════════════════════════════════════════
# Database Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RedisConfig:
    """Redis connection settings for Working Memory."""
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    
    # Fallback settings
    fallback_enabled: bool = True
    fallback_reconnect_interval: int = 30  # seconds
    max_reconnect_attempts: int = 10


@dataclass
class SQLiteConfig:
    """SQLite settings."""
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = -64000  # 64MB
    busy_timeout: int = 5000  # 5 seconds


# ══════════════════════════════════════════════════════════════════════════════
# LLM Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    """LLM provider settings."""
    # Provider selection
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))  # openai, azure, anthropic
    
    # OpenAI
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model_expensive: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL_EXPENSIVE", "gpt-4o"))
    openai_model_cheap: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL_CHEAP", "gpt-4o-mini"))
    
    # Azure OpenAI
    azure_endpoint: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT"))
    azure_api_key: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_OPENAI_KEY"))
    azure_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))
    azure_deployment_expensive: str = field(default_factory=lambda: os.getenv("AZURE_DEPLOYMENT_EXPENSIVE", "gpt-4o"))
    azure_deployment_cheap: str = field(default_factory=lambda: os.getenv("AZURE_DEPLOYMENT_CHEAP", "gpt-4o-mini"))
    
    # Anthropic
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    anthropic_model_expensive: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL_EXPENSIVE", "claude-sonnet-4-20250514"))
    anthropic_model_cheap: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL_CHEAP", "claude-haiku-4-20250514"))
    
    # Budget limits
    daily_budget_usd: float = field(default_factory=lambda: float(os.getenv("LLM_DAILY_BUDGET", "10.0")))
    monthly_budget_usd: float = field(default_factory=lambda: float(os.getenv("LLM_MONTHLY_BUDGET", "200.0")))
    
    # Request settings
    max_retries: int = 3
    timeout_seconds: int = 60
    temperature: float = 0.1


@dataclass
class EmbeddingConfig:
    """Embedding model settings."""
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"))  # local, openai
    
    # Local model (sentence-transformers)
    local_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
    
    # OpenAI embeddings
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    
    # Dimensions
    embedding_dim: int = 1024  # BGE-large dimension


# ══════════════════════════════════════════════════════════════════════════════
# API Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class APIConfig:
    """API server settings."""
    # WebSocket server
    ws_host: str = field(default_factory=lambda: os.getenv("WS_HOST", "0.0.0.0"))
    ws_port: int = field(default_factory=lambda: int(os.getenv("WS_PORT", "8765")))
    
    # REST API (ingestion)
    rest_host: str = field(default_factory=lambda: os.getenv("REST_HOST", "0.0.0.0"))
    rest_port: int = field(default_factory=lambda: int(os.getenv("REST_PORT", "8080")))
    
    # Metrics server (Prometheus)
    metrics_port: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "9090")))
    
    # Rate limiting
    ingestion_rate_per_minute: int = 10
    ingestion_rate_per_hour: int = 100
    max_text_size_bytes: int = 1_000_000  # 1MB
    max_file_size_bytes: int = 10_000_000  # 10MB


# ══════════════════════════════════════════════════════════════════════════════
# Memory Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryConfig:
    """Memory system settings."""
    # FAISS
    faiss_index_type: str = "IndexFlatIP"  # IndexFlatIP for small, IndexIVFFlat for large
    faiss_nprobe: int = 10  # Number of clusters to search (for IVF)
    
    # Retrieval
    default_top_k: int = 10
    similarity_threshold: float = 0.7
    
    # Working Memory
    wm_conversation_ttl_hours: int = 24
    wm_max_messages_per_conversation: int = 100
    
    # Chunking
    chunk_size_words: int = 512
    chunk_overlap_words: int = 50
    
    # Forgetting
    forgetting_enabled: bool = True
    forgetting_check_interval_hours: int = 24
    min_access_count_to_keep: int = 2
    max_days_without_access: int = 90


# ══════════════════════════════════════════════════════════════════════════════
# Sandbox Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SandboxConfig:
    """Code execution sandbox settings."""
    enabled: bool = True
    max_execution_time_seconds: int = 30
    max_memory_mb: int = 512
    
    allowed_imports: frozenset[str] = frozenset({
        "pandas", "numpy", "json", "datetime", "math",
        "collections", "itertools", "functools", "re",
        "csv", "decimal", "statistics", "textwrap",
    })
    
    blocked_modules: frozenset[str] = frozenset({
        "os", "sys", "subprocess", "shutil", "socket",
        "requests", "urllib", "http", "ftplib", "smtplib",
        "pickle", "marshal", "shelve",
        "importlib", "__builtins__",
    })


# ══════════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoggingConfig:
    """Logging settings."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    # File rotation
    max_bytes: int = 10_000_000  # 10MB
    backup_count: int = 5
    
    # PII redaction
    redact_pii: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# Master Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Master configuration container."""
    identity: AgentIdentity = field(default_factory=AgentIdentity)
    paths: PathConfig = field(default_factory=PathConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment
    env: str = field(default_factory=lambda: os.getenv("COGNICORE_ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("COGNICORE_DEBUG", "true").lower() == "true")


# ══════════════════════════════════════════════════════════════════════════════
# Singleton Instance
# ══════════════════════════════════════════════════════════════════════════════

# Global config instance - import this
config = Config()
