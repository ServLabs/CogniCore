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
# Domain Configuration (Platform-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DomainConfig:
    """
    Configurable domain-specific values.
    
    Change these to adapt CogniCore to any domain without code changes.
    """
    # Knowledge domains (e.g., for personal finance: budgeting, investments, bills)
    domains: tuple[str, ...] = (
        "budgeting",
        "investments", 
        "bills",
        "savings",
        "taxes",
        "general",
    )
    
    # Entity types in knowledge graph
    entity_types: tuple[str, ...] = (
        "concept",
        "dataset",
        "column",
        "user",
        "task",
        "error",
        "account",
        "category",
        "merchant",
    )
    
    # Relationship types between entities
    relation_types: tuple[str, ...] = (
        "RELATED_TO",
        "CAUSED_BY",
        "CONTAINS",
        "ASKED_ABOUT",
        "BELONGS_TO",
        "SIMILAR_TO",
    )
    
    # Task types for scheduled tasks
    task_types: tuple[str, ...] = (
        "data_pull",
        "report",
        "alert",
        "sync",
        "cleanup",
    )
    
    # Sentiment triggers (what causes sentiment changes)
    sentiment_triggers: tuple[str, ...] = (
        "task_failure",
        "slow_response",
        "good_result",
        "confusion",
        "repeated_question",
    )
    
    # Sentiment values
    sentiments: tuple[str, ...] = (
        "frustrated",
        "neutral",
        "satisfied",
    )
    
    # User preference options
    response_lengths: tuple[str, ...] = (
        "short",
        "medium",
        "detailed",
    )
    
    formality_levels: tuple[str, ...] = (
        "casual",
        "professional",
        "formal",
    )
    
    detail_levels: tuple[str, ...] = (
        "summary",
        "standard",
        "deep_dive",
    )
    
    @property
    def default_domain(self) -> str:
        """Default domain when none specified."""
        return self.domains[-1] if self.domains else "general"
    
    @property
    def default_sentiment(self) -> str:
        """Default sentiment."""
        return "neutral"
    
    @property
    def default_response_length(self) -> str:
        """Default response length preference."""
        return "medium"
    
    @property
    def default_formality(self) -> str:
        """Default formality level."""
        return "professional"
    
    @property
    def default_detail_level(self) -> str:
        """Default detail level."""
        return "standard"


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
    
    @property
    def sandbox_tmp(self) -> Path:
        return self.data_dir / "sandbox_tmp"
    
    @property
    def analytics_db(self) -> Path:
        return self.duckdb_path


# ══════════════════════════════════════════════════════════════════════════════
# Database Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RedisConfig:
    """Redis connection settings — separate DBs for isolation."""
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    
    # Separate Redis DBs for logical isolation (0-15 available)
    db_working_memory: int = field(default_factory=lambda: int(os.getenv("REDIS_DB_WM", "0")))
    db_mml_staging: int = field(default_factory=lambda: int(os.getenv("REDIS_DB_MML", "1")))
    db_cache: int = field(default_factory=lambda: int(os.getenv("REDIS_DB_CACHE", "2")))
    
    # Fallback settings
    fallback_enabled: bool = True
    fallback_reconnect_interval: int = 30  # seconds
    max_reconnect_attempts: int = 10
    
    def url(self, db: int) -> str:
        """Build Redis URL for a specific DB."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{db}"


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
    
    # Cost estimation (USD per 1K output characters)
    cost_per_1k_chars_cheap: float = field(default_factory=lambda: float(os.getenv("LLM_COST_1K_CHARS_CHEAP", "0.001")))
    cost_per_1k_chars_default: float = field(default_factory=lambda: float(os.getenv("LLM_COST_1K_CHARS_DEFAULT", "0.01")))
    cost_per_1k_chars_expensive: float = field(default_factory=lambda: float(os.getenv("LLM_COST_1K_CHARS_EXPENSIVE", "0.03")))

    @property
    def api_key(self) -> Optional[str]:
        """Active API key for selected provider."""
        match self.provider:
            case "openai" | "local":
                return self.openai_api_key
            case "azure":
                return self.azure_api_key
            case "anthropic":
                return self.anthropic_api_key
            case _:
                return self.openai_api_key

    @property
    def api_base(self) -> Optional[str]:
        """Active API base URL for selected provider."""
        match self.provider:
            case "azure":
                return self.azure_endpoint
            case "local":
                return os.getenv("LLM_LOCAL_BASE_URL", "http://localhost:11434/v1")
            case _:
                return None

    @property
    def model(self) -> str:
        """Active expensive model for selected provider."""
        match self.provider:
            case "openai" | "local":
                return self.openai_model_expensive
            case "azure":
                return self.azure_deployment_expensive
            case "anthropic":
                return self.anthropic_model_expensive
            case _:
                return self.openai_model_expensive

    @property
    def cheap_model(self) -> str:
        """Active cheap model for selected provider."""
        match self.provider:
            case "openai" | "local":
                return self.openai_model_cheap
            case "azure":
                return self.azure_deployment_cheap
            case "anthropic":
                return self.anthropic_model_cheap
            case _:
                return self.openai_model_cheap

    @property
    def max_tokens(self) -> int:
        """Maximum tokens per response."""
        return int(os.getenv("LLM_MAX_TOKENS", "4096"))


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

    # Batch
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    @property
    def model_name(self) -> str:
        """Active model name for the configured provider."""
        return self.openai_model if self.use_api else self.local_model

    @property
    def use_api(self) -> bool:
        """True if using API-based embeddings."""
        return self.provider == "openai"

    @property
    def api_key(self) -> Optional[str]:
        """API key for OpenAI embeddings."""
        return os.getenv("OPENAI_API_KEY")

    @property
    def model_path(self) -> Optional[str]:
        """Local model path override."""
        return os.getenv("EMBEDDING_MODEL_PATH")

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.embedding_dim


# ══════════════════════════════════════════════════════════════════════════════
# API Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NLIConfig:
    """NLI (Natural Language Inference) model settings."""
    model_name: str = field(default_factory=lambda: os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small"))
    model_path: Optional[str] = field(default_factory=lambda: os.getenv("NLI_MODEL_PATH"))
    contradiction_threshold: float = float(os.getenv("NLI_CONTRADICTION_THRESHOLD", "0.7"))
    entailment_threshold: float = float(os.getenv("NLI_ENTAILMENT_THRESHOLD", "0.7"))


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
    python_timeout_seconds: int = int(os.getenv("SANDBOX_PYTHON_TIMEOUT", "30"))
    sql_timeout_seconds: int = int(os.getenv("SANDBOX_SQL_TIMEOUT", "60"))
    max_memory_mb: int = 512
    max_output_bytes: int = int(os.getenv("SANDBOX_MAX_OUTPUT", "65536"))
    
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
    name: str = field(default_factory=lambda: os.getenv("LOG_NAME", "cognicore"))
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "DEBUG"))
    format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    log_to_console: bool = field(default_factory=lambda: os.getenv("LOG_TO_CONSOLE", "true").lower() == "true")
    # File rotation
    max_file_size_mb: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_FILE_SIZE_MB", "10")))
    backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")))
    # PII redaction
    redact_pii: bool = field(default_factory=lambda: os.getenv("LOG_REDACT_PII", "true").lower() == "true")


@dataclass
class PipelineConfig:
    """Response pipeline configuration."""
    max_clarification_rounds: int = field(default_factory=lambda: int(os.getenv("COGNICORE_MAX_CLARIFICATION_ROUNDS", "3")))
    sub_agent_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("COGNICORE_SUB_AGENT_TIMEOUT", "60")))
    default_model_tier: str = field(default_factory=lambda: os.getenv("COGNICORE_DEFAULT_MODEL_TIER", "default"))
    

@dataclass
class ControlConfig:
    """Control layer configuration."""
    # Worker pool
    worker_pool_size: int = int(os.getenv("COGNICORE_WORKER_POOL_SIZE", "30"))
    
    # Mode detection
    idle_threshold_seconds: int = int(os.getenv("COGNICORE_IDLE_THRESHOLD", "300"))
    mode_check_interval_seconds: int = int(os.getenv("COGNICORE_MODE_CHECK_INTERVAL", "60"))
    
    # Governor defaults
    max_llm_calls_per_hour: int = int(os.getenv("COGNICORE_MAX_LLM_CALLS_HOUR", "1000"))
    max_llm_cost_per_day_usd: float = float(os.getenv("COGNICORE_MAX_LLM_COST_DAY", "50.0"))
    max_requests_per_minute: int = int(os.getenv("COGNICORE_MAX_REQUESTS_MINUTE", "60"))


# ══════════════════════════════════════════════════════════════════════════════
# Master Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Master configuration container."""
    # Domain configuration (platform-agnostic)
    domain_config: DomainConfig = field(default_factory=DomainConfig)
    
    # Agent identity
    identity: AgentIdentity = field(default_factory=AgentIdentity)
    
    # Infrastructure
    paths: PathConfig = field(default_factory=PathConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)
    api: APIConfig = field(default_factory=APIConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    
    # Environment
    env: str = field(default_factory=lambda: os.getenv("COGNICORE_ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("COGNICORE_DEBUG", "true").lower() == "true")
    
    # ── Convenience accessors ──
    
    @property
    def domains(self) -> tuple[str, ...]:
        """Get configured domains."""
        return self.domain_config.domains
    
    @property
    def entity_types(self) -> tuple[str, ...]:
        """Get configured entity types."""
        return self.domain_config.entity_types
    
    @property
    def relation_types(self) -> tuple[str, ...]:
        """Get configured relation types."""
        return self.domain_config.relation_types
    
    def is_valid_domain(self, domain: str) -> bool:
        """Check if a domain is valid."""
        return domain in self.domain_config.domains
    
    def is_valid_entity_type(self, entity_type: str) -> bool:
        """Check if an entity type is valid."""
        return entity_type in self.domain_config.entity_types
    
    def is_valid_relation_type(self, relation_type: str) -> bool:
        """Check if a relation type is valid."""
        return relation_type in self.domain_config.relation_types


# ══════════════════════════════════════════════════════════════════════════════
# Singleton Instance
# ══════════════════════════════════════════════════════════════════════════════

# Global config instance - import this
config = Config()
