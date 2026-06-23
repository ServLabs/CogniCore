# ══════════════════════════════════════════════════════════════════════════════
# CogniCore — Production Dockerfile
# ══════════════════════════════════════════════════════════════════════════════
#
# Single container for CogniCore. Redis must be provided separately
# (sidecar container, managed service, or host network).
#
# Build:
#   docker build -t cognicore:latest .
#
# Run:
#   docker run -d \
#     --name cognicore \
#     -p 8765:8765 \
#     -p 8080:8080 \
#     -v cognicore_data:/data \
#     -e AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
#     -e AZURE_OPENAI_KEY="your-key" \
#     -e AZURE_DEPLOYMENT_EXPENSIVE="gpt-4o" \
#     -e AZURE_DEPLOYMENT_CHEAP="gpt-4o-mini" \
#     -e OPENAI_API_KEY="your-openai-key" \
#     -e REDIS_HOST="your-redis-host" \
#     cognicore:latest
#
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.13-slim AS base

# System dependencies for faiss-cpu, numpy, and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash cognicore

# Application directory
WORKDIR /opt/cognicore

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory structure
RUN mkdir -p /data/{sqlite,redis,kuzu/graph,faiss,duckdb,jsonl,md,yaml,models,audit,logs,sandbox_tmp} \
    && chown -R cognicore:cognicore /data /opt/cognicore

# Switch to non-root user
USER cognicore

# ══════════════════════════════════════════════════════════════════════════════
# TIER 3: USER MUST SUPPLY AT RUNTIME (via -e flags or orchestrator secrets)
#
# These are declared but left empty — container will not function without them.
# ══════════════════════════════════════════════════════════════════════════════

# Azure OpenAI (Primary LLM)
ENV AZURE_OPENAI_ENDPOINT=""
ENV AZURE_OPENAI_KEY=""
ENV AZURE_DEPLOYMENT_EXPENSIVE=""
ENV AZURE_DEPLOYMENT_CHEAP=""

# OpenAI (Embeddings)
ENV OPENAI_API_KEY=""

# Snowflake (optional — leave empty if not using)
ENV SNOWFLAKE_ACCOUNT=""
ENV SNOWFLAKE_USER=""
ENV SNOWFLAKE_PASSWORD=""
ENV SNOWFLAKE_WAREHOUSE=""
ENV SNOWFLAKE_DATABASE=""
ENV SNOWFLAKE_SCHEMA="PUBLIC"
ENV SNOWFLAKE_ROLE=""

# Databricks (optional — leave empty if not using)
ENV DATABRICKS_HOST=""
ENV DATABRICKS_TOKEN=""
ENV DATABRICKS_HTTP_PATH=""
ENV DATABRICKS_CATALOG=""
ENV DATABRICKS_SCHEMA=""

# Azure SQL (optional — leave empty if not using)
ENV AZURE_SQL_CONN_STRING=""
ENV AZURE_SQL_SERVER=""
ENV AZURE_SQL_DATABASE=""

# Redis auth
ENV REDIS_PASSWORD=""

# ══════════════════════════════════════════════════════════════════════════════
# TIER 2: PREDEFINED (correct for containerized deployment)
# ══════════════════════════════════════════════════════════════════════════════

# Paths
ENV COGNICORE_DATA_DIR="/data"
ENV COGNICORE_APP_DIR="/opt/cognicore"

# Server ports
ENV WS_HOST="0.0.0.0"
ENV WS_PORT="8765"
ENV REST_HOST="0.0.0.0"
ENV REST_PORT="8080"
ENV METRICS_PORT="9090"

# Redis connection (override REDIS_HOST to point at your Redis instance)
ENV REDIS_HOST="host.docker.internal"
ENV REDIS_PORT="6379"

# Redis DB isolation
ENV REDIS_DB_WM="0"
ENV REDIS_DB_MML="1"
ENV REDIS_DB_CACHE="2"

# LLM provider
ENV LLM_PROVIDER="azure"
ENV AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# Embedding
ENV EMBEDDING_PROVIDER="openai"
ENV OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

# NLI (auto-downloads on first run)
ENV NLI_MODEL="cross-encoder/nli-deberta-v3-small"

# Runtime
ENV COGNICORE_ENV="production"
ENV COGNICORE_DEBUG="false"
ENV LOG_LEVEL="INFO"
ENV LOG_TO_CONSOLE="true"

# Proxy bypass
ENV NO_PROXY="*"
ENV no_proxy="*"

# ══════════════════════════════════════════════════════════════════════════════
# TIER 1: OPTIONAL (uncomment/override at runtime if needed)
#
#   LLM_DAILY_BUDGET=10.0
#   LLM_MONTHLY_BUDGET=200.0
#   LLM_MAX_TOKENS=4096
#   COGNICORE_WORKER_POOL_SIZE=30
#   COGNICORE_IDLE_THRESHOLD=300
#   COGNICORE_MAX_LLM_CALLS_HOUR=1000
#   COGNICORE_MAX_LLM_COST_DAY=50.0
#   COGNICORE_MAX_REQUESTS_MINUTE=60
#   SANDBOX_PYTHON_TIMEOUT=30
#   SANDBOX_SQL_TIMEOUT=60
#   EMBEDDING_BATCH_SIZE=64
#   LOG_REDACT_PII=true
#
# ══════════════════════════════════════════════════════════════════════════════

# Expose ports
EXPOSE 8765 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/admin/health || exit 1

# Entrypoint
CMD ["python3", "run.py"]
