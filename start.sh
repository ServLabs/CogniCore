#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# CogniCore — Full Startup Script
# ══════════════════════════════════════════════════════════════════════════════
#
# This script:
#   1. Sets all environment variables
#   2. Bypasses proxy settings
#   3. Creates data directories
#   4. Activates Python venv & installs dependencies
#   5. Starts Redis
#   6. Starts CogniCore (WebSocket + REST API)
#
# Prerequisites:
#   - Python 3.11+ with venv module
#   - Redis installed (brew install redis / apt install redis-server)
#   - Internet access for first-run dependency install
#
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PROXY BYPASS
# ══════════════════════════════════════════════════════════════════════════════
# Unset all proxy environment variables so all traffic stays local/direct.
# If you ARE behind a corporate proxy and need external LLM/embedding calls
# to go through it, comment out this section and set proxies explicitly.

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy 2>/dev/null || true
export NO_PROXY="*"
export no_proxy="*"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ENVIRONMENT VARIABLES — USER MUST FILL (Tier 3)
# ══════════════════════════════════════════════════════════════════════════════
# These have NO defaults. The agent WILL NOT function without them.
# Fill in your values below.

# ── Azure OpenAI (Primary LLM) ──────────────────────────────────────────────
export AZURE_OPENAI_ENDPOINT=""           # e.g. https://your-resource.openai.azure.com/
export AZURE_OPENAI_KEY=""                # Your Azure OpenAI API key
export AZURE_DEPLOYMENT_EXPENSIVE=""      # e.g. gpt-4o
export AZURE_DEPLOYMENT_CHEAP=""          # e.g. gpt-4o-mini

# ── OpenAI (Used for embeddings) ────────────────────────────────────────────
export OPENAI_API_KEY=""                  # Required for text-embedding-3-small

# ── Snowflake (Data connector — leave empty if not using) ────────────────────
export SNOWFLAKE_ACCOUNT=""               # e.g. xy12345.us-east-1
export SNOWFLAKE_USER=""
export SNOWFLAKE_PASSWORD=""
export SNOWFLAKE_WAREHOUSE=""             # e.g. COMPUTE_WH
export SNOWFLAKE_DATABASE=""
export SNOWFLAKE_SCHEMA="PUBLIC"
export SNOWFLAKE_ROLE=""

# ── Databricks (Data connector — leave empty if not using) ───────────────────
export DATABRICKS_HOST=""                 # e.g. https://adb-123.azuredatabricks.net
export DATABRICKS_TOKEN=""
export DATABRICKS_HTTP_PATH=""            # e.g. /sql/1.0/warehouses/abc123
export DATABRICKS_CATALOG=""
export DATABRICKS_SCHEMA=""

# ── Azure SQL (Data connector — leave empty if not using) ────────────────────
export AZURE_SQL_CONN_STRING=""           # Full ODBC connection string (preferred)
export AZURE_SQL_SERVER=""                # e.g. your-server.database.windows.net
export AZURE_SQL_DATABASE=""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ENVIRONMENT VARIABLES — PREDEFINED (Tier 2)
# ══════════════════════════════════════════════════════════════════════════════
# These are important and required, but we already know the correct values
# since everything runs on the same host. Modify only if you change the setup.

# ── Paths ────────────────────────────────────────────────────────────────────
export COGNICORE_DATA_DIR="$SCRIPT_DIR/.local_data"
export COGNICORE_APP_DIR="$SCRIPT_DIR"

# ── Server Ports (all on localhost) ──────────────────────────────────────────
export WS_HOST="0.0.0.0"
export WS_PORT="8765"
export REST_HOST="0.0.0.0"
export REST_PORT="8080"
export METRICS_PORT="9090"

# ── Redis (assumes running on same host) ─────────────────────────────────────
export REDIS_HOST="127.0.0.1"
export REDIS_PORT="6379"
export REDIS_PASSWORD=""

# Redis DB isolation (each subsystem gets its own logical database)
export REDIS_DB_WM="0"                   # Working Memory — hot conversations
export REDIS_DB_MML="1"                  # MML — insight/proposal staging
export REDIS_DB_CACHE="2"                # General cache (future use)

# ── LLM Provider Selection ───────────────────────────────────────────────────
export LLM_PROVIDER="azure"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# ── Embedding (OpenAI API) ───────────────────────────────────────────────────
export EMBEDDING_PROVIDER="openai"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

# ── NLI Model (local, auto-downloads ~100MB on first run) ────────────────────
export NLI_MODEL="cross-encoder/nli-deberta-v3-small"

# ── Runtime ──────────────────────────────────────────────────────────────────
export COGNICORE_ENV="development"
export COGNICORE_DEBUG="true"
export LOG_LEVEL="INFO"

# ══════════════════════════════════════════════════════════════════════════════
# NOTE: Tier 1 variables (optional customization) are NOT listed here.
# They live in config.py with sensible defaults. Add them to this script
# only if you want to override. Examples:
#
#   export LLM_DAILY_BUDGET="10.0"
#   export LLM_MONTHLY_BUDGET="200.0"
#   export LLM_MAX_TOKENS="4096"
#   export COGNICORE_WORKER_POOL_SIZE="30"
#   export COGNICORE_IDLE_THRESHOLD="300"
#   export COGNICORE_MAX_LLM_CALLS_HOUR="1000"
#   export COGNICORE_MAX_LLM_COST_DAY="50.0"
#   export COGNICORE_MAX_REQUESTS_MINUTE="60"
#   export COGNICORE_MAX_CLARIFICATION_ROUNDS="3"
#   export COGNICORE_SUB_AGENT_TIMEOUT="60"
#   export COGNICORE_DEFAULT_MODEL_TIER="default"
#   export SANDBOX_PYTHON_TIMEOUT="30"
#   export SANDBOX_SQL_TIMEOUT="60"
#   export SANDBOX_MAX_OUTPUT="65536"
#   export EMBEDDING_BATCH_SIZE="64"
#   export NLI_CONTRADICTION_THRESHOLD="0.7"
#   export NLI_ENTAILMENT_THRESHOLD="0.7"
#   export LOG_FORMAT="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
#   export LOG_TO_CONSOLE="true"
#   export LOG_MAX_FILE_SIZE_MB="10"
#   export LOG_BACKUP_COUNT="5"
#   export LOG_REDACT_PII="true"
#   export AZURE_SQL_DRIVER="ODBC Driver 18 for SQL Server"
#
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

echo "══════════════════════════════════════════════════════════════"
echo " CogniCore Startup"
echo "══════════════════════════════════════════════════════════════"

# Check required Tier 3 variables are filled
MISSING=()
[[ -z "$AZURE_OPENAI_ENDPOINT" ]] && MISSING+=("AZURE_OPENAI_ENDPOINT")
[[ -z "$AZURE_OPENAI_KEY" ]] && MISSING+=("AZURE_OPENAI_KEY")
[[ -z "$AZURE_DEPLOYMENT_EXPENSIVE" ]] && MISSING+=("AZURE_DEPLOYMENT_EXPENSIVE")
[[ -z "$AZURE_DEPLOYMENT_CHEAP" ]] && MISSING+=("AZURE_DEPLOYMENT_CHEAP")
[[ -z "$OPENAI_API_KEY" ]] && MISSING+=("OPENAI_API_KEY")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: The following required variables are empty:"
    for var in "${MISSING[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Edit start.sh Section 2 and fill in your credentials."
    exit 1
fi

echo "[✓] Required credentials configured"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: PYTHON VIRTUAL ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "[✓] Python venv activated: $(python3 --version)"

# Install/update dependencies
echo "[*] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "[✓] Dependencies installed"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DATA DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════════

echo "[*] Creating data directories..."
mkdir -p "$COGNICORE_DATA_DIR"/{sqlite,redis,kuzu/graph,faiss,duckdb,jsonl,md,yaml,models,audit,logs,sandbox_tmp}
echo "[✓] Data directories ready at $COGNICORE_DATA_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: START REDIS
# ══════════════════════════════════════════════════════════════════════════════
# Redis must be installed on this machine.
#   macOS:  brew install redis
#   Linux:  sudo apt install redis-server
#   Windows: Use WSL or Docker
#
# The command below starts Redis as a background daemon on the configured port.
# If Redis is already running (e.g., via systemd/launchd), comment this out.

echo "[*] Starting Redis on port $REDIS_PORT..."
redis-server \
    --port "$REDIS_PORT" \
    --daemonize yes \
    --dir "$COGNICORE_DATA_DIR/redis" \
    --dbfilename cognicore.rdb \
    --save 60 1 \
    --loglevel warning \
    --bind 127.0.0.1 \
    --databases 16

# Verify Redis is up
if redis-cli -p "$REDIS_PORT" ping | grep -q PONG; then
    echo "[✓] Redis running on 127.0.0.1:$REDIS_PORT (DBs: WM=$REDIS_DB_WM, MML=$REDIS_DB_MML, Cache=$REDIS_DB_CACHE)"
else
    echo "[!] Redis failed to start — agent will use local fallback (degraded WM performance)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: START COGNICORE
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════════════════════"
echo " Launching CogniCore Agent"
echo "══════════════════════════════════════════════════════════════"
echo "  WebSocket Chat: ws://localhost:$WS_PORT/ws"
echo "  REST API:       http://localhost:$REST_PORT"
echo "  Admin:          http://localhost:$REST_PORT/admin/health"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 run.py
