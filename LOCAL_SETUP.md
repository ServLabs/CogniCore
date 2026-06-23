# Local Setup

Get CogniCore running on your machine.

## Prerequisites

- Python 3.11+ (3.13 recommended)
- Redis 7+ (required — used for Working Memory, MML staging, and analytics cache)

## Setup

```bash
# Clone and enter directory
git clone https://github.com/ServLabs/CogniCore.git
cd CogniCore

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The easiest way is to use `start.sh` which sets all environment variables for you:

```bash
# Edit start.sh — fill in your API keys in Section 2 (Tier 3 vars)
vim start.sh

# Make executable and run
chmod +x start.sh
./start.sh
```

### Environment Variable Tiers

CogniCore uses a 3-tier environment variable system:

| Tier | Purpose | Where |
|------|---------|-------|
| **Tier 3** (required) | API keys, credentials | Must fill in `start.sh` Section 2 |
| **Tier 2** (predefined) | Ports, paths, Redis config | Pre-set in `start.sh` Section 3 |
| **Tier 1** (optional) | Tuning knobs, limits | Defaults in `config.py`, override if needed |

### Minimum Required Variables

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_KEY="your-key"
export AZURE_DEPLOYMENT_EXPENSIVE="gpt-4o"
export AZURE_DEPLOYMENT_CHEAP="gpt-4o-mini"
export OPENAI_API_KEY="your-openai-key"   # For embeddings
```

### Redis DB Isolation

CogniCore uses 3 separate Redis logical databases:

| DB | Purpose | Env Var |
|----|---------|---------|
| 0 | Working Memory (hot conversations) | `REDIS_DB_WM` |
| 1 | MML staging (insight proposals) | `REDIS_DB_MML` |
| 2 | Analytics cache (60s TTL) | `REDIS_DB_CACHE` |

### Data Directory

All persistent data is stored under `COGNICORE_DATA_DIR` (defaults to `.local_data/` in dev):

```
.local_data/
├── sqlite/       # Episodic, procedural, prospective memory
├── faiss/        # Vector indices (SFM, LFM)
├── kuzu/graph/   # Associative memory graph
├── duckdb/       # Analytics metrics + eval results
├── audit/        # JSONL audit logs (daily rotation)
├── jsonl/        # Working memory backup
├── md/           # ABM identity files
├── yaml/         # Motor memory procedures
├── models/       # NLI model cache
├── logs/         # Application logs
└── sandbox_tmp/  # Ephemeral sandbox execution
```

## Run

### Using start.sh (recommended)

```bash
./start.sh
```

This handles Redis startup, directory creation, and environment setup automatically.

### Manual

```bash
# Start Redis
redis-server --daemonize yes --databases 16

# Set env vars (at minimum)
export COGNICORE_DATA_DIR="$PWD/.local_data"
export REDIS_HOST="127.0.0.1"
export REDIS_PORT="6379"

# Run
python run.py
```

You should see:
```
CogniCore starting...
  WebSocket: ws://localhost:8765/ws
  REST API:  http://localhost:8080
```

## Verify

```bash
# Health check
curl http://localhost:8080/admin/health

# Metrics dashboard
curl http://localhost:8080/metrics/summary

# Pipeline stats
curl http://localhost:8080/metrics/pipeline/stats

# Run evals
curl -X POST http://localhost:8080/evals/run-all
```

## Docker

```bash
# Build
docker build -t cognicore:latest .

# Run (provide Redis externally)
docker run -d \
  --name cognicore \
  -p 8765:8765 \
  -p 8080:8080 \
  -v cognicore_data:/data \
  -e AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
  -e AZURE_OPENAI_KEY="your-key" \
  -e AZURE_DEPLOYMENT_EXPENSIVE="gpt-4o" \
  -e AZURE_DEPLOYMENT_CHEAP="gpt-4o-mini" \
  -e OPENAI_API_KEY="your-openai-key" \
  -e REDIS_HOST="host.docker.internal" \
  cognicore:latest
```

## Troubleshooting

**Port already in use?**
```bash
export WS_PORT=8766
export REST_PORT=8081
```

**No API keys?** The agent will start but LLM calls will fail. You can still test the API structure and memory systems.

**Redis not running?** CogniCore requires Redis. Install with `brew install redis` (macOS) or `apt install redis-server` (Linux).

**`/datadrive` permission error?** Set `COGNICORE_DATA_DIR` to a writable path:
```bash
export COGNICORE_DATA_DIR="$PWD/.local_data"
```

**NLI model download slow?** The NLI model (~100MB) downloads on first run. Ensure internet access or pre-download:
```bash
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-small')"
```
