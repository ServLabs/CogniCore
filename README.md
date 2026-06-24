# CogniCore

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/ServLabs/CogniCore)](https://github.com/ServLabs/CogniCore/issues)
[![GitHub Stars](https://img.shields.io/github/stars/ServLabs/CogniCore)](https://github.com/ServLabs/CogniCore/stargazers)

A cognitive architecture for AI agents with human-like memory systems.

**[Documentation](https://github.com/ServLabs/CogniCore/wiki)** · **[Report Bug](https://github.com/ServLabs/CogniCore/issues/new?template=bug_report.md)** · **[Request Feature](https://github.com/ServLabs/CogniCore/issues/new?template=feature_request.md)**

## Quick Start

### Prerequisites

- Python 3.11+ (3.13 recommended)
- Redis 7+ (required — Working Memory, MML staging, analytics cache)

### Installation

```bash
# Clone the repository
git clone https://github.com/ServLabs/CogniCore.git
cd CogniCore

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

CogniCore uses a **3-tier environment variable system** (see [LOCAL_SETUP.md](LOCAL_SETUP.md) for full details):

```bash
# Required: LLM credentials (Tier 3)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_KEY="your-key"
export AZURE_DEPLOYMENT_EXPENSIVE="gpt-4o"
export AZURE_DEPLOYMENT_CHEAP="gpt-4o-mini"
export OPENAI_API_KEY="your-openai-key"   # For embeddings

# Predefined: Ports, paths, Redis (Tier 2 — usually no changes needed)
export COGNICORE_WS_PORT=8765
export COGNICORE_REST_PORT=8080
export COGNICORE_DATA_DIR=./.local_data
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379

# Redis DB isolation
export REDIS_DB_WM=0      # Working Memory
export REDIS_DB_MML=1     # MML staging
export REDIS_DB_CACHE=2   # Analytics cache (60s TTL)
```

The recommended way to start is via `./start.sh` which sets everything automatically.

### Running

```bash
# Start the server
python run.py
```

This starts two servers:
- **WebSocket** (`ws://localhost:8765/ws`) - Chat/agent conversations
- **REST API** (`http://localhost:8080`) - Ingestion, metrics, evals, admin

### Usage

**WebSocket Chat:**
```javascript
const ws = new WebSocket('ws://localhost:8765/ws?user_id=user123');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.type, data.content);
};

ws.send(JSON.stringify({
    type: 'message',
    text: 'What is my spending this month?'
}));
```

**REST API:**
```bash
# Ingest text
curl -X POST http://localhost:8080/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers detailed explanations", "source": "preferences"}'

# Check health
curl http://localhost:8080/admin/health

# Get metrics
curl http://localhost:8080/metrics/summary
```

---

## Architecture

CogniCore implements a cognitive architecture inspired by human memory systems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CogniCore                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ interfaces/ │  │  response/  │  │  control/   │  │observability│        │
│  │ chat/admin  │  │  Pipeline   │  │  CEN/DMN    │  │analytics/eval│       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                         │
│  ┌────────────────────────────────┴────────────────────────────────────┐   │
│  │                     Memory Management Layer (MML)                    │   │
│  │   Recall Engine │ Budget Manager │ Consolidation │ Learning         │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│  ┌────────────────────────────────┴────────────────────────────────────┐   │
│  │                          Memory Layer                                │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │   │
│  │  │ ABM │ │ WM  │ │ PM  │ │ EM  │ │ AM  │ │ MM  │ │ SFM │ │ LFM │   │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Connectors                                   │   │
│  │   LLM │ Embeddings │ NLI │ REST/File │ Sandbox         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Types

| Memory | Purpose | Storage |
|--------|---------|---------|
| **ABM** | Autobiographical Memory - Agent identity | Immutable config |
| **WM** | Working Memory - Active conversation | Redis + JSONL |
| **PM** | Prospective Memory - Scheduled tasks | SQLite |
| **EM** | Emotional Memory - User sentiment | SQLite |
| **AM** | Associative Memory - Knowledge graph | Kuzu |
| **MM** | Motor Memory - Learned procedures | YAML + SQLite |
| **SFM** | Short-Form Memory - Compressed facts | SQLite + FAISS |
| **LFM** | Long-Form Memory - Raw documents | SQLite + FAISS |

### Project Structure

```
CogniCore/
├── run.py                # Entry point - start servers
├── config.py             # 3-tier configuration (env vars → dataclasses)
├── start.sh              # Full startup script (env + Redis + app)
├── Dockerfile            # Production container
├── interfaces/           # External APIs
│   ├── admin/            # System administration
│   ├── service/          # Developer APIs (ingestion, metrics, evals, traces)
│   ├── chat/             # WebSocket chat + streaming
│   └── scheduled/        # Background task triggers
├── memory/               # 9 memory types + MML
│   ├── types/            # ABM, WM, PM, EM, AM, MM, SFM, LFM, Meta
│   └── management/       # MML, recall, budget, learning, maintenance
├── response/             # Gate, pipeline, thinking, decision, synthesis, sub-agents
├── control/              # Salience Network, CEN, DMN, Governor
├── connectors/           # AI (LLM, embeddings, NLI), data, sandbox
├── observability/        # Full telemetry stack
│   ├── audit.py          # JSONL append-only audit logging
│   ├── tracing.py        # Distributed tracing (traces + spans → DuckDB)
│   ├── analytics/        # DuckDB metrics + Redis-cached reads
│   └── evals/            # Evaluation system (5 built-in eval types)
└── prompts/              # LLM prompt templates (Jinja2 markdown)
```

---

## API Reference

### WebSocket Events

**Client → Server:**
```json
{"type": "message", "text": "Your question here"}
{"type": "ping"}
```

**Server → Client:**
```json
{"type": "thinking", "content": "Analyzing..."}
{"type": "memory", "content": "Found 3 relevant facts", "source": "sfm"}
{"type": "response_chunk", "content": "Here is your answer..."}
{"type": "done", "metadata": {"processing_time_ms": 150}}
```

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/text` | Ingest text content |
| POST | `/ingest/file` | Upload and ingest file |
| POST | `/ingest/facts` | Ingest structured facts |
| GET | `/metrics/summary` | System metrics summary |
| GET | `/metrics/memory` | Memory usage stats |
| GET | `/metrics/timeseries` | Time-series metric data |
| GET | `/metrics/traces` | List recent traces |
| GET | `/metrics/traces/{id}` | Get trace with spans |
| GET | `/metrics/latency` | Latency percentiles |
| GET | `/metrics/costs` | LLM cost breakdown |
| GET | `/metrics/throughput` | Request throughput |
| GET | `/metrics/errors` | Error rates |
| GET | `/metrics/pipeline/stats` | Pipeline execution stats |
| GET | `/evals` | List evaluation results |
| GET | `/evals/types` | Available eval types |
| GET | `/evals/summary` | Eval score summary |
| GET | `/evals/trend` | Eval scores over time |
| POST | `/evals/run` | Run single evaluation |
| POST | `/evals/run-all` | Run all evaluations |
| GET | `/admin/health` | Health check |
| GET | `/admin/config` | Current configuration |
| POST | `/admin/maintenance` | Trigger maintenance |
| GET | `/scheduled/tasks` | List background tasks |
| POST | `/scheduled/consolidation` | Trigger memory consolidation |
| POST | `/scheduled/learning/*` | Trigger learning algorithms |

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format
black .
isort .

# Lint
ruff check .
mypy .
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Security

To report security vulnerabilities, please see [SECURITY.md](SECURITY.md).

---

Made with ❤️ by [ServLabs](https://github.com/ServLabs)
