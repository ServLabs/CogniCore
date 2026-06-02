# CogniCore

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/ServLabs/CogniCore)](https://github.com/ServLabs/CogniCore/issues)
[![GitHub Stars](https://img.shields.io/github/stars/ServLabs/CogniCore)](https://github.com/ServLabs/CogniCore/stargazers)

A cognitive architecture for AI agents with human-like memory systems.

**[Documentation](https://github.com/ServLabs/CogniCore/wiki)** · **[Report Bug](https://github.com/ServLabs/CogniCore/issues/new?template=bug_report.md)** · **[Request Feature](https://github.com/ServLabs/CogniCore/issues/new?template=feature_request.md)**

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (optional, for hot memory caching)
- SQLite (included with Python)

### Installation

```bash
# Clone the repository
git clone https://github.com/ServLabs/CogniCore.git
cd CogniCore

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set environment variables or use defaults:

```bash
# Optional: Configure ports and paths
export COGNICORE_WS_PORT=8765        # WebSocket server port
export COGNICORE_REST_PORT=8080      # REST API port
export COGNICORE_DATA_DIR=/datadrive # Data directory
export COGNICORE_DEBUG=false         # Debug mode

# Optional: Configure LLM
export OPENAI_API_KEY=your-key-here
export COGNICORE_LLM_PROVIDER=openai  # or "anthropic", "local"
```

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
│  │   LLM │ Embeddings │ NLI │ Snowflake │ Databricks │ Sandbox         │   │
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
├── core/                 # Infrastructure (config, logger, audit, errors, migrations)
├── interfaces/           # External APIs
│   ├── admin/            # System administration
│   ├── service/          # Developer APIs (ingestion, metrics, evals)
│   ├── chat/             # WebSocket chat
│   └── scheduled/        # Background task triggers
├── memory/               # 9 memory types + MML
│   ├── types/            # ABM, WM, PM, EM, AM, MM, SFM, LFM, Meta
│   └── management/       # MML, recall, budget, learning
├── response/             # Gate, pipeline, thinking, decision, synthesis
├── control/              # Salience Network, CEN, DMN, Governor
├── connectors/           # AI, data, sandbox connectors
├── observability/        # Monitoring
│   ├── analytics/        # DuckDB metrics
│   └── evals/            # Evaluation system
└── prompts/              # LLM prompt templates
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
| GET | `/evals` | List evaluation results |
| POST | `/evals/run` | Trigger evaluation run |
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
