# Local Setup

Get CogniCore running on your machine.

## Prerequisites

- Python 3.11+
- Redis (optional, falls back to local files)

## Setup

```bash
# Clone and enter directory
git clone https://github.com/your-org/cognicore.git
cd cognicore

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data/sqlite data/faiss data/redis data/audit
```

## Configuration

Create a `.env` file or export these:

```bash
export OPENAI_API_KEY=your-key-here
export COGNICORE_DATA_DIR=./data
export COGNICORE_DEBUG=true
```

## Run

```bash
python run.py
```

You should see:
```
CogniCore starting...
  WebSocket: ws://localhost:8765/ws
  REST API:  http://localhost:8080
```

## Test it

```bash
# Health check
curl http://localhost:8080/admin/health

# Or connect via WebSocket and send a message
```

## Troubleshooting

**Port already in use?**
```bash
export COGNICORE_WS_PORT=8766
export COGNICORE_REST_PORT=8081
```

**No OpenAI key?** The agent will start but LLM calls will fail. You can still test the API structure.

**Redis not running?** That's fine - it falls back to local file storage automatically.
