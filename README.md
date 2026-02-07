# AmpleNote API Wrapper

HTTP API wrapper for the AmpleNote MCP Server. Exposes REST endpoints that connect to the MCP server, making it easy to integrate AmpleNote data with LLM APIs like OpenAI.

## Architecture

```
Amplenote Chat UI ─► This API Wrapper (port 8001) ─► MCP Server (persistent session) ─► Amplenote API
```

## Prerequisites

- Python 3.11+
- The **AmpleNote_MCP** server (sibling directory) must be set up with a valid `.env`
- An OpenAI API key for chat and summarization features

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

4. Run the API server:
```bash
python3 main.py
```

The API will be available at `http://localhost:8001`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/notes?days=7` | Get recent notes |
| `GET` | `/api/notes/{uuid}` | Get specific note content |
| `GET` | `/api/tasks` | Get all tasks |
| `POST` | `/api/chat` | General chat (no note context) |
| `POST` | `/api/chat-with-context` | Chat with note context |
| `POST` | `/api/summarize-notes` | Summarize recent notes |

## Key Design Decisions

- **Persistent MCP connection** — The wrapper maintains a single long-lived MCP session instead of spawning a new subprocess per request
- **Auto-reconnect** — If the MCP session drops, the next request will automatically reconnect
- **Auth passthrough** — 401 errors from the MCP server are forwarded to the client with the auth URL

## Usage with curl

```bash
# Health check
curl http://localhost:8001/api/health

# Get recent notes
curl http://localhost:8001/api/notes?days=7

# General chat
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'

# Chat with note context
curl -X POST http://localhost:8001/api/chat-with-context \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize my recent notes", "days": 7}'
```
