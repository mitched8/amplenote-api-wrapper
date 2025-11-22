# AmpleNote API Wrapper

HTTP API wrapper for the AmpleNote MCP Server. This service exposes REST endpoints that connect to the MCP server, making it easy to integrate AmpleNote data with LLM APIs.

## Architecture

```
LLM API → HTTP → This API Wrapper → MCP Protocol → AmpleNote MCP Server → AmpleNote API
```

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the MCP server path (optional):
   - The API wrapper automatically finds your MCP server at `../AmpleNote_MCP/amplenote_mcp_server.py` (sibling directory)
   - Or set the `MCP_SERVER_PATH` environment variable:
   ```bash
   export MCP_SERVER_PATH="/path/to/AmpleNote_MCP/amplenote_mcp_server.py"
   ```

4. Run the API server:
```bash
python3 main.py
# Or: uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /api/notes?days=7` - Get recent notes
- `GET /api/notes/{uuid}` - Get specific note content
- `GET /api/tasks` - Get all tasks
- `GET /api/health` - Health check

## Usage with LLM APIs

Example with OpenAI:
```python
import requests

# Get notes
response = requests.get("http://localhost:8000/api/notes?days=7")
notes = response.json()

# Use in LLM prompt
prompt = f"Summarize these notes: {notes}"
```

