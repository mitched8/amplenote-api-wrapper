#!/usr/bin/env python3
"""
AmpleNote API Wrapper
HTTP API that wraps the AmpleNote MCP Server for LLM integration
"""

import os
import json
import asyncio
import logging
import traceback
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("amplenote-api-wrapper")

# Configuration
MCP_SERVER_PATH = os.getenv(
    "MCP_SERVER_PATH",
    str(Path(__file__).parent.parent / "AmpleNote_MCP" / "amplenote_mcp_server.py")
)
API_PORT = int(os.getenv("API_PORT", "8001"))

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Persistent MCP Connection Manager
# ---------------------------------------------------------------------------
class MCPConnectionManager:
    """Manages a persistent connection to the MCP server subprocess."""

    def __init__(self, server_path: str):
        self._server_path = server_path
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._lock = asyncio.Lock()

    def _build_server_params(self) -> StdioServerParameters:
        mcp_server_dir = Path(self._server_path).parent
        mcp_venv_python = mcp_server_dir / "venv" / "bin" / "python3"
        python_cmd = str(mcp_venv_python) if mcp_venv_python.exists() else "python3"
        return StdioServerParameters(
            command=python_cmd,
            args=[self._server_path],
            env=None,
            cwd=str(mcp_server_dir),
        )

    async def connect(self) -> None:
        """Establish a persistent MCP session."""
        if self._session is not None:
            return

        logger.info("Connecting to MCP server at %s", self._server_path)
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        params = self._build_server_params()
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        logger.info("MCP session established")

    async def disconnect(self) -> None:
        """Tear down the MCP session and subprocess."""
        if self._exit_stack:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Error during MCP disconnect: %s", exc)
            finally:
                self._exit_stack = None
                self._session = None
        logger.info("MCP session closed")

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool, reconnecting if the session is stale."""
        async with self._lock:
            # Ensure connected
            if self._session is None:
                await self.connect()

            try:
                result = await self._session.call_tool(tool_name, arguments)
                response = (
                    json.loads(result.content[0].text) if result.content else {"error": "No content"}
                )
            except Exception as exc:
                logger.error("MCP call failed (%s), reconnecting: %s", tool_name, exc)
                await self.disconnect()
                # One retry with a fresh connection
                await self.connect()
                result = await self._session.call_tool(tool_name, arguments)
                response = (
                    json.loads(result.content[0].text) if result.content else {"error": "No content"}
                )

            # Handle authentication errors
            if isinstance(response, dict) and response.get("error") == "authentication_required":
                logger.warning("Authentication required — attempting to fetch auth URL")
                try:
                    auth_init = await self._session.call_tool("start_authentication", {})
                    auth_info = json.loads(auth_init.content[0].text)
                    if "url" in auth_info:
                        response["auth_url"] = auth_info["url"]
                        response["instructions"] = auth_info.get("instructions", "")
                except Exception as auth_exc:
                    logger.error("Failed to retrieve auth URL: %s", auth_exc)

            return response


# Singleton connection manager
mcp_manager = MCPConnectionManager(MCP_SERVER_PATH)


# ---------------------------------------------------------------------------
# FastAPI lifespan — connect / disconnect MCP on startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await mcp_manager.connect()
    except Exception as exc:
        logger.warning("MCP server not available at startup (will retry on first request): %s", exc)
    yield
    await mcp_manager.disconnect()


app = FastAPI(
    title="AmpleNote API Wrapper",
    description="HTTP API wrapper for AmpleNote MCP Server",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


class ChatWithContextRequest(BaseModel):
    message: str
    days: int = 7


class SummarizeNotesRequest(BaseModel):
    days: int = 7


# ---------------------------------------------------------------------------
# Note content extraction helper (single source of truth)
# ---------------------------------------------------------------------------
CONTENT_KEYS = ("content", "text", "body", "html", "markdown", "note_content")


def extract_note_content(note: dict) -> str:
    """Extract the text content from a note dict, trying known field names."""
    for key in CONTENT_KEYS:
        value = note.get(key)
        if value:
            return value

    # Check nested data object
    data = note.get("data")
    if isinstance(data, dict):
        for key in CONTENT_KEYS:
            value = data.get(key)
            if value:
                return value

    return ""


def format_note_for_context(note: dict, content_override: str = "") -> str:
    """Format a single note dict into a context string for the LLM."""
    title = note.get("name", note.get("title", "Untitled"))
    tags = ", ".join(note.get("tags", []))
    content = content_override or extract_note_content(note) or "(No text content found)"
    return f"Title: {title}\nTags: {tags}\nContent: {content}\n---\n"


async def build_notes_context(notes_list: list[dict]) -> tuple[str, bool]:
    """Build the combined notes context string from a list of note dicts.

    Returns (context_string, has_real_content).
    """
    parts: list[str] = []
    has_real_content = False

    for note in notes_list:
        content = extract_note_content(note)

        # If no content and we have a UUID, try fetching explicitly
        if not content and note.get("uuid"):
            try:
                detail = await mcp_manager.call_tool("get_note_content", {"note_uuid": note["uuid"]})
                if isinstance(detail, dict):
                    content = extract_note_content(detail)
            except Exception:
                pass  # Continue without content

        if content:
            has_real_content = True

        parts.append(format_note_for_context(note, content))

    return "\n".join(parts), has_real_content


# ---------------------------------------------------------------------------
# LLM Helper Functions
# ---------------------------------------------------------------------------
async def call_openai_chat(message: str) -> str:
    """Call OpenAI for general chat."""
    if not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
        )
    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant with general knowledge."},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {exc}")


async def call_openai_with_context(message: str, notes_context: str) -> str:
    """Call OpenAI with AmpleNote notes as context."""
    if not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
        )

    system_prompt = (
        "You are a helpful assistant. Use the provided AmpleNote notes context to answer questions. "
        "You can also use your general knowledge when appropriate. "
        "When referencing information from notes, cite the note title if available."
    )
    user_content = f"Context from AmpleNote notes:\n{notes_context}\n\nUser Question: {message}"

    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {exc}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AmpleNote API Wrapper",
        "version": "1.0.0",
        "endpoints": {
            "notes": "/api/notes",
            "note_content": "/api/notes/{uuid}",
            "tasks": "/api/tasks",
            "chat": "/api/chat",
            "chat_with_context": "/api/chat-with-context",
            "summarize_notes": "/api/summarize-notes",
            "health": "/api/health",
        },
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "mcp_server_path": MCP_SERVER_PATH}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """General knowledge chat endpoint (no AmpleNote context)."""
    try:
        response_text = await call_openai_chat(request.message)
        return JSONResponse(content={"response": response_text})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat-with-context")
async def chat_with_context(request: ChatWithContextRequest):
    """Chat endpoint with AmpleNote notes as context."""
    try:
        notes_result = await mcp_manager.call_tool("get_recent_notes", {"days": request.days})

        # Auth / error pass-through
        if isinstance(notes_result, dict) and "error" in notes_result:
            status_code = 401 if notes_result.get("error") == "authentication_required" else 500
            return JSONResponse(content=notes_result, status_code=status_code)

        notes_list = notes_result if isinstance(notes_result, list) else []
        notes_context, has_real_content = await build_notes_context(notes_list)

        if not has_real_content:
            notes_context += (
                "\nNOTE: The user's notes appear to have no text content, only titles and tags. "
                "This might mean they are empty 'Daily Jots' or task lists. "
                "Please summarize based on the titles (dates) and tags provided, "
                "and mention that the notes seem empty of text."
            )

        response_text = await call_openai_with_context(request.message, notes_context)
        return JSONResponse(content={"response": response_text, "notes_count": len(notes_list)})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/summarize-notes")
async def summarize_notes(request: SummarizeNotesRequest):
    """Summarize AmpleNote notes using LLM."""
    try:
        notes_result = await mcp_manager.call_tool("get_recent_notes", {"days": request.days})

        if isinstance(notes_result, dict) and "error" in notes_result:
            status_code = 401 if notes_result.get("error") == "authentication_required" else 500
            return JSONResponse(content=notes_result, status_code=status_code)

        notes_list = notes_result if isinstance(notes_result, list) else []
        if not notes_list:
            return JSONResponse(content={"summary": "No notes found for the specified time period.", "notes_count": 0})

        notes_context, _ = await build_notes_context(notes_list)

        summary_prompt = (
            f"Please analyze and summarize the following {len(notes_list)} notes from AmpleNote. "
            "Identify the main topics, themes, and any patterns. Provide a concise summary."
        )
        summary_text = await call_openai_with_context(summary_prompt, notes_context)
        return JSONResponse(content={"summary": summary_text, "notes_count": len(notes_list)})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/notes")
async def get_notes(days: int = Query(7, ge=1, le=365, description="Number of days to look back")):
    """Get recent notes from AmpleNote."""
    try:
        result = await mcp_manager.call_tool("get_recent_notes", {"days": days})

        if isinstance(result, dict) and "error" in result:
            status_code = 401 if result.get("error") == "authentication_required" else 500
            return JSONResponse(content=result, status_code=status_code)

        if isinstance(result, list):
            return JSONResponse(content={"notes": result, "count": len(result)})

        return JSONResponse(content={"notes": result, "count": 0})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/notes/{note_uuid}")
async def get_note_content(note_uuid: str):
    """Get full content of a specific note by UUID."""
    try:
        result = await mcp_manager.call_tool("get_note_content", {"note_uuid": note_uuid})

        if isinstance(result, dict) and "error" in result:
            status_code = 401 if result.get("error") == "authentication_required" else 500
            return JSONResponse(content=result, status_code=status_code)

        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks from AmpleNote."""
    try:
        result = await mcp_manager.call_tool("get_all_tasks", {})

        if isinstance(result, dict) and "error" in result:
            status_code = 401 if result.get("error") == "authentication_required" else 500
            return JSONResponse(content=result, status_code=status_code)

        if isinstance(result, dict) and "tasks" in result:
            tasks = result["tasks"]
        elif isinstance(result, list):
            tasks = result
        else:
            tasks = []

        return JSONResponse(content={"tasks": tasks, "count": len(tasks)})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting AmpleNote API Wrapper on port %d", API_PORT)
    logger.info("MCP Server: %s", MCP_SERVER_PATH)
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
