#!/usr/bin/env python3
"""
AmpleNote API Wrapper
HTTP API that wraps the AmpleNote MCP Server for LLM integration
"""

import os
import json
import asyncio
import traceback
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuration
MCP_SERVER_PATH = os.getenv(
    "MCP_SERVER_PATH",
    str(Path(__file__).parent.parent / "AmpleNote_MCP" / "amplenote_mcp_server.py")
)
API_PORT = int(os.getenv("API_PORT", "8000"))

app = FastAPI(
    title="AmpleNote API Wrapper",
    description="HTTP API wrapper for AmpleNote MCP Server",
    version="1.0.0"
)


async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool and return the result"""
    try:
        # Use the venv's Python interpreter if available, otherwise system python3
        venv_python = Path(__file__).parent / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        
        # Set working directory to MCP server's directory so it can find config files
        mcp_server_dir = Path(MCP_SERVER_PATH).parent
        
        server_params = StdioServerParameters(
            command=python_cmd,
            args=[MCP_SERVER_PATH],
            env=None,
            cwd=str(mcp_server_dir)
        )
        
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # 1. Try calling the tool
                result = await session.call_tool(tool_name, arguments)
                response = json.loads(result.content[0].text) if result.content else {"error": "No content"}
                
                # 2. Check for auth error
                print(f"DEBUG call_mcp_tool: response type={type(response)}, keys={list(response.keys()) if isinstance(response, dict) else 'N/A'}")
                if isinstance(response, dict) and response.get("error") == "authentication_required":
                    print("DEBUG: Auth error detected, getting auth URL...")
                    # For HTTP API, return auth info in response instead of blocking on input()
                    # Start auth to get URL
                    try:
                        auth_init = await session.call_tool("start_authentication", {})
                        auth_info = json.loads(auth_init.content[0].text)
                        
                        # Merge auth URL into error response
                        if "url" in auth_info:
                            response["auth_url"] = auth_info.get("url")
                            response["instructions"] = auth_info.get("instructions", "")
                    except Exception as e:
                        print(f"DEBUG: Error getting auth URL: {e}")
                        # If we can't start auth, just return the original error
                        pass
                    
                    print(f"DEBUG call_mcp_tool: Returning response={response}")
                    return response
                
                print(f"DEBUG call_mcp_tool: Returning response (no auth error)={response}")
                return response
                    
    except Exception as e:
        error_detail = str(e)
        error_type = type(e).__name__
        # Include traceback for debugging
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"MCP Error ({error_type}): {error_detail}")
        print(f"Traceback:\n{tb_str}")
        raise HTTPException(
            status_code=500, 
            detail=f"MCP server error ({error_type}): {error_detail}"
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AmpleNote API Wrapper",
        "version": "1.0.0",
        "endpoints": {
            "notes": "/api/notes",
            "note_content": "/api/notes/{uuid}",
            "tasks": "/api/tasks",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "mcp_server_path": MCP_SERVER_PATH}


@app.get("/api/notes")
async def get_notes(days: int = Query(7, ge=1, le=365, description="Number of days to look back")):
    """Get recent notes from AmpleNote"""
    try:
        result = await call_mcp_tool("get_recent_notes", {"days": days})
        
        # Check if result is an error response - if so, return it directly (don't wrap)
        if isinstance(result, dict) and "error" in result:
            status_code = 401 if result.get("error") == "authentication_required" else 500
            return JSONResponse(content=result, status_code=status_code)
        
        # Success case: wrap list results
        if isinstance(result, list):
            return JSONResponse(content={"notes": result, "count": len(result)})
        
        # Fallback: wrap any other format
        return JSONResponse(content={"notes": result, "count": 0})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/notes/{note_uuid}")
async def get_note_content(note_uuid: str):
    """Get full content of a specific note by UUID"""
    try:
        result = await call_mcp_tool("get_note_content", {"note_uuid": note_uuid})
        
        # Check if result is an error response - return directly (don't wrap)
        if isinstance(result, dict) and "error" in result:
            status_code = 401 if result.get("error") == "authentication_required" else 500
            return JSONResponse(content=result, status_code=status_code)
        
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks from AmpleNote"""
    try:
        result = await call_mcp_tool("get_all_tasks", {})
        
        # Check if result is an error response - return directly (don't wrap)
        if isinstance(result, dict) and "error" in result:
            status_code = 401 if result.get("error") == "authentication_required" else 500
            return JSONResponse(content=result, status_code=status_code)
        
        # Handle different response formats
        if isinstance(result, dict) and "tasks" in result:
            tasks = result["tasks"]
        elif isinstance(result, list):
            tasks = result
        else:
            tasks = []
        return JSONResponse(content={"tasks": tasks, "count": len(tasks)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting AmpleNote API Wrapper on port {API_PORT}")
    print(f"📡 MCP Server: {MCP_SERVER_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)

