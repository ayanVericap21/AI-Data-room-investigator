"""Tool clients used by the agent. Same interface, two transports:
LocalToolClient (in-process; unit tests) and MCPToolClient (spawns the MCP server over stdio; default at runtime).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Protocol

from ..tools import TOOL_NAMES, DataRoomTools


class ToolError(Exception):
    pass


class ToolClient(Protocol):
    def list_tools(self) -> list[dict[str, str]]: ...
    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


class LocalToolClient:
    def __init__(self, tools: DataRoomTools):
        self._tools = tools

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": n, "description": (getattr(DataRoomTools, n).__doc__ or "").strip().split("\n")[0]} for n in TOOL_NAMES]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool '{name}'")
        try:
            return getattr(self._tools, name)(**args)
        except TypeError as e:
            raise ToolError(f"bad arguments for {name}: {e}") from e

    def close(self) -> None:
        pass


class MCPToolClient:
    """Synchronous facade over an async MCP stdio session running on a private event-loop thread."""

    def __init__(self, database_url: str, schema: str | None = None, document_ids: list[int] | None = None,
                 timeout: float = 60.0):
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._stack: AsyncExitStack | None = None
        self._client = None
        try:
            self._run(self._start(database_url, schema or "", document_ids or []))
        except BaseException:
            self.close()
            raise

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self._timeout)

    async def _start(self, database_url: str, schema: str, ids: list[int]):
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters

        root = Path(__file__).resolve().parents[2]
        # The URL may contain a password: it travels in the child's environment, never on the command line.
        env = {**os.environ, "DRI_DATABASE_URL": database_url, "DRI_DB_SCHEMA": schema,
               "DRI_DOC_IDS": ",".join(map(str, ids)), "PYTHONPATH": str(root)}
        params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], env=env, cwd=str(root))
        self._stack = AsyncExitStack()
        self._client = await self._stack.enter_async_context(Client(params))

    def list_tools(self) -> list[dict[str, str]]:
        """Ask the running MCP server which tools it exposes (a real protocol round-trip)."""
        res = self._run(self._client.list_tools())
        return [{"name": t.name, "description": (t.description or "").strip().split("\n")[0]} for t in res.tools]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool '{name}'")
        try:
            res = self._run(self._client.call_tool(name, args))
        except Exception as e:
            raise ToolError(f"MCP call {name} failed: {e}") from e
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            raise ToolError(f"{name} returned an error: {_text(res)}")
        structured = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if structured is not None:
            return structured
        return json.loads(_text(res))

    def close(self) -> None:
        try:
            if self._stack is not None:
                asyncio.run_coroutine_threadsafe(self._stack.aclose(), self._loop).result(10)
        except Exception:
            pass
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)


def _text(res) -> str:
    return "".join(getattr(c, "text", "") for c in getattr(res, "content", []) or [])
