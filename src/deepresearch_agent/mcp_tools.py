from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen

from deepresearch_agent.schemas import Source


class McpError(RuntimeError):
    pass


class McpClient(Protocol):
    async def list_tools(self) -> list[dict[str, Any]]:
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class McpServerConfig:
    transport: str = "stdio"
    command: str = ""
    args: str = ""
    http_url: str = ""
    search_tool: str = ""
    query_argument: str = "query"
    timeout_seconds: float = 4.0


class HttpMcpClient:
    def __init__(self, url: str, timeout_seconds: float) -> None:
        if not url:
            raise McpError("MCP_HTTP_URL is required for http MCP transport")
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._request_id = 0

    async def list_tools(self) -> list[dict[str, Any]]:
        payload = await asyncio.to_thread(self._request, "tools/list", {})
        return _extract_tools(payload)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._request,
            "tools/call",
            {"name": name, "arguments": arguments},
        )

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        body = json.dumps(
            {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        ).encode("utf-8")
        request = Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "deepresearch-agent/0.1 local interview project",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if "error" in payload:
            raise McpError(str(payload["error"]))
        result = payload.get("result")
        if not isinstance(result, dict):
            raise McpError("MCP response missing object result")
        return result


class StdioMcpClient:
    def __init__(self, command: str, args: str, timeout_seconds: float) -> None:
        if not command:
            raise McpError("MCP_COMMAND is required for stdio MCP transport")
        self.command = command
        self.args = shlex.split(args) if args else []
        self.timeout_seconds = timeout_seconds

    async def list_tools(self) -> list[dict[str, Any]]:
        payload = await asyncio.to_thread(
            self._run_requests,
            [
                ("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "deepresearch-agent", "version": "0.1"}}),
                ("tools/list", {}),
            ],
        )
        return _extract_tools(payload[-1])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        responses = await asyncio.to_thread(
            self._run_requests,
            [
                ("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "deepresearch-agent", "version": "0.1"}}),
                ("tools/call", {"name": name, "arguments": arguments}),
            ],
        )
        return responses[-1]

    def _run_requests(self, requests: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        process = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        responses: list[dict[str, Any]] = []
        try:
            for index, (method, params) in enumerate(requests, start=1):
                process.stdin.write(_frame({"jsonrpc": "2.0", "id": index, "method": method, "params": params}))
                process.stdin.flush()
                result = _read_framed_response(process.stdout)
                if "error" in result:
                    raise McpError(str(result["error"]))
                payload = result.get("result")
                if not isinstance(payload, dict):
                    raise McpError("MCP response missing object result")
                responses.append(payload)
                if method == "initialize":
                    process.stdin.write(
                        _frame({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
                    )
                    process.stdin.flush()
            return responses
        finally:
            try:
                process.stdin.close()
            except Exception:
                pass
            try:
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()


class McpToolSearchAdapter:
    name = "mcp"

    def __init__(self, client: McpClient, tool_name: str, query_argument: str = "query") -> None:
        if not tool_name:
            raise McpError("MCP_SEARCH_TOOL is required for mcp search provider")
        self.client = client
        self.tool_name = tool_name
        self.query_argument = query_argument

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del timeout
        result = await self.client.call_tool(
            self.tool_name,
            {self.query_argument: query, "max_results": max_results},
        )
        sources = mcp_tool_result_to_sources(
            result,
            provider=self.name,
            query=query,
            max_results=max_results,
            tool_name=self.tool_name,
        )
        if not sources:
            raise McpError("MCP tool returned no source-like content")
        return sources


def build_mcp_client(config: McpServerConfig) -> McpClient:
    transport = config.transport.strip().lower()
    if transport == "http":
        return HttpMcpClient(config.http_url, config.timeout_seconds)
    if transport == "stdio":
        return StdioMcpClient(config.command, config.args, config.timeout_seconds)
    raise McpError(f"unknown MCP transport: {config.transport}")


def mcp_tool_result_to_sources(
    result: dict[str, Any],
    *,
    provider: str,
    query: str,
    max_results: int,
    tool_name: str,
) -> list[Source]:
    rows = _extract_source_rows(result)
    sources: list[Source] = []
    for index, row in enumerate(rows[:max_results]):
        title = str(row.get("title") or row.get("name") or f"MCP tool result {index + 1}")
        url = str(row.get("url") or row.get("uri") or f"mcp://{tool_name}/{index + 1}")
        content = str(row.get("content") or row.get("text") or row.get("description") or title)
        sources.append(
            Source(
                title=title,
                url=url,
                content=content,
                provider=provider,
                query=query,
                score=float(max_results - index),
                metadata={"mcp_tool": tool_name, "raw_keys": sorted(row.keys())},
            )
        )
    return sources


def _extract_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        raise McpError("MCP tools/list result missing tools list")
    return [tool for tool in tools if isinstance(tool, dict)]


def _extract_source_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(result.get("sources"), list):
        return [row for row in result["sources"] if isinstance(row, dict)]
    content = result.get("content", [])
    if isinstance(content, list):
        rows: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                rows.extend(_rows_from_text(str(item.get("text", ""))))
        if rows:
            return rows
    return _rows_from_text(json.dumps(result, ensure_ascii=False))


def _rows_from_text(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return [{"title": "MCP text result", "content": stripped}]
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        if isinstance(parsed.get("sources"), list):
            return [row for row in parsed["sources"] if isinstance(row, dict)]
        return [parsed]
    return [{"title": "MCP text result", "content": stripped}]


def _frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def _read_framed_response(stdout) -> dict[str, Any]:
    headers = {}
    while True:
        line = stdout.readline()
        if not line:
            raise McpError("MCP stdio server closed stdout before response")
        decoded = line.decode("ascii", errors="replace").strip()
        if not decoded:
            break
        if ":" in decoded:
            key, value = decoded.split(":", 1)
            headers[key.lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        raise McpError("MCP response missing Content-Length")
    body = stdout.read(length)
    return json.loads(body.decode("utf-8"))
