from __future__ import annotations

import asyncio

import pytest

from deepresearch_agent.config import Settings
from deepresearch_agent.mcp_tools import McpError, McpToolSearchAdapter, mcp_tool_result_to_sources
from deepresearch_agent.search import build_search_adapter


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls = []

    async def list_tools(self):
        return [{"name": "search_docs"}]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {
            "content": [
                {
                    "type": "text",
                    "text": '[{"title":"MCP result","url":"https://example.com/mcp","content":"MCP tool returned document content."}]',
                }
            ]
        }


def test_mcp_tool_result_to_sources_accepts_sources_array() -> None:
    sources = mcp_tool_result_to_sources(
        {
            "sources": [
                {
                    "title": "Doc",
                    "url": "https://example.com/doc",
                    "content": "Document body",
                }
            ]
        },
        provider="mcp",
        query="agent tools",
        max_results=3,
        tool_name="search_docs",
    )

    assert sources[0].title == "Doc"
    assert sources[0].provider == "mcp"
    assert sources[0].metadata["mcp_tool"] == "search_docs"


def test_mcp_search_adapter_calls_configured_tool() -> None:
    client = FakeMcpClient()
    adapter = McpToolSearchAdapter(client, tool_name="search_docs", query_argument="q")

    sources = asyncio.run(adapter.search("citation grounding", max_results=2, timeout=1.0))

    assert client.calls == [("search_docs", {"q": "citation grounding", "max_results": 2})]
    assert sources[0].url == "https://example.com/mcp"
    assert "document content" in sources[0].content


def test_mcp_adapter_requires_tool_name() -> None:
    with pytest.raises(McpError, match="MCP_SEARCH_TOOL"):
        McpToolSearchAdapter(FakeMcpClient(), tool_name="")


def test_build_search_adapter_supports_mcp_provider() -> None:
    adapter = build_search_adapter(
        Settings(
            search_provider="mcp",
            mcp_transport="http",
            mcp_http_url="https://mcp.local/rpc",
            mcp_search_tool="search_docs",
        ),
        "mcp",
    )

    assert adapter.name == "mcp"
