from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from deepresearch_agent.config import Settings
from deepresearch_agent.search import (
    JinaReaderCrawler,
    JinaSearchAdapter,
    SearchError,
    SearxngSearchAdapter,
    build_search_adapter,
    build_search_service,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any] | str) -> None:
        if isinstance(payload, str):
            self.payload = payload.encode("utf-8")
        else:
            self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class StubCrawler:
    name = "stub_crawler"

    async def crawl(self, url: str, timeout: float) -> str:
        del timeout
        return f"Full crawled content from {url} about citation grounding."


def test_searxng_adapter_uses_crawler_content(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        del timeout
        requested_urls.append(request.full_url)
        return FakeResponse(
            {
                "results": [
                    {
                        "title": "Citation grounding",
                        "url": "https://example.com/grounding",
                        "content": "Short snippet",
                        "engine": "stub",
                    }
                ]
            }
        )

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    adapter = SearxngSearchAdapter(
        "https://search.local",
        crawler=StubCrawler(),
        max_chars=200,
    )

    sources = asyncio.run(adapter.search("citation grounding", max_results=1, timeout=1.0))

    assert requested_urls[0].startswith("https://search.local/search?")
    assert sources[0].provider == "searxng"
    assert sources[0].content == "Full crawled content from https://example.com/grounding about citation grounding."
    assert sources[0].metadata["crawler"] == "stub_crawler"


def test_searxng_requires_base_url() -> None:
    adapter = SearxngSearchAdapter("", crawler=None)

    with pytest.raises(SearchError, match="SEARXNG_BASE_URL"):
        asyncio.run(adapter.search("agent tools", max_results=1, timeout=1.0))


def test_jina_reader_crawler_prefixes_target_url(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_urls: list[str] = []
    monkeypatch.delenv("JINA_API_KEY", raising=False)

    def fake_urlopen(request, timeout):
        del timeout
        requested_urls.append(request.full_url)
        return FakeResponse("Title: Example\n\nClean markdown body about agents.")

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    crawler = JinaReaderCrawler(base_url="https://r.jina.ai/", max_chars=50)

    content = asyncio.run(crawler.crawl("https://example.com/page", timeout=1.0))

    assert requested_urls == ["https://r.jina.ai/https://example.com/page"]
    assert content == "Title: Example Clean markdown body about agents."


def test_jina_search_adapter_parses_json_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)

    def fake_urlopen(request, timeout):
        del timeout
        assert request.full_url == "https://s.jina.ai/agent+observability"
        return FakeResponse(
            {
                "data": [
                    {
                        "title": "Agent observability",
                        "url": "https://example.com/trace",
                        "content": "Trace logs and per-stage cost make agents debuggable.",
                    }
                ]
            }
        )

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    adapter = JinaSearchAdapter(base_url="https://s.jina.ai/", max_chars=200)

    sources = asyncio.run(adapter.search("agent observability", max_results=1, timeout=1.0))

    assert sources[0].provider == "jina"
    assert sources[0].url == "https://example.com/trace"
    assert "per-stage cost" in sources[0].content


def test_build_search_service_selects_new_providers() -> None:
    searxng = build_search_adapter(
        Settings(searxng_base_url="https://search.local", web_crawler_provider="jina"),
        "searxng",
    )
    jina_service = build_search_service(Settings(search_provider="jina"), None)

    assert isinstance(searxng, SearxngSearchAdapter)
    assert jina_service.primary.name == "jina"


def test_unknown_search_provider_fails_fast() -> None:
    with pytest.raises(ValueError, match="unknown search provider"):
        build_search_adapter(Settings(), "unknown")
