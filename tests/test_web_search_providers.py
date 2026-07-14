from __future__ import annotations

import asyncio
import json
import socket
from typing import Any

import pytest

from deepresearch_agent.config import Settings
from deepresearch_agent.search import (
    BenchmarkContaminationError,
    BingRssSearchAdapter,
    BraveSearchAdapter,
    HtmlTextCrawler,
    JinaReaderCrawler,
    JinaSearchAdapter,
    SearchError,
    SearxngSearchAdapter,
    TavilySearchAdapter,
    build_search_adapter,
    build_search_service,
)
from deepresearch_agent.schemas import Source
from deepresearch_agent.search import MockSearchAdapter, SearchService
from deepresearch_agent.search import FetchedPage
from deepresearch_agent.url_policy import (
    SafeHTTPError,
    UnsupportedContentTypeError,
    URLPolicyError,
)

class FakeResponse:
    def __init__(self, payload: dict[str, Any] | str) -> None:
        if isinstance(payload, str):
            self.payload = payload.encode("utf-8")
            self.headers = {"Content-Type": "text/plain; charset=utf-8"}
        else:
            self.payload = json.dumps(payload).encode("utf-8")
            self.headers = {"Content-Type": "application/json; charset=utf-8"}
        self.status = 200
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.payload) - self.offset
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        return None


def test_bing_rss_adapter_parses_search_snippets(monkeypatch) -> None:
    rss = '''<?xml version="1.0" encoding="utf-8"?>
    <rss><channel><item><title>Official release</title>
    <link>https://example.com/release</link>
    <description>Version 3.14.6 is the latest stable release.</description>
    <pubDate>Mon, 13 Jul 2026 00:00:00 GMT</pubDate></item></channel></rss>'''
    response = FakeResponse(rss)
    response.headers = {"Content-Type": "application/rss+xml; charset=utf-8"}
    monkeypatch.setattr("deepresearch_agent.search.urlopen", lambda *args, **kwargs: response)

    rows = asyncio.run(
        BingRssSearchAdapter(base_url="https://global.bing.com/search").search(
            "Python latest stable release", 3, 1.0
        )
    )

    assert rows[0].url == "https://example.com/release"
    assert rows[0].metadata["snippet_only"] is True
    assert rows[0].metadata["search_indexed_at"].startswith("Mon")
    assert "published_at" not in rows[0].metadata


def test_bing_rss_uses_direct_official_domain_when_site_filter_is_ignored(
    monkeypatch,
) -> None:
    rss = '''<?xml version="1.0" encoding="utf-8"?>
    <rss><channel><item><title>Unrelated downloads</title>
    <link>https://unrelated.example/downloads</link>
    <description>Windows downloads.</description></item></channel></rss>'''
    response = FakeResponse(rss)
    response.headers = {"Content-Type": "application/rss+xml; charset=utf-8"}
    monkeypatch.setattr("deepresearch_agent.search.urlopen", lambda *args, **kwargs: response)

    rows = asyncio.run(
        BingRssSearchAdapter().search(
            "site:python.org downloads latest stable Python version", 5, 1.0
        )
    )

    assert [row.url for row in rows] == [
        "https://python.org/",
        "https://python.org/downloads/",
    ]
    assert all(row.metadata["direct_domain_fallback"] for row in rows)


class _BenchmarkLeakAdapter:
    name = "leak-fixture"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del max_results, timeout
        return [
            Source(
                title="SimpleQA answer key",
                url="https://github.com/example/simpleqa/data.jsonl",
                content=f'{{"query": "{query}", "reference_answer": "leaked"}}',
                provider=self.name,
                query=query,
                metadata={"extract_status": "ok", "snippet_only": False},
            )
        ]


class _LegitimateGitHubDocsAdapter:
    name = "github-docs"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del max_results, timeout
        return [
            Source(
                title="FastAPI documentation",
                url="https://github.com/tiangolo/fastapi/blob/master/README.md",
                content="FastAPI documentation and release notes.",
                provider=self.name,
                query=query,
                metadata={"extract_status": "ok", "snippet_only": False},
            )
        ]


def test_benchmark_source_exclusion_blocks_known_answer_key_paths() -> None:
    service = SearchService(
        primary=_BenchmarkLeakAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(
            benchmark_source_exclusion=True,
            local_retrieval_mode="none",
        ),
        fallback_policy="fail",
    )

    with pytest.raises(BenchmarkContaminationError, match="benchmark contamination"):
        asyncio.run(service.search("What year was San Carlos founded?", max_results=1))


def test_benchmark_source_exclusion_keeps_non_benchmark_github_docs() -> None:
    service = SearchService(
        primary=_LegitimateGitHubDocsAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(
            benchmark_source_exclusion=True,
            local_retrieval_mode="none",
        ),
        fallback_policy="fail",
    )

    outcome = asyncio.run(service.search("FastAPI documentation", max_results=1))

    assert outcome.sources[0].title == "FastAPI documentation"


def _public_getaddrinfo(host: str, port: int, **kwargs: Any) -> list[tuple[Any, ...]]:
    del host, kwargs
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


class StubCrawler:
    name = "stub_crawler"

    async def crawl(self, url: str, timeout: float) -> str:
        del timeout
        return f"Full crawled content from {url} about citation grounding."


class RedirectingCrawler:
    name = "redirecting"

    async def crawl(self, url: str, timeout: float) -> FetchedPage:
        del timeout
        return FetchedPage(
            content="Canonical body about citation grounding.",
            final_url="https://example.com/canonical",
            redirect_chain=(url, "https://example.com/canonical"),
        )


class _OneCandidateAdapter:
    name = "remote"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del max_results, timeout
        return [
            Source(
                title="Remote candidate",
                url="https://example.com/candidate?token=strip-me",
                content="unverified snippet",
                provider=self.name,
                query=query,
            )
        ]


class _FailingAuditCrawler:
    name = "failing_audit"

    def __init__(self, error_factory) -> None:
        self.error_factory = error_factory
        self.calls = 0

    async def crawl(self, url: str, timeout: float) -> str:
        del url, timeout
        self.calls += 1
        raise self.error_factory()


@pytest.mark.parametrize(
    ("error_factory", "expected_class", "expected_attempts"),
    [
        (lambda: TimeoutError("timed out"), "timeout", 2),
        (lambda: ConnectionError("connection reset"), "connection_failure", 2),
        (lambda: SafeHTTPError("HTTP request failed with status 429"), "http_429", 2),
        (lambda: SafeHTTPError("HTTP request failed with status 503"), "http_5xx", 2),
        (lambda: SafeHTTPError("HTTP request failed with status 404"), "http_4xx", 1),
        (lambda: URLPolicyError("blocked hostname: localhost"), "policy_rejected", 1),
        (
            lambda: UnsupportedContentTypeError(
                "unsupported response Content-Type: image/png"
            ),
            "non_html",
            1,
        ),
    ],
)
def test_crawler_retries_only_transient_failures_and_records_error_class(
    error_factory,
    expected_class: str,
    expected_attempts: int,
) -> None:
    crawler = _FailingAuditCrawler(error_factory)
    service = SearchService(
        primary=_OneCandidateAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(
            local_retrieval_mode="none",
            search_retry_backoff_seconds=0,
        ),
        crawler=crawler,
        fallback_policy="degraded",
    )

    outcome = asyncio.run(service.search("audit query", max_results=1))

    assert crawler.calls == expected_attempts
    assert outcome.tool_attempts == 1 + expected_attempts
    assert outcome.sources[0].metadata["crawl_error_class"] == expected_class
    assert outcome.sources[0].metadata["crawl_attempts"] == expected_attempts
    assert outcome.retrieval_audit["error_classes"] == {expected_class: 1}
    assert outcome.failed_candidate_hints[0]["error_class"] == expected_class
    assert outcome.failed_candidate_hints[0]["url"] == "https://example.com/candidate"
    assert "strip-me" not in str(outcome.failed_candidate_hints)


@pytest.mark.parametrize(
    ("error_factory", "expected_class", "expected_attempts"),
    [
        (
            lambda: URLPolicyError("DNS resolution failed for example.com"),
            "dns_failure",
            2,
        ),
        (
            lambda: URLPolicyError("blocked hostname: localhost"),
            "policy_rejected",
            1,
        ),
        (
            lambda: UnsupportedContentTypeError(
                "unsupported response Content-Type: image/png"
            ),
            "non_html",
            1,
        ),
    ],
)
def test_html_crawler_preserves_safe_error_types_for_retry_classification(
    monkeypatch: pytest.MonkeyPatch,
    error_factory,
    expected_class: str,
    expected_attempts: int,
) -> None:
    calls = 0

    def fail_fetch(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise error_factory()

    monkeypatch.setattr("deepresearch_agent.search.fetch_text_url", fail_fetch)
    service = SearchService(
        primary=_OneCandidateAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(
            local_retrieval_mode="none",
            search_retry_backoff_seconds=0,
        ),
        crawler=HtmlTextCrawler(),
        fallback_policy="degraded",
    )

    outcome = asyncio.run(service.search("audit query", max_results=1))

    assert calls == expected_attempts
    assert outcome.retrieval_audit["error_classes"] == {expected_class: 1}
    assert outcome.failed_candidate_hints[0]["crawl_attempts"] == expected_attempts


class _TwoAliasAdapter:
    name = "aliases"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del max_results, timeout
        return [
            Source(
                title="Alias A",
                url="https://example.com/alias-a",
                content="snippet",
                provider=self.name,
                query=query,
            ),
            Source(
                title="Alias B",
                url="https://example.com/alias-b",
                content="snippet",
                provider=self.name,
                query=query,
            ),
        ]


def test_crawler_uses_final_redirect_url_for_source_provenance() -> None:
    service = SearchService(
        primary=_TwoAliasAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(local_retrieval_mode="none"),
        crawler=RedirectingCrawler(),
        fallback_policy="fail",
    )

    outcome = asyncio.run(service.search("citation grounding", max_results=2))

    assert {source.url for source in outcome.sources} == {"https://example.com/canonical"}
    assert all(source.metadata["redirect_chain"][-1] == source.url for source in outcome.sources)


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
    monkeypatch.setattr(
        "deepresearch_agent.url_policy.socket.getaddrinfo",
        _public_getaddrinfo,
    )

    def fake_urlopen(request, timeout):
        del timeout
        requested_urls.append(request.full_url)
        return FakeResponse("Title: Example\n\nClean markdown body about agents.")

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    crawler = JinaReaderCrawler(base_url="https://r.jina.ai/", max_chars=50)

    content = asyncio.run(crawler.crawl("https://example.com/page", timeout=1.0))

    assert requested_urls == ["https://r.jina.ai/https://example.com/page"]
    assert content == "Title: Example Clean markdown body about agents."


def test_html_text_crawler_strips_script_and_style(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_urls: list[str] = []
    monkeypatch.setattr(
        "deepresearch_agent.url_policy.socket.getaddrinfo",
        _public_getaddrinfo,
    )

    def fake_urlopen(request, timeout):
        del timeout
        requested_urls.append(request.full_url)
        return FakeResponse(
            """
            <html>
              <head>
                <style>.hidden{display:none}</style>
                <script>window.secret = "ignore";</script>
              </head>
              <body>
                <h1>Agent Evidence</h1>
                <p>HTML crawler extracts readable source text.</p>
              </body>
            </html>
            """
        )

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    crawler = HtmlTextCrawler(max_chars=200)

    content = asyncio.run(crawler.crawl("https://example.com/page", timeout=1.0))

    assert requested_urls == ["https://example.com/page"]
    assert "Agent Evidence" in content
    assert "readable source text" in content
    assert "window.secret" not in content
    assert "display:none" not in content


def test_html_text_crawler_strips_nav_chrome_and_keeps_article_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nav-heavy pages must not let menus crowd out the article body.

    Reproduces the sherdog/MacTutor failure: a long nav bar used to fill the
    char budget before the answer-bearing body text was reached.
    """
    monkeypatch.setattr(
        "deepresearch_agent.url_policy.socket.getaddrinfo",
        _public_getaddrinfo,
    )

    nav_menu = "NEWS FEATURES FIGHT FINDER PODCASTS VIDEOS RANKINGS FORUM " * 40
    fake_html = f"""
    <html>
      <head><title>Andrew Tate</title></head>
      <body>
        <nav>{nav_menu}</nav>
        <header>Site header chrome</header>
        <article>
          <h1>Andrew Tate kickboxing</h1>
          <p>Tate's kickboxing nickname was "King Cobra".</p>
        </article>
        <footer>Copyright footer links</footer>
      </body>
    </html>
    """

    def fake_urlopen(request, timeout):
        del timeout
        return FakeResponse(fake_html)

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    crawler = HtmlTextCrawler(max_chars=4000)

    content = asyncio.run(crawler.crawl("https://example.com/tate", timeout=1.0))

    # Article body survives
    assert "King Cobra" in content
    assert "kickboxing nickname" in content
    # Nav chrome is stripped, not dominating the budget
    assert "FIGHT FINDER" not in content
    assert "Site header chrome" not in content
    assert "Copyright footer" not in content


def test_html_text_crawler_prefers_article_over_sidebar_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When <article> is present, prefer it over aside/sidebar content."""
    monkeypatch.setattr(
        "deepresearch_agent.url_policy.socket.getaddrinfo",
        _public_getaddrinfo,
    )

    fake_html = """
    <html><body>
      <aside>Related articles sidebar noise</aside>
      <article>
        <p>San Carlos Antioquia was founded in 1786.</p>
      </article>
      <aside>More sidebar promotions</aside>
    </body></html>
    """

    def fake_urlopen(request, timeout):
        del timeout
        return FakeResponse(fake_html)

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    crawler = HtmlTextCrawler(max_chars=4000)

    content = asyncio.run(crawler.crawl("https://example.com/sancarlos", timeout=1.0))
    assert "1786" in content
    assert "sidebar noise" not in content
    assert "sidebar promotions" not in content


def test_html_text_crawler_falls_back_to_nav_stripped_text_without_article(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pages without <article>/<main> still get nav-stripped full text."""
    monkeypatch.setattr(
        "deepresearch_agent.url_policy.socket.getaddrinfo",
        _public_getaddrinfo,
    )

    fake_html = """
    <html><body>
      <nav>Menu links</nav>
      <p>The answer is hidden in a plain paragraph.</p>
      <footer>Footer</footer>
    </body></html>
    """

    def fake_urlopen(request, timeout):
        del timeout
        return FakeResponse(fake_html)

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    crawler = HtmlTextCrawler(max_chars=4000)

    content = asyncio.run(crawler.crawl("https://example.com/plain", timeout=1.0))
    assert "hidden in a plain paragraph" in content
    assert "Menu links" not in content
    assert "Footer" not in content


def test_jina_reader_rejects_private_target_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        del timeout
        requested_urls.append(request.full_url)
        return FakeResponse("must not be reached")

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    crawler = JinaReaderCrawler(base_url="https://r.jina.ai/")

    with pytest.raises(SearchError, match="non-global"):
        asyncio.run(crawler.crawl("http://169.254.169.254/latest/meta-data/", timeout=1.0))

    assert requested_urls == []


def test_jina_search_adapter_parses_json_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.setattr(
        "deepresearch_agent.url_policy.socket.getaddrinfo",
        _public_getaddrinfo,
    )

    def fake_urlopen(request, timeout):
        del timeout
        assert request.full_url == "https://s.jina.ai/agent%20observability"
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


def test_jina_search_drops_api_key_on_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RedirectResponse(FakeResponse):
        def __init__(self) -> None:
            super().__init__("")
            self.status = 302
            self.headers = {"Location": "https://other.example/results"}

    monkeypatch.setenv("JINA_API_KEY", "jina-test-key")
    monkeypatch.setattr(
        "deepresearch_agent.url_policy.socket.getaddrinfo",
        _public_getaddrinfo,
    )
    requests = []

    def fake_urlopen(request, timeout):
        del timeout
        requests.append(request)
        if len(requests) == 1:
            return RedirectResponse()
        return FakeResponse(
            {
                "data": [
                    {
                        "title": "Result",
                        "url": "https://example.com/result",
                        "content": "Safe search result.",
                    }
                ]
            }
        )

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    adapter = JinaSearchAdapter(base_url="https://s.jina.ai/", max_chars=200)

    sources = asyncio.run(adapter.search("safe query", max_results=1, timeout=1.0))

    assert sources[0].title == "Result"
    assert len(requests) == 2
    assert requests[0].get_header("Authorization") == "Bearer jina-test-key"
    assert requests[1].get_header("Authorization") is None


def test_brave_search_adapter_parses_web_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-test-key")
    requested = {}

    def fake_urlopen(request, timeout):
        del timeout
        requested["url"] = request.full_url
        requested["token"] = request.get_header("X-subscription-token")
        return FakeResponse(
            {
                "web": {
                    "results": [
                        {
                            "title": "Agent search result",
                            "url": "https://example.com/agent-search",
                            "description": "Brave web search can retrieve agent sources.",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    adapter = BraveSearchAdapter(
        base_url="https://api.search.brave.com/res/v1/web/search",
        max_chars=200,
    )

    sources = asyncio.run(adapter.search("agent search", max_results=1, timeout=1.0))

    assert requested["url"].startswith("https://api.search.brave.com/res/v1/web/search?")
    assert requested["token"] == "brave-test-key"
    assert sources[0].provider == "brave"
    assert sources[0].url == "https://example.com/agent-search"
    assert "agent sources" in sources[0].content


def test_brave_search_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    adapter = BraveSearchAdapter("https://api.search.brave.com/res/v1/web/search")

    with pytest.raises(SearchError, match="BRAVE_SEARCH_API_KEY"):
        asyncio.run(adapter.search("agent search", max_results=1, timeout=1.0))


def test_tavily_search_adapter_posts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")
    requested = {}

    def fake_urlopen(request, timeout):
        del timeout
        requested["url"] = request.full_url
        requested["auth"] = request.get_header("Authorization")
        requested["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "results": [
                    {
                        "title": "Tavily result",
                        "url": "https://example.com/tavily",
                        "content": "Tavily search returns NLP content for research agents.",
                        "score": 0.8,
                    }
                ]
            }
        )

    monkeypatch.setattr("deepresearch_agent.search.urlopen", fake_urlopen)
    adapter = TavilySearchAdapter(
        base_url="https://api.tavily.com/search",
        search_depth="basic",
        max_chars=200,
    )

    sources = asyncio.run(adapter.search("agent search", max_results=1, timeout=1.0))

    assert requested["url"] == "https://api.tavily.com/search"
    assert requested["auth"] == "Bearer " + "tavily-test-key"
    assert requested["body"]["query"] == "agent search"
    assert requested["body"]["search_depth"] == "basic"
    assert sources[0].provider == "tavily"
    assert "NLP content" in sources[0].content


def test_tavily_search_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    adapter = TavilySearchAdapter("https://api.tavily.com/search")

    with pytest.raises(SearchError, match="TAVILY_API_KEY"):
        asyncio.run(adapter.search("agent search", max_results=1, timeout=1.0))


def test_build_search_service_selects_new_providers() -> None:
    searxng = build_search_adapter(
        Settings(searxng_base_url="https://search.local", web_crawler_provider="jina"),
        "searxng",
    )
    searxng_html = build_search_adapter(
        Settings(searxng_base_url="https://search.local", web_crawler_provider="html"),
        "searxng",
    )
    jina_service = build_search_service(Settings(search_provider="jina"), None)
    brave = build_search_adapter(Settings(), "brave")
    tavily = build_search_adapter(Settings(), "tavily")

    assert isinstance(searxng, SearxngSearchAdapter)
    assert isinstance(searxng_html, SearxngSearchAdapter)
    assert searxng_html.crawler is not None
    assert searxng_html.crawler.name == "html"
    assert jina_service.primary.name == "jina"
    assert isinstance(brave, BraveSearchAdapter)
    assert isinstance(tavily, TavilySearchAdapter)


def test_unknown_search_provider_fails_fast() -> None:
    with pytest.raises(ValueError, match="unknown search provider"):
        build_search_adapter(Settings(), "unknown")
