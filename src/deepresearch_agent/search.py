from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from deepresearch_agent.config import Settings
from deepresearch_agent.mcp_tools import McpServerConfig, McpToolSearchAdapter, build_mcp_client
from deepresearch_agent.schemas import Source


class SearchError(RuntimeError):
    pass


class SearchAdapter(Protocol):
    name: str

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        ...


class WebCrawler(Protocol):
    name: str

    async def crawl(self, url: str, timeout: float) -> str:
        ...


@dataclass
class SearchOutcome:
    sources: list[Source]
    provider: str
    fallback_used: bool = False
    error: str | None = None


class MockSearchAdapter:
    name = "mock"

    def __init__(self) -> None:
        self._corpus = [
            (
                "Supervisor researcher pattern",
                "https://example.local/supervisor-researcher",
                "A supervisor-researcher deep research agent decomposes a broad question into independent subquestions, runs worker researchers concurrently, and merges their findings with source IDs for later citation checks.",
            ),
            (
                "Agentic RAG evidence handling",
                "https://example.local/agentic-rag",
                "Agentic RAG differs from normal RAG because the system can plan searches, call tools, inspect intermediate evidence, and revise the retrieval path before producing a final answer.",
            ),
            (
                "Citation faithfulness check",
                "https://example.local/citation-faithfulness",
                "Citation faithfulness checks compare each cited claim with the referenced source text, flagging claims whose words or facts are not supported by the cited source.",
            ),
            (
                "Tool failure fallback",
                "https://example.local/tool-fallback",
                "Reliable agent tools should use timeout budgets, bounded retries, circuit breakers, and deterministic fallback providers so a failed search API does not stop the full research workflow.",
            ),
            (
                "Trace and cost attribution",
                "https://example.local/trace-cost",
                "Structured trace logs and per-stage token accounting make it possible to explain latency, cost, and failure recovery for each stage of a multi-agent research pipeline.",
            ),
            (
                "Benchmark harness design",
                "https://example.local/benchmark-harness",
                "A reproducible benchmark harness records seed, configuration, latency, token estimates, source counts, citation retention, and task success for every research case.",
            ),
        ]

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del timeout
        query_terms = _tokens(query)
        ranked: list[tuple[float, tuple[str, str, str]]] = []
        for item in self._corpus:
            text = " ".join(item)
            overlap = len(query_terms & _tokens(text))
            ranked.append((overlap + 0.1, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        sources = []
        for score, (title, url, content) in ranked[:max_results]:
            sources.append(
                Source(
                    title=title,
                    url=url,
                    content=content,
                    provider=self.name,
                    query=query,
                    score=score,
                )
            )
        return sources


class WikipediaSearchAdapter:
    name = "wikipedia"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        return await asyncio.to_thread(self._search_sync, query, max_results, timeout)

    def _search_sync(self, query: str, max_results: int, timeout: float) -> list[Source]:
        errors: list[str] = []
        for search_query in _wikipedia_query_candidates(query):
            try:
                sources = self._search_once(search_query, query, max_results, timeout)
            except SearchError as exc:
                errors.append(str(exc))
                continue
            if sources:
                return sources
        raise SearchError("; ".join(error for error in errors if error) or "wikipedia returned no results")

    def _search_once(
        self, search_query: str, original_query: str, max_results: int, timeout: float
    ) -> list[Source]:
        params = urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srlimit": max_results,
                "format": "json",
                "utf8": "1",
            }
        )
        request = Request(
            f"https://en.wikipedia.org/w/api.php?{params}",
            headers={"User-Agent": "deepresearch-agent/0.1 local interview project"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - depends on live network
            raise SearchError(str(exc)) from exc

        results = payload.get("query", {}).get("search", [])
        sources: list[Source] = []
        query_terms = _tokens(search_query)
        for index, result in enumerate(results):
            title = html.unescape(str(result.get("title", "")))
            snippet = _strip_html(str(result.get("snippet", "")))
            page_id = result.get("pageid")
            url_title = quote_plus(title.replace(" ", "_"))
            content = f"{title}. {snippet}"
            overlap = len(query_terms & _tokens(content))
            rank_bonus = max_results - index
            sources.append(
                Source(
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{url_title}",
                    content=content,
                    provider=self.name,
                    query=original_query,
                    score=float(overlap * 10 + rank_bonus),
                    metadata={"pageid": page_id, "search_query": search_query},
                )
            )
        if not sources:
            raise SearchError(f"wikipedia returned no results for query: {search_query}")
        return sources


class JinaReaderCrawler:
    name = "jina_reader"

    def __init__(self, base_url: str = "https://r.jina.ai/", max_chars: int = 4000) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.max_chars = max_chars

    async def crawl(self, url: str, timeout: float) -> str:
        return await asyncio.to_thread(self._crawl_sync, url, timeout)

    def _crawl_sync(self, url: str, timeout: float) -> str:
        headers = {"User-Agent": "deepresearch-agent/0.1 local interview project"}
        api_key = os.environ.get("JINA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(
            f"{self.base_url}{url}",
            headers=headers,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - depends on live network
            raise SearchError(str(exc)) from exc
        return _normalize_content(text, self.max_chars)


class HtmlTextCrawler:
    name = "html"

    def __init__(self, max_chars: int = 4000) -> None:
        self.max_chars = max_chars

    async def crawl(self, url: str, timeout: float) -> str:
        return await asyncio.to_thread(self._crawl_sync, url, timeout)

    def _crawl_sync(self, url: str, timeout: float) -> str:
        request = Request(
            url,
            headers={"User-Agent": "deepresearch-agent/0.1 local interview project"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                charset = None
                if getattr(response, "headers", None) is not None:
                    charset = response.headers.get_content_charset()
                raw = response.read().decode(charset or "utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - depends on live network
            raise SearchError(str(exc)) from exc
        parser = _HtmlTextParser()
        parser.feed(raw)
        return _normalize_content(parser.text(), self.max_chars)


class JinaSearchAdapter:
    name = "jina"

    def __init__(self, base_url: str = "https://s.jina.ai/", max_chars: int = 4000) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.max_chars = max_chars

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        return await asyncio.to_thread(self._search_sync, query, max_results, timeout)

    def _search_sync(self, query: str, max_results: int, timeout: float) -> list[Source]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "deepresearch-agent/0.1 local interview project",
        }
        api_key = os.environ.get("JINA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(
            f"{self.base_url}{quote_plus(query)}",
            headers=headers,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - depends on live network
            raise SearchError(str(exc)) from exc
        rows = _jina_rows(raw)
        if not rows:
            rows = [
                {
                    "title": f"Jina search: {query}",
                    "url": self.base_url,
                    "content": raw,
                }
            ]
        sources = _rows_to_sources(
            rows[:max_results],
            provider=self.name,
            query=query,
            max_results=max_results,
            max_chars=self.max_chars,
            metadata={"search_api": "jina_search"},
        )
        if not sources:
            raise SearchError("jina search returned no results")
        return sources


class BraveSearchAdapter:
    name = "brave"

    def __init__(self, base_url: str, max_chars: int = 4000) -> None:
        self.base_url = base_url
        self.max_chars = max_chars

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        return await asyncio.to_thread(self._search_sync, query, max_results, timeout)

    def _search_sync(self, query: str, max_results: int, timeout: float) -> list[Source]:
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise SearchError("BRAVE_SEARCH_API_KEY is required for brave search")
        params = urlencode({"q": query, "count": min(max(max_results, 1), 20)})
        request = Request(
            f"{self.base_url}?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "deepresearch-agent/0.1 local interview project",
                "X-Subscription-Token": api_key,
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - depends on live network
            raise SearchError(str(exc)) from exc
        rows = payload.get("web", {}).get("results", [])
        if not isinstance(rows, list) or not rows:
            raise SearchError("brave search returned no web results")
        sources = _rows_to_sources(
            rows[:max_results],
            provider=self.name,
            query=query,
            max_results=max_results,
            max_chars=self.max_chars,
            metadata={"search_api": "brave"},
        )
        if not sources:
            raise SearchError("brave search returned no parseable results")
        return sources


class TavilySearchAdapter:
    name = "tavily"

    def __init__(
        self,
        base_url: str,
        search_depth: str = "basic",
        max_chars: int = 4000,
    ) -> None:
        self.base_url = base_url
        self.search_depth = search_depth
        self.max_chars = max_chars

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        return await asyncio.to_thread(self._search_sync, query, max_results, timeout)

    def _search_sync(self, query: str, max_results: int, timeout: float) -> list[Source]:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise SearchError("TAVILY_API_KEY is required for tavily search")
        body = json.dumps(
            {
                "query": query,
                "search_depth": self.search_depth,
                "max_results": min(max(max_results, 1), 20),
                "include_answer": False,
                "include_raw_content": False,
            }
        ).encode("utf-8")
        request = Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "deepresearch-agent/0.1 local interview project",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - depends on live network
            raise SearchError(str(exc)) from exc
        rows = payload.get("results", [])
        if not isinstance(rows, list) or not rows:
            raise SearchError("tavily search returned no results")
        normalized_rows = []
        for row in rows[:max_results]:
            if isinstance(row, dict):
                normalized = dict(row)
                normalized["content"] = row.get("raw_content") or row.get("content") or ""
                normalized_rows.append(normalized)
        sources = _rows_to_sources(
            normalized_rows,
            provider=self.name,
            query=query,
            max_results=max_results,
            max_chars=self.max_chars,
            metadata={"search_api": "tavily", "search_depth": self.search_depth},
        )
        if not sources:
            raise SearchError("tavily search returned no parseable results")
        return sources


class SearxngSearchAdapter:
    name = "searxng"

    def __init__(
        self,
        base_url: str,
        crawler: WebCrawler | None = None,
        max_chars: int = 4000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.crawler = crawler
        self.max_chars = max_chars

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        rows = await asyncio.to_thread(self._fetch_rows_sync, query, timeout)
        sources: list[Source] = []
        for index, row in enumerate(rows[:max_results]):
            if not isinstance(row, dict):
                continue
            url = str(row.get("url", ""))
            title = str(row.get("title") or url or "Untitled search result")
            snippet = str(row.get("content") or row.get("snippet") or "")
            crawler_error = None
            content = snippet
            if self.crawler is not None and url:
                try:
                    content = await self.crawler.crawl(url, timeout)
                except Exception as exc:
                    crawler_error = str(exc)
                    content = snippet
            content = _normalize_content(content or title, self.max_chars)
            query_terms = _tokens(query)
            overlap = len(query_terms & _tokens(f"{title} {content}"))
            rank_bonus = max_results - index
            metadata = {
                "search_api": "searxng",
                "engine": row.get("engine"),
                "engines": row.get("engines"),
                "snippet": snippet,
                "crawler": self.crawler.name if self.crawler is not None else "none",
            }
            if crawler_error:
                metadata["crawler_error"] = crawler_error
            sources.append(
                Source(
                    title=html.unescape(title),
                    url=url,
                    content=content,
                    provider=self.name,
                    query=query,
                    score=float(overlap * 5 + rank_bonus),
                    metadata=metadata,
                )
            )
        if not sources:
            raise SearchError("searxng returned no parseable results")
        return sources

    def _fetch_rows_sync(self, query: str, timeout: float) -> list[dict[str, Any]]:
        if not self.base_url:
            raise SearchError("SEARXNG_BASE_URL is required for searxng search")
        params = urlencode({"q": query, "format": "json", "language": "en", "safesearch": "0"})
        request = Request(
            f"{self.base_url}/search?{params}",
            headers={"User-Agent": "deepresearch-agent/0.1 local interview project"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - depends on live network
            raise SearchError(str(exc)) from exc

        rows = payload.get("results", [])
        if not isinstance(rows, list) or not rows:
            raise SearchError("searxng returned no results")
        return rows


class CircuitBreaker:
    def __init__(self, failure_threshold: int, cooldown_seconds: float) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failures = 0
        self.opened_at: float | None = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.cooldown_seconds:
            self.failures = 0
            self.opened_at = None
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()


class SearchRateLimiter:
    def __init__(
        self,
        rate_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.rate_per_second = rate_per_second
        self.clock = clock
        self.sleep = sleep
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    @property
    def enabled(self) -> bool:
        return self.rate_per_second > 0

    async def wait(self) -> None:
        if not self.enabled:
            return
        interval = 1.0 / self.rate_per_second
        async with self._lock:
            now = self.clock()
            if now < self._next_allowed_at:
                await self.sleep(self._next_allowed_at - now)
                now = self.clock()
            self._next_allowed_at = max(now, self._next_allowed_at) + interval


class SearchService:
    def __init__(
        self,
        primary: SearchAdapter,
        fallback: SearchAdapter,
        settings: Settings,
        rate_limiter: SearchRateLimiter | None = None,
        retry_sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.settings = settings
        self.rate_limiter = rate_limiter or SearchRateLimiter(
            settings.search_rate_limit_per_second
        )
        self.retry_sleep = retry_sleep
        self.breaker = CircuitBreaker(
            settings.circuit_breaker_failure_threshold,
            settings.circuit_breaker_cooldown_seconds,
        )

    async def search(self, query: str, max_results: int) -> SearchOutcome:
        if self.primary.name == self.fallback.name:
            sources = await self.fallback.search(query, max_results, self.settings.request_timeout_seconds)
            return SearchOutcome(sources=sources, provider=self.fallback.name)

        last_error: str | None = None
        if self.breaker.allow():
            for attempt in range(self.settings.max_retries + 1):
                try:
                    await self.rate_limiter.wait()
                    sources = await asyncio.wait_for(
                        self.primary.search(query, max_results, self.settings.request_timeout_seconds),
                        timeout=self.settings.request_timeout_seconds + 0.5,
                    )
                    self.breaker.record_success()
                    return SearchOutcome(sources=sources, provider=self.primary.name)
                except Exception as exc:
                    last_error = str(exc)
                    self.breaker.record_failure()
                    if attempt < self.settings.max_retries:
                        await self._retry_backoff(attempt)
        else:
            last_error = "circuit breaker open"

        fallback_sources = await self.fallback.search(
            query, max_results, self.settings.request_timeout_seconds
        )
        return SearchOutcome(
            sources=fallback_sources,
            provider=self.fallback.name,
            fallback_used=True,
            error=last_error,
        )

    async def _retry_backoff(self, attempt: int) -> None:
        base_delay = self.settings.search_retry_backoff_seconds
        if base_delay <= 0:
            return
        await self.retry_sleep(base_delay * (2**attempt))


def build_search_service(settings: Settings, provider: str | None = None) -> SearchService:
    selected = (provider or settings.search_provider).strip().lower()
    fallback = MockSearchAdapter()
    primary = build_search_adapter(settings, selected, fallback)
    return SearchService(primary=primary, fallback=fallback, settings=settings)


def build_search_adapter(
    settings: Settings,
    selected: str,
    fallback: SearchAdapter | None = None,
) -> SearchAdapter:
    if selected == "mock":
        return fallback or MockSearchAdapter()
    if selected == "wikipedia":
        return WikipediaSearchAdapter()
    if selected == "searxng":
        return SearxngSearchAdapter(
            base_url=settings.searxng_base_url,
            crawler=build_crawler(settings),
            max_chars=settings.crawler_max_chars,
        )
    if selected == "jina":
        return JinaSearchAdapter(
            base_url=settings.jina_search_base_url,
            max_chars=settings.crawler_max_chars,
        )
    if selected == "brave":
        return BraveSearchAdapter(
            base_url=settings.brave_search_base_url,
            max_chars=settings.crawler_max_chars,
        )
    if selected == "tavily":
        return TavilySearchAdapter(
            base_url=settings.tavily_search_base_url,
            search_depth=settings.tavily_search_depth,
            max_chars=settings.crawler_max_chars,
        )
    if selected == "mcp":
        config = McpServerConfig(
            transport=settings.mcp_transport,
            command=settings.mcp_command,
            args=settings.mcp_args,
            http_url=settings.mcp_http_url,
            search_tool=settings.mcp_search_tool,
            query_argument=settings.mcp_query_argument,
            timeout_seconds=settings.request_timeout_seconds,
        )
        return McpToolSearchAdapter(
            client=build_mcp_client(config),
            tool_name=config.search_tool,
            query_argument=config.query_argument,
        )
    raise ValueError(f"unknown search provider: {selected}")


def build_crawler(settings: Settings) -> WebCrawler | None:
    provider = settings.web_crawler_provider.strip().lower()
    if provider in {"", "none"}:
        return None
    if provider in {"jina", "jina_reader"}:
        return JinaReaderCrawler(
            base_url=settings.jina_reader_base_url,
            max_chars=settings.crawler_max_chars,
        )
    if provider in {"html", "basic_html", "local_html"}:
        return HtmlTextCrawler(max_chars=settings.crawler_max_chars)
    raise ValueError(f"unknown web crawler provider: {provider}")


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _normalize_content(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", _strip_html(text)).strip()
    if max_chars > 0 and len(normalized) > max_chars:
        return normalized[:max_chars].rstrip()
    return normalized


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", text)}


def _jina_rows(raw: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _jina_markdown_rows(raw)
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        data = data.get("results", [])
    if not isinstance(data, list):
        return []
    rows = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _jina_markdown_rows(raw: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: dict[str, str] = {}
    content_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("Title:"):
            if current:
                current["content"] = "\n".join(content_lines).strip()
                rows.append(current)
            current = {"title": line.removeprefix("Title:").strip()}
            content_lines = []
        elif line.startswith("URL Source:") or line.startswith("URL:"):
            key = "URL Source:" if line.startswith("URL Source:") else "URL:"
            current["url"] = line.removeprefix(key).strip()
        elif current:
            content_lines.append(line)
    if current:
        current["content"] = "\n".join(content_lines).strip()
        rows.append(current)
    return rows


def _rows_to_sources(
    rows: list[dict[str, Any]],
    *,
    provider: str,
    query: str,
    max_results: int,
    max_chars: int,
    metadata: dict[str, Any],
) -> list[Source]:
    sources = []
    query_terms = _tokens(query)
    for index, row in enumerate(rows):
        title = str(row.get("title") or row.get("name") or row.get("url") or "Untitled search result")
        url = str(row.get("url") or row.get("link") or row.get("href") or "")
        content = _normalize_content(str(row.get("content") or row.get("description") or title), max_chars)
        overlap = len(query_terms & _tokens(f"{title} {content}"))
        rank_bonus = max_results - index
        sources.append(
            Source(
                title=html.unescape(title),
                url=url,
                content=content,
                provider=provider,
                query=query,
                score=float(overlap * 5 + rank_bonus),
                metadata=dict(metadata),
            )
        )
    return sources


class _HtmlTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)


WIKIPEDIA_STOPWORDS = {
    "about",
    "agent",
    "agents",
    "and",
    "are",
    "between",
    "compare",
    "compared",
    "common",
    "does",
    "from",
    "have",
    "how",
    "implement",
    "implemented",
    "into",
    "needed",
    "should",
    "that",
    "the",
    "their",
    "used",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def _wikipedia_query_candidates(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", query)
    keywords = []
    for word in words:
        lowered = word.lower()
        if lowered in WIKIPEDIA_STOPWORDS or len(lowered) <= 2:
            continue
        if lowered not in {item.lower() for item in keywords}:
            keywords.append(word)
    candidates = []
    if keywords:
        candidates.append(" ".join(keywords[:8]))
    if len(keywords) > 4:
        candidates.append(" ".join(keywords[:4]))
    candidates.append(query)
    deduped = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped
