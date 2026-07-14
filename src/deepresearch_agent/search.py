from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import re
import socket
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import quote, quote_plus, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from deepresearch_agent.config import Settings
from deepresearch_agent.gateway_search import (
    GatewayWebSearchAdapter,
    GatewayWebSearchNoResultsError,
)
from deepresearch_agent.mcp_tools import McpServerConfig, McpToolSearchAdapter, build_mcp_client
from deepresearch_agent.schemas import Source
from deepresearch_agent.text_utils import tokenize
from deepresearch_agent.url_policy import (
    DEFAULT_MAX_RESPONSE_BYTES,
    ResponseTooLargeError,
    SafeHTTPError,
    UnsupportedContentTypeError,
    URLPolicyError,
    fetch_text_url,
    no_redirect_urlopen,
    validate_url,
)


_DEFAULT_URLOPEN = urlopen


class SearchError(RuntimeError):
    pass


class BenchmarkContaminationError(SearchError):
    """A public benchmark answer page was discovered during an evaluation run."""


class SearchEvidenceUnavailableError(SearchError):
    """Search found candidates, but none yielded safely crawled page evidence."""

    def __init__(
        self,
        message: str,
        *,
        failed_candidate_hints: list[dict[str, Any]],
        retrieval_audit: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.failed_candidate_hints = failed_candidate_hints
        self.retrieval_audit = retrieval_audit


def _crawler_urlopen(request: Request, timeout: float) -> Any:
    """Use a non-redirecting transport while retaining test opener injection."""

    if urlopen is _DEFAULT_URLOPEN:
        return no_redirect_urlopen(request, timeout=timeout)
    return urlopen(request, timeout=timeout)


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
    degraded: bool = False
    error: str | None = None
    tool_attempts: int = 0
    failed_candidate_hints: list[dict[str, Any]] = field(default_factory=list)
    retrieval_audit: dict[str, Any] = field(default_factory=dict)


class FetchedPage(str):
    """Crawler text with canonical redirect provenance.

    It intentionally behaves as ``str`` so third-party crawler callers and
    existing adapters can continue to use ordinary string operations.
    """

    final_url: str
    redirect_chain: tuple[str, ...]

    def __new__(
        cls,
        content: str,
        *,
        final_url: str,
        redirect_chain: tuple[str, ...] = (),
    ) -> "FetchedPage":
        result = super().__new__(cls, content)
        result.final_url = final_url
        result.redirect_chain = redirect_chain
        return result

    @property
    def content(self) -> str:
        return str(self)


@dataclass(frozen=True)
class CrawlErrorInfo:
    error_class: str
    retryable: bool


def _classify_crawl_error(error: Exception) -> CrawlErrorInfo:
    """Classify crawl failures without exposing raw upstream diagnostics."""

    message = str(error).lower()
    status_match = re.search(r"\bstatus\s+(\d{3})\b|\bhttp\s+(\d{3})\b", message)
    status = int(next(value for value in status_match.groups() if value)) if status_match else None
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, socket.timeout)) or any(
        marker in message for marker in ("timed out", "timeout")
    ):
        return CrawlErrorInfo("timeout", True)
    if status == 429:
        return CrawlErrorInfo("http_429", True)
    if status is not None and 500 <= status <= 599:
        return CrawlErrorInfo("http_5xx", True)
    if status is not None and 400 <= status <= 499:
        return CrawlErrorInfo("http_4xx", False)
    if isinstance(error, UnsupportedContentTypeError):
        return CrawlErrorInfo("non_html", False)
    if isinstance(error, ResponseTooLargeError):
        return CrawlErrorInfo("response_too_large", False)
    if isinstance(error, URLPolicyError):
        if "dns resolution failed" in message:
            return CrawlErrorInfo("dns_failure", True)
        return CrawlErrorInfo("policy_rejected", False)
    if isinstance(error, (ConnectionError, OSError)) or any(
        marker in message
        for marker in (
            "connection refused",
            "connection reset",
            "connection aborted",
            "remote end closed",
            "temporary failure",
            "network is unreachable",
            "ssl",
        )
    ):
        return CrawlErrorInfo("connection_failure", True)
    if isinstance(error, SafeHTTPError):
        return CrawlErrorInfo("transport_failure", False)
    if "empty content" in message:
        return CrawlErrorInfo("empty_content", False)
    return CrawlErrorInfo("crawler_error", False)


def _safe_audit_text(value: Any, *, max_chars: int = 240) -> str:
    normalized = " ".join(str(value or "").split())
    return normalized[:max_chars]


def _safe_audit_url(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme.lower(), host, parsed.path, "", ""))[:2048]
    except ValueError:
        return ""


def _failed_candidate_hints(sources: list[Source]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for index, source in enumerate(sources, 1):
        if source.metadata.get("extract_status") != "crawl_failed":
            continue
        hints.append(
            {
                "title": _safe_audit_text(source.title),
                "url": _safe_audit_url(source.url),
                "query": _safe_audit_text(source.query, max_chars=320),
                "provider": _safe_audit_text(source.provider, max_chars=80),
                "rank": int(source.metadata.get("search_rank") or index),
                "crawl_status": "failed",
                "error_class": _safe_audit_text(
                    source.metadata.get("crawl_error_class") or "crawler_error",
                    max_chars=80,
                ),
                "crawl_attempts": int(source.metadata.get("crawl_attempts") or 1),
                "actual_model": _safe_audit_text(
                    source.metadata.get("gateway_model"), max_chars=120
                )
                or None,
            }
        )
    return hints


def _retrieval_audit(
    candidates: list[Source],
    verified: list[Source],
) -> dict[str, Any]:
    error_classes = Counter(
        str(source.metadata.get("crawl_error_class") or "crawler_error")
        for source in candidates
        if source.metadata.get("extract_status") == "crawl_failed"
    )
    return {
        "candidate_count": len(candidates),
        "fetchable_count": sum(
            1 for source in candidates if source.url.startswith(("http://", "https://"))
        ),
        "verified_count": len(verified),
        "crawl_attempts": sum(
            int(source.metadata.get("crawl_attempts") or 0) for source in candidates
        ),
        "error_classes": dict(sorted(error_classes.items())),
    }


def _merge_failed_candidate_hints(
    *groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for hint in group:
            fingerprint = json.dumps(hint, sort_keys=True, ensure_ascii=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            merged.append(hint)
    return merged


def _merge_retrieval_audits(*audits: dict[str, Any]) -> dict[str, Any]:
    valid = [audit for audit in audits if audit]
    if not valid:
        return {}
    errors: Counter[str] = Counter()
    for audit in valid:
        errors.update(audit.get("error_classes") or {})
    return {
        "candidate_count": sum(int(audit.get("candidate_count") or 0) for audit in valid),
        "fetchable_count": sum(int(audit.get("fetchable_count") or 0) for audit in valid),
        "verified_count": sum(int(audit.get("verified_count") or 0) for audit in valid),
        "crawl_attempts": sum(int(audit.get("crawl_attempts") or 0) for audit in valid),
        "error_classes": dict(sorted(errors.items())),
    }


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
                    metadata={
                        "pageid": page_id,
                        "search_query": search_query,
                        "snippet_only": True,
                        "extract_status": "snippet",
                        "content_type": "text/plain",
                    },
                )
            )
        if not sources:
            raise SearchError(f"wikipedia returned no results for query: {search_query}")
        return sources


class BingRssSearchAdapter:
    """No-key candidate URL discovery for personal/demo use.

    Bing RSS descriptions are search snippets, not source publication dates or
    full evidence. A configured crawler should fetch the result pages before
    synthesis when evidence-grade content is required.
    """

    name = "bing"

    def __init__(
        self,
        base_url: str = "https://global.bing.com/search",
        max_chars: int = 4000,
    ) -> None:
        self.base_url = base_url
        self.max_chars = max_chars

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        return await asyncio.to_thread(self._search_sync, query, max_results, timeout)

    def _search_sync(self, query: str, max_results: int, timeout: float) -> list[Source]:
        errors: list[str] = []
        candidates = _bing_query_candidates(query)
        for request_attempt, candidate in enumerate(candidates, 1):
            try:
                sources = self._search_once(
                    candidate,
                    query,
                    max_results,
                    timeout,
                    request_attempt,
                )
            except SearchError as exc:
                errors.append(str(exc))
                continue
            if sources:
                return sources
        error = SearchError("; ".join(errors) or "bing RSS returned no parseable results")
        error.request_attempts = max(len(candidates), 1)  # type: ignore[attr-defined]
        raise error

    def _search_once(
        self,
        search_query: str,
        original_query: str,
        max_results: int,
        timeout: float,
        request_attempt: int,
    ) -> list[Source]:
        params = urlencode(
            {
                "q": search_query,
                "format": "rss",
                "setlang": "en-us",
                "cc": "us",
            }
        )
        separator = "&" if "?" in self.base_url else "?"
        try:
            raw = fetch_text_url(
                f"{self.base_url}{separator}{params}",
                timeout=timeout,
                headers={
                    "Accept": "application/rss+xml, application/xml, text/xml",
                    "User-Agent": "deepresearch-agent/0.1 local interview project",
                },
                opener=_crawler_urlopen,
            )
            root = ET.fromstring(raw)
        except Exception as exc:  # pragma: no cover - depends on live network
            raise SearchError(str(exc)) from exc

        query_terms = _tokens(search_query)
        sources: list[Source] = []
        for index, item in enumerate(root.findall("./channel/item")[:max_results]):
            title = html.unescape((item.findtext("title") or "").strip())
            url = (item.findtext("link") or "").strip()
            description = _normalize_content(item.findtext("description") or "", self.max_chars)
            if not title or not url:
                continue
            overlap = len(query_terms & _tokens(f"{title} {description}"))
            rank_bonus = max_results - index
            sources.append(
                Source(
                    title=title,
                    url=url,
                    content=description or title,
                    provider=self.name,
                    query=original_query,
                    score=float(overlap * 8 + rank_bonus),
                    metadata={
                        "search_api": "bing_rss",
                        "search_query": search_query,
                        "provider_request_attempts": request_attempt,
                        "snippet_only": True,
                        "extract_status": "snippet",
                        "content_type": "text/plain",
                        # RSS pubDate is Bing's crawl/index timestamp, not the
                        # page's publication date, so do not map it to published_at.
                        "search_indexed_at": item.findtext("pubDate"),
                    },
                )
            )
        if not sources:
            raise SearchError(f"bing RSS returned no results for query: {search_query}")
        requested_domain = _query_domain(search_query)
        if requested_domain:
            domain_sources = [
                source
                for source in sources
                if (urlsplit(source.url).hostname or "").lower().removeprefix("www.")
                == requested_domain.removeprefix("www.")
            ]
            if domain_sources:
                return domain_sources
            direct_urls = [f"https://{requested_domain}/"]
            if re.search(r"\bdownloads?\b", search_query, flags=re.I):
                direct_urls.append(f"https://{requested_domain}/downloads/")
            return [
                Source(
                    title=f"{requested_domain} official site",
                    url=url,
                    content=(
                        "Direct official-domain fallback generated from the validated "
                        "search query; fetch the page body before using it as evidence."
                    ),
                    provider=self.name,
                    query=original_query,
                    score=float(100 - index),
                    metadata={
                        "search_api": "bing_rss",
                        "search_query": search_query,
                        "provider_request_attempts": request_attempt,
                        "direct_domain_fallback": True,
                        "snippet_only": True,
                        "extract_status": "snippet",
                        "content_type": "text/plain",
                    },
                )
                for index, url in enumerate(dict.fromkeys(direct_urls))
            ]
        return sources


class JinaReaderCrawler:
    name = "jina_reader"

    def __init__(
        self,
        base_url: str = "https://r.jina.ai/",
        max_chars: int = 4000,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.max_chars = max_chars
        self.max_response_bytes = max_response_bytes

    async def crawl(self, url: str, timeout: float) -> str:
        return await asyncio.to_thread(self._crawl_sync, url, timeout)

    def _crawl_sync(self, url: str, timeout: float) -> str:
        headers = {"User-Agent": "deepresearch-agent/0.1 local interview project"}
        api_key = os.environ.get("JINA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            # Validate the raw target before encoding it into the Reader URL;
            # validating only the outer r.jina.ai URL would leave a server-side
            # fetch primitive for private and metadata addresses.
            validate_url(url)
            encoded_target = quote(url, safe=":/?&=%")
            text = fetch_text_url(
                f"{self.base_url}{encoded_target}",
                timeout=timeout,
                headers=headers,
                max_response_bytes=self.max_response_bytes,
                opener=_crawler_urlopen,
            )
        except Exception as exc:  # pragma: no cover - depends on live network
            raise SearchError(str(exc)) from exc
        return _normalize_content(text, self.max_chars)


class HtmlTextCrawler:
    name = "html"

    def __init__(
        self,
        max_chars: int = 4000,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.max_chars = max_chars
        self.max_response_bytes = max_response_bytes

    async def crawl(self, url: str, timeout: float) -> str:
        return await asyncio.to_thread(self._crawl_sync, url, timeout)

    def _crawl_sync(self, url: str, timeout: float) -> FetchedPage:
        response = fetch_text_url(
            url,
            timeout=timeout,
            headers={"User-Agent": "deepresearch-agent/0.1 local interview project"},
            max_response_bytes=self.max_response_bytes,
            opener=_crawler_urlopen,
        )
        parser = _HtmlTextParser()
        parser.feed(response)
        return FetchedPage(
            content=_normalize_content(parser.text(), self.max_chars),
            final_url=getattr(response, "final_url", url),
            redirect_chain=tuple(getattr(response, "redirect_chain", (url,))),
        )


class JinaSearchAdapter:
    name = "jina"

    def __init__(
        self,
        base_url: str = "https://s.jina.ai/",
        max_chars: int = 4000,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.max_chars = max_chars
        self.max_response_bytes = max_response_bytes

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
        try:
            # The token-bearing search request uses the same explicit
            # redirect and SSRF policy as Jina Reader.  In particular, a 3xx
            # to another origin must never receive JINA_API_KEY.
            raw = str(
                fetch_text_url(
                    f"{self.base_url}{quote(query)}",
                    timeout=timeout,
                    headers=headers,
                    max_response_bytes=self.max_response_bytes,
                    opener=_crawler_urlopen,
                )
            )
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
            metadata={
                "search_api": "jina_search",
                "snippet_only": False,
                "extract_status": "ok",
            },
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
                    crawled = await self.crawler.crawl(url, timeout)
                    if not crawled.strip():
                        raise SearchError("crawler returned empty content")
                    content = crawled
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
                "snippet_only": self.crawler is None,
                "extract_status": "snippet" if self.crawler is None else "ok",
                "content_type": "text/plain",
            }
            if crawler_error:
                metadata.update(
                    {
                        "crawler_error": crawler_error,
                        "snippet_only": True,
                        "extract_status": "crawl_failed",
                        "degrade_reason": crawler_error,
                    }
                )
            elif self.crawler is not None:
                metadata["snippet_only"] = False
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
        crawler: WebCrawler | None = None,
        fallback_policy: str = "mock",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.settings = settings
        self.rate_limiter = rate_limiter or SearchRateLimiter(
            settings.search_rate_limit_per_second
        )
        self.retry_sleep = retry_sleep
        self.crawler = crawler
        self.fallback_policy = fallback_policy.strip().lower()
        if self.fallback_policy not in {"mock", "degraded", "fail"}:
            raise ValueError(f"unknown fallback policy: {fallback_policy}")
        self.gateway_chain = self.primary.name == "gateway-web"
        self.benchmark_source_exclusion = settings.benchmark_source_exclusion
        if self.gateway_chain and self.crawler is None:
            raise ValueError("gateway-web requires a configured safe web crawler")
        if self.gateway_chain and self.fallback.name == "mock":
            raise ValueError("gateway-web requires a real non-mock fallback provider")
        self.breaker = CircuitBreaker(
            settings.circuit_breaker_failure_threshold,
            settings.circuit_breaker_cooldown_seconds,
        )

    async def search(self, query: str, max_results: int) -> SearchOutcome:
        if self.primary.name == self.fallback.name:
            request_timeout = self._primary_timeout_seconds()
            sources = await self.fallback.search(query, max_results, request_timeout)
            enriched = enrich_source_metadata(sources)
            return SearchOutcome(
                sources=enriched,
                provider=self.fallback.name,
                tool_attempts=1,
                failed_candidate_hints=_failed_candidate_hints(enriched),
                retrieval_audit=_retrieval_audit(enriched, enriched),
            )

        last_error: str | None = None
        tool_attempts = 0
        failed_candidate_hints: list[dict[str, Any]] = []
        retrieval_audit: dict[str, Any] = {}
        if self.breaker.allow():
            request_timeout = self._primary_timeout_seconds()
            service_attempts = 1 if self.gateway_chain else self.settings.max_retries + 1
            for attempt in range(service_attempts):
                try:
                    await self.rate_limiter.wait()
                    tool_attempts += 1
                    sources = await asyncio.wait_for(
                        self.primary.search(query, max_results, request_timeout),
                        timeout=request_timeout + 0.5,
                    )
                    if not sources:
                        raise SearchError(f"{self.primary.name} returned no results")
                    self._raise_if_benchmark_contaminated(sources, query=query)
                    tool_attempts += self._provider_extra_attempts(sources)
                    self.breaker.record_success()
                    sources = await self._crawl_sources(sources)
                    tool_attempts += self._crawl_attempt_count(sources)
                    self._raise_if_benchmark_contaminated(sources, query=query)
                    (
                        sources,
                        crawl_errors,
                        current_hints,
                        current_audit,
                    ) = self._evidence_ready_sources(
                        sources, provider=self.primary.name
                    )
                    if crawl_errors:
                        error = "; ".join(dict.fromkeys(crawl_errors))
                        if not self.gateway_chain and self.fallback_policy != "degraded":
                            sources = [
                                source
                                for source in sources
                                if source.metadata.get("extract_status") == "ok"
                                and source.metadata.get("snippet_only") is False
                            ]
                        if not sources:
                            message = (
                                f"{self.primary.name} returned only unverified candidates; "
                                f"safe crawl required: {error}"
                                if self.gateway_chain
                                else f"crawler extraction failed: {error}"
                            )
                            raise SearchEvidenceUnavailableError(
                                message,
                                failed_candidate_hints=current_hints,
                                retrieval_audit=current_audit,
                            )
                        failed_candidate_hints = _merge_failed_candidate_hints(
                            failed_candidate_hints, current_hints
                        )
                        retrieval_audit = _merge_retrieval_audits(
                            retrieval_audit, current_audit
                        )
                        return SearchOutcome(
                            sources=enrich_source_metadata(sources),
                            provider=self.primary.name,
                            degraded=True,
                            error=error,
                            tool_attempts=tool_attempts,
                            failed_candidate_hints=failed_candidate_hints,
                            retrieval_audit=retrieval_audit,
                        )
                    failed_candidate_hints = _merge_failed_candidate_hints(
                        failed_candidate_hints, current_hints
                    )
                    retrieval_audit = _merge_retrieval_audits(
                        retrieval_audit, current_audit
                    )
                    return SearchOutcome(
                        sources=enrich_source_metadata(sources),
                        provider=self.primary.name,
                        tool_attempts=tool_attempts,
                        failed_candidate_hints=failed_candidate_hints,
                        retrieval_audit=retrieval_audit,
                    )
                except BenchmarkContaminationError:
                    # A benchmark answer page is not an ordinary transient search
                    # failure. Do not silently fall through to another provider and
                    # score the case as though retrieval were clean.
                    raise
                except Exception as exc:
                    last_error = str(exc)
                    failed_candidate_hints = _merge_failed_candidate_hints(
                        failed_candidate_hints,
                        getattr(exc, "failed_candidate_hints", []),
                    )
                    retrieval_audit = _merge_retrieval_audits(
                        retrieval_audit,
                        getattr(exc, "retrieval_audit", {}),
                    )
                    tool_attempts += self._exception_extra_attempts(exc)
                    self.breaker.record_failure()
                    if isinstance(exc, GatewayWebSearchNoResultsError):
                        break
                    if attempt < service_attempts - 1:
                        await self._retry_backoff(attempt)
        else:
            last_error = "circuit breaker open"

        if self.fallback.name != "mock":
            return await self._search_real_fallback(
                query,
                max_results,
                primary_error=last_error,
                tool_attempts=tool_attempts,
                failed_candidate_hints=failed_candidate_hints,
                retrieval_audit=retrieval_audit,
            )
        if self.fallback_policy == "fail":
            if failed_candidate_hints:
                raise SearchEvidenceUnavailableError(
                    last_error or f"{self.primary.name} search failed",
                    failed_candidate_hints=failed_candidate_hints,
                    retrieval_audit=retrieval_audit,
                )
            raise SearchError(last_error or f"{self.primary.name} search failed")
        if self.fallback_policy == "degraded":
            return SearchOutcome(
                sources=[],
                provider=self.primary.name,
                degraded=True,
                error=last_error,
                tool_attempts=tool_attempts,
                failed_candidate_hints=failed_candidate_hints,
                retrieval_audit=retrieval_audit,
            )

        tool_attempts += 1
        fallback_sources = await self.fallback.search(
            query, max_results, self.settings.request_timeout_seconds
        )
        return SearchOutcome(
            sources=enrich_source_metadata(
                fallback_sources,
                fallback_used=True,
                degrade_reason=last_error,
            ),
            provider=self.fallback.name,
            fallback_used=True,
            error=last_error,
            tool_attempts=tool_attempts,
            retrieval_audit=_retrieval_audit(fallback_sources, fallback_sources),
        )

    async def _search_real_fallback(
        self,
        query: str,
        max_results: int,
        *,
        primary_error: str | None,
        tool_attempts: int,
        failed_candidate_hints: list[dict[str, Any]],
        retrieval_audit: dict[str, Any],
    ) -> SearchOutcome:
        fallback_error: str | None = None
        try:
            await self.rate_limiter.wait()
            tool_attempts += 1
            timeout = self.settings.request_timeout_seconds
            fallback_sources = await asyncio.wait_for(
                self.fallback.search(query, max_results, timeout),
                timeout=timeout + 0.5,
            )
            if not fallback_sources:
                raise SearchError(f"{self.fallback.name} returned no results")
            self._raise_if_benchmark_contaminated(fallback_sources, query=query)
            tool_attempts += self._provider_extra_attempts(fallback_sources)
            fallback_sources = await self._crawl_sources(fallback_sources)
            tool_attempts += self._crawl_attempt_count(fallback_sources)
            self._raise_if_benchmark_contaminated(fallback_sources, query=query)
            (
                fallback_sources,
                crawl_errors,
                fallback_hints,
                fallback_audit,
            ) = self._evidence_ready_sources(
                fallback_sources, provider=self.fallback.name
            )
            fallback_error = "; ".join(dict.fromkeys(crawl_errors)) or None
            if not fallback_sources:
                detail = fallback_error or "no page body was extracted"
                raise SearchEvidenceUnavailableError(
                    f"crawler extraction failed: {detail}",
                    failed_candidate_hints=fallback_hints,
                    retrieval_audit=fallback_audit,
                )
            failed_candidate_hints = _merge_failed_candidate_hints(
                failed_candidate_hints, fallback_hints
            )
            retrieval_audit = _merge_retrieval_audits(
                retrieval_audit, fallback_audit
            )
            audit_error = "; ".join(
                item
                for item in (
                    f"{self.primary.name}: {primary_error}" if primary_error else None,
                    f"{self.fallback.name}: {fallback_error}" if fallback_error else None,
                )
                if item
            )
            marked_sources = []
            for source in fallback_sources:
                metadata = {
                    **source.metadata,
                    "fallback_used": True,
                    "fallback_from": self.primary.name,
                    "fallback_provider": self.fallback.name,
                    "fallback_search_attempts": int(
                        source.metadata.get("provider_request_attempts") or 1
                    ),
                    "primary_error": primary_error,
                }
                marked_sources.append(source.model_copy(update={"metadata": metadata}))
            return SearchOutcome(
                sources=enrich_source_metadata(
                    marked_sources,
                    fallback_used=True,
                    degrade_reason=audit_error or None,
                ),
                provider=self.fallback.name,
                fallback_used=True,
                degraded=bool(fallback_error),
                error=audit_error or None,
                tool_attempts=tool_attempts,
                failed_candidate_hints=failed_candidate_hints,
                retrieval_audit=retrieval_audit,
            )
        except BenchmarkContaminationError:
            raise
        except Exception as exc:
            failed_candidate_hints = _merge_failed_candidate_hints(
                failed_candidate_hints,
                getattr(exc, "failed_candidate_hints", []),
            )
            retrieval_audit = _merge_retrieval_audits(
                retrieval_audit,
                getattr(exc, "retrieval_audit", {}),
            )
            tool_attempts += self._exception_extra_attempts(exc)
            fallback_error = str(exc)

        combined_error = "; ".join(
            item
            for item in (
                f"{self.primary.name}: {primary_error}" if primary_error else None,
                f"{self.fallback.name}: {fallback_error}" if fallback_error else None,
            )
            if item
        )
        if self.fallback_policy == "degraded":
            return SearchOutcome(
                sources=[],
                provider=self.fallback.name,
                fallback_used=True,
                degraded=True,
                error=combined_error or None,
                tool_attempts=tool_attempts,
                failed_candidate_hints=failed_candidate_hints,
                retrieval_audit=retrieval_audit,
            )
        if failed_candidate_hints:
            raise SearchEvidenceUnavailableError(
                combined_error or "real search providers failed",
                failed_candidate_hints=failed_candidate_hints,
                retrieval_audit=retrieval_audit,
            )
        raise SearchError(combined_error or "real search providers failed")

    def _provider_extra_attempts(self, sources: list[Source]) -> int:
        attempts = [
            source.metadata.get("provider_request_attempts")
            for source in sources
            if source.metadata.get("provider_request_attempts") is not None
        ]
        parsed = [int(value) for value in attempts if str(value).isdigit()]
        return max(max(parsed, default=1) - 1, 0)

    def _raise_if_benchmark_contaminated(
        self,
        sources: list[Source],
        *,
        query: str,
    ) -> None:
        if not self.benchmark_source_exclusion:
            return
        for source in sources:
            reason = _benchmark_contamination_reason(source, query=query)
            if reason:
                raise BenchmarkContaminationError(
                    f"benchmark contamination blocked: {reason}"
                )

    @staticmethod
    def _exception_extra_attempts(error: Exception) -> int:
        attempts = getattr(error, "request_attempts", 1)
        try:
            return max(int(attempts) - 1, 0)
        except (TypeError, ValueError):
            return 0

    def _evidence_ready_sources(
        self,
        sources: list[Source],
        *,
        provider: str,
    ) -> tuple[
        list[Source],
        list[str],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        crawl_errors = [
            str(source.metadata.get("crawl_error_class") or "crawler_error")
            for source in sources
            if source.metadata.get("extract_status") == "crawl_failed"
        ]
        requires_crawl = self.gateway_chain and provider in {
            self.primary.name,
            self.fallback.name,
        }
        if not requires_crawl:
            return sources, crawl_errors, _failed_candidate_hints(sources), _retrieval_audit(
                sources, sources
            )
        evidence_ready = [
            source
            for source in sources
            if source.metadata.get("extract_status") == "ok"
            and source.metadata.get("snippet_only") is False
            and source.metadata.get("crawler") not in {None, "", "none"}
        ]
        del provider
        return (
            evidence_ready,
            crawl_errors,
            _failed_candidate_hints(sources),
            _retrieval_audit(sources, evidence_ready),
        )

    def _crawl_attempt_count(self, sources: list[Source]) -> int:
        return sum(
            int(source.metadata.get("crawl_attempts") or 0) for source in sources
        )

    def _primary_timeout_seconds(self) -> float:
        timeout = getattr(self.primary, "timeout_seconds", None)
        if isinstance(timeout, (int, float)) and timeout > 0:
            return float(timeout)
        return self.settings.request_timeout_seconds

    async def _crawl_sources(self, sources: list[Source]) -> list[Source]:
        if self.crawler is None:
            return sources

        async def crawl_one(source: Source) -> Source:
            if (
                source.provider == "mock"
                or not source.url.startswith(("http://", "https://"))
                or source.metadata.get("crawler") not in {None, "", "none"}
            ):
                return source
            metadata = dict(source.metadata)
            metadata.setdefault("search_snippet", source.content)
            metadata["crawler"] = self.crawler.name
            for attempt in (1, 2):
                try:
                    crawled = await self.crawler.crawl(
                        source.url,
                        self.settings.request_timeout_seconds,
                    )
                    if isinstance(crawled, FetchedPage):
                        content = crawled.content
                        final_url = crawled.final_url
                        redirect_chain = crawled.redirect_chain
                    else:
                        content = str(crawled)
                        final_url = source.url
                        redirect_chain = (source.url,)
                    if not content.strip():
                        raise SearchError("crawler returned empty content")
                    # The fetcher validated each redirect target. Preserve the final
                    # canonical URL so evidence, deduplication and diversity metrics
                    # do not count several aliases as independent pages.
                    validate_url(final_url)
                    metadata.update(
                        {
                            "extract_status": "ok",
                            "content_type": "text/plain",
                            "snippet_only": False,
                            "candidate_only": False,
                            "requires_crawl": False,
                            "verification_status": "crawled",
                            "redirect_chain": list(redirect_chain),
                            "crawl_attempts": attempt,
                        }
                    )
                    return source.model_copy(
                        update={"url": final_url, "content": content, "metadata": metadata}
                    )
                except Exception as exc:  # noqa: BLE001 - classify before bounded retry.
                    error_info = _classify_crawl_error(exc)
                    if attempt == 1 and error_info.retryable:
                        await self._retry_backoff(0)
                        continue
                    metadata.update(
                        {
                            "extract_status": "crawl_failed",
                            "crawler_error": str(exc),
                            "crawl_error_class": error_info.error_class,
                            "crawl_retryable": error_info.retryable,
                            "crawl_attempts": attempt,
                            "degrade_reason": error_info.error_class,
                            "snippet_only": True,
                            "candidate_only": True,
                            "verification_status": "crawl_failed",
                        }
                    )
                    return source.model_copy(update={"metadata": metadata})

            raise AssertionError("crawler retry loop exhausted without a result")

        return list(await asyncio.gather(*(crawl_one(source) for source in sources)))

    async def _retry_backoff(self, attempt: int) -> None:
        base_delay = self.settings.search_retry_backoff_seconds
        if base_delay <= 0:
            return
        await self.retry_sleep(base_delay * (2**attempt))


def enrich_source_metadata(
    sources: list[Source],
    *,
    fallback_used: bool = False,
    degrade_reason: str | None = None,
) -> list[Source]:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    enriched: list[Source] = []
    for source in sources:
        web_search_provider = source.provider in {
            "wikipedia",
            "bing",
            "searxng",
            "jina",
            "brave",
            "tavily",
            "gateway-web",
            "mcp",
        }
        default_extract_status = "snippet" if web_search_provider else (
            "ok" if source.content.strip() else "empty"
        )
        metadata = {
            **source.metadata,
            "retrieved_at": source.metadata.get("retrieved_at", retrieved_at),
            "content_hash": hashlib.sha256(source.content.encode("utf-8")).hexdigest(),
            "content_type": source.metadata.get("content_type", "text/plain"),
            "extract_status": source.metadata.get(
                "extract_status", default_extract_status
            ),
            "snippet_only": source.metadata.get(
                "snippet_only", web_search_provider
            ),
            "fallback_used": source.metadata.get("fallback_used", fallback_used),
            "degrade_reason": degrade_reason or source.metadata.get("degrade_reason"),
            "published_at": source.metadata.get("published_at"),
        }
        enriched.append(source.model_copy(update={"metadata": metadata}))
    return enriched


def build_search_service(
    settings: Settings,
    provider: str | None = None,
    fallback_policy: str = "mock",
) -> SearchService:
    selected = (provider or settings.search_provider).strip().lower()
    crawler = build_crawler(settings)
    if selected in {"gateway-web", "gateway_web"}:
        fallback: SearchAdapter = BingRssSearchAdapter(
            base_url=settings.bing_search_base_url,
            max_chars=settings.crawler_max_chars,
        )
    else:
        fallback = MockSearchAdapter()
    primary = build_search_adapter(settings, selected, fallback)
    return SearchService(
        primary=primary,
        fallback=fallback,
        settings=settings,
        crawler=crawler,
        fallback_policy=fallback_policy,
    )


def build_search_adapter(
    settings: Settings,
    selected: str,
    fallback: SearchAdapter | None = None,
) -> SearchAdapter:
    if selected == "mock":
        return fallback or MockSearchAdapter()
    if selected == "wikipedia":
        return WikipediaSearchAdapter()
    if selected == "bing":
        return BingRssSearchAdapter(
            base_url=settings.bing_search_base_url,
            max_chars=settings.crawler_max_chars,
        )
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
    if selected in {"gateway-web", "gateway_web"}:
        return GatewayWebSearchAdapter(
            base_url=settings.llm_gateway_base_url,
            model=settings.gateway_web_search_model,
            max_chars=settings.crawler_max_chars,
            timeout_seconds=settings.llm_gateway_timeout_seconds,
            require_response_model_match=(
                settings.llm_gateway_require_response_model_match
            ),
            thinking_budget_tokens=settings.llm_gateway_thinking_budget_tokens,
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
    return tokenize(text)


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
                metadata={
                    **metadata,
                    "snippet_only": metadata.get("snippet_only", True),
                    "extract_status": metadata.get("extract_status", "snippet"),
                    "content_type": metadata.get("content_type", "text/plain"),
                    "published_at": row.get("published_at") or row.get("published") or row.get("date"),
                },
            )
        )
    return sources


class _HtmlTextParser(HTMLParser):
    # Non-content tags: their text is navigation, chrome, or boilerplate that
    # crowds out the article body within a fixed char budget.
    _SKIP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "menu",
    }
    # Main-content tags: when present, prefer their text over the rest so a
    # nav-heavy page does not waste the budget on menus before the article.
    _MAIN_TAGS = {"article", "main"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._main_depth = 0
        self._parts: list[str] = []
        self._main_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lower = tag.lower()
        if lower in self._SKIP_TAGS:
            self._skip_depth += 1
        elif lower in self._MAIN_TAGS:
            self._main_depth += 1

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif lower in self._MAIN_TAGS and self._main_depth:
            self._main_depth -= 1

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        if self._skip_depth:
            return
        self._parts.append(data.strip())
        if self._main_depth:
            self._main_parts.append(data.strip())

    def text(self) -> str:
        # Prefer article/main body when the page exposed it; otherwise fall
        # back to nav-stripped full text.
        return " ".join(self._main_parts or self._parts)


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

_BENCHMARK_PATH_MARKER = re.compile(
    r"(?:^|[/_\-.])(?:simpleqa|simple-evals?|livedrbench)(?:$|[/_\-.])",
    re.IGNORECASE,
)
_BENCHMARK_ANSWER_FIELD = re.compile(
    r"(?:\"|\b)(?:reference_?answer|ground_?truths?|expected_?answer|answers?)(?:\"|\b)\s*[:=]",
    re.IGNORECASE,
)


def _benchmark_contamination_reason(source: Source, *, query: str) -> str | None:
    """Identify known public benchmark pages before they can become evidence.

    We deliberately do not block GitHub or Hugging Face broadly: legitimate
    technical documentation remains useful. The policy targets known benchmark
    paths, plus pages that expose a query together with answer-key fields.
    """

    parsed = urlsplit(source.url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path_and_title = f"{parsed.path} {source.title}".lower()
    if host in {
        "github.com",
        "raw.githubusercontent.com",
        "huggingface.co",
        "hf.co",
        "datasets-server.huggingface.co",
    } and _BENCHMARK_PATH_MARKER.search(path_and_title):
        return "known benchmark repository or dataset path"

    content = source.content.lower()
    normalized_query = re.sub(r"\s+", " ", query.strip().lower())
    if normalized_query and len(normalized_query) >= 16:
        compact_content = re.sub(r"\s+", " ", content)
        if normalized_query in compact_content and _BENCHMARK_ANSWER_FIELD.search(content):
            return "page contains the queried prompt and benchmark answer fields"
    return None


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


def _bing_query_candidates(query: str) -> list[str]:
    candidates = [query.strip()]
    domain_match = re.search(r"\b(?:www\.)?([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", query)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", query)
    keywords: list[str] = []
    for word in words:
        lowered = word.lower().strip(".")
        if lowered in WIKIPEDIA_STOPWORDS or lowered in {"who", "latest", "current"}:
            continue
        if len(lowered) <= 2 and not lowered.isdigit():
            continue
        if lowered not in {item.lower() for item in keywords}:
            keywords.append(word)
    if keywords:
        compact = " ".join(keywords[:10])
        if domain_match:
            domain = domain_match.group(1).lower()
            compact = re.sub(rf"\b{re.escape(domain)}\b", "", compact, flags=re.I)
            compact = f"{compact.strip()} site:{domain}"
        candidates.append(compact)
    quoted = re.findall(r'"([^"\n]{3,100})"', query)
    if quoted:
        candidates.append(" ".join(f'"{item}"' for item in quoted[:2]))
    deduped: list[str] = []
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _query_domain(query: str) -> str | None:
    match = re.search(r"(?:site:)?(?:www\.)?([A-Za-z0-9.-]+\.[A-Za-z]{2,})", query)
    return match.group(1).lower() if match else None
