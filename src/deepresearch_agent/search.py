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
from urllib.parse import quote, quote_plus, unquote, urlencode, urlsplit, urlunsplit
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
    canonical_url_identity,
    url_identity_matches_blocked,
    fetch_text_url,
    no_redirect_urlopen,
    validate_url,
)


_DEFAULT_URLOPEN = urlopen


class SearchError(RuntimeError):
    pass


class BenchmarkContaminationError(SearchError):
    """A public benchmark answer page was discovered during an evaluation run."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "content",
        benchmark_contamination: bool = True,
        protocol_violation: dict[str, Any] | None = None,
        retrieval_audit: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.benchmark_contamination = benchmark_contamination
        self.protocol_violation = protocol_violation
        self.retrieval_audit = retrieval_audit or {}


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


class SearchDeadlineExceeded(TimeoutError):
    """A search phase exhausted its caller-owned absolute deadline."""

    def __init__(
        self,
        message: str = "search deadline exceeded",
        *,
        failed_candidate_hints: list[dict[str, Any]] | None = None,
        retrieval_audit: dict[str, Any] | None = None,
        tool_attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.failed_candidate_hints = failed_candidate_hints or []
        self.retrieval_audit = retrieval_audit or {}
        self.tool_attempts = max(int(tool_attempts), 0)


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
    denylist_enforcement_hit: bool = False
    benchmark_contamination: bool = False
    protocol_violations: list[dict[str, Any]] = field(default_factory=list)


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
        protocol_violation = source.metadata.get("protocol_violation")
        if isinstance(protocol_violation, dict):
            hints.append(
                {
                    "title": _safe_audit_text(source.title),
                    "url_identity_sha256": _safe_audit_text(
                        protocol_violation.get("url_identity_sha256"), max_chars=64
                    ),
                    "query": _safe_audit_text(source.query, max_chars=320),
                    "provider": _safe_audit_text(source.provider, max_chars=80),
                    "rank": int(source.metadata.get("search_rank") or index),
                    "crawl_status": "blocked",
                    "error_class": "denylist_enforcement",
                    "protocol_violation": dict(protocol_violation),
                    "crawl_attempts": int(source.metadata.get("crawl_attempts") or 0),
                    "actual_model": _safe_audit_text(
                        source.metadata.get("gateway_model"), max_chars=120
                    )
                    or None,
                }
            )
            continue
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
                "crawl_attempts": (
                    int(source.metadata["crawl_attempts"])
                    if "crawl_attempts" in source.metadata
                    else 1
                ),
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
    audit = {
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
    protocol_violations = [
        dict(violation)
        for source in candidates
        if isinstance((violation := source.metadata.get("protocol_violation")), dict)
    ]
    if protocol_violations:
        audit.update(
            {
                "denylist_enforcement_hit": True,
                "benchmark_contamination": False,
                "blocked_count": len(protocol_violations),
                "protocol_violation_count": len(protocol_violations),
                "protocol_violations": protocol_violations,
            }
        )
    return audit


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
    merged = {
        "candidate_count": sum(int(audit.get("candidate_count") or 0) for audit in valid),
        "fetchable_count": sum(int(audit.get("fetchable_count") or 0) for audit in valid),
        "verified_count": sum(int(audit.get("verified_count") or 0) for audit in valid),
        "crawl_attempts": sum(int(audit.get("crawl_attempts") or 0) for audit in valid),
        "error_classes": dict(sorted(errors.items())),
    }
    protocol_violations = [
        dict(violation)
        for audit in valid
        for violation in (audit.get("protocol_violations") or [])
        if isinstance(violation, dict)
    ]
    if protocol_violations:
        merged.update(
            {
                "denylist_enforcement_hit": True,
                "benchmark_contamination": False,
                "blocked_count": sum(
                    int(audit.get("blocked_count") or 0) for audit in valid
                ),
                "protocol_violation_count": sum(
                    int(audit.get("protocol_violation_count") or 0)
                    for audit in valid
                ),
                "protocol_violations": protocol_violations,
            }
        )
    if any(bool(audit.get("benchmark_contamination")) for audit in valid):
        merged["benchmark_contamination"] = True
    return merged


def _audit_has_denylist_hit(audit: dict[str, Any]) -> bool:
    return bool(audit.get("denylist_enforcement_hit"))


def _protocol_violations(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(violation)
        for violation in (audit.get("protocol_violations") or [])
        if isinstance(violation, dict)
    ]


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
        deadline_at = time.monotonic() + timeout
        for request_attempt, candidate in enumerate(candidates, 1):
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                break
            try:
                sources = self._search_once(
                    candidate,
                    query,
                    max_results,
                    remaining,
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

    async def crawl(
        self,
        url: str,
        timeout: float,
        *,
        url_guard: Callable[[str], None] | None = None,
    ) -> str:
        return await asyncio.to_thread(self._crawl_sync, url, timeout, url_guard)

    def _crawl_sync(
        self,
        url: str,
        timeout: float,
        url_guard: Callable[[str], None] | None = None,
    ) -> FetchedPage:
        response = fetch_text_url(
            url,
            timeout=timeout,
            headers={"User-Agent": "deepresearch-agent/0.1 local interview project"},
            max_response_bytes=self.max_response_bytes,
            opener=_crawler_urlopen,
            url_guard=url_guard,
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
        self.crawler_concurrency_per_search = max(
            int(settings.crawler_concurrency_per_search), 1
        )
        if self.gateway_chain and self.crawler is None:
            raise ValueError("gateway-web requires a configured safe web crawler")
        if self.gateway_chain and self.fallback.name == "mock":
            raise ValueError("gateway-web requires a real non-mock fallback provider")
        self.breaker = CircuitBreaker(
            settings.circuit_breaker_failure_threshold,
            settings.circuit_breaker_cooldown_seconds,
        )

    async def search(
        self,
        query: str,
        max_results: int,
        *,
        blocked_source_urls: list[str] | tuple[str, ...] = (),
        deadline_at: float | None = None,
    ) -> SearchOutcome:
        blocked_source_identities = _blocked_source_identities(blocked_source_urls)
        if self.primary.name == self.fallback.name:
            request_timeout = self._operation_timeout(
                self._primary_timeout_seconds(), deadline_at=deadline_at
            )
            candidates = await asyncio.wait_for(
                self.fallback.search(query, max_results, request_timeout),
                timeout=request_timeout,
            )
            clean_sources, blocked_sources = self._partition_blocked_sources(
                candidates,
                blocked_source_identities=blocked_source_identities,
                stage="candidate",
            )
            self._raise_if_benchmark_contaminated(
                clean_sources,
                query=query,
                stage="candidate_content",
            )
            audit_candidates = [*blocked_sources, *clean_sources]
            audit = _retrieval_audit(audit_candidates, clean_sources)
            if not clean_sources:
                raise SearchEvidenceUnavailableError(
                    f"{self.fallback.name} returned no safely usable candidates",
                    failed_candidate_hints=_failed_candidate_hints(audit_candidates),
                    retrieval_audit=audit,
                )
            self._assert_final_sources_clean(
                clean_sources,
                blocked_source_identities=blocked_source_identities,
            )
            enriched = enrich_source_metadata(clean_sources)
            return SearchOutcome(
                sources=enriched,
                provider=self.fallback.name,
                tool_attempts=1,
                failed_candidate_hints=_failed_candidate_hints(audit_candidates),
                retrieval_audit=audit,
                denylist_enforcement_hit=_audit_has_denylist_hit(audit),
                protocol_violations=_protocol_violations(audit),
            )

        last_error: str | None = None
        tool_attempts = 0
        failed_candidate_hints: list[dict[str, Any]] = []
        retrieval_audit: dict[str, Any] = {}
        if self.breaker.allow():
            request_timeout = self._operation_timeout(
                self._primary_timeout_seconds(), deadline_at=deadline_at
            )
            service_attempts = 1 if self.gateway_chain else self.settings.max_retries + 1
            for attempt in range(service_attempts):
                try:
                    await self.rate_limiter.wait()
                    request_timeout = self._operation_timeout(
                        self._primary_timeout_seconds(), deadline_at=deadline_at
                    )
                    tool_attempts += 1
                    sources = await asyncio.wait_for(
                        self.primary.search(query, max_results, request_timeout),
                        timeout=request_timeout,
                    )
                    if not sources:
                        raise SearchError(f"{self.primary.name} returned no results")
                    tool_attempts += self._provider_extra_attempts(sources)
                    sources, blocked_sources = self._partition_blocked_sources(
                        sources,
                        blocked_source_identities=blocked_source_identities,
                        stage="candidate",
                    )
                    self._raise_if_benchmark_contaminated(
                        sources,
                        query=query,
                        stage="candidate_content",
                    )
                    self.breaker.record_success()
                    crawled_sources = await self._crawl_sources(
                        sources,
                        blocked_source_identities=blocked_source_identities,
                        deadline_at=deadline_at,
                    )
                    sources = [*blocked_sources, *crawled_sources]
                    tool_attempts += self._crawl_attempt_count(crawled_sources)
                    self._raise_if_benchmark_contaminated(
                        crawled_sources,
                        query=query,
                        stage="crawled_content",
                    )
                    (
                        sources,
                        crawl_errors,
                        current_hints,
                        current_audit,
                    ) = self._evidence_ready_sources(
                        sources, provider=self.primary.name
                    )
                    if not sources:
                        detail = "; ".join(dict.fromkeys(crawl_errors))
                        if detail:
                            message = (
                                f"{self.primary.name} returned only unverified candidates; "
                                f"safe crawl required: {detail}"
                                if self.gateway_chain
                                else f"crawler extraction failed: {detail}"
                            )
                        else:
                            message = (
                                f"{self.primary.name} returned no safely usable candidates"
                            )
                        raise SearchEvidenceUnavailableError(
                            message,
                            failed_candidate_hints=current_hints,
                            retrieval_audit=current_audit,
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
                        self._assert_final_sources_clean(
                            sources,
                            blocked_source_identities=blocked_source_identities,
                        )
                        return SearchOutcome(
                            sources=enrich_source_metadata(sources),
                            provider=self.primary.name,
                            degraded=True,
                            error=error,
                            tool_attempts=tool_attempts,
                            failed_candidate_hints=failed_candidate_hints,
                            retrieval_audit=retrieval_audit,
                            denylist_enforcement_hit=_audit_has_denylist_hit(
                                retrieval_audit
                            ),
                            protocol_violations=_protocol_violations(retrieval_audit),
                        )
                    failed_candidate_hints = _merge_failed_candidate_hints(
                        failed_candidate_hints, current_hints
                    )
                    retrieval_audit = _merge_retrieval_audits(
                        retrieval_audit, current_audit
                    )
                    self._assert_final_sources_clean(
                        sources,
                        blocked_source_identities=blocked_source_identities,
                    )
                    return SearchOutcome(
                        sources=enrich_source_metadata(sources),
                        provider=self.primary.name,
                        tool_attempts=tool_attempts,
                        failed_candidate_hints=failed_candidate_hints,
                        retrieval_audit=retrieval_audit,
                        denylist_enforcement_hit=_audit_has_denylist_hit(
                            retrieval_audit
                        ),
                        protocol_violations=_protocol_violations(retrieval_audit),
                    )
                except BenchmarkContaminationError:
                    # A benchmark answer page is not an ordinary transient search
                    # failure. Do not silently fall through to another provider and
                    # score the case as though retrieval were clean.
                    raise
                except SearchDeadlineExceeded as exc:
                    exc.failed_candidate_hints = _merge_failed_candidate_hints(
                        failed_candidate_hints, exc.failed_candidate_hints
                    )
                    exc.retrieval_audit = _merge_retrieval_audits(
                        retrieval_audit, exc.retrieval_audit
                    )
                    exc.tool_attempts += tool_attempts
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
                    if not _audit_has_denylist_hit(
                        getattr(exc, "retrieval_audit", {})
                    ):
                        self.breaker.record_failure()
                    if isinstance(exc, GatewayWebSearchNoResultsError):
                        break
                    if _audit_has_denylist_hit(
                        getattr(exc, "retrieval_audit", {})
                    ):
                        # Retrying the same provider can rediscover the same
                        # forbidden URL. Move directly to a configured real fallback.
                        break
                    if attempt < service_attempts - 1:
                        await self._retry_backoff(attempt, deadline_at=deadline_at)
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
                blocked_source_identities=blocked_source_identities,
                deadline_at=deadline_at,
            )
        if _audit_has_denylist_hit(retrieval_audit):
            raise SearchEvidenceUnavailableError(
                last_error or f"{self.primary.name} candidates were denylist-blocked",
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

        fallback_timeout = self._operation_timeout(
            self.settings.request_timeout_seconds, deadline_at=deadline_at
        )
        tool_attempts += 1
        fallback_sources = await asyncio.wait_for(
            self.fallback.search(query, max_results, fallback_timeout),
            timeout=fallback_timeout,
        )
        fallback_sources, blocked_fallback_sources = self._partition_blocked_sources(
            fallback_sources,
            blocked_source_identities=blocked_source_identities,
            stage="candidate",
        )
        self._raise_if_benchmark_contaminated(
            fallback_sources, query=query, stage="candidate_content"
        )
        fallback_audit = _retrieval_audit(
            [*blocked_fallback_sources, *fallback_sources], fallback_sources
        )
        if not fallback_sources:
            raise SearchEvidenceUnavailableError(
                f"{self.fallback.name} returned no safely usable candidates",
                failed_candidate_hints=_failed_candidate_hints(
                    [*blocked_fallback_sources, *fallback_sources]
                ),
                retrieval_audit=_merge_retrieval_audits(
                    retrieval_audit, fallback_audit
                ),
            )
        self._assert_final_sources_clean(
            fallback_sources,
            blocked_source_identities=blocked_source_identities,
        )
        retrieval_audit = _merge_retrieval_audits(retrieval_audit, fallback_audit)
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
            retrieval_audit=retrieval_audit,
            denylist_enforcement_hit=_audit_has_denylist_hit(retrieval_audit),
            protocol_violations=_protocol_violations(retrieval_audit),
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
        blocked_source_identities: frozenset[tuple[str, str, int | None, str]],
        deadline_at: float | None,
    ) -> SearchOutcome:
        fallback_error: str | None = None
        try:
            await self.rate_limiter.wait()
            timeout = self._operation_timeout(
                self.settings.request_timeout_seconds, deadline_at=deadline_at
            )
            tool_attempts += 1
            fallback_sources = await asyncio.wait_for(
                self.fallback.search(query, max_results, timeout),
                timeout=timeout,
            )
            if not fallback_sources:
                raise SearchError(f"{self.fallback.name} returned no results")
            tool_attempts += self._provider_extra_attempts(fallback_sources)
            fallback_sources, blocked_fallback_sources = self._partition_blocked_sources(
                fallback_sources,
                blocked_source_identities=blocked_source_identities,
                stage="candidate",
            )
            self._raise_if_benchmark_contaminated(
                fallback_sources,
                query=query,
                stage="candidate_content",
            )
            crawled_fallback_sources = await self._crawl_sources(
                fallback_sources,
                blocked_source_identities=blocked_source_identities,
                deadline_at=deadline_at,
            )
            fallback_sources = [*blocked_fallback_sources, *crawled_fallback_sources]
            tool_attempts += self._crawl_attempt_count(crawled_fallback_sources)
            self._raise_if_benchmark_contaminated(
                crawled_fallback_sources,
                query=query,
                stage="crawled_content",
            )
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
                detail = fallback_error or "no safely usable candidates remained"
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
            self._assert_final_sources_clean(
                fallback_sources,
                blocked_source_identities=blocked_source_identities,
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
                denylist_enforcement_hit=_audit_has_denylist_hit(retrieval_audit),
                protocol_violations=_protocol_violations(retrieval_audit),
            )
        except BenchmarkContaminationError:
            raise
        except SearchDeadlineExceeded as exc:
            exc.failed_candidate_hints = _merge_failed_candidate_hints(
                failed_candidate_hints, exc.failed_candidate_hints
            )
            exc.retrieval_audit = _merge_retrieval_audits(
                retrieval_audit, exc.retrieval_audit
            )
            exc.tool_attempts += tool_attempts
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
            if deadline_at is not None and time.monotonic() >= deadline_at:
                raise SearchDeadlineExceeded(
                    failed_candidate_hints=failed_candidate_hints,
                    retrieval_audit=retrieval_audit,
                    tool_attempts=tool_attempts,
                ) from exc

        combined_error = "; ".join(
            item
            for item in (
                f"{self.primary.name}: {primary_error}" if primary_error else None,
                f"{self.fallback.name}: {fallback_error}" if fallback_error else None,
            )
            if item
        )
        if _audit_has_denylist_hit(retrieval_audit):
            raise SearchEvidenceUnavailableError(
                combined_error or "all real search candidates were denylist-blocked",
                failed_candidate_hints=failed_candidate_hints,
                retrieval_audit=retrieval_audit,
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
        stage: str = "content",
    ) -> None:
        for source in sources:
            if source.metadata.get("extract_status") == "blocked":
                continue
            reason = (
                _benchmark_content_contamination_reason(source, query=query)
                if self.benchmark_source_exclusion
                else None
            )
            if reason:
                violation = self._protocol_violation(
                    source.url,
                    stage=stage,
                    reason="benchmark answer content reached retrieval",
                )
                audit = {
                    "benchmark_contamination": True,
                    "denylist_enforcement_hit": False,
                    "protocol_violation_count": 1,
                    "protocol_violations": [violation],
                }
                raise BenchmarkContaminationError(
                    f"benchmark contamination detected: {reason}",
                    stage=stage,
                    benchmark_contamination=True,
                    protocol_violation=violation,
                    retrieval_audit=audit,
                )

    def _partition_blocked_sources(
        self,
        sources: list[Source],
        *,
        blocked_source_identities: frozenset[
            tuple[str, str, int | None, str]
        ],
        stage: str,
    ) -> tuple[list[Source], list[Source]]:
        clean: list[Source] = []
        blocked: list[Source] = []
        for source in sources:
            violation = self._url_protocol_violation(
                source.url,
                blocked_source_identities=blocked_source_identities,
                stage=stage,
            )
            if violation is None:
                clean.append(source)
                continue
            blocked.append(self._mark_source_blocked(source, violation=violation))
        return clean, blocked

    def _url_contamination_reason(
        self,
        url: str,
        *,
        blocked_source_identities: frozenset[
            tuple[str, str, int | None, str]
        ],
    ) -> str | None:
        if blocked_source_identities and url.startswith(("http://", "https://")):
            candidate_identity = canonical_url_identity(url)
            if any(
                url_identity_matches_blocked(candidate_identity, blocked_identity)
                for blocked_identity in blocked_source_identities
            ):
                return "URL matches this evaluation case's blocked reference source"
        if self.benchmark_source_exclusion:
            return _benchmark_url_contamination_reason(url)
        return None

    @staticmethod
    def _protocol_violation(
        url: str,
        *,
        stage: str,
        reason: str,
    ) -> dict[str, Any]:
        try:
            identity = canonical_url_identity(url)
            serialized_identity = json.dumps(identity, ensure_ascii=True, separators=(",", ":"))
        except URLPolicyError:
            serialized_identity = _safe_audit_url(url)
        return {
            "type": "BenchmarkContaminationError",
            "category": "denylist_enforcement",
            "stage": stage,
            "action": "blocked_before_evidence",
            "reason": reason,
            "blocked_count": 1,
            "url_identity_sha256": hashlib.sha256(
                serialized_identity.encode("utf-8")
            ).hexdigest(),
        }

    def _url_protocol_violation(
        self,
        url: str,
        *,
        blocked_source_identities: frozenset[
            tuple[str, str, int | None, str]
        ],
        stage: str,
    ) -> dict[str, Any] | None:
        reason = self._url_contamination_reason(
            url,
            blocked_source_identities=blocked_source_identities,
        )
        if reason is None:
            return None
        return self._protocol_violation(url, stage=stage, reason=reason)

    @staticmethod
    def _mark_source_blocked(
        source: Source,
        *,
        violation: dict[str, Any],
        crawl_attempts: int = 0,
    ) -> Source:
        metadata = {
            **source.metadata,
            "extract_status": "blocked",
            "snippet_only": True,
            "candidate_only": True,
            "requires_crawl": False,
            "verification_status": "denylist_blocked",
            "degrade_reason": "denylist_enforcement",
            "protocol_violation": dict(violation),
            "crawl_attempts": crawl_attempts,
        }
        return source.model_copy(update={"content": "", "metadata": metadata})

    def _raise_if_url_contaminated(
        self,
        url: str,
        *,
        blocked_source_identities: frozenset[
            tuple[str, str, int | None, str]
        ],
        stage: str,
    ) -> None:
        violation = self._url_protocol_violation(
            url,
            blocked_source_identities=blocked_source_identities,
            stage=stage,
        )
        if violation:
            raise BenchmarkContaminationError(
                "benchmark source denylist enforcement blocked a URL before evidence",
                stage=stage,
                benchmark_contamination=False,
                protocol_violation=violation,
            )

    def _assert_final_sources_clean(
        self,
        sources: list[Source],
        *,
        blocked_source_identities: frozenset[
            tuple[str, str, int | None, str]
        ],
    ) -> None:
        for source in sources:
            violation = self._url_protocol_violation(
                source.url,
                blocked_source_identities=blocked_source_identities,
                stage="evidence",
            )
            if violation is None and not source.metadata.get("protocol_violation"):
                continue
            violation = violation or dict(source.metadata["protocol_violation"])
            audit = {
                "benchmark_contamination": True,
                "denylist_enforcement_hit": False,
                "protocol_violation_count": 1,
                "protocol_violations": [violation],
            }
            raise BenchmarkContaminationError(
                "blocked benchmark source crossed the final evidence boundary",
                stage="evidence",
                benchmark_contamination=True,
                protocol_violation=violation,
                retrieval_audit=audit,
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
        safe_sources = [
            source
            for source in sources
            if source.metadata.get("extract_status") != "blocked"
            and not source.metadata.get("protocol_violation")
        ]
        crawl_errors = [
            str(source.metadata.get("crawl_error_class") or "crawler_error")
            for source in safe_sources
            if source.metadata.get("extract_status") == "crawl_failed"
        ]
        requires_crawl = self.gateway_chain and provider in {
            self.primary.name,
            self.fallback.name,
        }
        if not requires_crawl:
            return safe_sources, crawl_errors, _failed_candidate_hints(sources), _retrieval_audit(
                sources, safe_sources
            )
        evidence_ready = [
            source
            for source in safe_sources
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

    @staticmethod
    def _operation_timeout(
        configured_timeout: float,
        *,
        deadline_at: float | None,
    ) -> float:
        if configured_timeout <= 0:
            raise ValueError("configured timeout must be positive")
        if deadline_at is None:
            return configured_timeout
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            raise SearchDeadlineExceeded
        return min(configured_timeout, remaining)

    def _mark_source_crawl_timeout(
        self,
        source: Source,
        *,
        crawl_attempts: int,
    ) -> Source:
        metadata = {
            **source.metadata,
            "search_snippet": source.metadata.get("search_snippet", source.content),
            "crawler": self.crawler.name if self.crawler is not None else "none",
            "extract_status": "crawl_failed",
            "crawler_error": "crawl batch deadline exceeded",
            "crawl_error_class": "batch_timeout",
            "crawl_retryable": True,
            "crawl_attempts": crawl_attempts,
            "degrade_reason": "batch_timeout",
            "snippet_only": True,
            "candidate_only": True,
            "verification_status": "crawl_failed",
        }
        return source.model_copy(update={"metadata": metadata})

    async def _crawl_sources(
        self,
        sources: list[Source],
        *,
        blocked_source_identities: frozenset[
            tuple[str, str, int | None, str]
        ] = frozenset(),
        deadline_at: float | None = None,
    ) -> list[Source]:
        if self.crawler is None:
            return sources

        semaphore = asyncio.Semaphore(self.crawler_concurrency_per_search)
        attempts_by_index: dict[int, int] = {}

        async def crawl_one(index: int, source: Source) -> Source:
            if (
                source.provider == "mock"
                or not source.url.startswith(("http://", "https://"))
                or source.metadata.get("crawler") not in {None, "", "none"}
            ):
                return source
            metadata = dict(source.metadata)
            metadata.setdefault("search_snippet", source.content)
            metadata["crawler"] = self.crawler.name
            async with semaphore:
                for attempt in (1, 2):
                    attempts_by_index[index] = attempt
                    try:
                        crawl_timeout = self._operation_timeout(
                            self.settings.request_timeout_seconds,
                            deadline_at=deadline_at,
                        )
                        self._raise_if_url_contaminated(
                            source.url,
                            blocked_source_identities=blocked_source_identities,
                            stage="candidate",
                        )
                        if isinstance(self.crawler, HtmlTextCrawler) and (
                            blocked_source_identities or self.benchmark_source_exclusion
                        ):
                            crawled = await self.crawler.crawl(
                                source.url,
                                crawl_timeout,
                                url_guard=lambda url: self._raise_if_url_contaminated(
                                    url,
                                    blocked_source_identities=blocked_source_identities,
                                    stage="redirect",
                                ),
                            )
                        else:
                            # Preserve compatibility with custom two-argument
                            # crawlers when no pre-request redirect guard is needed.
                            crawled = await self.crawler.crawl(
                                source.url,
                                crawl_timeout,
                            )
                        if isinstance(crawled, FetchedPage):
                            content = crawled.content
                            final_url = crawled.final_url
                            redirect_chain = crawled.redirect_chain
                        else:
                            content = str(crawled)
                            final_url = source.url
                            redirect_chain = (source.url,)
                        for redirect_url in redirect_chain[1:-1]:
                            self._raise_if_url_contaminated(
                                redirect_url,
                                blocked_source_identities=blocked_source_identities,
                                stage="redirect",
                            )
                        self._raise_if_url_contaminated(
                            final_url,
                            blocked_source_identities=blocked_source_identities,
                            stage="final_url",
                        )
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
                            update={
                                "url": final_url,
                                "content": content,
                                "metadata": metadata,
                            }
                        )
                    except BenchmarkContaminationError as exc:
                        if exc.benchmark_contamination:
                            raise
                        violation = exc.protocol_violation or self._protocol_violation(
                            source.url,
                            stage=exc.stage,
                            reason="benchmark source denylist enforcement",
                        )
                        return self._mark_source_blocked(
                            source,
                            violation=violation,
                            crawl_attempts=attempt,
                        )
                    except Exception as exc:  # noqa: BLE001 - classify before bounded retry.
                        error_info = _classify_crawl_error(exc)
                        if attempt == 1 and error_info.retryable:
                            await self._retry_backoff(0, deadline_at=deadline_at)
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

        tasks = {
            asyncio.create_task(crawl_one(index, source)): index
            for index, source in enumerate(sources)
        }
        if deadline_at is None:
            return list(await asyncio.gather(*tasks))

        remaining = max(deadline_at - time.monotonic(), 0.0)
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        results: list[Source | None] = [None] * len(sources)
        fatal_error: BaseException | None = None
        for task in done:
            index = tasks[task]
            try:
                results[index] = task.result()
            except SearchDeadlineExceeded:
                results[index] = self._mark_source_crawl_timeout(
                    sources[index],
                    crawl_attempts=attempts_by_index.get(index, 0),
                )
            except BaseException as exc:  # preserve fatal contamination/cancellation.
                fatal_error = exc
                break

        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if fatal_error is not None:
            raise fatal_error
        for task in pending:
            index = tasks[task]
            results[index] = self._mark_source_crawl_timeout(
                sources[index],
                crawl_attempts=attempts_by_index.get(index, 0),
            )
        return [
            result
            if result is not None
            else self._mark_source_crawl_timeout(
                sources[index],
                crawl_attempts=attempts_by_index.get(index, 0),
            )
            for index, result in enumerate(results)
        ]

    async def _retry_backoff(
        self,
        attempt: int,
        *,
        deadline_at: float | None = None,
    ) -> None:
        base_delay = self.settings.search_retry_backoff_seconds
        if base_delay <= 0:
            return
        delay = base_delay * (2**attempt)
        if deadline_at is not None:
            remaining = deadline_at - time.monotonic()
            if remaining <= 0 or delay >= remaining:
                raise SearchDeadlineExceeded
        await self.retry_sleep(delay)


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
    r"(?:^|[/_\-.])(?:simpleqa|simple-evals?|livedrbench|"
    r"deepresearch-bench|deepresearchbench|drb2|deep_research_bench)"
    r"(?:$|[/_\-.])",
    re.IGNORECASE,
)
_BENCHMARK_ANSWER_FIELD = re.compile(
    r"(?:\"|\b)(?:reference_?answer|ground_?truths?|expected_?answer|answers?)(?:\"|\b)\s*[:=]",
    re.IGNORECASE,
)


_BENCHMARK_DATA_HOSTS = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
        "huggingface.co",
        "hf.co",
        "datasets-server.huggingface.co",
    }
)


def _blocked_source_identities(
    urls: list[str] | tuple[str, ...],
) -> frozenset[tuple[str, str, int | None, str]]:
    try:
        return frozenset(canonical_url_identity(url) for url in urls)
    except URLPolicyError as exc:
        raise ValueError(f"invalid blocked source URL: {exc}") from exc


def _benchmark_url_contamination_reason(url: str) -> str | None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".").removeprefix("www.")
    marker_target = parsed.path.lower()
    if host == "datasets-server.huggingface.co":
        # The dataset identity for HF's rows/parquet endpoints lives in the
        # query string rather than the path. This is a static benchmark marker
        # check, distinct from dynamic source canonicalization (which ignores
        # query and fragment by design).
        marker_target = f"{marker_target} {unquote(parsed.query).lower()}"
    if host in _BENCHMARK_DATA_HOSTS and _BENCHMARK_PATH_MARKER.search(marker_target):
        return "known benchmark repository or dataset path"
    return None


def _benchmark_content_contamination_reason(
    source: Source,
    *,
    query: str,
) -> str | None:
    """Detect answer-key content that has already crossed the URL filter."""

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
