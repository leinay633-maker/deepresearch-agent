from __future__ import annotations

import asyncio
import html
import json
import re
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from deepresearch_agent.config import Settings
from deepresearch_agent.schemas import Source


class SearchError(RuntimeError):
    pass


class SearchAdapter(Protocol):
    name: str

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
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
        params = urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
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
        query_terms = _tokens(query)
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
                    query=query,
                    score=float(overlap * 10 + rank_bonus),
                    metadata={"pageid": page_id},
                )
            )
        if not sources:
            raise SearchError("wikipedia returned no results")
        return sources


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


class SearchService:
    def __init__(
        self,
        primary: SearchAdapter,
        fallback: SearchAdapter,
        settings: Settings,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.settings = settings
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
            for _ in range(self.settings.max_retries + 1):
                try:
                    sources = await asyncio.wait_for(
                        self.primary.search(query, max_results, self.settings.request_timeout_seconds),
                        timeout=self.settings.request_timeout_seconds + 0.5,
                    )
                    self.breaker.record_success()
                    return SearchOutcome(sources=sources, provider=self.primary.name)
                except Exception as exc:
                    last_error = str(exc)
                    self.breaker.record_failure()
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


def build_search_service(settings: Settings, provider: str | None = None) -> SearchService:
    selected = (provider or settings.search_provider).strip().lower()
    fallback = MockSearchAdapter()
    if selected == "wikipedia":
        primary: SearchAdapter = WikipediaSearchAdapter()
    else:
        primary = fallback
    return SearchService(primary=primary, fallback=fallback, settings=settings)


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", text)}
