from __future__ import annotations

import asyncio

import pytest

from deepresearch_agent.config import Settings
from deepresearch_agent.search import (
    MockSearchAdapter,
    SearchError,
    SearchRateLimiter,
    SearchService,
    _wikipedia_query_candidates,
)
from deepresearch_agent.schemas import Source


class FailingSearchAdapter:
    name = "failing"

    async def search(self, query: str, max_results: int, timeout: float):
        del query, max_results, timeout
        raise SearchError("forced failure")


class SuccessfulSearchAdapter:
    name = "primary"

    async def search(self, query: str, max_results: int, timeout: float):
        del timeout
        return [
            Source(
                title="Primary result",
                url="https://example.com/primary",
                content=f"Primary content for {query}",
                provider=self.name,
                query=query,
                score=float(max_results),
            )
        ]


class CountingRateLimiter:
    def __init__(self) -> None:
        self.calls = 0

    async def wait(self) -> None:
        self.calls += 1


def test_search_service_falls_back_after_primary_failure() -> None:
    service = SearchService(
        primary=FailingSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(max_retries=0, circuit_breaker_failure_threshold=1),
    )

    outcome = asyncio.run(service.search("agent fallback", max_results=2))

    assert outcome.fallback_used is True
    assert outcome.provider == "mock"
    assert outcome.error == "forced failure"
    assert len(outcome.sources) == 2


def test_circuit_breaker_skips_open_primary() -> None:
    service = SearchService(
        primary=FailingSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(max_retries=0, circuit_breaker_failure_threshold=1),
    )

    first = asyncio.run(service.search("agent fallback", max_results=1))
    second = asyncio.run(service.search("agent fallback", max_results=1))

    assert first.error == "forced failure"
    assert second.error == "circuit breaker open"
    assert second.fallback_used is True


def test_search_rate_limiter_waits_between_primary_calls() -> None:
    current_time = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        return current_time

    async def fake_sleep(delay: float) -> None:
        nonlocal current_time
        sleeps.append(round(delay, 3))
        current_time += delay

    limiter = SearchRateLimiter(rate_per_second=2.0, clock=clock, sleep=fake_sleep)

    async def run_waits() -> None:
        await limiter.wait()
        await limiter.wait()
        await limiter.wait()

    asyncio.run(run_waits())

    assert sleeps == [0.5, 0.5]


def test_search_service_applies_rate_limiter_before_primary_call() -> None:
    limiter = CountingRateLimiter()
    service = SearchService(
        primary=SuccessfulSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(search_rate_limit_per_second=10.0),
        rate_limiter=limiter,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(service.search("agent rate limit", max_results=1))

    assert limiter.calls == 1
    assert outcome.provider == "primary"
    assert outcome.fallback_used is False


def test_pytest_is_available() -> None:
    assert pytest.__version__


def test_wikipedia_query_candidates_compact_long_questions() -> None:
    candidates = _wikipedia_query_candidates(
        "What empirical evidence exists from 2020 onwards that citation faithfulness checks reduce hallucination in LLM-based agent reports?"
    )

    assert candidates[0] != candidates[-1]
    assert "citation" in candidates[0].lower()
    assert "faithfulness" in candidates[0].lower()
    assert len(candidates[0]) < len(candidates[-1])
