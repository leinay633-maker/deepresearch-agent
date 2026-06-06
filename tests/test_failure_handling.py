from __future__ import annotations

import asyncio

import pytest

from deepresearch_agent.config import Settings
from deepresearch_agent.search import (
    MockSearchAdapter,
    SearchError,
    SearchService,
    _wikipedia_query_candidates,
)


class FailingSearchAdapter:
    name = "failing"

    async def search(self, query: str, max_results: int, timeout: float):
        del query, max_results, timeout
        raise SearchError("forced failure")


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
