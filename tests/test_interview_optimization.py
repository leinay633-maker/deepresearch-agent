from __future__ import annotations

import asyncio

import pytest

from deepresearch_agent.citation import CitationChecker
from deepresearch_agent.config import Settings
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest, Source
from deepresearch_agent.search import MockSearchAdapter, SearchError, SearchService
from deepresearch_agent.text_utils import tokenize


class EmptyRag:
    async def retrieve(self, query: str, max_results: int) -> list[Source]:
        del query, max_results
        return []


class CountingSearchAdapter:
    name = "counting"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del max_results, timeout
        self.calls.append(query)
        return [
            Source(
                title="同一份证据",
                url="https://example.com/evidence",
                content="研究系统需要保留证据片段，并在预算耗尽时停止继续搜索。",
                provider=self.name,
                query=query,
                score=1.0,
            )
        ]


class IrrelevantSearchAdapter:
    name = "fixture"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del max_results, timeout
        return [
            Source(
                title="旅游信息",
                url="https://example.com/travel",
                content="旅游系统提供酒店和景点信息。",
                provider=self.name,
                query=query,
                score=1.0,
            )
        ]


class SlowSearchAdapter:
    name = "slow"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del query, max_results, timeout
        await asyncio.sleep(0.05)
        return []


class FastEvidenceSearchAdapter:
    name = "fast-evidence"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del max_results, timeout
        return [
            Source(
                title="Deadline evidence",
                url="https://example.com/deadline",
                content="Research deadlines must cover retrieval and decision actions.",
                provider=self.name,
                query=query,
                score=1.0,
                metadata={"extract_status": "ok", "snippet_only": False},
            )
        ]


class SlowRag:
    async def retrieve(self, query: str, max_results: int) -> list[Source]:
        del query, max_results
        await asyncio.sleep(0.05)
        return []


class SlowDecisionLLM:
    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    async def decide_research(self, *args, **kwargs):
        await asyncio.sleep(0.05)
        return await self.delegate.decide_research(*args, **kwargs)


class FailingSearchAdapter:
    name = "failing"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del query, max_results, timeout
        raise SearchError("forced failure")


class EmptySearchAdapter:
    name = "empty"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del query, max_results, timeout
        return []


class SuccessfulSearchAdapter:
    name = "wikipedia"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del max_results, timeout
        return [
            Source(
                title="Search result",
                url="https://example.com/article",
                content="short search snippet",
                provider=self.name,
                query=query,
            )
        ]


class StubCrawler:
    name = "stub"

    async def crawl(self, url: str, timeout: float) -> str:
        del url, timeout
        return "Full article evidence extracted by the crawler."


class FailingCrawler:
    name = "failing-crawler"

    async def crawl(self, url: str, timeout: float) -> str:
        del url, timeout
        raise SearchError("forced crawl failure")


def test_chinese_citation_grounding_keeps_traceable_evidence() -> None:
    source = Source(
        id="S1",
        title="引用校验说明",
        url="https://example.com/citation",
        content="引用校验会把报告中的结论与来源证据逐条对齐，从而暴露不受支持的结论。",
        provider="fixture",
        query="引用校验",
        metadata={"retrieved_at": "2026-07-12T00:00:00+00:00"},
    )

    report = CitationChecker().check(
        ["引用校验会把结论与来源证据逐条对齐 [S1]"],
        [source],
    )

    assert report.supported_claims == 1
    assert report.citation_coverage == 1.0
    assert report.citation_precision == 1.0
    quote = report.assessments[0].evidence_quotes[0]
    assert quote.source_url == source.url
    assert quote.retrieved_at == source.metadata["retrieved_at"]
    assert "逐条对齐" in quote.quote
    assert {"引用", "引用校验", "citation"} & tokenize("引用校验 citation")


def test_chinese_weak_character_overlap_is_not_marked_supported() -> None:
    source = Source(
        id="S1",
        title="旅游信息",
        url="https://example.com/travel",
        content="旅游系统提供酒店和景点信息。",
        provider="fixture",
        query="旅游",
        metadata={"extract_status": "ok", "snippet_only": False},
    )

    report = CitationChecker().check(
        ["人工智能系统需要验证来源和结论 [S1]"],
        [source],
    )

    assert report.assessments[0].supported is False


def test_bounded_research_loop_stops_at_tool_budget_and_remaps_evidence_ids() -> None:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="keyword",
        max_retries=0,
    )
    primary = CountingSearchAdapter()
    service = SearchService(
        primary=primary,
        fallback=MockSearchAdapter(),
        settings=settings,
    )
    orchestrator = DeepResearchOrchestrator(settings=settings, search_service=service)
    orchestrator.rag = EmptyRag()  # type: ignore[assignment]

    report = asyncio.run(
        orchestrator.run(
            ResearchRequest(
                query="如何用有限预算收集可追溯证据？",
                llm_provider="mock",
                search_provider="mock",
                max_researchers=1,
                max_results_per_researcher=1,
                max_rounds=5,
                max_tool_calls=2,
                min_evidence_items=3,
            )
        )
    )

    research = report.findings[0].research
    assert research is not None
    assert research.tool_calls == 2
    assert len(research.rounds) == 2
    assert research.termination_reason == "max_tool_calls"
    assert len(primary.calls) == 2
    assert research.evidence[0].source_id == "S1"
    assert research.evidence[0].retrieved_at
    assert report.metrics["execution_success"] is True
    assert report.metrics["success"] is True
    assert report.metrics["answer_quality"] is None
    assert report.metrics["citation_precision"] == report.citation_check.citation_precision


def test_research_loop_does_not_stop_on_irrelevant_evidence() -> None:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="keyword",
        max_retries=0,
    )
    orchestrator = DeepResearchOrchestrator(
        settings=settings,
        search_service=SearchService(
            primary=IrrelevantSearchAdapter(),
            fallback=MockSearchAdapter(),
            settings=settings,
        ),
    )
    orchestrator.rag = EmptyRag()  # type: ignore[assignment]

    report = asyncio.run(
        orchestrator.run(
            ResearchRequest(
                query="人工智能系统如何验证来源和结论？",
                llm_provider="mock",
                max_researchers=1,
                max_results_per_researcher=1,
                max_rounds=2,
                max_tool_calls=2,
                min_evidence_items=1,
            )
        )
    )

    research = report.findings[0].research
    assert research is not None
    assert research.evidence == []
    assert research.termination_reason == "max_rounds"


def test_timed_out_search_still_consumes_agent_tool_budget() -> None:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="keyword",
        max_retries=0,
    )
    orchestrator = DeepResearchOrchestrator(
        settings=settings,
        search_service=SearchService(
            primary=SlowSearchAdapter(),
            fallback=MockSearchAdapter(),
            settings=settings,
        ),
    )
    orchestrator.rag = EmptyRag()  # type: ignore[assignment]

    report = asyncio.run(
        orchestrator.run(
            ResearchRequest(
                query="How should deadline budgets stop research?",
                llm_provider="mock",
                max_researchers=1,
                max_results_per_researcher=1,
                max_rounds=3,
                max_tool_calls=3,
                deadline_seconds=0.01,
                min_evidence_items=1,
            )
        )
    )

    research = report.findings[0].research
    assert research is not None
    assert research.tool_calls == 1
    assert research.termination_reason == "deadline"
    assert research.budget_exhausted is True
    assert report.metrics["degraded_count"] == 1
    assert report.metrics["budget_exhausted_count"] == 1
    assert next(
        event for event in report.trace_events if event.stage == "run" and event.status == "success"
    ).payload["degraded"] is True


def test_research_deadline_covers_rag_and_decision() -> None:
    settings = Settings(local_retrieval_mode="keyword", max_retries=0)
    service = SearchService(
        FastEvidenceSearchAdapter(), MockSearchAdapter(), settings, fallback_policy="fail"
    )

    rag_orchestrator = DeepResearchOrchestrator(settings=settings, search_service=service)
    rag_orchestrator.rag = SlowRag()  # type: ignore[assignment]
    rag_report = asyncio.run(
        rag_orchestrator.run(
            ResearchRequest(
                query="How should a RAG deadline work?",
                max_researchers=1,
                max_results_per_researcher=1,
                deadline_seconds=0.01,
                fallback_policy="fail",
            )
        )
    )

    decision_orchestrator = DeepResearchOrchestrator(
        settings=settings,
        search_service=service,
        llm_provider=SlowDecisionLLM(
            DeepResearchOrchestrator(settings=settings)._build_llm_provider(
                ResearchRequest(query="How should a decision deadline work?")
            )
        ),
    )
    decision_orchestrator.rag = EmptyRag()  # type: ignore[assignment]
    decision_report = asyncio.run(
        decision_orchestrator.run(
            ResearchRequest(
                query="How should a decision deadline work?",
                max_researchers=1,
                max_results_per_researcher=1,
                deadline_seconds=0.01,
                fallback_policy="fail",
            )
        )
    )

    rag_research = rag_report.findings[0].research
    decision_research = decision_report.findings[0].research
    assert rag_research is not None and rag_research.termination_reason == "deadline"
    assert decision_research is not None and decision_research.termination_reason == "deadline"
    assert rag_research.tool_calls == decision_research.tool_calls == 1
    assert rag_research.budget_exhausted is True
    # The decision call timed out only after one grounded item met the minimum.
    assert decision_research.budget_exhausted is False


def test_search_fallback_policy_can_degrade_or_fail_without_mock_pollution() -> None:
    settings = Settings(max_retries=0, circuit_breaker_failure_threshold=1)
    degraded_service = SearchService(
        primary=FailingSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="degraded",
    )
    failed_service = SearchService(
        primary=FailingSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="fail",
    )

    degraded = asyncio.run(degraded_service.search("query", max_results=1))

    assert degraded.sources == []
    assert degraded.degraded is True
    assert degraded.fallback_used is False
    assert degraded.provider == "failing"
    with pytest.raises(SearchError, match="forced failure"):
        asyncio.run(failed_service.search("query", max_results=1))


def test_empty_search_results_follow_fallback_policy() -> None:
    settings = Settings(max_retries=0, circuit_breaker_failure_threshold=1)
    degraded_service = SearchService(
        primary=EmptySearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="degraded",
    )
    failed_service = SearchService(
        primary=EmptySearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="fail",
    )

    degraded = asyncio.run(degraded_service.search("query", max_results=1))

    assert degraded.degraded is True
    assert degraded.sources == []
    assert "returned no results" in str(degraded.error)
    with pytest.raises(SearchError, match="returned no results"):
        asyncio.run(failed_service.search("query", max_results=1))


def test_crawler_failure_is_explicitly_degraded_or_failed() -> None:
    settings = Settings(max_retries=0, circuit_breaker_failure_threshold=1)
    degraded_service = SearchService(
        primary=SuccessfulSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        crawler=FailingCrawler(),
        fallback_policy="degraded",
    )
    failed_service = SearchService(
        primary=SuccessfulSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        crawler=FailingCrawler(),
        fallback_policy="fail",
    )

    degraded = asyncio.run(degraded_service.search("query", max_results=1))

    assert degraded.degraded is True
    assert degraded.sources[0].metadata["extract_status"] == "crawl_failed"
    assert degraded.sources[0].metadata["snippet_only"] is True
    with pytest.raises(SearchError, match="crawler extraction failed"):
        asyncio.run(failed_service.search("query", max_results=1))


def test_search_service_applies_crawler_and_records_evidence_metadata() -> None:
    service = SearchService(
        primary=SuccessfulSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(max_retries=0),
        crawler=StubCrawler(),
    )

    outcome = asyncio.run(service.search("evidence", max_results=1))
    source = outcome.sources[0]

    assert source.content.startswith("Full article evidence")
    assert source.metadata["search_snippet"] == "short search snippet"
    assert source.metadata["crawler"] == "stub"
    assert source.metadata["extract_status"] == "ok"
    assert source.metadata["snippet_only"] is False
    assert source.metadata["content_hash"]
    assert source.metadata["retrieved_at"]


def test_search_service_marks_uncrawled_web_content_as_snippet_only() -> None:
    service = SearchService(
        primary=SuccessfulSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=Settings(max_retries=0),
    )

    outcome = asyncio.run(service.search("evidence", max_results=1))
    source = outcome.sources[0]

    assert source.metadata["snippet_only"] is True
    assert source.metadata["extract_status"] == "snippet"
    assert source.metadata["published_at"] is None
