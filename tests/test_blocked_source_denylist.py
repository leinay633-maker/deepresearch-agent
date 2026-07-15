from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from deepresearch_agent.config import Settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.deep_research_eval import _run_case
from deepresearch_agent.llm import MockLLMProvider
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest, Source, SubQuestion
from deepresearch_agent.search import (
    BenchmarkContaminationError,
    FetchedPage,
    HtmlTextCrawler,
    MockSearchAdapter,
    SearchEvidenceUnavailableError,
    SearchService,
)
from deepresearch_agent.tracing import TraceLogger
from deepresearch_agent.url_policy import (
    canonical_url_identity,
    url_identity_matches_blocked,
)


class _CandidateAdapter:
    name = "candidate"

    def __init__(self, url: str) -> None:
        self.url = url

    async def search(
        self,
        query: str,
        max_results: int,
        timeout: float,
    ) -> list[Source]:
        del max_results, timeout
        return [
            Source(
                title="Candidate source",
                url=self.url,
                content="Unverified search snippet.",
                provider=self.name,
                query=query,
            )
        ]


class _CandidatesAdapter:
    def __init__(self, name: str, urls: list[str]) -> None:
        self.name = name
        self.urls = urls

    async def search(
        self,
        query: str,
        max_results: int,
        timeout: float,
    ) -> list[Source]:
        del timeout
        return [
            Source(
                title=f"Candidate {index}",
                url=url,
                content="Unverified search snippet.",
                provider=self.name,
                query=query,
            )
            for index, url in enumerate(self.urls[:max_results], 1)
        ]


class _RecordingCrawler:
    name = "recording"

    def __init__(self, result: str | FetchedPage = "Crawled evidence body.") -> None:
        self.result = result
        self.calls: list[str] = []

    async def crawl(self, url: str, timeout: float) -> str:
        del timeout
        self.calls.append(url)
        return self.result


class _FakeResponse:
    def __init__(
        self,
        body: bytes = b"ok",
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        return None


def _service(
    url: str,
    *,
    crawler: Any,
    benchmark_source_exclusion: bool = False,
) -> SearchService:
    return SearchService(
        primary=_CandidateAdapter(url),
        fallback=MockSearchAdapter(),
        settings=Settings(
            benchmark_source_exclusion=benchmark_source_exclusion,
            local_retrieval_mode="none",
            max_retries=0,
            search_retry_backoff_seconds=0,
        ),
        crawler=crawler,
        fallback_policy="fail",
    )


def test_canonical_blocked_url_ignores_query_fragment_host_case_and_default_port() -> None:
    assert canonical_url_identity(
        "HTTPS://Example.COM:443/reference/report?utm_source=search#section-2"
    ) == canonical_url_identity("https://example.com/reference/report")
    assert canonical_url_identity("http://example.com:80/") == canonical_url_identity(
        "http://EXAMPLE.com"
    )
    assert canonical_url_identity("http://example.com/report") != canonical_url_identity(
        "https://example.com/report"
    )


def test_blocked_html_report_matches_descendant_chapter_but_not_sibling() -> None:
    blocked = canonical_url_identity(
        "https://www.oecd.org/en/publications/report_abc/full-report.html"
    )

    assert url_identity_matches_blocked(
        canonical_url_identity(
            "https://www.oecd.org/en/publications/report_abc/full-report/chapter_1.html"
        ),
        blocked,
    )
    assert not url_identity_matches_blocked(
        canonical_url_identity(
            "https://www.oecd.org/en/publications/another-report/full-report/chapter_1.html"
        ),
        blocked,
    )


def test_blocked_candidate_is_audited_while_clean_sibling_succeeds() -> None:
    crawler = _RecordingCrawler()
    service = SearchService(
        primary=_CandidatesAdapter(
            "primary",
            [
                "https://EXAMPLE.com:443/reference/report?tracking=1#abstract",
                "https://example.com/clean",
            ],
        ),
        fallback=MockSearchAdapter(),
        settings=Settings(local_retrieval_mode="none", max_retries=0),
        crawler=crawler,
        fallback_policy="fail",
    )

    outcome = asyncio.run(
        service.search(
            "research question",
            max_results=2,
            blocked_source_urls=["https://example.com/reference/report"],
        )
    )

    assert crawler.calls == ["https://example.com/clean"]
    assert [source.url for source in outcome.sources] == [
        "https://example.com/clean"
    ]
    assert outcome.denylist_enforcement_hit is True
    assert outcome.benchmark_contamination is False
    assert outcome.retrieval_audit["blocked_count"] == 1
    violation = outcome.protocol_violations[0]
    assert violation["type"] == "BenchmarkContaminationError"
    assert violation["stage"] == "candidate"
    assert len(violation["url_identity_sha256"]) == 64


def test_blocked_report_descendant_candidate_is_never_crawled() -> None:
    crawler = _RecordingCrawler()
    blocked_report = "https://www.oecd.org/en/publications/report_abc/full-report.html"
    descendant = (
        "https://www.oecd.org/en/publications/report_abc/"
        "full-report/expert-chapter_1234.html"
    )
    service = SearchService(
        primary=_CandidatesAdapter(
            "primary",
            [descendant, "https://www.oecd.org/en/publications/clean-report.html"],
        ),
        fallback=MockSearchAdapter(),
        settings=Settings(local_retrieval_mode="none", max_retries=0),
        crawler=crawler,
        fallback_policy="fail",
    )

    outcome = asyncio.run(
        service.search(
            "research question",
            max_results=2,
            blocked_source_urls=[blocked_report],
        )
    )

    assert crawler.calls == [
        "https://www.oecd.org/en/publications/clean-report.html"
    ]
    assert [source.url for source in outcome.sources] == crawler.calls
    assert outcome.retrieval_audit["blocked_count"] == 1


def test_blocked_only_primary_uses_clean_real_fallback() -> None:
    blocked_url = "https://reference.example/expert-report"
    crawler = _RecordingCrawler()
    service = SearchService(
        primary=_CandidatesAdapter("primary", [blocked_url]),
        fallback=_CandidatesAdapter("real-fallback", ["https://example.com/clean"]),
        settings=Settings(local_retrieval_mode="none", max_retries=2),
        crawler=crawler,
        fallback_policy="fail",
    )

    outcome = asyncio.run(
        service.search(
            "research question",
            max_results=2,
            blocked_source_urls=[blocked_url],
        )
    )

    assert outcome.fallback_used is True
    assert outcome.provider == "real-fallback"
    assert [source.url for source in outcome.sources] == [
        "https://example.com/clean"
    ]
    assert crawler.calls == ["https://example.com/clean"]
    assert outcome.retrieval_audit["blocked_count"] == 1


def test_blocked_html_redirect_keeps_clean_sibling_and_never_opens_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []
    blocked_url = "https://reference.example/expert-report"

    def opener(request: Any, timeout: float) -> _FakeResponse:
        del timeout
        requested_urls.append(request.full_url)
        if request.full_url == "https://example.com/redirect-start":
            return _FakeResponse(status=302, headers={"Location": blocked_url + "?via=302"})
        if request.full_url == "https://example.com/clean":
            return _FakeResponse(body=b"<main>Clean evidence body.</main>")
        raise AssertionError(f"unexpected or blocked URL opened: {request.full_url}")

    monkeypatch.setattr("deepresearch_agent.search.urlopen", opener)
    monkeypatch.setattr(
        "deepresearch_agent.url_policy.validate_url",
        lambda url, **kwargs: url,
    )
    monkeypatch.setattr("deepresearch_agent.search.validate_url", lambda url: url)
    service = SearchService(
        primary=_CandidatesAdapter(
            "primary",
            [
                "https://example.com/redirect-start",
                "https://example.com/clean",
            ],
        ),
        fallback=MockSearchAdapter(),
        settings=Settings(local_retrieval_mode="none", max_retries=0),
        crawler=HtmlTextCrawler(),
        fallback_policy="fail",
    )

    outcome = asyncio.run(
        service.search(
            "research question",
            max_results=2,
            blocked_source_urls=[blocked_url],
        )
    )

    assert set(requested_urls) == {
        "https://example.com/redirect-start",
        "https://example.com/clean",
    }
    assert blocked_url not in requested_urls
    assert [source.url for source in outcome.sources] == [
        "https://example.com/clean"
    ]
    assert outcome.protocol_violations[0]["stage"] == "redirect"


def test_custom_crawler_blocked_final_url_is_removed_with_protocol_audit() -> None:
    blocked_url = "https://reference.example/expert-report"
    crawler = _RecordingCrawler(
        FetchedPage(
            "Expert article body.",
            final_url=blocked_url + "?download=1#full-text",
            redirect_chain=("https://public.example/start", blocked_url),
        )
    )
    service = _service("https://public.example/start", crawler=crawler)

    with pytest.raises(SearchEvidenceUnavailableError) as exc_info:
        asyncio.run(
            service.search(
                "research question",
                max_results=1,
                blocked_source_urls=[blocked_url],
            )
        )

    assert crawler.calls == ["https://public.example/start"]
    audit = exc_info.value.retrieval_audit
    assert audit["blocked_count"] == 1
    assert audit["protocol_violations"][0]["stage"] == "final_url"


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/example/deepresearch-bench/blob/main/data/tasks.jsonl",
        "https://raw.githubusercontent.com/example/deepresearchbench/main/rubrics.jsonl",
        "https://huggingface.co/datasets/example/drb2/resolve/main/tasks.jsonl",
        "https://hf.co/datasets/example/deep_research_bench/data.jsonl",
        "https://datasets-server.huggingface.co/rows?dataset=example%2Fdeepresearchbench",
    ],
)
def test_static_drb_dataset_paths_are_blocked(url: str) -> None:
    crawler = _RecordingCrawler()
    service = _service(
        url,
        crawler=crawler,
        benchmark_source_exclusion=True,
    )

    with pytest.raises(SearchEvidenceUnavailableError) as exc_info:
        asyncio.run(service.search("research question", max_results=1))

    assert crawler.calls == []
    assert exc_info.value.retrieval_audit["blocked_count"] == 1
    assert (
        exc_info.value.retrieval_audit["protocol_violations"][0]["stage"]
        == "candidate"
    )


def test_all_primary_and_real_fallback_candidates_blocked_returns_protocol_audit() -> None:
    primary_blocked = "https://reference.example/expert-report"
    fallback_blocked = "https://reference.example/expert-report-mirror"
    crawler = _RecordingCrawler()
    service = SearchService(
        primary=_CandidatesAdapter("primary", [primary_blocked]),
        fallback=_CandidatesAdapter("real-fallback", [fallback_blocked]),
        settings=Settings(local_retrieval_mode="none", max_retries=2),
        crawler=crawler,
        fallback_policy="fail",
    )

    with pytest.raises(SearchEvidenceUnavailableError) as exc_info:
        asyncio.run(
            service.search(
                "research question",
                max_results=2,
                blocked_source_urls=[primary_blocked, fallback_blocked],
            )
        )

    assert crawler.calls == []
    audit = exc_info.value.retrieval_audit
    assert audit["denylist_enforcement_hit"] is True
    assert audit["benchmark_contamination"] is False
    assert audit["blocked_count"] == 2
    assert audit["protocol_violation_count"] == 2
    assert all(
        item["type"] == "BenchmarkContaminationError"
        for item in audit["protocol_violations"]
    )


def test_static_policy_does_not_block_ordinary_github_documentation() -> None:
    crawler = _RecordingCrawler()
    service = _service(
        "https://github.com/tiangolo/fastapi/blob/master/README.md",
        crawler=crawler,
        benchmark_source_exclusion=True,
    )

    outcome = asyncio.run(service.search("FastAPI documentation", max_results=1))

    assert outcome.sources[0].url.endswith("/README.md")
    assert crawler.calls == ["https://github.com/tiangolo/fastapi/blob/master/README.md"]


class _CapturingSearchService:
    primary = SimpleNamespace(name="capture")

    def __init__(self) -> None:
        self.blocked_source_urls: list[str] | None = None

    async def search(
        self,
        query: str,
        max_results: int,
        *,
        blocked_source_urls: list[str],
    ) -> Any:
        del query, max_results
        self.blocked_source_urls = blocked_source_urls
        raise BenchmarkContaminationError("benchmark contamination blocked in fixture")


def test_orchestrator_forwards_request_blocklist_to_search_service() -> None:
    blocked_urls = ["https://reference.example/expert-report"]
    request = ResearchRequest(
        query="Compare two systems with cited evidence.",
        blocked_source_urls=blocked_urls,
        max_researchers=1,
        max_rounds=1,
        max_tool_calls=1,
    )
    service = _CapturingSearchService()
    orchestrator = DeepResearchOrchestrator(
        settings=Settings(local_retrieval_mode="none"),
        search_service=service,  # type: ignore[arg-type]
    )
    llm = MockLLMProvider()

    with pytest.raises(BenchmarkContaminationError):
        asyncio.run(
            orchestrator._research_one(
                SubQuestion(
                    id="Q1",
                    question=request.query,
                    rationale="Collect evidence.",
                    search_query=request.query,
                ),
                request,
                service,  # type: ignore[arg-type]
                asyncio.Semaphore(1),
                TraceLogger("blocked-source-test", write_enabled=False),
                None,
                llm,
                CostTracker(provider=llm.name, model=llm.model),
            )
        )

    assert service.blocked_source_urls == blocked_urls


def test_eval_forwards_blocklist_and_records_contamination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_urls = ["https://reference.example/expert-report"]
    captured: dict[str, ResearchRequest] = {}

    class _ContaminatedOrchestrator:
        def __init__(self, settings: Settings) -> None:
            del settings

        async def run(self, request: ResearchRequest) -> Any:
            captured["request"] = request
            raise BenchmarkContaminationError("blocked reference source")

    monkeypatch.setattr(
        "deepresearch_agent.deep_research_eval.DeepResearchOrchestrator",
        _ContaminatedOrchestrator,
    )
    args = argparse.Namespace(
        max_results=4,
        seed=20260714,
        reflection_enabled=False,
        max_reflection_rounds=1,
        reflection_min_sources=4,
        max_rounds=1,
        max_tool_calls=1,
        deadline_seconds=None,
        min_evidence_items=1,
        fallback_policy="fail",
    )
    settings = SimpleNamespace(
        max_researchers=3,
        llm_provider="mock",
        search_provider="mock",
        citation_judge_provider="none",
        citation_judge_model=None,
    )
    case = {
        "id": "drb2-anchor",
        "query": "Produce a detailed comparative report.",
        "category": "test",
        "benchmark_name": "drb2-public12",
        "metadata": {
            "report_depth": "deep",
            "expected_format": "markdown",
            "blocked_source_urls": blocked_urls,
        },
    }

    record = asyncio.run(
        _run_case(
            case,
            args,
            settings,
            "mock-structured-tool-model",
            {"brief_generation": "", "planning": "", "synthesis": ""},
        )
    )

    assert captured["request"].blocked_source_urls == blocked_urls
    assert captured["request"].report_depth == "deep"
    assert record["benchmark_contamination"] is True
    assert record["error_category"] == "benchmark_contamination"
    assert record["answer"] == ""
