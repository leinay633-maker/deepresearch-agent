from __future__ import annotations

import asyncio
from types import SimpleNamespace

from deepresearch_agent.config import Settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.orchestrator import _coverage_status
from deepresearch_agent.schemas import (
    ResearchDecision,
    ResearchRequest,
    Source,
    SubQuestion,
)
from deepresearch_agent.search import SearchOutcome
from deepresearch_agent.tracing import TraceLogger


class _EmptyRag:
    async def retrieve(self, query: str, max_results: int) -> list[Source]:
        del query, max_results
        return []


class _AlwaysStopLLM:
    async def decide_research(self, **kwargs) -> ResearchDecision:
        del kwargs
        return ResearchDecision(action="stop", reason="fixture requested stop")


def _country_source(country: str, suffix: str) -> Source:
    return Source(
        title=f"{country} pension authority {suffix}",
        url=f"https://{country.lower()}.example/{suffix}",
        content=(
            f"{country} mandatory and voluntary pension schemes cover private and "
            f"self-employed workers. {country} publishes plan type and coverage rules."
        ),
        provider="fixture",
        query="pension schemes",
        score=1.0,
        metadata={"extract_status": "ok", "snippet_only": False},
    )


class _CoverageSearchService:
    primary = SimpleNamespace(name="fixture")

    def __init__(self, rounds: list[list[Source]]) -> None:
        self.rounds = rounds
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int, **kwargs) -> SearchOutcome:
        del max_results, kwargs
        self.calls.append(query)
        sources = self.rounds[min(len(self.calls) - 1, len(self.rounds) - 1)]
        return SearchOutcome(sources=sources, provider="fixture", tool_attempts=1)


def _three_country_subquestion() -> SubQuestion:
    return SubQuestion(
        id="Q1",
        question=(
            "Which mandatory and voluntary pension schemes cover Indonesia, Malaysia, "
            "and Pakistan?"
        ),
        rationale="Supply the three-country scheme profile.",
        search_query="Indonesia Malaysia Pakistan mandatory voluntary pension schemes",
        required_entities=["Indonesia", "Malaysia", "Pakistan"],
        required_aspects=["mandatory", "voluntary"],
    )


def _run_research_one(service: _CoverageSearchService):
    settings = Settings(local_retrieval_mode="none", trace_write_enabled=False)
    orchestrator = DeepResearchOrchestrator(settings=settings)
    orchestrator.rag = _EmptyRag()  # type: ignore[assignment]
    request = ResearchRequest(
        query="Compare pension schemes in Indonesia, Malaysia, and Pakistan.",
        report_depth="deep",
        max_researchers=1,
        max_rounds=2,
        max_tool_calls=2,
        min_evidence_items=2,
        fallback_policy="fail",
    )
    trace = TraceLogger("coverage-aware-test", write_enabled=False)
    finding, _outcome = asyncio.run(
        orchestrator._research_one(
            _three_country_subquestion(),
            request,
            service,  # type: ignore[arg-type]
            asyncio.Semaphore(1),
            trace,
            None,
            _AlwaysStopLLM(),
            CostTracker(provider="fixture", model="fixture"),
        )
    )
    return finding, trace


def test_deep_research_rejects_early_stop_and_focuses_missing_entities() -> None:
    service = _CoverageSearchService(
        [
            [_country_source("Indonesia", "one"), _country_source("Indonesia", "two")],
            [_country_source("Malaysia", "one"), _country_source("Pakistan", "one")],
        ]
    )

    finding, trace = _run_research_one(service)

    assert finding.research is not None
    assert len(finding.research.rounds) == 2
    assert finding.research.rounds[0].decision.action == "need_follow_up"
    assert "Malaysia" in service.calls[1]
    assert "Pakistan" in service.calls[1]
    assert "Indonesia" not in service.calls[1]
    assert finding.research.rounds[1].decision.action == "stop"
    assert finding.research.termination_reason == "evidence_sufficient"
    assert finding.research.budget_exhausted is False
    event = next(item for item in trace.events if item.stage == "researcher.Q1")
    assert event.payload["coverage"]["complete"] is True
    assert event.payload["coverage"]["covered_entities"] == [
        "Indonesia",
        "Malaysia",
        "Pakistan",
    ]


def test_deep_research_stops_when_coverage_contract_is_complete() -> None:
    service = _CoverageSearchService(
        [[
            _country_source("Indonesia", "one"),
            _country_source("Malaysia", "one"),
            _country_source("Pakistan", "one"),
        ]]
    )

    finding, _trace = _run_research_one(service)

    assert finding.research is not None
    assert len(service.calls) == 1
    assert finding.research.rounds[0].decision.action == "stop"
    assert finding.research.termination_reason == "evidence_sufficient"


def test_multicountry_branch_accepts_single_country_page_but_rejects_generic_page() -> None:
    orchestrator = DeepResearchOrchestrator(
        settings=Settings(local_retrieval_mode="none", trace_write_enabled=False)
    )
    subquestion = SubQuestion(
        id="Q3",
        question=(
            "For mandatory Defined Benefit pension schemes in Indonesia, Malaysia, "
            "Pakistan, the Philippines, Sri Lanka, Thailand, and Vietnam, what are the "
            "annual accrual rate and salary base?"
        ),
        rationale="Supply mandatory DB comparison-table evidence.",
        search_query="mandatory DB pension accrual rate salary base seven countries",
        required_entities=[
            "Indonesia",
            "Malaysia",
            "Pakistan",
            "Philippines",
            "Sri Lanka",
            "Thailand",
            "Vietnam",
        ],
        required_aspects=["annual accrual rate", "salary base"],
    )
    philippines = Source(
        title="Philippines SSS official pension formula",
        url="https://philippines.example/sss-formula",
        content=(
            "Philippines pension formula specifies the annual accrual rate and salary base. "
            "Philippines workers receive a pension after the minimum contribution period."
        ),
        provider="fixture",
        query=subquestion.search_query or "",
        metadata={"extract_status": "ok", "snippet_only": False},
    )
    generic = Source(
        title="Regional pension formula guidance",
        url="https://regional.example/formula",
        content=(
            "Mandatory pension formula guidance describes annual accrual rate and salary base "
            "without identifying any jurisdiction."
        ),
        provider="fixture",
        query=subquestion.search_query or "",
        metadata={"extract_status": "ok", "snippet_only": False},
    )

    assert orchestrator._source_is_relevant_for_subquestion(subquestion, philippines) is True
    assert orchestrator._source_is_relevant_for_subquestion(subquestion, generic) is False


def test_coverage_counts_repeated_entity_mentions_in_content() -> None:
    subquestion = SubQuestion(
        id="Q1",
        question="What pension rules apply in Indonesia?",
        rationale="Cover the named country.",
        required_entities=["Indonesia"],
        required_aspects=[],
    )
    source = Source(
        title="Official pension overview",
        url="https://authority.example/overview",
        content=(
            "Indonesia publishes mandatory pension rules. "
            "Indonesia also publishes voluntary pension rules."
        ),
        provider="fixture",
        query="Indonesia pension rules",
        metadata={"extract_status": "ok", "snippet_only": False},
    )

    status = _coverage_status(subquestion, [source], [])

    assert status["covered_entities"] == ["Indonesia"]
    assert status["complete"] is True
