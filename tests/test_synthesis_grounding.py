from __future__ import annotations

import asyncio
import json

import pytest

from deepresearch_agent.config import Settings
from deepresearch_agent.llm import (
    MockLLMProvider,
    _deterministic_synthesis,
    _evidence_abstention_synthesis,
    _is_evidence_abstention_error,
    _synthesis_from_payload,
)
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import (
    CitationAssessment,
    CitationCheckReport,
    EvidenceQuote,
    ResearchRequest,
    Source,
    Finding,
    ResearchBrief,
    ResearchResult,
)
from deepresearch_agent.search import MockSearchAdapter, SearchService


def test_synthesis_validation_caps_claims_and_requires_query_language() -> None:
    with pytest.raises(ValueError, match="maximum is 3"):
        _synthesis_from_payload(
            {
                "answer": "Four facts [S1].",
                "claims": [f"Fact {index} [S1]" for index in range(4)],
            },
            allowed_source_ids={"S1"},
            query="What is the answer?",
            max_claims=3,
        )


def test_synthesis_recovers_cited_answer_when_parallel_claims_omit_citations() -> None:
    answer, claims = _synthesis_from_payload(
        {
            "answer": "William Carter's house was the location [S1].",
            "claims": ["William Carter's house was the location"],
        },
        allowed_source_ids={"S1"},
        query="At whose house did the event occur?",
    )

    assert answer == "William Carter's house was the location [S1]."
    assert claims == ["William Carter's house was the location [S1]."]


def test_synthesis_rebuilds_uncited_visible_answer_from_cited_claims() -> None:
    answer, claims = _synthesis_from_payload(
        {
            "answer": "William Carter's house was the location.",
            "claims": ["William Carter's house was the location [S1]."],
        },
        allowed_source_ids={"S1"},
        query="At whose house did the event occur?",
    )

    assert answer == "William Carter's house was the location [S1]."
    assert claims == ["William Carter's house was the location [S1]."]


def test_synthesis_discards_answer_facts_outside_checked_claims() -> None:
    sanitization_audit: dict[str, int | bool] = {}
    answer, claims = _synthesis_from_payload(
        {
            "answer": "San Carlos was founded in 1900 [S1]. Python was created in 1991 [S1].",
            "claims": ["Python was created in 1991 [S1]"],
        },
        allowed_source_ids={"S1"},
        query="When was San Carlos founded?",
        sanitization_audit=sanitization_audit,
    )
    assert answer == "Python was created in 1991 [S1]"
    assert claims == ["Python was created in 1991 [S1]"]
    assert sanitization_audit == {
        "enabled": False,
        "applied": False,
        "dropped_uncited_sentence_count": 0,
        "dropped_uncited_line_count": 0,
        "dropped_uncited_table_row_count": 0,
    }

    with pytest.raises(ValueError, match="only contain claims"):
        _synthesis_from_payload(
            {
                "answer": {
                    "founded_year": 1900,
                    "claims": ["Python was created in 1991 [S1]"],
                },
                "claims": ["Python was created in 1991 [S1]"],
            },
            allowed_source_ids={"S1"},
            expected_format="json",
            query="When was San Carlos founded?",
        )

    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "Python was created in 1991 [S1]. "
                "San Carlos was founded in 1900."
            ),
            "claims": ["Python was created in 1991 [S1]."],
        },
        allowed_source_ids={"S1"},
        query="When was San Carlos founded?",
    )
    assert answer == "Python was created in 1991 [S1]."
    assert claims == ["Python was created in 1991 [S1]."]

    with pytest.raises(ValueError, match="claims must exactly match"):
        _synthesis_from_payload(
            {
                "answer": {"claims": ["San Carlos was founded in 1900 [S1]"]},
                "claims": ["Python was created in 1991 [S1]"],
            },
            allowed_source_ids={"S1"},
            expected_format="json",
            query="When was San Carlos founded?",
        )

    with pytest.raises(ValueError, match="limitations are reserved"):
        _synthesis_from_payload(
            {
                "answer": {
                    "claims": ["Python was created in 1991 [S1]"],
                    "limitations": {"hidden_fact": "San Carlos was founded in 1900"},
                },
                "claims": ["Python was created in 1991 [S1]"],
            },
            allowed_source_ids={"S1"},
            expected_format="json",
            query="When was San Carlos founded?",
        )

    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "Python was created in 1991 [S1]. "
                "Limitations: San Carlos was founded in 1900."
            ),
            "claims": ["Python was created in 1991 [S1]."],
        },
        allowed_source_ids={"S1"},
        query="When was San Carlos founded?",
    )
    assert answer == "Python was created in 1991 [S1]."
    assert claims == ["Python was created in 1991 [S1]."]

    with pytest.raises(ValueError, match="must be an object"):
        _synthesis_from_payload(
            {
                "answer": ["Python was created in 1991 [S1]"],
                "claims": ["Python was created in 1991 [S1]"],
            },
            allowed_source_ids={"S1"},
            expected_format="json",
            query="When was San Carlos founded?",
        )


def test_deterministic_synthesis_is_bounded_and_uses_claims_only() -> None:
    brief = ResearchBrief(
        original_query="What year was San Carlos founded?",
        normalized_query="What year was San Carlos founded?",
        scope="fact",
        constraints=[],
        assumptions=[],
        expected_format="text",
    )
    findings = [
        Finding(
            subquestion_id=f"Q{index}",
            subquestion="founding year",
            summary="ignored raw summary",
            source_ids=[f"S{index}"],
            sources=[],
            research=ResearchResult(),
        )
        for index in range(1, 6)
    ]
    sources = [
        Source(
            id=f"S{index}",
            title=f"Source {index}",
            url=f"https://example.com/{index}",
            content=f"Evidence {index}.",
            provider="fixture",
            query="San Carlos founding",
        )
        for index in range(1, 6)
    ]

    answer, claims = _deterministic_synthesis(brief, findings, sources)

    assert len(claims) == 3
    assert answer.splitlines() == claims

    with pytest.raises(ValueError, match="must use Chinese"):
        _synthesis_from_payload(
            {
                "answer": "Python 3.14.6 is the latest release [S1].",
                "claims": ["Python 3.14.6 is the latest release [S1]"],
            },
            allowed_source_ids={"S1"},
            query="Python 最新版本是什么？",
            max_claims=3,
        )


def test_evidence_abstention_is_a_claim_free_user_language_response() -> None:
    english = ResearchBrief(
        original_query="Who was named in the record?",
        normalized_query="Who was named in the record?",
        scope="fact",
        constraints=[],
        assumptions=[],
        expected_format="text",
    )
    chinese = english.model_copy(
        update={"original_query": "记录中提到谁？", "expected_format": "json"}
    )

    english_answer, english_claims = _evidence_abstention_synthesis(english)
    chinese_answer, chinese_claims = _evidence_abstention_synthesis(chinese)

    assert english_claims == []
    assert english_answer == (
        "The available sources are insufficient to support a citation-verified answer."
    )
    assert chinese_claims == []
    assert json.loads(chinese_answer)["claims"] == []
    assert "现有来源不足" in json.loads(chinese_answer)["limitations"][0]


def test_empty_synthesis_is_an_abstention_only_without_verified_sources() -> None:
    error = RuntimeError("LLM synthesis response contains no usable claims")

    assert _is_evidence_abstention_error(error) is True
    assert (
        _is_evidence_abstention_error(error, has_verified_sources=True) is False
    )


@pytest.mark.parametrize(
    "message",
    [
        "LLM synthesis claim is missing a citation ID",
        "LLM synthesis answer is missing source citations",
    ],
)
def test_missing_citations_remain_validation_failures(message: str) -> None:
    assert _is_evidence_abstention_error(RuntimeError(message)) is False


class _EvidenceSearchAdapter:
    name = "evidence"

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del max_results, timeout
        return [
            Source(
                title="Python Source Releases",
                url="https://www.python.org/downloads/source/",
                content=(
                    "Python Source Releases. Latest Python 3 Release - Python 3.14.6. "
                    "Stable Releases include Python 3.14.6."
                ),
                provider=self.name,
                query=query,
                score=100.0,
                metadata={"extract_status": "ok", "snippet_only": False},
            )
        ]


class _RepairingMockLLM(MockLLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.repair_called = False

    async def synthesize(self, brief, plan, findings, sources, cost):
        del brief, plan, findings, sources
        claims = [
            "The latest Python 3 release is Python 3.14.6 [S1]",
            "Redis caching is fully implemented [S1]",
        ]
        answer = ".\n".join(claims)
        cost.add("synthesis", "draft", answer)
        return answer, claims

    async def repair_synthesis(self, brief, answer, citation_report, sources, cost):
        del brief, answer, citation_report, sources
        self.repair_called = True
        claim = "The latest Python 3 release is Python 3.14.6 [S1]"
        cost.add("synthesis_repair", "draft", claim)
        return claim, [claim]


class _AllPartialRepairingMockLLM(_RepairingMockLLM):
    async def synthesize(self, brief, plan, findings, sources, cost):
        del brief, plan, findings, sources
        claim = "The latest Python 3 release is Python 3.14.6 and Redis is implemented [S1]"
        cost.add("synthesis", "draft", claim)
        return claim, [claim]


class _UnsupportedDraftMockLLM(MockLLMProvider):
    async def synthesize(self, brief, plan, findings, sources, cost):
        del plan, findings, sources
        claim = "Redis caching is fully implemented [S1]"
        answer = (
            json.dumps({"claims": [claim]}, ensure_ascii=False)
            if brief.expected_format == "json"
            else claim
        )
        cost.add("synthesis", "draft", answer)
        return answer, [claim]


class _FallbackSynthesisMockLLM(MockLLMProvider):
    async def synthesize(self, brief, plan, findings, sources, cost):
        del brief, plan, findings, sources
        self.last_synthesis_context = {
            "synthesis_fallback": True,
            "synthesis_fallback_reason": "fixture structured-output validation failed",
            "estimated_tokens": 1234,
            "attempt_ledger": [
                {
                    "attempt": 1,
                    "failure_class": "structured_output_validation",
                }
            ],
        }
        claim = "The latest Python 3 release is Python 3.14.6 [S1]"
        cost.add("synthesis", "fallback", claim)
        return claim, [claim]


class _ExplodingSynthesisMockLLM(MockLLMProvider):
    async def synthesize(self, brief, plan, findings, sources, cost):
        del brief, plan, findings, sources, cost
        raise RuntimeError("fixture synthesis transport failed")


class _PartialThenSupportedChecker:
    def __init__(self) -> None:
        self.calls = 0

    def minimum_overlap_for_claim(self, claim: str) -> float:
        del claim
        return 0.0

    def check(self, claims, sources, judge_provider=None, cost=None):
        del judge_provider, cost
        self.calls += 1
        source = sources[0]
        assessment = CitationAssessment(
            claim=claims[0],
            citation_ids=["S1"],
            supported=self.calls > 1,
            support_level="supported" if self.calls > 1 else "partial",
            reason="test citation result",
            overlap_score=1.0,
            evidence_quotes=[
                EvidenceQuote(
                    source_id="S1",
                    source_title=source.title,
                    source_url=source.url,
                    quote="Latest Python 3 Release - Python 3.14.6.",
                    overlap_score=1.0,
                )
            ],
        )
        supported = int(assessment.supported)
        return CitationCheckReport(
            total_claims=1,
            supported_claims=supported,
            unsupported_claims=1 - supported,
            retention_rate=float(supported),
            assessments=[assessment],
            citation_grounding=float(supported),
            citation_coverage=1.0,
            unsupported_claim_rate=float(1 - supported),
            citation_precision=1.0,
        )


def test_orchestrator_repairs_and_returns_only_fully_grounded_claims() -> None:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="none",
        max_researchers=1,
        max_retries=0,
    )
    service = SearchService(
        primary=_EvidenceSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="fail",
    )
    llm = _RepairingMockLLM()
    report = asyncio.run(
        DeepResearchOrchestrator(
            settings=settings,
            search_service=service,
            llm_provider=llm,
        ).run(
            ResearchRequest(
                query="What is the latest Python 3 release?",
                max_researchers=1,
                max_results_per_researcher=1,
                fallback_policy="fail",
            )
        )
    )

    assert llm.repair_called is True
    assert report.claims == ["The latest Python 3 release is Python 3.14.6 [S1]"]
    assert report.citation_check.citation_grounding == 1.0
    assert report.citation_check.unsupported_claims == 0
    assert "Redis caching" not in report.answer
    assert {event.stage for event in report.trace_events} >= {
        "citation_check.initial",
        "synthesis_repair",
        "citation_check",
    }


def test_orchestrator_repairs_when_every_initial_claim_is_only_partial() -> None:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="none",
        max_researchers=1,
        max_retries=0,
    )
    service = SearchService(
        primary=_EvidenceSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="fail",
    )
    llm = _AllPartialRepairingMockLLM()
    orchestrator = DeepResearchOrchestrator(
        settings=settings,
        search_service=service,
        llm_provider=llm,
    )
    orchestrator.citation_checker = _PartialThenSupportedChecker()

    report = asyncio.run(
        orchestrator.run(
            ResearchRequest(
                query="What is the latest Python 3 release?",
                max_researchers=1,
                max_results_per_researcher=1,
                fallback_policy="fail",
            )
        )
    )

    assert llm.repair_called is True
    assert report.citation_check.citation_grounding == 1.0
    assert report.claims == ["The latest Python 3 release is Python 3.14.6 [S1]"]


@pytest.mark.parametrize("expected_format", ["text", "json"])
def test_orchestrator_abstains_when_every_claim_is_unsupported(
    expected_format: str,
) -> None:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="none",
        max_researchers=1,
        max_retries=0,
    )
    service = SearchService(
        primary=_EvidenceSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="fail",
    )
    report = asyncio.run(
        DeepResearchOrchestrator(
            settings=settings,
            search_service=service,
            llm_provider=_UnsupportedDraftMockLLM(),
        ).run(
            ResearchRequest(
                query="What is the latest Python 3 release?",
                max_researchers=1,
                max_results_per_researcher=1,
                fallback_policy="fail",
                expected_format=expected_format,
            )
        )
    )

    assert report.claims == []
    assert "Redis" not in report.answer
    if expected_format == "json":
        assert json.loads(report.answer) == {
            "claims": [],
            "limitations": [
                "The available sources are insufficient to support a citation-verified answer."
            ],
        }
    else:
        assert report.answer == (
            "The available sources are insufficient to support a citation-verified answer."
        )

    # Keep the rejected draft assessments for diagnosis instead of rewriting history
    # as though the verifier saw no claims.
    assert report.citation_check.total_claims == 1
    assert report.citation_check.supported_claims == 0
    assert report.citation_check.unsupported_claims == 1
    assert len(report.citation_check.assessments) == 1
    assert report.citation_check.assessments[0].claim == (
        "Redis caching is fully implemented [S1]"
    )
    filter_event = next(
        event for event in report.trace_events if event.stage == "grounded_answer_filter"
    )
    assert filter_event.payload["retained_claim_count"] == 0
    assert "abstention" in filter_event.payload["reason"]


def test_orchestrator_rejects_synthesis_fallback_when_policy_is_fail() -> None:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="none",
        max_researchers=1,
        max_retries=0,
    )
    service = SearchService(
        primary=_EvidenceSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="fail",
    )

    with pytest.raises(
        RuntimeError,
        match="synthesis fallback is disallowed by fallback_policy=fail",
    ) as error:
        asyncio.run(
            DeepResearchOrchestrator(
                settings=settings,
                search_service=service,
                llm_provider=_FallbackSynthesisMockLLM(),
            ).run(
                ResearchRequest(
                    query="What is the latest Python 3 release?",
                    max_researchers=1,
                    max_results_per_researcher=1,
                    fallback_policy="fail",
                )
            )
        )

    synthesizer_event = error.value.deepresearch_trace_events[-1]
    assert synthesizer_event.stage == "synthesizer"
    assert synthesizer_event.payload["synthesis_fallback_reason"] == (
        "fixture structured-output validation failed"
    )
    assert synthesizer_event.payload["context"]["estimated_tokens"] == 1234
    assert synthesizer_event.payload["context"]["attempt_ledger"] == [
        {
            "attempt": 1,
            "failure_class": "structured_output_validation",
        }
    ]


def test_orchestrator_preserves_direct_synthesis_error_and_trace_context() -> None:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="none",
        max_researchers=1,
    )
    service = SearchService(
        primary=_EvidenceSearchAdapter(),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="fail",
    )

    with pytest.raises(RuntimeError, match="fixture synthesis transport failed") as error:
        asyncio.run(
            DeepResearchOrchestrator(
                settings=settings,
                search_service=service,
                llm_provider=_ExplodingSynthesisMockLLM(),
            ).run(
                ResearchRequest(
                    query="What is the latest Python 3 release?",
                    max_researchers=1,
                    max_results_per_researcher=1,
                    fallback_policy="fail",
                )
            )
        )

    event = error.value.deepresearch_trace_events[-1]
    assert event.stage == "synthesizer"
    assert event.payload["error"] == "fixture synthesis transport failed"
    assert event.payload["synthesis_fallback_reason"] is None
