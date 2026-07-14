from __future__ import annotations

from types import SimpleNamespace

import pytest

from deepresearch_agent.benchmark import build_case_evaluation_metrics
from deepresearch_agent.citation import CitationChecker, source_is_relevant_to_claim
from deepresearch_agent.cost import CostTracker, deepseek_usage_cost_usd
from deepresearch_agent.dedup import SourceDeduplicator
from deepresearch_agent.schemas import Source
from deepresearch_agent.verifier import SourceVerifier


def test_dedup_keeps_highest_quality_source() -> None:
    deduper = SourceDeduplicator()
    low = Source(
        title="Same",
        url="https://example.com/a?x=1",
        content="short",
        provider="mock",
        query="q",
        score=0.1,
        quality_score=0.1,
    )
    high = low.model_copy(update={"url": "https://example.com/a?x=2", "quality_score": 0.9})

    result = deduper.dedup([low, high])

    assert len(result) == 1
    assert result[0].quality_score == 0.9


def test_verifier_filters_low_quality_content() -> None:
    verifier = SourceVerifier()
    source = Source(
        title="Sponsored",
        url="http://bad.local/post",
        content="click here",
        provider="unknown",
        query="q",
    )

    assert verifier.verify([source]) == []


def test_citation_checker_flags_unsupported_claim() -> None:
    source = Source(
        id="S1",
        title="Trace source",
        url="https://example.local/trace",
        content="Structured trace logs explain latency and cost by stage.",
        provider="mock",
        query="trace",
    )
    checker = CitationChecker()
    report = checker.check(
        [
            "Structured trace logs explain latency and cost by stage [S1]",
            "Redis caching is fully implemented [S1]",
        ],
        [source],
    )

    assert report.supported_claims == 1
    assert report.unsupported_claims == 1
    assert report.assessments[0].support_level == "supported"
    assert report.assessments[0].evidence_quotes[0].source_id == "S1"
    assert "Structured trace logs" in report.assessments[0].evidence_quotes[0].quote
    assert report.assessments[1].support_level == "unsupported"


def test_citation_checker_marks_missing_source_unverifiable() -> None:
    checker = CitationChecker()

    report = checker.check(["Postgres checkpointing is implemented [S9]"], [])

    assert report.supported_claims == 0
    assert report.unsupported_claims == 1
    assert report.assessments[0].support_level == "unverifiable"
    assert report.assessments[0].evidence_quotes == []
    assert report.assessments[0].missing_citation_ids == ["S9"]


def test_empty_claims_do_not_receive_perfect_citation_scores() -> None:
    report = CitationChecker().check([], [])

    assert report.total_claims == 0
    assert report.claim_extraction_valid is False
    assert report.retention_rate == 0.0
    assert report.citation_grounding == 0.0
    assert report.citation_coverage == 0.0

    case_metrics = build_case_evaluation_metrics(
        {"expected_format": "text"},
        SimpleNamespace(
            citation_check=report,
            sources=[],
            answer="Answer without extracted claims.",
            metrics={},
            cost=SimpleNamespace(total_tokens=0, total_estimated_cost_usd=0.0),
        ),
    )
    assert case_metrics["claim_extraction_valid"] is False
    assert case_metrics["unsupported_claim_rate"] is None


def test_citation_checker_does_not_hide_missing_ids_behind_valid_citation() -> None:
    source = Source(
        id="S1",
        title="Trace source",
        url="https://example.local/trace",
        content="Structured trace logs explain latency and cost by stage.",
        provider="mock",
        query="trace",
    )

    report = CitationChecker().check(
        ["Structured trace logs explain latency [S1] [S999]"],
        [source],
    )

    assessment = report.assessments[0]
    assert assessment.support_level == "partial"
    assert assessment.supported is False
    assert assessment.missing_citation_ids == ["S999"]
    assert report.citation_precision == 0.5


def test_snippet_only_citation_is_unverifiable() -> None:
    source = Source(
        id="S1",
        title="Search snippet",
        url="https://example.com/snippet",
        content="Structured trace logs explain latency and cost by stage.",
        provider="search",
        query="trace",
        metadata={"snippet_only": True, "extract_status": "snippet"},
    )

    report = CitationChecker().check(
        ["Structured trace logs explain latency and cost by stage [S1]"],
        [source],
    )

    assert report.assessments[0].support_level == "unverifiable"
    assert report.citation_grounding == 0.0


def test_evidence_quote_selects_relevant_window_inside_long_unpunctuated_page() -> None:
    source = Source(
        id="S1",
        title="Python.org",
        url="https://www.python.org/",
        content=(
            "Navigation and unrelated fallback text " * 80
            + "Download Python source code. Latest: Python 3.14.6. "
            + "More unrelated footer text " * 40
        ),
        provider="bing",
        query="Python latest stable version",
        metadata={"extract_status": "ok", "snippet_only": False},
    )

    report = CitationChecker().check(
        ["The latest Python version is 3.14.6 [S1]"],
        [source],
    )

    assert "3.14.6" in report.assessments[0].evidence_quotes[0].quote


def test_evidence_quote_keeps_date_sentence_for_founding_year_question() -> None:
    source = Source(
        id="S1",
        title="San Carlos history",
        url="https://example.com/san-carlos",
        content=(
            "San Carlos is a municipality in Antioquia, Colombia. "
            "It has dams and a population of 14,480 people. "
            "The town itself was officially started on August 14, 1786."
        ),
        provider="web",
        query="San Carlos Antioquia founding year",
        metadata={"extract_status": "ok", "snippet_only": False},
    )

    report = CitationChecker().check(
        ["What year was San Carlos, Antioquia founded? [S1]"],
        [source],
    )

    assert "1786" in report.assessments[0].evidence_quotes[0].quote


def test_founding_date_heuristic_rejects_correct_date_for_wrong_entity() -> None:
    source = Source(
        id="S1",
        title="Python history",
        url="https://example.com/python-history",
        content="Python was created in 1991.",
        provider="web",
        query="San Carlos founding year",
        metadata={"extract_status": "ok", "snippet_only": False},
    )

    report = CitationChecker().check(
        ["San Carlos was founded in 1991 [S1]"],
        [source],
    )

    assert report.citation_grounding == 0.0
    assert report.assessments[0].support_level == "unsupported"


def test_source_relevance_rejects_single_place_overlap_for_entity_rich_question() -> None:
    query = (
        "In June 1637, Thomas Ballard of Wandsworth accused Richard Kestian of "
        "calling him a liar at which man's house in Putney?"
    )
    unrelated_putney_page = Source(
        id="S1",
        title="Putney Debates",
        url="https://example.com/putney-debates",
        content=(
            "The Putney Debates concerned the Levellers and the New Model Army in 1647."
        ),
        provider="web",
        query=query,
        metadata={"extract_status": "ok", "snippet_only": False},
    )
    relevant_record = Source(
        id="S2",
        title="Thomas Ballard v Richard Kestian",
        url="https://example.com/court-record",
        content=(
            "Thomas Ballard of Wandsworth complained that Richard Kestian called him "
            "a liar at William Carter's house in Putney."
        ),
        provider="web",
        query=query,
        metadata={"extract_status": "ok", "snippet_only": False},
    )

    assert source_is_relevant_to_claim(query, unrelated_putney_page) is False
    assert source_is_relevant_to_claim(query, relevant_record) is True


def test_source_relevance_rejects_single_acronym_collision_in_multi_entity_question() -> None:
    query = (
        "In the video game Project Firebreak, what does the acronym CYAN stand for?"
    )
    unrelated_cyan_page = Source(
        id="S1",
        title="CyAN color terminology",
        url="https://example.com/cyan-color",
        content=(
            "CYAN is discussed as a blue-green color term and printing component. "
            "This reference explains what the acronym CYAN can mean in color systems."
        ),
        provider="web",
        query=query,
        metadata={"extract_status": "ok", "snippet_only": False},
    )
    project_page = Source(
        id="S2",
        title="Project Firebreak game guide",
        url="https://example.com/project-firebreak",
        content=(
            "Project Firebreak includes an organization abbreviated CYAN. "
            "The game guide expands the acronym in its story reference section."
        ),
        provider="web",
        query=query,
        metadata={"extract_status": "ok", "snippet_only": False},
    )

    assert source_is_relevant_to_claim(query, unrelated_cyan_page) is False
    assert source_is_relevant_to_claim(query, project_page) is True


def test_source_relevance_keeps_single_acronym_query_compatible() -> None:
    query = "What does TPLF stand for?"
    source = Source(
        id="S1",
        title="TPLF overview",
        url="https://example.com/tplf",
        content="TPLF stands for the Tigray People's Liberation Front.",
        provider="web",
        query=query,
        metadata={"extract_status": "ok", "snippet_only": False},
    )

    assert source_is_relevant_to_claim(query, source) is True


def test_cost_tracker_can_record_provider_usage() -> None:
    tracker = CostTracker(provider="deepseek", model="deepseek-v4-flash")

    record = tracker.add_usage(
        stage="planning",
        input_tokens=1000,
        output_tokens=200,
        estimated_cost_usd=0.00049,
    )
    summary = tracker.summary()

    assert record.input_tokens == 1000
    assert record.output_tokens == 200
    assert summary.total_tokens == 1200
    assert summary.total_estimated_cost_usd == 0.00049


def test_deepseek_v4_flash_usage_cost_uses_cache_buckets() -> None:
    input_tokens, output_tokens, cost = deepseek_usage_cost_usd(
        "deepseek-v4-flash",
        {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_cache_hit_tokens": 250,
            "prompt_cache_miss_tokens": 750,
        },
    )

    assert input_tokens == 1000
    assert output_tokens == 200
    assert cost == (250 * 0.0028 + 750 * 0.14 + 200 * 0.28) / 1_000_000


def test_deepseek_usage_cost_rejects_unpriced_model_aliases() -> None:
    with pytest.raises(ValueError, match="pricing is not configured"):
        deepseek_usage_cost_usd(
            "deepseek-chat",
            {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
            },
        )
