from __future__ import annotations

import pytest

from deepresearch_agent.citation import CitationChecker
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
