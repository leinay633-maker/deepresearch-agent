from __future__ import annotations

from deepresearch_agent.citation import CitationChecker
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
