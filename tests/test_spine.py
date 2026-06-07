from __future__ import annotations

import asyncio

from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest


def test_deep_research_spine_returns_cited_report() -> None:
    report = asyncio.run(
        DeepResearchOrchestrator().run(
            ResearchRequest(
                query="How does citation checking reduce hallucination in agentic RAG?",
                search_provider="mock",
            )
        )
    )

    assert report.metrics["success"] is True
    assert report.brief.normalized_query.endswith("?")
    assert len(report.plan) == 3
    assert len(report.findings) == 3
    assert report.sources
    assert report.citation_check.retention_rate >= 0.8
    assert report.metrics["source_provider_count"] >= 1
    assert report.metrics["source_domain_count"] >= 1
    assert report.metrics["source_provider_counts"]
    assert report.metrics["source_domain_counts"]
    assert report.cost.total_tokens > 0
    assert any(event.stage == "source_dedup" for event in report.trace_events)
