from __future__ import annotations

import asyncio

from deepresearch_agent.config import Settings
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest


def test_reflection_adds_bounded_follow_up_question() -> None:
    request = ResearchRequest(
        query="How should citation grounding work in a research agent?",
        max_researchers=1,
        max_results_per_researcher=1,
        search_provider="mock",
        llm_provider="mock",
        reflection_enabled=True,
        max_reflection_rounds=1,
        reflection_min_sources=4,
    )
    orchestrator = DeepResearchOrchestrator(settings=Settings(local_retrieval_mode="keyword"))

    report = asyncio.run(orchestrator.run(request))

    stages = [event.stage for event in report.trace_events]
    assert len(report.plan) == 2
    assert report.plan[1].id == "R1"
    assert report.plan[1].question == request.query
    assert "exact answer independent source" in (report.plan[1].search_query or "")
    assert "compression.round1" in stages
    assert "reflection.round1" in stages
    assert any(finding.subquestion_id == "R1" for finding in report.findings)
