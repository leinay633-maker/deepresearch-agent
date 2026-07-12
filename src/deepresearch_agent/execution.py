from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from deepresearch_agent.cost import CostTracker
from deepresearch_agent.schemas import (
    CitationCheckReport,
    Finding,
    ResearchBrief,
    ResearchRequest,
    Source,
    SubQuestion,
)
from deepresearch_agent.search import SearchOutcome, SearchService
from deepresearch_agent.tracing import TraceLogger

if TYPE_CHECKING:
    from deepresearch_agent.orchestrator import DeepResearchOrchestrator, Emit


@dataclass
class ResearchStageArtifacts:
    findings: list[Finding]
    raw_search_count: int
    fallback_count: int
    degraded_count: int
    all_sources: list[Source]
    sources: list[Source]


class ResearchExecutionEngine:
    """Shared deterministic stage runner for direct and durable executions."""

    def __init__(self, orchestrator: DeepResearchOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def run_clarify_stage(
        self,
        *,
        request: ResearchRequest,
        llm: Any,
        cost: CostTracker,
    ) -> ResearchBrief:
        return await llm.create_brief(request, cost)

    async def run_planner_stage(
        self,
        *,
        brief: ResearchBrief,
        max_researchers: int,
        llm: Any,
        cost: CostTracker,
    ) -> list[SubQuestion]:
        return await llm.plan(brief, max_researchers=max_researchers, cost=cost)

    async def run_synthesizer_stage(
        self,
        *,
        brief: ResearchBrief,
        plan: list[SubQuestion],
        findings: list[Finding],
        sources: list[Source],
        llm: Any,
        cost: CostTracker,
    ) -> tuple[str, list[str]]:
        return await llm.synthesize(brief, plan, findings, sources, cost)

    def run_verifier_stage(
        self,
        *,
        request: ResearchRequest,
        claims: list[str],
        sources: list[Source],
        cost: CostTracker,
    ) -> CitationCheckReport:
        return self.orchestrator.citation_checker.check(
            claims,
            sources,
            judge_provider=self.orchestrator._build_citation_judge_provider(request),
            cost=cost,
        )

    async def run_research_stage(
        self,
        *,
        plan: list[SubQuestion],
        request: ResearchRequest,
        search_service: SearchService,
        trace: TraceLogger,
        llm: Any,
        cost: CostTracker,
        max_researchers: int,
        emit: Emit | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> ResearchStageArtifacts:
        semaphore = asyncio.Semaphore(max_researchers)
        research_results: list[tuple[Finding, SearchOutcome]] = list(
            await asyncio.gather(
                *[
                    self.orchestrator._research_one(
                        subquestion,
                        request,
                        search_service,
                        semaphore,
                        trace,
                        emit,
                        llm,
                        cost,
                        cancel_check,
                    )
                    for subquestion in plan
                ]
            )
        )
        research_results = await self.orchestrator._run_reflection_rounds(
            plan=plan,
            research_results=research_results,
            request=request,
            search_service=search_service,
            semaphore=semaphore,
            trace=trace,
            emit=emit,
            llm=llm,
            cost=cost,
            cancel_check=cancel_check,
        )

        raw_search_count = sum(len(outcome.sources) for _, outcome in research_results)
        fallback_count = sum(1 for _, outcome in research_results if outcome.fallback_used)
        degraded_count = sum(1 for _, outcome in research_results if outcome.degraded)
        preliminary_findings = [finding for finding, _ in research_results]

        stage_start = trace.now()
        all_sources = [source for finding in preliminary_findings for source in finding.sources]
        deduped_sources = self.orchestrator.deduper.dedup(all_sources)
        sources = self.orchestrator._assign_source_ids(deduped_sources)
        source_by_key = {
            self.orchestrator.deduper.key(source): source for source in sources
        }
        findings = self.orchestrator._remap_findings(preliminary_findings, source_by_key)
        await self.orchestrator._record(
            trace,
            "source_dedup",
            "success",
            {"before": len(all_sources), "after": len(sources)},
            stage_start,
            emit,
        )
        return ResearchStageArtifacts(
            findings=findings,
            raw_search_count=raw_search_count,
            fallback_count=fallback_count,
            degraded_count=degraded_count,
            all_sources=all_sources,
            sources=sources,
        )


def is_retryable_error(error: Any) -> bool:
    """Classify stage errors consistently for direct and durable traces."""

    if not error:
        return False
    lowered = str(error).lower()
    permanent_markers = (
        "api_key",
        "authentication",
        "unauthorized",
        "forbidden",
        "invalid",
        "schema",
        "json validation",
        "unknown provider",
        "pricing is not configured",
    )
    if any(marker in lowered for marker in permanent_markers):
        return False
    retryable_markers = (
        "timeout",
        "timed out",
        "429",
        "rate limit",
        "connection",
        "temporar",
        "unavailable",
        "circuit breaker",
        "network",
    )
    if any(marker in lowered for marker in retryable_markers):
        return True
    return True
