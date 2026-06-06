from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from deepresearch_agent.citation import CitationChecker
from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.dedup import SourceDeduplicator
from deepresearch_agent.llm import DeepSeekLLMProvider, MockLLMProvider, summarize_sources
from deepresearch_agent.rag import LocalRagRetriever
from deepresearch_agent.schemas import Finding, ResearchRequest, Source, StructuredReport, TraceEvent
from deepresearch_agent.search import SearchOutcome, SearchService, build_search_service
from deepresearch_agent.tracing import TraceLogger
from deepresearch_agent.verifier import SourceVerifier

Emit = Callable[[dict[str, Any]], Awaitable[None]]


class DeepResearchOrchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        search_service: SearchService | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.search_service = search_service
        self.rag = LocalRagRetriever()
        self.deduper = SourceDeduplicator()
        self.verifier = SourceVerifier()
        self.citation_checker = CitationChecker()

    async def run(self, request: ResearchRequest, emit: Emit | None = None) -> StructuredReport:
        run_id = uuid.uuid4().hex[:12]
        trace = TraceLogger(run_id=run_id, trace_dir=self.settings.trace_dir)
        llm = self._build_llm_provider(request)
        cost = CostTracker(
            provider=llm.name,
            model=llm.model,
            input_cost_per_1m=self.settings.mock_input_cost_per_1m_tokens,
            output_cost_per_1m=self.settings.mock_output_cost_per_1m_tokens,
        )
        run_start = time.perf_counter()

        await self._record(trace, "run", "start", {"query": request.query}, emit=emit)

        stage_start = trace.now()
        brief = await llm.create_brief(request, cost)
        await self._record(
            trace,
            "clarify_normalize",
            "success",
            {"normalized_query": brief.normalized_query},
            stage_start,
            emit,
        )

        stage_start = trace.now()
        max_researchers = min(request.max_researchers, self.settings.max_researchers)
        plan = await llm.plan(brief, max_researchers=max_researchers, cost=cost)
        await self._record(
            trace,
            "planner",
            "success",
            {"subquestions": [item.model_dump() for item in plan]},
            stage_start,
            emit,
        )

        search_service = self.search_service or build_search_service(
            self.settings, request.search_provider
        )
        semaphore = asyncio.Semaphore(max_researchers)
        research_tasks = [
            self._research_one(subquestion, request, search_service, semaphore, trace, emit)
            for subquestion in plan
        ]
        research_results = await asyncio.gather(*research_tasks)

        raw_search_count = sum(len(outcome.sources) for _, outcome in research_results)
        fallback_count = sum(1 for _, outcome in research_results if outcome.fallback_used)
        preliminary_findings = [finding for finding, _ in research_results]

        stage_start = trace.now()
        all_sources = [source for finding in preliminary_findings for source in finding.sources]
        deduped_sources = self.deduper.dedup(all_sources)
        sources = self._assign_source_ids(deduped_sources)
        source_by_key = {self.deduper.key(source): source for source in sources}
        findings = self._remap_findings(preliminary_findings, source_by_key)
        await self._record(
            trace,
            "source_dedup",
            "success",
            {"before": len(all_sources), "after": len(sources)},
            stage_start,
            emit,
        )

        stage_start = trace.now()
        answer, claims = await llm.synthesize(brief, plan, findings, sources, cost)
        await self._record(
            trace,
            "synthesizer",
            "success",
            {"claim_count": len(claims), "source_count": len(sources)},
            stage_start,
            emit,
        )

        stage_start = trace.now()
        citation_report = self.citation_checker.check(claims, sources)
        await self._record(
            trace,
            "citation_check",
            "success",
            citation_report.model_dump(mode="json"),
            stage_start,
            emit,
        )

        latency_ms = round((time.perf_counter() - run_start) * 1000, 3)
        metrics = {
            "latency_ms": latency_ms,
            "raw_search_result_count": raw_search_count,
            "verified_source_count": len(all_sources),
            "deduped_source_count": len(sources),
            "fallback_count": fallback_count,
            "citation_retention_rate": citation_report.retention_rate,
            "success": citation_report.retention_rate >= 0.8 and len(sources) > 0,
        }
        await self._record(trace, "run", "success", metrics, start=run_start, emit=emit)

        return StructuredReport(
            run_id=run_id,
            query=request.query,
            brief=brief,
            plan=plan,
            answer=answer,
            claims=claims,
            findings=findings,
            sources=sources,
            citation_check=citation_report,
            cost=cost.summary(),
            metrics=metrics,
            trace_events=trace.events,
        )

    def _build_llm_provider(self, request: ResearchRequest):
        provider = (request.llm_provider or self.settings.llm_provider).strip().lower()
        if provider == "deepseek":
            return DeepSeekLLMProvider(model=request.llm_model or self.settings.deepseek_model)
        return MockLLMProvider(self.settings.mock_model_name)

    async def _research_one(
        self,
        subquestion,
        request: ResearchRequest,
        search_service: SearchService,
        semaphore: asyncio.Semaphore,
        trace: TraceLogger,
        emit: Emit | None,
    ) -> tuple[Finding, SearchOutcome]:
        async with semaphore:
            stage = f"researcher.{subquestion.id}"
            stage_start = trace.now()
            outcome = await search_service.search(
                subquestion.question,
                max_results=request.max_results_per_researcher,
            )
            rag_sources = await self.rag.retrieve(subquestion.question, max_results=2)
            combined = self.deduper.dedup([*outcome.sources, *rag_sources])
            verified = self.verifier.verify(combined)
            summary = summarize_sources(subquestion.question, verified)
            finding = Finding(
                subquestion_id=subquestion.id,
                subquestion=subquestion.question,
                summary=summary,
                source_ids=[],
                sources=verified,
            )
            status = "fallback" if outcome.fallback_used else "success"
            payload = {
                "subquestion": subquestion.question,
                "provider": outcome.provider,
                "fallback_used": outcome.fallback_used,
                "error": outcome.error,
                "source_count": len(verified),
            }
            await self._record(trace, stage, status, payload, stage_start, emit)
            return finding, outcome

    def _assign_source_ids(self, sources: list[Source]) -> list[Source]:
        return [source.model_copy(update={"id": f"S{i + 1}"}) for i, source in enumerate(sources)]

    def _remap_findings(
        self, findings: list[Finding], source_by_key: dict[str, Source]
    ) -> list[Finding]:
        remapped: list[Finding] = []
        for finding in findings:
            mapped_sources = []
            for source in finding.sources:
                mapped = source_by_key.get(self.deduper.key(source))
                if mapped is not None:
                    mapped_sources.append(mapped)
            remapped.append(
                finding.model_copy(
                    update={
                        "source_ids": [source.id for source in mapped_sources],
                        "sources": mapped_sources,
                    }
                )
            )
        return remapped

    async def _record(
        self,
        trace: TraceLogger,
        stage: str,
        status: str,
        payload: dict[str, Any],
        start: float | None = None,
        emit: Emit | None = None,
    ) -> TraceEvent:
        event = trace.record(stage=stage, status=status, payload=payload, start=start)
        if emit is not None:
            await emit({"event": "stage", "data": event.model_dump(mode="json")})
        return event
