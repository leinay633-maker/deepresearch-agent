from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from deepresearch_agent.citation import CitationChecker
from deepresearch_agent.citation_judge import build_citation_judge_provider
from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.dedup import SourceDeduplicator
from deepresearch_agent.llm import DeepSeekLLMProvider, MockLLMProvider, summarize_sources
from deepresearch_agent.rag import LocalRagRetriever
from deepresearch_agent.schemas import (
    Finding,
    ResearchRequest,
    Source,
    StructuredReport,
    SubQuestion,
    TraceEvent,
)
from deepresearch_agent.search import SearchOutcome, SearchService, build_search_service
from deepresearch_agent.tracing import TraceLogger, build_trace_exporter
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
        self.rag = LocalRagRetriever(settings=self.settings)
        self.deduper = SourceDeduplicator()
        self.verifier = SourceVerifier()
        self.citation_checker = CitationChecker()

    async def run(self, request: ResearchRequest, emit: Emit | None = None) -> StructuredReport:
        run_id = uuid.uuid4().hex[:12]
        trace = TraceLogger(
            run_id=run_id,
            trace_dir=self.settings.trace_dir,
            exporter=build_trace_exporter(self.settings),
        )
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
        research_results = await self._run_reflection_rounds(
            plan=plan,
            research_results=list(research_results),
            request=request,
            search_service=search_service,
            semaphore=semaphore,
            trace=trace,
            emit=emit,
        )

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
        citation_report = self.citation_checker.check(
            claims,
            sources,
            judge_provider=self._build_citation_judge_provider(request),
            cost=cost,
        )
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
        stage_models = self._stage_models(request)
        if provider == "deepseek":
            return DeepSeekLLMProvider(
                model=request.llm_model or self.settings.deepseek_model,
                stage_models=stage_models,
            )
        return MockLLMProvider(
            request.llm_model or self.settings.mock_model_name,
            stage_models=stage_models,
        )

    def _stage_models(self, request: ResearchRequest) -> dict[str, str]:
        candidates = {
            "brief_generation": request.brief_model or self.settings.llm_brief_model,
            "planning": request.planner_model or self.settings.llm_planner_model,
            "synthesis": request.synthesis_model or self.settings.llm_synthesis_model,
        }
        return {stage: model for stage, model in candidates.items() if model}

    def _build_citation_judge_provider(self, request: ResearchRequest):
        return build_citation_judge_provider(
            self.settings,
            provider_name=request.citation_judge_provider,
            model=request.citation_judge_model,
        )

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

    async def _run_reflection_rounds(
        self,
        *,
        plan: list[SubQuestion],
        research_results: list[tuple[Finding, SearchOutcome]],
        request: ResearchRequest,
        search_service: SearchService,
        semaphore: asyncio.Semaphore,
        trace: TraceLogger,
        emit: Emit | None,
    ) -> list[tuple[Finding, SearchOutcome]]:
        if not request.reflection_enabled or request.max_reflection_rounds <= 0:
            return research_results

        for round_index in range(1, request.max_reflection_rounds + 1):
            findings = [finding for finding, _ in research_results]
            outcomes = [outcome for _, outcome in research_results]
            compressed_context = self._compress_findings(findings)
            await self._record(
                trace,
                f"compression.round{round_index}",
                "success",
                {
                    "finding_count": len(findings),
                    "compressed_chars": len(compressed_context),
                    "compressed_context": compressed_context,
                },
                emit=emit,
            )
            reflection = self._reflect_on_evidence(
                request=request,
                plan=plan,
                findings=findings,
                outcomes=outcomes,
                round_index=round_index,
            )
            await self._record(
                trace,
                f"reflection.round{round_index}",
                "success",
                reflection,
                emit=emit,
            )
            if not reflection["should_add_question"]:
                break
            subquestion = SubQuestion.model_validate(reflection["subquestion"])
            plan.append(subquestion)
            new_results = await asyncio.gather(
                self._research_one(subquestion, request, search_service, semaphore, trace, emit)
            )
            research_results.extend(new_results)
        return research_results

    def _compress_findings(self, findings: list[Finding], max_chars: int = 1200) -> str:
        lines = []
        for finding in findings:
            titles = ", ".join(source.title for source in finding.sources[:3]) or "no sources"
            lines.append(
                f"{finding.subquestion_id}: {finding.summary} Sources: {titles}"
            )
        compressed = "\n".join(lines)
        if len(compressed) > max_chars:
            return compressed[:max_chars].rstrip()
        return compressed

    def _reflect_on_evidence(
        self,
        *,
        request: ResearchRequest,
        plan: list[SubQuestion],
        findings: list[Finding],
        outcomes: list[SearchOutcome],
        round_index: int,
    ) -> dict[str, Any]:
        unique_urls = {source.url for finding in findings for source in finding.sources}
        fallback_count = sum(1 for outcome in outcomes if outcome.fallback_used)
        low_source_questions = [
            finding.subquestion_id
            for finding in findings
            if len({source.url for source in finding.sources}) < request.reflection_min_sources
        ]
        should_add = bool(low_source_questions or fallback_count) and len(plan) < request.max_researchers + request.max_reflection_rounds
        if not should_add:
            return {
                "should_add_question": False,
                "reason": "current evidence meets heuristic source coverage",
                "unique_source_count": len(unique_urls),
                "fallback_count": fallback_count,
                "low_source_questions": low_source_questions,
            }
        subquestion = SubQuestion(
            id=f"R{round_index}",
            question=(
                "What additional independent evidence, counterexamples, or primary sources are "
                f"needed to verify: {request.query}"
            ),
            rationale=(
                "Reflection found low source coverage or fallback in earlier research; "
                "run one bounded follow-up search before synthesis."
            ),
        )
        return {
            "should_add_question": True,
            "reason": "evidence coverage below reflection threshold",
            "unique_source_count": len(unique_urls),
            "fallback_count": fallback_count,
            "low_source_questions": low_source_questions,
            "subquestion": subquestion.model_dump(mode="json"),
        }

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
