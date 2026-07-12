from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from deepresearch_agent.citation import CitationChecker, best_evidence_quote
from deepresearch_agent.citation_judge import build_citation_judge_provider
from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.dedup import SourceDeduplicator
from deepresearch_agent.execution import ResearchExecutionEngine, is_retryable_error
from deepresearch_agent.llm import (
    DeepSeekLLMProvider,
    MockLLMProvider,
    OpenAICompatibleLLMProvider,
)
from deepresearch_agent.rag import LocalRagRetriever
from deepresearch_agent.report_metrics import build_execution_metrics
from deepresearch_agent.schemas import (
    Finding,
    EvidenceItem,
    ResearchRequest,
    ResearchResult,
    ResearchRoundResult,
    Source,
    StructuredReport,
    SubQuestion,
    TraceEvent,
)
from deepresearch_agent.search import (
    SearchOutcome,
    SearchService,
    build_search_service,
    enrich_source_metadata,
)
from deepresearch_agent.tracing import TraceLogger, build_trace_exporter
from deepresearch_agent.verifier import SourceVerifier

Emit = Callable[[dict[str, Any]], Awaitable[None]]

class DeepResearchOrchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        search_service: SearchService | None = None,
        llm_provider: Any | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.search_service = search_service
        self.llm_provider = llm_provider
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
        llm = self.llm_provider or self._build_llm_provider(request)
        cost = CostTracker(
            provider=llm.name,
            model=llm.model,
            input_cost_per_1m=self.settings.mock_input_cost_per_1m_tokens,
            output_cost_per_1m=self.settings.mock_output_cost_per_1m_tokens,
        )
        engine = ResearchExecutionEngine(self)
        run_start = time.perf_counter()

        await self._record(trace, "run", "start", {"query": request.query}, emit=emit)

        stage_start = trace.now()
        try:
            brief = await engine.run_clarify_stage(request=request, llm=llm, cost=cost)
        except Exception as exc:
            await self._record(
                trace,
                "clarify_normalize",
                "error",
                {"error": str(exc), "retryable": is_retryable_error(exc)},
                stage_start,
                emit,
            )
            raise
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
        try:
            plan = await engine.run_planner_stage(
                brief=brief,
                max_researchers=max_researchers,
                llm=llm,
                cost=cost,
            )
        except Exception as exc:
            await self._record(
                trace,
                "planner",
                "error",
                {"error": str(exc), "retryable": is_retryable_error(exc)},
                stage_start,
                emit,
            )
            raise
        await self._record(
            trace,
            "planner",
            "success",
            {"subquestions": [item.model_dump() for item in plan]},
            stage_start,
            emit,
        )

        search_service = self.search_service or build_search_service(
            self.settings,
            request.search_provider,
            self._fallback_policy(request),
        )
        try:
            research = await engine.run_research_stage(
                plan=plan,
                request=request,
                search_service=search_service,
                trace=trace,
                llm=llm,
                cost=cost,
                max_researchers=max_researchers,
                emit=emit,
                cancel_check=None,
            )
        except Exception as exc:
            await self._record(
                trace,
                "researcher",
                "error",
                {"error": str(exc), "retryable": is_retryable_error(exc)},
                emit=emit,
            )
            raise
        findings = research.findings
        sources = research.sources

        stage_start = trace.now()
        try:
            answer, claims = await engine.run_synthesizer_stage(
                brief=brief,
                plan=plan,
                findings=findings,
                sources=sources,
                llm=llm,
                cost=cost,
            )
        except Exception as exc:
            await self._record(
                trace,
                "synthesizer",
                "error",
                {"error": str(exc), "retryable": is_retryable_error(exc)},
                stage_start,
                emit,
            )
            raise
        await self._record(
            trace,
            "synthesizer",
            "success",
            {"claim_count": len(claims), "source_count": len(sources)},
            stage_start,
            emit,
        )

        stage_start = trace.now()
        try:
            citation_report = engine.run_verifier_stage(
                request=request,
                claims=claims,
                sources=sources,
                cost=cost,
            )
        except Exception as exc:
            await self._record(
                trace,
                "verifier",
                "error",
                {"error": str(exc), "retryable": is_retryable_error(exc)},
                stage_start,
                emit,
            )
            raise
        await self._record(
            trace,
            "citation_check",
            "success",
            citation_report.model_dump(mode="json"),
            stage_start,
            emit,
        )

        latency_ms = round((time.perf_counter() - run_start) * 1000, 3)
        metrics = build_execution_metrics(
            latency_ms=latency_ms,
            raw_search_result_count=research.raw_search_count,
            verified_source_count=len(research.all_sources),
            deduped_source_count=len(sources),
            fallback_count=research.fallback_count,
            degraded_count=research.degraded_count,
            budget_exhausted_count=research.budget_exhausted_count,
            sources=sources,
            citation_report=citation_report,
        )
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
        provider = provider.replace("_", "-")
        stage_models = self._stage_models(request)
        if provider == "deepseek":
            return DeepSeekLLMProvider(
                model=request.llm_model or self.settings.deepseek_model,
                stage_models=stage_models,
            )
        if provider == "openai-compatible":
            return OpenAICompatibleLLMProvider(
                model=request.llm_model or self.settings.openai_compatible_model,
                base_url=self.settings.openai_compatible_base_url,
                api_key_env=self.settings.openai_compatible_api_key_env,
                api_key_required=self.settings.openai_compatible_api_key_required,
                stage_models=stage_models,
                input_cost_per_1m_tokens=(
                    self.settings.openai_compatible_input_cost_per_1m_tokens
                ),
                output_cost_per_1m_tokens=(
                    self.settings.openai_compatible_output_cost_per_1m_tokens
                ),
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

    def _fallback_policy(self, request: ResearchRequest) -> str:
        if request.fallback_policy:
            return request.fallback_policy
        provider = (request.search_provider or self.settings.search_provider).strip().lower()
        return "mock" if provider == "mock" else "degraded"

    async def _research_one(
        self,
        subquestion,
        request: ResearchRequest,
        search_service: SearchService,
        semaphore: asyncio.Semaphore,
        trace: TraceLogger,
        emit: Emit | None,
        llm: Any,
        cost: CostTracker,
        cancel_check: Callable[[], None] | None = None,
    ) -> tuple[Finding, SearchOutcome]:
        async with semaphore:
            stage = f"researcher.{subquestion.id}"
            stage_start = trace.now()
            budget = request.research_budget()
            started_at = time.perf_counter()
            query = subquestion.question
            tool_calls = 0
            provider_tool_attempts = 0
            raw_sources: list[Source] = []
            rag_sources: list[Source] = []
            verified: list[Source] = []
            evidence: list[EvidenceItem] = []
            rounds: list[ResearchRoundResult] = []
            gaps: list[str] = []
            conflicts: list[str] = []
            fallback_used = False
            degraded = False
            errors: list[str] = []
            provider = search_service.primary.name
            termination_reason = "max_rounds"

            async def await_with_deadline(awaitable):
                if budget.deadline_seconds is None:
                    return await awaitable
                remaining_seconds = budget.deadline_seconds - (
                    time.perf_counter() - started_at
                )
                if remaining_seconds <= 0:
                    if hasattr(awaitable, "close"):
                        awaitable.close()
                    raise TimeoutError
                return await asyncio.wait_for(awaitable, timeout=remaining_seconds)

            for round_index in range(1, budget.max_rounds + 1):
                await asyncio.sleep(0)  # explicit cancellation boundary
                if cancel_check is not None:
                    cancel_check()
                if tool_calls >= budget.max_tool_calls:
                    termination_reason = "max_tool_calls"
                    break
                # Count the agent-level search action before awaiting it so a
                # timeout or provider exception still consumes budget. Internal
                # SearchService retries/crawls are recorded separately as tool
                # implementation details, not extra research-loop actions.
                tool_calls += 1
                try:
                    search_call = search_service.search(
                        query,
                        max_results=request.max_results_per_researcher,
                    )
                    outcome = await await_with_deadline(search_call)
                except TimeoutError:
                    termination_reason = "deadline"
                    errors.append("research deadline exceeded")
                    break

                if cancel_check is not None:
                    cancel_check()
                provider = outcome.provider
                provider_tool_attempts += outcome.tool_attempts
                fallback_used = fallback_used or outcome.fallback_used
                degraded = degraded or outcome.degraded
                if outcome.error:
                    errors.append(outcome.error)
                raw_sources.extend(outcome.sources)
                try:
                    round_rag_sources = enrich_source_metadata(
                        await await_with_deadline(
                            self.rag.retrieve(query, max_results=2)
                        )
                    )
                except TimeoutError:
                    termination_reason = "deadline"
                    errors.append("research deadline exceeded during local retrieval")
                    break
                if cancel_check is not None:
                    cancel_check()
                degraded = degraded or any(
                    bool(source.metadata.get("retrieval_degraded"))
                    for source in round_rag_sources
                )
                rag_sources.extend(round_rag_sources)
                combined = self.deduper.dedup([*raw_sources, *rag_sources])
                verified = self.verifier.verify(combined)
                evidence = self._evidence_items(subquestion.question, verified)
                if (
                    budget.deadline_seconds is not None
                    and time.perf_counter() - started_at >= budget.deadline_seconds
                ):
                    termination_reason = "deadline"
                    errors.append("research deadline exceeded")
                    break
                try:
                    decision = await await_with_deadline(
                        llm.decide_research(
                            subquestion=subquestion,
                            evidence=evidence,
                            min_evidence_items=budget.min_evidence_items,
                            round_index=round_index,
                            cost=cost,
                        )
                    )
                except TimeoutError:
                    termination_reason = "deadline"
                    errors.append("research deadline exceeded during evidence decision")
                    break
                if decision.action == "stop" and len(evidence) < budget.min_evidence_items:
                    gap = (
                        f"need {budget.min_evidence_items - len(evidence)} additional "
                        "grounded evidence items"
                    )
                    decision = decision.model_copy(
                        update={
                            "action": "need_follow_up",
                            "reason": (
                                "model requested stop, but the Python executor rejected it "
                                "because grounded evidence is below the configured minimum"
                            ),
                            "evidence_gap": gap,
                            "follow_up_query": decision.follow_up_query
                            or f"{subquestion.question} {gap}",
                        }
                    )
                if cancel_check is not None:
                    cancel_check()
                if decision.evidence_gap:
                    gaps.append(decision.evidence_gap)
                if decision.action == "conflict_found":
                    conflicts.append(decision.reason)
                rounds.append(
                    ResearchRoundResult(
                        round_index=round_index,
                        query=query,
                        source_count=len(verified),
                        evidence_count=len(evidence),
                        tool_attempts=outcome.tool_attempts,
                        fallback_used=outcome.fallback_used,
                        decision=decision,
                    )
                )
                if decision.action == "stop":
                    termination_reason = "evidence_sufficient"
                    break
                if round_index >= budget.max_rounds:
                    termination_reason = "max_rounds"
                    break
                if tool_calls >= budget.max_tool_calls:
                    termination_reason = "max_tool_calls"
                    break
                query = decision.follow_up_query or f"{subquestion.question} {decision.evidence_gap or 'additional evidence'}"

            budget_exhausted = (
                termination_reason in {"deadline", "max_rounds", "max_tool_calls"}
                and len(evidence) < budget.min_evidence_items
            )
            if budget_exhausted:
                degraded = True
                gaps.append(
                    f"research budget ended with {len(evidence)} of "
                    f"{budget.min_evidence_items} required evidence items"
                )
            research_result = ResearchResult(
                rounds=rounds,
                evidence=evidence,
                gaps=list(dict.fromkeys(gaps)),
                conflicts=list(dict.fromkeys(conflicts)),
                tool_calls=tool_calls,
                termination_reason=termination_reason,
                budget_exhausted=budget_exhausted,
            )
            summary = self._summarize_evidence(research_result)
            finding = Finding(
                subquestion_id=subquestion.id,
                subquestion=subquestion.question,
                summary=summary,
                source_ids=[],
                sources=verified,
                research=research_result,
            )
            status = "fallback" if fallback_used or degraded else "success"
            payload = {
                "subquestion": subquestion.question,
                "provider": provider,
                "fallback_used": fallback_used,
                "degraded": degraded,
                "error": "; ".join(dict.fromkeys(errors)) or None,
                "source_count": len(verified),
                "evidence_count": len(evidence),
                "round_count": len(rounds),
                "tool_calls": tool_calls,
                "provider_tool_attempts": provider_tool_attempts,
                "termination_reason": termination_reason,
                "budget_exhausted": budget_exhausted,
            }
            await self._record(trace, stage, status, payload, stage_start, emit)
            return finding, SearchOutcome(
                sources=raw_sources,
                provider=provider,
                fallback_used=fallback_used,
                degraded=degraded,
                error=payload["error"],
                tool_attempts=provider_tool_attempts,
            )

    def _evidence_items(
        self, claim: str, sources: list[Source]
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for source in sources:
            if source.metadata.get("snippet_only") or source.metadata.get(
                "extract_status"
            ) in {"snippet", "crawl_failed", "empty"}:
                continue
            quote, overlap = best_evidence_quote(claim, source)
            if not quote or overlap < self.citation_checker.minimum_overlap_for_claim(claim):
                continue
            items.append(
                EvidenceItem(
                    source_id=source.id,
                    source_title=source.title,
                    source_url=source.url,
                    quote=quote,
                    query=source.query,
                    overlap_score=round(overlap, 3),
                    retrieved_at=source.metadata.get("retrieved_at"),
                )
            )
        return items

    def _summarize_evidence(self, result: ResearchResult) -> str:
        if not result.evidence:
            return "No verified evidence survived filtering for this subquestion."
        excerpts = " | ".join(
            f"{item.source_title}: {item.quote}" for item in result.evidence[:3]
        )
        return (
            f"Collected {len(result.evidence)} verified evidence items across "
            f"{len(result.rounds)} bounded research rounds. Evidence: {excerpts}"
        )

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
        llm: Any,
        cost: CostTracker,
        cancel_check: Callable[[], None] | None = None,
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
                self._research_one(
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
            research = finding.research
            if research is not None:
                mapped_evidence = []
                mapped_by_url = {source.url: source for source in mapped_sources}
                mapped_by_key = {self.deduper.key(source): source for source in mapped_sources}
                for item in research.evidence:
                    mapped = mapped_by_url.get(item.source_url)
                    if mapped is None and item.source_url:
                        evidence_source = Source(
                            title=item.source_title,
                            url=item.source_url,
                            content=item.quote,
                            provider="evidence",
                            query=item.query,
                        )
                        mapped = mapped_by_key.get(self.deduper.key(evidence_source))
                    if mapped is None:
                        continue
                    quote, overlap = best_evidence_quote(finding.subquestion, mapped)
                    if (
                        not quote
                        or overlap
                        < self.citation_checker.minimum_overlap_for_claim(finding.subquestion)
                        or mapped.metadata.get("snippet_only")
                    ):
                        continue
                    mapped_evidence.append(
                        item.model_copy(
                            update={
                                "source_id": mapped.id,
                                "source_title": mapped.title,
                                "source_url": mapped.url,
                                "quote": quote,
                                "overlap_score": round(overlap, 3),
                                "retrieved_at": mapped.metadata.get("retrieved_at"),
                            }
                        )
                    )
                research = research.model_copy(update={"evidence": mapped_evidence})
            remapped.append(
                finding.model_copy(
                    update={
                        "source_ids": [source.id for source in mapped_sources],
                        "sources": mapped_sources,
                        "research": research,
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
        event_payload = dict(payload)
        event_payload.setdefault("attempt", 1)
        event_payload.setdefault(
            "retryable",
            status == "error" and is_retryable_error(event_payload.get("error")),
        )
        event_payload.setdefault(
            "degraded",
            status == "fallback"
            or bool(event_payload.get("fallback_used"))
            or bool(event_payload.get("degraded"))
            or bool(event_payload.get("degraded_count", 0)),
        )
        event = trace.record(stage=stage, status=status, payload=event_payload, start=start)
        if emit is not None:
            await emit({"event": "stage", "data": event.model_dump(mode="json")})
        return event
