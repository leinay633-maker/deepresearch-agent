from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from deepresearch_agent.citation import (
    CitationChecker,
    best_evidence_quote,
    entity_anchor_coverage,
    source_is_relevant_to_claim,
)
from deepresearch_agent.citation_judge import build_citation_judge_provider
from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.dedup import SourceDeduplicator
from deepresearch_agent.execution import ResearchExecutionEngine, is_retryable_error
from deepresearch_agent.guardrails import safe_follow_up_query
from deepresearch_agent.gateway_search import (
    GatewayWebSearchUsage,
    capture_gateway_web_search_usage,
)
from deepresearch_agent.llm import (
    DeepSeekLLMProvider,
    LLMGatewayLLMProvider,
    MockLLMProvider,
    OpenAICompatibleLLMProvider,
)
from deepresearch_agent.rag import LocalRagRetriever
from deepresearch_agent.report_metrics import build_execution_metrics
from deepresearch_agent.schemas import (
    CitationCheckReport,
    Finding,
    EvidenceItem,
    ResearchBrief,
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
from deepresearch_agent.text_utils import tokenize
from deepresearch_agent.tracing import TraceLogger, build_trace_exporter
from deepresearch_agent.verifier import SourceVerifier

Emit = Callable[[dict[str, Any]], Awaitable[None]]

_COVERAGE_TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _coverage_term_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in tokenize(value)
        if token not in _COVERAGE_TERM_STOPWORDS
    )


def _text_covers_term(text: str, term: str, *, minimum_occurrences: int = 1) -> bool:
    term_tokens = _coverage_term_tokens(term)
    if not term_tokens:
        return False
    lowered = text.lower()
    counts = Counter(
        re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*|[\u3400-\u9fff]", lowered)
    )
    # ``tokenize`` also exposes CJK bigrams for lightweight semantic matching;
    # count those directly because the lexical occurrence stream above is
    # intentionally character-based for Chinese text.
    for token in term_tokens:
        if any("\u3400" <= char <= "\u9fff" for char in token) and len(token) > 1:
            counts[token] = lowered.count(token)
    return min((counts[token] for token in set(term_tokens)), default=0) >= minimum_occurrences


def _source_covers_entity(
    source: Source,
    entity: str,
    *,
    evidence_quotes: list[str] | None = None,
    require_substantive_content: bool,
) -> bool:
    if _text_covers_term(source.title, entity):
        return True
    if any(_text_covers_term(quote, entity) for quote in (evidence_quotes or [])):
        return True
    return _text_covers_term(
        source.content,
        entity,
        minimum_occurrences=2 if require_substantive_content else 1,
    )


def _coverage_status(
    subquestion: SubQuestion,
    sources: list[Source],
    evidence: list[EvidenceItem],
) -> dict[str, Any]:
    required_entities = list(dict.fromkeys(subquestion.required_entities))
    required_aspects = list(dict.fromkeys(subquestion.required_aspects))
    quotes_by_url: dict[str, list[str]] = {}
    for item in evidence:
        quotes_by_url.setdefault(item.source_url, []).append(item.quote)

    covered_entities = [
        entity
        for entity in required_entities
        if any(
            _source_covers_entity(
                source,
                entity,
                evidence_quotes=quotes_by_url.get(source.url, []),
                require_substantive_content=True,
            )
            for source in sources
        )
    ]
    covered_aspects = [
        aspect
        for aspect in required_aspects
        if any(
            _text_covers_term(
                " ".join(
                    [
                        source.title,
                        source.content,
                        *quotes_by_url.get(source.url, []),
                    ]
                ),
                aspect,
            )
            for source in sources
        )
    ]
    missing_entities = [
        entity for entity in required_entities if entity not in covered_entities
    ]
    missing_aspects = [
        aspect for aspect in required_aspects if aspect not in covered_aspects
    ]
    return {
        "required_entities": required_entities,
        "covered_entities": covered_entities,
        "missing_entities": missing_entities,
        "required_aspects": required_aspects,
        "covered_aspects": covered_aspects,
        "missing_aspects": missing_aspects,
        "complete": not missing_entities and not missing_aspects,
    }


def _coverage_gap(status: dict[str, Any]) -> str | None:
    parts = []
    if status["missing_entities"]:
        parts.append(f"entities={', '.join(status['missing_entities'])}")
    if status["missing_aspects"]:
        parts.append(f"aspects={', '.join(status['missing_aspects'])}")
    if not parts:
        return None
    return "missing required coverage: " + "; ".join(parts)


def _coverage_follow_up_query(status: dict[str, Any]) -> str:
    targets = [*status["missing_entities"], *status["missing_aspects"]]
    return " ".join([*targets, "official primary source"]).strip()

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
            write_enabled=self.settings.trace_write_enabled,
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
            _attach_failure_context(exc, cost=cost, trace=trace)
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
            _attach_failure_context(exc, cost=cost, trace=trace)
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
            _attach_failure_context(exc, cost=cost, trace=trace)
            raise
        findings = research.findings
        sources = research.sources

        stage_start = trace.now()
        synthesis_context: dict[str, Any] | None = None
        try:
            answer, claims = await engine.run_synthesizer_stage(
                brief=brief,
                plan=plan,
                findings=findings,
                sources=sources,
                llm=llm,
                cost=cost,
            )
            raw_synthesis_context = getattr(llm, "last_synthesis_context", None)
            synthesis_context = (
                raw_synthesis_context
                if isinstance(raw_synthesis_context, dict)
                else None
            )
            if (
                self._fallback_policy(request) == "fail"
                and isinstance(synthesis_context, dict)
                and synthesis_context.get("synthesis_fallback")
            ):
                raise RuntimeError(
                    "synthesis fallback is disallowed by fallback_policy=fail"
                )
        except Exception as exc:
            if synthesis_context is None:
                raw_synthesis_context = getattr(llm, "last_synthesis_context", None)
                synthesis_context = (
                    raw_synthesis_context
                    if isinstance(raw_synthesis_context, dict)
                    else None
                )
            await self._record(
                trace,
                "synthesizer",
                "error",
                {
                    "error": str(exc),
                    "retryable": is_retryable_error(exc),
                    "synthesis_fallback_reason": (
                        str(synthesis_context.get("synthesis_fallback_reason") or "")[:500]
                        if synthesis_context
                        and synthesis_context.get("synthesis_fallback")
                        else None
                    ),
                    "context": synthesis_context,
                },
                stage_start,
                emit,
            )
            _attach_failure_context(exc, cost=cost, trace=trace)
            raise
        await self._record(
            trace,
            "synthesizer",
            "success",
            {
                "claim_count": len(claims),
                "source_count": len(sources),
                "context": synthesis_context,
            },
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
            _attach_failure_context(exc, cost=cost, trace=trace)
            raise

        if citation_report.unsupported_claims:
            await self._record(
                trace,
                "citation_check.initial",
                "success",
                citation_report.model_dump(mode="json"),
                stage_start,
                emit,
            )
            repair_synthesis = getattr(llm, "repair_synthesis", None)
            if callable(repair_synthesis) and any(
                assessment.evidence_quotes for assessment in citation_report.assessments
            ):
                repair_start = trace.now()
                try:
                    repaired_answer, repaired_claims = await repair_synthesis(
                        brief,
                        answer,
                        citation_report,
                        sources,
                        cost,
                    )
                    repaired_report = engine.run_verifier_stage(
                        request=request,
                        claims=repaired_claims,
                        sources=sources,
                        cost=cost,
                    )
                    if _citation_report_is_better(repaired_report, citation_report):
                        answer = repaired_answer
                        claims = repaired_claims
                        citation_report = repaired_report
                        await self._record(
                            trace,
                            "synthesis_repair",
                            "success",
                            {
                                "claim_count": len(claims),
                                "citation_grounding": citation_report.citation_grounding,
                                "unsupported_claims": citation_report.unsupported_claims,
                            },
                            repair_start,
                            emit,
                        )
                    else:
                        await self._record(
                            trace,
                            "synthesis_repair",
                            "fallback",
                            {
                                "reason": "repaired answer did not improve citation grounding",
                                "candidate_grounding": repaired_report.citation_grounding,
                                "candidate_unsupported_claims": repaired_report.unsupported_claims,
                            },
                            repair_start,
                            emit,
                        )
                except Exception as exc:  # noqa: BLE001 - retain the verified draft on repair failure.
                    await self._record(
                        trace,
                        "synthesis_repair",
                        "fallback",
                        {"reason": f"{type(exc).__name__}: {str(exc)[:240]}"},
                        repair_start,
                        emit,
                    )

        if citation_report.unsupported_claims:
            answer, claims, citation_report, dropped_claims = _retain_supported_claims(
                brief,
                citation_report,
            )
            await self._record(
                trace,
                "grounded_answer_filter",
                "fallback",
                {
                    "dropped_claims": dropped_claims,
                    "retained_claim_count": len(claims),
                    "reason": (
                        "removed claims that citation verification did not fully support"
                        if claims
                        else "all claims failed citation verification; returned an abstention"
                    ),
                },
                emit=emit,
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
        if provider == "llm-gateway":
            return LLMGatewayLLMProvider(
                model=request.llm_model or self.settings.llm_gateway_model,
                base_url=self.settings.llm_gateway_base_url,
                timeout_seconds=self.settings.llm_gateway_timeout_seconds,
                synthesis_timeout_seconds=(
                    self.settings.llm_synthesis_timeout_seconds
                ),
                max_retries=self.settings.max_retries,
                stage_models=stage_models,
                thinking_budget_tokens=(
                    self.settings.llm_gateway_thinking_budget_tokens
                ),
                require_response_model_match=(
                    self.settings.llm_gateway_require_response_model_match
                ),
            )
        if provider == "mock":
            return MockLLMProvider(
                request.llm_model or self.settings.mock_model_name,
                stage_models=stage_models,
            )
        raise ValueError(f"unknown LLM provider: {provider}")

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
            deadline_at = (
                time.monotonic() + budget.deadline_seconds
                if budget.deadline_seconds is not None
                else None
            )
            query = safe_follow_up_query(
                subquestion.search_query or subquestion.question,
                original_question=subquestion.question,
            )
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
            failed_candidate_hints: list[dict[str, Any]] = []
            retrieval_rounds: list[dict[str, Any]] = []
            provider = search_service.primary.name
            termination_reason = "max_rounds"
            coverage_status = _coverage_status(subquestion, verified, evidence)

            def record_gateway_web_search_usage(
                usage: GatewayWebSearchUsage,
            ) -> None:
                cost.add_usage(
                    stage="gateway_web_search",
                    provider="llm-gateway",
                    model=usage.model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_creation_input_tokens=usage.cache_creation_input_tokens,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                    # Gateway pricing is unpublished; do not invent dollar cost.
                    estimated_cost_usd=0.0,
                )

            async def await_with_deadline(awaitable):
                if deadline_at is None:
                    return await awaitable
                remaining_seconds = deadline_at - time.monotonic()
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
                    search_kwargs: dict[str, Any] = {}
                    if request.blocked_source_urls:
                        search_kwargs["blocked_source_urls"] = (
                            request.blocked_source_urls
                        )
                    if deadline_at is not None:
                        remaining_global = max(deadline_at - time.monotonic(), 0.0)
                        rounds_left = budget.max_rounds - round_index + 1
                        # Keep every configured round reachable and reserve 20%
                        # of this round's slice for local retrieval and the
                        # evidence-decision LLM call.
                        round_slice = remaining_global / max(rounds_left, 1)
                        search_kwargs["deadline_at"] = min(
                            deadline_at,
                            time.monotonic() + (round_slice * 0.8),
                        )
                    search_call = search_service.search(
                        query,
                        max_results=request.search_results_per_researcher(),
                        **search_kwargs,
                    )
                    with capture_gateway_web_search_usage(
                        record_gateway_web_search_usage
                    ):
                        outcome = await await_with_deadline(search_call)
                except TimeoutError as exc:
                    termination_reason = "deadline"
                    errors.append("research deadline exceeded")
                    provider_tool_attempts += int(
                        getattr(exc, "tool_attempts", 0) or 0
                    )
                    timeout_hints = list(
                        getattr(exc, "failed_candidate_hints", []) or []
                    )
                    failed_candidate_hints = self._merge_audit_hints(
                        failed_candidate_hints,
                        timeout_hints,
                    )
                    timeout_audit = dict(
                        getattr(exc, "retrieval_audit", {}) or {}
                    )
                    if timeout_audit:
                        retrieval_rounds.append(
                            self._retrieval_round_audit(
                                claim=subquestion.question,
                                query=query,
                                round_index=round_index,
                                audit=timeout_audit,
                                sources=[],
                                failed_candidate_hints=timeout_hints,
                            )
                        )
                    break
                except Exception as exc:
                    failure_hints = list(
                        getattr(exc, "failed_candidate_hints", []) or []
                    )
                    failure_audit = dict(
                        getattr(exc, "retrieval_audit", {}) or {}
                    )
                    if failure_audit:
                        failure_audit = self._retrieval_round_audit(
                            claim=subquestion.question,
                            query=query,
                            round_index=round_index,
                            audit=failure_audit,
                            sources=[],
                            failed_candidate_hints=failure_hints,
                        )
                        retrieval_rounds.append(failure_audit)
                    await self._record(
                        trace,
                        stage,
                        "error",
                        {
                            "subquestion": subquestion.question,
                            "provider": provider,
                            "error": str(exc),
                            "failed_candidate_hints": failure_hints,
                            "retrieval_rounds": retrieval_rounds,
                        },
                        stage_start,
                        emit,
                    )
                    raise

                if cancel_check is not None:
                    cancel_check()
                provider = outcome.provider
                provider_tool_attempts += outcome.tool_attempts
                fallback_used = fallback_used or outcome.fallback_used
                degraded = degraded or outcome.degraded
                if outcome.error:
                    errors.append(outcome.error)
                failed_candidate_hints = self._merge_audit_hints(
                    failed_candidate_hints,
                    outcome.failed_candidate_hints,
                )
                retrieval_rounds.append(
                    self._retrieval_round_audit(
                        claim=subquestion.question,
                        query=query,
                        round_index=round_index,
                        audit=outcome.retrieval_audit,
                        sources=outcome.sources,
                        failed_candidate_hints=outcome.failed_candidate_hints,
                    )
                )
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
                verified = [
                    source
                    for source in verified
                    if not source.metadata.get("snippet_only")
                    and source.metadata.get("extract_status")
                    not in {"snippet", "crawl_failed", "empty"}
                    if self._source_is_relevant_for_subquestion(subquestion, source)
                ]
                evidence = self._evidence_items(subquestion, verified)
                coverage_status = _coverage_status(subquestion, verified, evidence)
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
                coverage_gap = _coverage_gap(coverage_status)
                if coverage_gap:
                    focused_query = _coverage_follow_up_query(coverage_status)
                    decision = decision.model_copy(
                        update={
                            "action": (
                                "conflict_found"
                                if decision.action == "conflict_found"
                                else "need_follow_up"
                            ),
                            "reason": (
                                "Python coverage contract rejected an early stop because "
                                "verified evidence does not cover every required target"
                            ),
                            "evidence_gap": coverage_gap,
                            "follow_up_query": focused_query,
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
                query = safe_follow_up_query(
                    decision.follow_up_query,
                    original_question=subquestion.question,
                    evidence_gap=decision.evidence_gap,
                )

            budget_exhausted = (
                termination_reason in {"deadline", "max_rounds", "max_tool_calls"}
                and (
                    len(evidence) < budget.min_evidence_items
                    or not coverage_status["complete"]
                )
            )
            if budget_exhausted:
                degraded = True
                gaps.append(
                    f"research budget ended with {len(evidence)} of "
                    f"{budget.min_evidence_items} required evidence items; "
                    f"coverage_complete={coverage_status['complete']}"
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
            aggregate_retrieval_audit = self._aggregate_retrieval_rounds(
                retrieval_rounds
            )
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
                "failed_candidate_hints": failed_candidate_hints,
                "retrieval_rounds": retrieval_rounds,
                "denylist_enforcement_hit": bool(
                    aggregate_retrieval_audit.get("denylist_enforcement_hit")
                ),
                "benchmark_contamination": bool(
                    aggregate_retrieval_audit.get("benchmark_contamination")
                ),
                "protocol_violation_count": int(
                    aggregate_retrieval_audit.get("protocol_violation_count") or 0
                ),
                "coverage": coverage_status,
            }
            await self._record(trace, stage, status, payload, stage_start, emit)
            return finding, SearchOutcome(
                sources=raw_sources,
                provider=provider,
                fallback_used=fallback_used,
                degraded=degraded,
                error=payload["error"],
                tool_attempts=provider_tool_attempts,
                failed_candidate_hints=failed_candidate_hints,
                retrieval_audit=aggregate_retrieval_audit,
                denylist_enforcement_hit=bool(
                    aggregate_retrieval_audit.get("denylist_enforcement_hit")
                ),
                benchmark_contamination=bool(
                    aggregate_retrieval_audit.get("benchmark_contamination")
                ),
                protocol_violations=list(
                    aggregate_retrieval_audit.get("protocol_violations") or []
                ),
            )

    def _source_is_relevant_for_subquestion(
        self,
        subquestion: SubQuestion,
        source: Source,
    ) -> bool:
        claim = subquestion.question
        if source_is_relevant_to_claim(claim, source):
            return True
        if not subquestion.required_entities:
            return False
        quote, overlap = best_evidence_quote(claim, source)
        if (
            not quote
            or overlap < self.citation_checker.minimum_overlap_for_claim(claim)
        ):
            return False
        return any(
            _source_covers_entity(
                source,
                entity,
                evidence_quotes=[quote],
                require_substantive_content=False,
            )
            for entity in subquestion.required_entities
        )

    def _evidence_items(
        self, subquestion: SubQuestion, sources: list[Source]
    ) -> list[EvidenceItem]:
        claim = subquestion.question
        items: list[EvidenceItem] = []
        for source in sources:
            if source.metadata.get("snippet_only") or source.metadata.get(
                "extract_status"
            ) in {"snippet", "crawl_failed", "empty"}:
                continue
            quote, overlap = best_evidence_quote(claim, source)
            if (
                not quote
                or overlap < self.citation_checker.minimum_overlap_for_claim(claim)
                or not self._source_is_relevant_for_subquestion(subquestion, source)
            ):
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

    def _retrieval_round_audit(
        self,
        *,
        claim: str,
        query: str,
        round_index: int,
        audit: dict[str, Any],
        sources: list[Source],
        failed_candidate_hints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        coverage_rows = [
            entity_anchor_coverage(claim, f"{source.title} {source.content}")
            for source in sources
        ]
        coverage_rows.extend(
            entity_anchor_coverage(claim, str(hint.get("title") or ""))
            for hint in failed_candidate_hints
        )
        anchor_count = max(
            (int(row["anchor_count"]) for row in coverage_rows),
            default=int(entity_anchor_coverage(claim, "")["anchor_count"]),
        )
        matched_distribution = Counter(
            str(int(row["matched_anchor_count"])) for row in coverage_rows
        )
        required = 2 if anchor_count >= 2 else 1
        eligible_count = (
            sum(
                1
                for row in coverage_rows
                if int(row["matched_anchor_count"]) >= required
                or bool(row["complete_multiword_entity"])
            )
            if anchor_count
            else 0
        )
        result = {
            "round_index": round_index,
            "query": query,
            "candidate_count": int(
                audit.get("candidate_count")
                or len(sources) + len(failed_candidate_hints)
            ),
            "fetchable_count": int(
                audit.get("fetchable_count")
                or len(sources) + len(failed_candidate_hints)
            ),
            "verified_count": int(audit.get("verified_count") or len(sources)),
            "crawl_attempts": int(audit.get("crawl_attempts") or 0),
            "error_classes": dict(audit.get("error_classes") or {}),
            "entity_coverage": {
                "anchor_count": anchor_count,
                "max_matched_anchor_count": max(
                    (
                        int(row["matched_anchor_count"])
                        for row in coverage_rows
                    ),
                    default=0,
                ),
                "eligible_candidate_count": eligible_count,
                "matched_anchor_distribution": dict(
                    sorted(matched_distribution.items())
                ),
            },
        }
        protocol_violations = [
            dict(violation)
            for violation in (audit.get("protocol_violations") or [])
            if isinstance(violation, dict)
        ]
        if protocol_violations:
            result.update(
                {
                    "denylist_enforcement_hit": True,
                    "benchmark_contamination": bool(
                        audit.get("benchmark_contamination")
                    ),
                    "blocked_count": int(audit.get("blocked_count") or 0),
                    "protocol_violation_count": int(
                        audit.get("protocol_violation_count") or 0
                    ),
                    "protocol_violations": protocol_violations,
                }
            )
        elif audit.get("benchmark_contamination"):
            result["benchmark_contamination"] = True
        return result

    @staticmethod
    def _merge_audit_hints(
        existing: list[dict[str, Any]],
        new: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hint in [*existing, *new]:
            fingerprint = json.dumps(hint, sort_keys=True, ensure_ascii=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            merged.append(hint)
        return merged

    @staticmethod
    def _aggregate_retrieval_rounds(
        rounds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        errors: Counter[str] = Counter()
        for item in rounds:
            errors.update(item.get("error_classes") or {})
        result = {
            "candidate_count": sum(int(item.get("candidate_count") or 0) for item in rounds),
            "fetchable_count": sum(int(item.get("fetchable_count") or 0) for item in rounds),
            "verified_count": sum(int(item.get("verified_count") or 0) for item in rounds),
            "crawl_attempts": sum(int(item.get("crawl_attempts") or 0) for item in rounds),
            "error_classes": dict(sorted(errors.items())),
        }
        protocol_violations = [
            dict(violation)
            for item in rounds
            for violation in (item.get("protocol_violations") or [])
            if isinstance(violation, dict)
        ]
        if protocol_violations:
            result.update(
                {
                    "denylist_enforcement_hit": True,
                    "benchmark_contamination": any(
                        bool(item.get("benchmark_contamination")) for item in rounds
                    ),
                    "blocked_count": sum(
                        int(item.get("blocked_count") or 0) for item in rounds
                    ),
                    "protocol_violation_count": sum(
                        int(item.get("protocol_violation_count") or 0)
                        for item in rounds
                    ),
                    "protocol_violations": protocol_violations,
                }
            )
        elif any(bool(item.get("benchmark_contamination")) for item in rounds):
            result["benchmark_contamination"] = True
        return result

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
            question=request.query,
            rationale=(
                "Reflection found low source coverage or fallback; repeat the user's exact "
                "question to seek one direct, independent corroborating source."
            ),
            search_query=f"{request.query} exact answer independent source",
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


def _citation_report_is_better(
    candidate: CitationCheckReport,
    current: CitationCheckReport,
) -> bool:
    if candidate.supported_claims <= 0:
        return False
    return (
        candidate.citation_grounding > current.citation_grounding
        or (
            candidate.citation_grounding == current.citation_grounding
            and candidate.unsupported_claims < current.unsupported_claims
        )
    )


def _retain_supported_claims(
    brief: ResearchBrief,
    report: CitationCheckReport,
) -> tuple[str, list[str], CitationCheckReport, list[dict[str, str]]]:
    supported = []
    seen_claims: set[str] = set()
    for assessment in report.assessments:
        if not assessment.supported or assessment.claim in seen_claims:
            continue
        seen_claims.add(assessment.claim)
        supported.append(assessment)
    claims = [assessment.claim for assessment in supported]
    if not claims:
        answer = _citation_abstention_answer(brief)
        dropped = [
            {
                "claim": assessment.claim,
                "support_level": assessment.support_level,
                "reason": assessment.judge_reason or assessment.reason,
            }
            for assessment in report.assessments
        ]
        # Keep the original failed assessments for diagnosis. The visible answer and
        # exported claims abstain, while citation_check still explains why they were
        # removed instead of pretending that no claims were ever checked.
        return answer, [], report, dropped
    if brief.expected_format == "json":
        answer = json.dumps({"claims": claims}, ensure_ascii=False, indent=2)
    elif brief.expected_format == "text" or len(claims) == 1:
        answer = "\n".join(claims)
    else:
        answer = "\n".join(f"- {claim}" for claim in claims)
    dropped = [
        {
            "claim": assessment.claim,
            "support_level": assessment.support_level,
            "reason": assessment.judge_reason or assessment.reason,
        }
        for assessment in report.assessments
        if not assessment.supported
    ]
    retained_count = len(supported)
    filtered_report = report.model_copy(
        update={
            "total_claims": retained_count,
            "supported_claims": retained_count,
            "unsupported_claims": 0,
            "retention_rate": 1.0,
            "assessments": supported,
            "citation_grounding": 1.0,
            "citation_coverage": 1.0,
            "unsupported_claim_rate": 0.0,
            "claim_extraction_valid": True,
        }
    )
    return answer, claims, filtered_report, dropped


def _citation_abstention_answer(brief: ResearchBrief) -> str:
    chinese = any("\u3400" <= char <= "\u9fff" for char in brief.original_query)
    limitation = (
        "现有来源不足以形成经过引用核查的结论。"
        if chinese
        else "The available sources are insufficient to support a citation-verified answer."
    )
    if brief.expected_format == "json":
        return json.dumps(
            {"claims": [], "limitations": [limitation]},
            ensure_ascii=False,
            indent=2,
        )
    return limitation


def _attach_failure_context(
    exc: Exception,
    *,
    cost: CostTracker,
    trace: TraceLogger,
) -> None:
    """Preserve auditable attempted usage and stages for an eval failure."""

    setattr(exc, "deepresearch_cost", cost.summary())
    setattr(exc, "deepresearch_trace_events", list(trace.events))
    setattr(exc, "deepresearch_run_id", trace.run_id)
