from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.run_models import (
    AgentEvent,
    AgentRun,
    ApprovalPayload,
    CreateRunRequest,
    EditPlanRequest,
    RecoverStaleRunsRequest,
    RejectRunRequest,
    RunLeaseRequest,
    RunActionResponse,
    RunTrace,
)
from deepresearch_agent.run_store import RunStore
from deepresearch_agent.schemas import (
    CostRecord,
    CostSummary,
    Finding,
    ResearchBrief,
    ResearchRequest,
    Source,
    StructuredReport,
    SubQuestion,
)
from deepresearch_agent.search import build_search_service
from deepresearch_agent.source_metrics import source_diversity_metrics
from deepresearch_agent.tracing import TraceLogger, build_trace_exporter


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class RunCancelledError(RuntimeError):
    pass


class RunController:
    def __init__(
        self,
        store: RunStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.store = store or RunStore(settings=self.settings)
        self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"

    async def create_run(self, request: CreateRunRequest) -> AgentRun:
        run_id = uuid.uuid4().hex[:12]
        run = self.store.create_run(
            run_id=run_id,
            query=request.query,
            require_approval=request.require_approval,
            request_json=request.model_dump(mode="json"),
        )
        self._event(run_id, "run", "queued", {"query": request.query})
        if request.defer_execution:
            return run
        try:
            return await self._run_planner(run_id, request)
        except RunCancelledError:
            return self._cancelled_run(run_id)
        except Exception as exc:  # noqa: BLE001
            return self._fail_run(run_id, "planner", exc)

    async def approve(self, run_id: str) -> AgentRun:
        run = self._require_status(run_id, {"waiting_approval"})
        self._event(run_id, "approval", "approved", {})
        return await self._continue_from_plan(run)

    async def edit(self, run_id: str, request: EditPlanRequest) -> AgentRun:
        run = self._require_status(run_id, {"waiting_approval"})
        plan_state = self._plan_state(run)
        plan_state["subquestions"] = [
            item.model_dump(mode="json") for item in request.subquestions
        ]
        self.store.update_run(run_id, plan_json=plan_state)
        self._event(
            run_id,
            "approval",
            "edited",
            {"subquestion_count": len(request.subquestions)},
        )
        return await self._continue_from_plan(self.store.require_run(run_id))

    def reject(self, run_id: str, request: RejectRunRequest) -> AgentRun:
        self._require_status(run_id, {"waiting_approval"})
        self._event(run_id, "approval", "rejected", {"reason": request.reason})
        self.store.update_run(
            run_id,
            status="cancelled",
            current_stage="completed",
            error_message=request.reason,
        )
        return self.store.clear_lease(run_id)

    def cancel(self, run_id: str) -> AgentRun:
        run = self.store.require_run(run_id)
        if run.status in TERMINAL_STATUSES:
            return run
        self._event(run_id, "run", "cancelled", {"previous_status": run.status})
        self.store.update_run(
            run_id,
            status="cancelled",
            current_stage="completed",
            error_message="run cancelled",
        )
        return self.store.clear_lease(run_id)

    async def retry(self, run_id: str) -> AgentRun:
        run = self._require_status(run_id, {"failed"})
        self._event(run_id, "run", "retrying", {})
        self.store.update_run(run_id, status="queued", error_message=None)
        run = self.store.require_run(run_id)
        if run.plan_json is not None:
            return await self._continue_from_plan(run, retry_count=1)
        request = self._request_from_run(run, require_approval=False)
        try:
            return await self._run_planner(run_id, request, retry_count=1)
        except RunCancelledError:
            return self._cancelled_run(run_id)
        except Exception as exc:  # noqa: BLE001
            return self._fail_run(run_id, "planner", exc, retry_count=1)

    async def process_next_queued(self) -> AgentRun | None:
        run = self.store.claim_next_queued_run(
            worker_id=self.worker_id,
            lease_seconds=self.settings.run_lease_seconds,
        )
        if run is None:
            return None
        self._event(
            run.run_id,
            "worker",
            "claimed",
            {"worker_id": self.worker_id},
        )
        request = self._request_from_run(run)
        try:
            return await self._run_planner(run.run_id, request)
        except RunCancelledError:
            return self._cancelled_run(run.run_id)
        except Exception as exc:  # noqa: BLE001
            return self._fail_run(run.run_id, "planner", exc)

    def get_run(self, run_id: str) -> AgentRun:
        return self.store.require_run(run_id)

    def list_runs(self, limit: int = 20) -> list[AgentRun]:
        return self.store.list_runs(limit=limit)

    def acquire_lease(self, run_id: str, request: RunLeaseRequest) -> AgentRun:
        run = self.store.acquire_lease(
            run_id,
            worker_id=request.worker_id,
            lease_seconds=request.lease_seconds,
        )
        if run is None:
            self.store.require_run(run_id)
            raise ValueError(f"run {run_id} lease is held by another worker or terminal")
        self._event(
            run_id,
            "lease",
            "acquired",
            {
                "worker_id": request.worker_id,
                "lease_expires_at": run.lease_expires_at.isoformat()
                if run.lease_expires_at
                else None,
            },
        )
        return self.store.require_run(run_id)

    def heartbeat_lease(self, run_id: str, request: RunLeaseRequest) -> AgentRun:
        run = self.store.heartbeat_lease(
            run_id,
            worker_id=request.worker_id,
            lease_seconds=request.lease_seconds,
        )
        if run is None:
            self.store.require_run(run_id)
            raise ValueError(f"run {run_id} lease heartbeat rejected")
        self._event(
            run_id,
            "lease",
            "heartbeat",
            {
                "worker_id": request.worker_id,
                "lease_expires_at": run.lease_expires_at.isoformat()
                if run.lease_expires_at
                else None,
            },
        )
        return self.store.require_run(run_id)

    def stale_runs(self) -> list[AgentRun]:
        return self.store.list_stale_runs()

    def recover_stale_runs(self, request: RecoverStaleRunsRequest) -> list[AgentRun]:
        recovered: list[AgentRun] = []
        for run in self.store.list_stale_runs():
            self._step(run.run_id, run.current_stage, "failed", error=request.reason)
            self._event(
                run.run_id,
                "lease",
                "stale_recovered",
                {"reason": request.reason, "leased_by": run.leased_by},
            )
            self.store.update_run(
                run.run_id,
                status="failed",
                error_message=request.reason,
            )
            recovered.append(self.store.clear_lease(run.run_id))
        return recovered

    def steps(self, run_id: str):
        self.store.require_run(run_id)
        return self.store.list_steps(run_id)

    def events(self, run_id: str, after_event_id: int | None = None) -> list[AgentEvent]:
        self.store.require_run(run_id)
        return self.store.list_events(run_id, after_event_id=after_event_id)

    def trace(self, run_id: str) -> RunTrace:
        run = self.store.require_run(run_id)
        steps = self.store.list_steps(run_id)
        events = self.store.list_events(run_id)
        latency_ms = None
        if steps:
            started_at = min(step.created_at for step in steps)
            ended_at = run.updated_at
            latency_ms = round((ended_at - started_at).total_seconds() * 1000, 3)
        return RunTrace(
            run=run,
            steps=steps,
            events=events,
            total_tokens=run.total_tokens,
            total_cost=run.total_cost,
            latency_ms=latency_ms,
        )

    def approval_payload(self, run_id: str) -> ApprovalPayload:
        run = self._require_status(run_id, {"waiting_approval"})
        plan_state = self._plan_state(run)
        subquestions = [
            SubQuestion.model_validate(item) for item in plan_state["subquestions"]
        ]
        return ApprovalPayload(
            plan=plan_state,
            subquestions=subquestions,
            estimated_researcher_count=len(subquestions),
            risk_note=(
                "Review the plan before researcher execution to avoid spending "
                "search/LLM budget on the wrong direction."
            ),
        )

    async def _run_planner(
        self,
        run_id: str,
        request: CreateRunRequest,
        retry_count: int = 0,
    ) -> AgentRun:
        self._acquire_execution_lease(run_id)
        self.store.update_run(run_id, status="running", current_stage="planner")
        self._event(run_id, "planner", "started", {"query": request.query})
        started_at = time.perf_counter()
        self._step(
            run_id,
            "planner",
            "started",
            input_json={"request": request.model_dump(mode="json")},
            retry_count=retry_count,
        )
        orchestrator = DeepResearchOrchestrator(settings=self.settings)
        llm = orchestrator._build_llm_provider(request)
        cost = self._new_cost_tracker(llm)
        brief = await llm.create_brief(request, cost)
        self._raise_if_cancelled(run_id)
        max_researchers = min(request.max_researchers, self.settings.max_researchers)
        plan = await llm.plan(brief, max_researchers=max_researchers, cost=cost)
        self._raise_if_cancelled(run_id)
        cost_summary = cost.summary()
        plan_state = {
            "request": request.model_dump(mode="json"),
            "brief": brief.model_dump(mode="json"),
            "subquestions": [item.model_dump(mode="json") for item in plan],
            "cost": cost_summary.model_dump(mode="json"),
        }
        self._step(
            run_id,
            "planner",
            "succeeded",
            output_json={
                "brief": plan_state["brief"],
                "subquestions": plan_state["subquestions"],
            },
            latency_ms=_elapsed_ms(started_at),
            token_usage=cost_summary.total_tokens,
            cost=cost_summary.total_estimated_cost_usd,
            retry_count=retry_count,
        )
        self._event(
            run_id,
            "planner",
            "planner_done",
            {"subquestion_count": len(plan)},
        )
        self.store.update_run(
            run_id,
            plan_json=plan_state,
            total_tokens=cost_summary.total_tokens,
            total_cost=cost_summary.total_estimated_cost_usd,
            error_message=None,
        )
        if request.require_approval:
            self._event(
                run_id,
                "approval",
                "waiting_approval",
                self.approval_payload_data(plan_state),
            )
            waiting = self.store.update_run(
                run_id,
                status="waiting_approval",
                current_stage="approval",
            )
            return self._release_execution_lease(waiting.run_id)
        return await self._continue_from_plan(self.store.require_run(run_id), retry_count=retry_count)

    async def _continue_from_plan(
        self,
        run: AgentRun,
        retry_count: int = 0,
    ) -> AgentRun:
        run = self._acquire_execution_lease(run.run_id)
        if run.status == "cancelled":
            return run
        plan_state = self._plan_state(run)
        request = ResearchRequest.model_validate(plan_state["request"])
        brief = ResearchBrief.model_validate(plan_state["brief"])
        plan = [SubQuestion.model_validate(item) for item in plan_state["subquestions"]]
        planner_cost = CostSummary.model_validate(plan_state["cost"])
        try:
            return await self._execute_research_flow(
                run.run_id,
                request,
                brief,
                plan,
                planner_cost,
                retry_count=retry_count,
            )
        except RunCancelledError:
            return self._cancelled_run(run.run_id)
        except Exception as exc:  # noqa: BLE001
            return self._fail_run(run.run_id, "researcher", exc, retry_count=retry_count)

    async def _execute_research_flow(
        self,
        run_id: str,
        request: ResearchRequest,
        brief: ResearchBrief,
        plan: list[SubQuestion],
        planner_cost: CostSummary,
        retry_count: int,
    ) -> AgentRun:
        orchestrator = DeepResearchOrchestrator(settings=self.settings)
        llm = orchestrator._build_llm_provider(request)
        cost = self._new_cost_tracker(llm)
        trace = TraceLogger(
            run_id=run_id,
            trace_dir=self.settings.trace_dir,
            exporter=build_trace_exporter(self.settings),
        )
        search_service = build_search_service(self.settings, request.search_provider)
        max_researchers = min(request.max_researchers, self.settings.max_researchers)

        self._raise_if_cancelled(run_id)
        findings, raw_search_count, fallback_count, all_sources, sources = await self._run_researcher_stage(
            run_id,
            request,
            plan,
            orchestrator,
            search_service,
            trace,
            max_researchers,
            retry_count,
        )

        self._raise_if_cancelled(run_id)
        answer, claims, _synthesis_cost = await self._run_synthesizer_stage(
            run_id,
            brief,
            plan,
            findings,
            sources,
            llm,
            cost,
            retry_count,
        )

        self._raise_if_cancelled(run_id)
        citation_report = await self._run_verifier_stage(
            run_id,
            request,
            claims,
            sources,
            orchestrator,
            cost,
            retry_count,
        )

        total_cost = _merge_costs(planner_cost, cost.summary())
        latency_ms = sum(step.latency_ms or 0.0 for step in self.store.list_steps(run_id))
        metrics = {
            "latency_ms": round(latency_ms, 3),
            "raw_search_result_count": raw_search_count,
            "verified_source_count": len(all_sources),
            "deduped_source_count": len(sources),
            "fallback_count": fallback_count,
            "citation_retention_rate": citation_report.retention_rate,
            "success": citation_report.retention_rate >= 0.8 and len(sources) > 0,
            **source_diversity_metrics(sources),
        }
        report = StructuredReport(
            run_id=run_id,
            query=request.query,
            brief=brief,
            plan=plan,
            answer=answer,
            claims=claims,
            findings=findings,
            sources=sources,
            citation_check=citation_report,
            cost=total_cost,
            metrics=metrics,
            trace_events=trace.events,
        )
        self._event(run_id, "run", "succeeded", metrics)
        self.store.update_run(
            run_id,
            status="succeeded",
            current_stage="completed",
            result_json=report.model_dump(mode="json"),
            total_tokens=total_cost.total_tokens,
            total_cost=total_cost.total_estimated_cost_usd,
            error_message=None,
        )
        return self._release_execution_lease(run_id)

    async def _run_researcher_stage(
        self,
        run_id: str,
        request: ResearchRequest,
        plan: list[SubQuestion],
        orchestrator: DeepResearchOrchestrator,
        search_service,
        trace: TraceLogger,
        max_researchers: int,
        retry_count: int,
    ) -> tuple[list[Finding], int, int, list[Source], list[Source]]:
        self._heartbeat_execution_lease(run_id)
        self.store.update_run(run_id, status="running", current_stage="researcher")
        self._event(run_id, "researcher", "started", {"subquestion_count": len(plan)})
        started_at = time.perf_counter()
        self._step(
            run_id,
            "researcher",
            "started",
            input_json={"subquestions": [item.model_dump(mode="json") for item in plan]},
            retry_count=retry_count,
        )
        semaphore = asyncio.Semaphore(max_researchers)
        research_results = await asyncio.gather(
            *[
                orchestrator._research_one(
                    subquestion,
                    request,
                    search_service,
                    semaphore,
                    trace,
                    emit=None,
                )
                for subquestion in plan
            ]
        )
        research_results = await orchestrator._run_reflection_rounds(
            plan=plan,
            research_results=list(research_results),
            request=request,
            search_service=search_service,
            semaphore=semaphore,
            trace=trace,
            emit=None,
        )
        raw_search_count = sum(len(outcome.sources) for _, outcome in research_results)
        fallback_count = sum(1 for _, outcome in research_results if outcome.fallback_used)
        preliminary_findings = [finding for finding, _ in research_results]
        all_sources = [
            source for finding in preliminary_findings for source in finding.sources
        ]
        deduped_sources = orchestrator.deduper.dedup(all_sources)
        sources = orchestrator._assign_source_ids(deduped_sources)
        source_by_key = {orchestrator.deduper.key(source): source for source in sources}
        findings = orchestrator._remap_findings(preliminary_findings, source_by_key)
        self._step(
            run_id,
            "researcher",
            "succeeded",
            output_json={
                "raw_search_result_count": raw_search_count,
                "fallback_count": fallback_count,
                "deduped_source_count": len(sources),
            },
            latency_ms=_elapsed_ms(started_at),
            retry_count=retry_count,
        )
        self._event(
            run_id,
            "researcher",
            "succeeded",
            {"finding_count": len(findings), "source_count": len(sources)},
        )
        return findings, raw_search_count, fallback_count, all_sources, sources

    async def _run_synthesizer_stage(
        self,
        run_id: str,
        brief: ResearchBrief,
        plan: list[SubQuestion],
        findings: list[Finding],
        sources: list[Source],
        llm,
        cost: CostTracker,
        retry_count: int,
    ) -> tuple[str, list[str], CostSummary]:
        self._heartbeat_execution_lease(run_id)
        self.store.update_run(run_id, status="running", current_stage="synthesizer")
        self._event(run_id, "synthesizer", "started", {"source_count": len(sources)})
        started_at = time.perf_counter()
        self._step(
            run_id,
            "synthesizer",
            "started",
            input_json={"finding_count": len(findings), "source_count": len(sources)},
            retry_count=retry_count,
        )
        before_tokens = cost.summary().total_tokens
        before_cost = cost.summary().total_estimated_cost_usd
        answer, claims = await llm.synthesize(brief, plan, findings, sources, cost)
        summary = cost.summary()
        self._step(
            run_id,
            "synthesizer",
            "succeeded",
            output_json={"claim_count": len(claims), "answer_chars": len(answer)},
            latency_ms=_elapsed_ms(started_at),
            token_usage=summary.total_tokens - before_tokens,
            cost=round(summary.total_estimated_cost_usd - before_cost, 8),
            retry_count=retry_count,
        )
        self._event(run_id, "synthesizer", "succeeded", {"claim_count": len(claims)})
        return answer, claims, summary

    async def _run_verifier_stage(
        self,
        run_id: str,
        request: ResearchRequest,
        claims: list[str],
        sources: list[Source],
        orchestrator: DeepResearchOrchestrator,
        cost: CostTracker,
        retry_count: int,
    ):
        self._heartbeat_execution_lease(run_id)
        self.store.update_run(run_id, status="running", current_stage="verifier")
        self._event(run_id, "verifier", "started", {"claim_count": len(claims)})
        started_at = time.perf_counter()
        self._step(
            run_id,
            "verifier",
            "started",
            input_json={"claim_count": len(claims), "source_count": len(sources)},
            retry_count=retry_count,
        )
        citation_report = orchestrator.citation_checker.check(
            claims,
            sources,
            judge_provider=orchestrator._build_citation_judge_provider(request),
            cost=cost,
        )
        self._step(
            run_id,
            "verifier",
            "succeeded",
            output_json=citation_report.model_dump(mode="json"),
            latency_ms=_elapsed_ms(started_at),
            retry_count=retry_count,
        )
        self._event(
            run_id,
            "verifier",
            "succeeded",
            {"retention_rate": citation_report.retention_rate},
        )
        return citation_report

    def _new_cost_tracker(self, llm) -> CostTracker:
        return CostTracker(
            provider=llm.name,
            model=llm.model,
            input_cost_per_1m=self.settings.mock_input_cost_per_1m_tokens,
            output_cost_per_1m=self.settings.mock_output_cost_per_1m_tokens,
        )

    def _step(
        self,
        run_id: str,
        stage: str,
        status: str,
        **kwargs: Any,
    ):
        return self.store.add_step(
            step_id=f"{run_id}-{stage}-{status}-{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            stage=stage,
            status=status,
            **kwargs,
        )

    def _event(
        self,
        run_id: str,
        stage: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        return self.store.add_event(
            run_id=run_id,
            stage=stage,
            status=status,
            payload=payload or {},
        )

    def _plan_state(self, run: AgentRun) -> dict[str, Any]:
        if run.plan_json is None:
            raise ValueError(f"run has no planner checkpoint: {run.run_id}")
        return run.plan_json

    def _require_status(self, run_id: str, statuses: set[str]) -> AgentRun:
        run = self.store.require_run(run_id)
        if run.status not in statuses:
            expected = ", ".join(sorted(statuses))
            raise ValueError(f"run {run_id} status is {run.status}; expected {expected}")
        return run

    def _request_from_run(
        self,
        run: AgentRun,
        *,
        require_approval: bool | None = None,
    ) -> CreateRunRequest:
        payload = dict(run.request_json or {"query": run.query})
        if require_approval is not None:
            payload["require_approval"] = require_approval
        else:
            payload.setdefault("require_approval", run.require_approval)
        payload["defer_execution"] = False
        return CreateRunRequest.model_validate(payload)

    def _raise_if_cancelled(self, run_id: str) -> None:
        if self.store.require_run(run_id).status == "cancelled":
            raise RunCancelledError("run cancelled")

    def _acquire_execution_lease(self, run_id: str) -> AgentRun:
        run = self.store.acquire_lease(
            run_id,
            worker_id=self.worker_id,
            lease_seconds=self.settings.run_lease_seconds,
        )
        if run is None:
            if self.store.require_run(run_id).status == "cancelled":
                raise RunCancelledError("run cancelled")
            raise RuntimeError(f"run {run_id} lease is unavailable")
        return run

    def _heartbeat_execution_lease(self, run_id: str) -> AgentRun:
        run = self.store.heartbeat_lease(
            run_id,
            worker_id=self.worker_id,
            lease_seconds=self.settings.run_lease_seconds,
        )
        if run is None:
            if self.store.require_run(run_id).status == "cancelled":
                raise RunCancelledError("run cancelled")
            raise RuntimeError(f"run {run_id} lease heartbeat rejected")
        return run

    def _release_execution_lease(self, run_id: str) -> AgentRun:
        return self.store.release_lease(run_id, worker_id=self.worker_id)

    def _cancelled_run(self, run_id: str) -> AgentRun:
        run = self.store.require_run(run_id)
        if run.status != "cancelled":
            self._event(run_id, "run", "cancelled", {"previous_status": run.status})
            self.store.update_run(
                run_id,
                status="cancelled",
                current_stage="completed",
                error_message="run cancelled",
            )
        return self.store.clear_lease(run_id)

    def _fail_run(
        self,
        run_id: str,
        stage: str,
        exc: Exception,
        retry_count: int = 0,
    ) -> AgentRun:
        message = str(exc)
        self._step(
            run_id,
            stage,
            "failed",
            error=message,
            retry_count=retry_count,
        )
        self._event(run_id, stage, "failed", {"error": message})
        self.store.update_run(
            run_id,
            status="failed",
            current_stage=stage if stage in {"planner", "researcher", "synthesizer", "verifier"} else "completed",
            error_message=message,
        )
        return self.store.clear_lease(run_id)

    def approval_payload_data(self, plan_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "plan": plan_state,
            "subquestions": plan_state["subquestions"],
            "estimated_researcher_count": len(plan_state["subquestions"]),
            "risk_note": (
                "Review the plan before researcher execution to avoid spending "
                "search/LLM budget on the wrong direction."
            ),
        }


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def _merge_costs(first: CostSummary, second: CostSummary) -> CostSummary:
    records = [
        *[CostRecord.model_validate(item.model_dump(mode="json")) for item in first.records],
        *[CostRecord.model_validate(item.model_dump(mode="json")) for item in second.records],
    ]
    input_tokens = sum(record.input_tokens for record in records)
    output_tokens = sum(record.output_tokens for record in records)
    cost = sum(record.estimated_cost_usd for record in records)
    return CostSummary(
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        total_estimated_cost_usd=round(cost, 8),
        records=records,
    )
