from __future__ import annotations

import time
import uuid
from typing import Any

from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.execution import ResearchExecutionEngine, is_retryable_error
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.report_metrics import build_execution_metrics
from deepresearch_agent.run_models import (
    AgentEvent,
    AgentRun,
    ApprovalPayload,
    CreateRunRequest,
    EditPlanRequest,
    RecoverStaleRunsRequest,
    RejectRunRequest,
    RunLeaseRequest,
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
        self._require_status(run_id, {"waiting_approval"})
        run = self.store.transition_run(
            run_id,
            expected_statuses={"waiting_approval"},
            require_unleased=True,
            status="queued",
            current_stage="researcher",
            error_message=None,
        )
        if run is None:
            current = self.store.require_run(run_id)
            raise ValueError(
                f"run {run_id} status is {current.status}; approval was already claimed"
            )
        self._event(run_id, "approval", "approved", {})
        return await self._continue_from_plan(run)

    async def edit(self, run_id: str, request: EditPlanRequest) -> AgentRun:
        run = self._require_status(run_id, {"waiting_approval"})
        plan_state = self._plan_state(run)
        plan_state["subquestions"] = [
            item.model_dump(mode="json") for item in request.subquestions
        ]
        claimed = self.store.transition_run(
            run_id,
            expected_statuses={"waiting_approval"},
            require_unleased=True,
            status="queued",
            current_stage="researcher",
            plan_json=plan_state,
            error_message=None,
        )
        if claimed is None:
            current = self.store.require_run(run_id)
            raise ValueError(
                f"run {run_id} status is {current.status}; edit was already claimed"
            )
        self._event(
            run_id,
            "approval",
            "edited",
            {"subquestion_count": len(request.subquestions)},
        )
        return await self._continue_from_plan(claimed)

    def reject(self, run_id: str, request: RejectRunRequest) -> AgentRun:
        self._require_status(run_id, {"waiting_approval"})
        cancelled = self.store.transition_run(
            run_id,
            expected_statuses={"waiting_approval"},
            require_unleased=True,
            status="cancelled",
            current_stage="completed",
            error_message=request.reason,
            leased_by=None,
            heartbeat_at=None,
            lease_expires_at=None,
        )
        if cancelled is None:
            current = self.store.require_run(run_id)
            raise ValueError(
                f"run {run_id} status is {current.status}; rejection lost a concurrent transition"
            )
        self._event(run_id, "approval", "rejected", {"reason": request.reason})
        return cancelled

    def cancel(self, run_id: str) -> AgentRun:
        run = self.store.require_run(run_id)
        if run.status in TERMINAL_STATUSES:
            return run
        cancelled = self.store.transition_run(
            run_id,
            expected_statuses={run.status},
            status="cancelled",
            current_stage="completed",
            error_message="run cancelled",
            leased_by=None,
            heartbeat_at=None,
            lease_expires_at=None,
        )
        if cancelled is None:
            current = self.store.require_run(run_id)
            if current.status in TERMINAL_STATUSES:
                return current
            raise ValueError(f"run {run_id} changed while cancellation was being applied")
        self._event(run_id, "run", "cancelled", {"previous_status": run.status})
        return cancelled

    async def retry(self, run_id: str) -> AgentRun:
        run = self._require_status(run_id, {"failed"})
        retry_count = self._next_retry_count(run_id)
        claimed = self.store.transition_run(
            run_id,
            expected_statuses={"failed"},
            require_unleased=True,
            status="queued",
            error_message=None,
        )
        if claimed is None:
            current = self.store.require_run(run_id)
            raise ValueError(
                f"run {run_id} status is {current.status}; retry was already claimed"
            )
        self._event(run_id, "run", "retrying", {"retry_count": retry_count})
        run = claimed
        if run.plan_json is not None:
            return await self._continue_from_plan(run, retry_count=retry_count)
        request = self._request_from_run(run, require_approval=False)
        try:
            return await self._run_planner(run_id, request, retry_count=retry_count)
        except RunCancelledError:
            return self._cancelled_run(run_id)
        except Exception as exc:  # noqa: BLE001
            return self._fail_run(run_id, "planner", exc, retry_count=retry_count)

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
            if run.leased_by is None or run.lease_expires_at is None:
                continue
            recovered_run = self.store.recover_stale_run(
                run.run_id,
                expected_worker_id=run.leased_by,
                expected_lease_expires_at=run.lease_expires_at,
                reason=request.reason,
            )
            if recovered_run is None:
                continue
            self._step(run.run_id, run.current_stage, "failed", error=request.reason)
            self._event(
                run.run_id,
                "lease",
                "stale_recovered",
                {"reason": request.reason, "leased_by": run.leased_by},
            )
            recovered.append(recovered_run)
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
        running = self.store.transition_run(
            run_id,
            expected_statuses={"queued", "running"},
            expected_worker_id=self.worker_id,
            status="running",
            current_stage="planner",
        )
        if running is None:
            self._check_execution_active(run_id)
            raise RuntimeError("planner transition rejected because execution ownership was lost")
        self._event(
            run_id,
            "planner",
            "started",
            {"query": request.query, "retry_count": retry_count},
        )
        started_at = time.perf_counter()
        self._step(
            run_id,
            "planner",
            "started",
            input_json={"request": request.model_dump(mode="json")},
            retry_count=retry_count,
        )
        orchestrator = DeepResearchOrchestrator(settings=self.settings)
        engine = ResearchExecutionEngine(orchestrator)
        llm = orchestrator._build_llm_provider(request)
        cost = self._new_cost_tracker(llm)
        brief = await engine.run_clarify_stage(request=request, llm=llm, cost=cost)
        self._raise_if_cancelled(run_id)
        max_researchers = min(request.max_researchers, self.settings.max_researchers)
        plan = await engine.run_planner_stage(
            brief=brief,
            max_researchers=max_researchers,
            llm=llm,
            cost=cost,
        )
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
            {"subquestion_count": len(plan), "retry_count": retry_count},
        )
        saved = self.store.transition_run(
            run_id,
            expected_statuses={"running"},
            expected_worker_id=self.worker_id,
            plan_json=plan_state,
            total_tokens=cost_summary.total_tokens,
            total_cost=cost_summary.total_estimated_cost_usd,
            error_message=None,
        )
        if saved is None:
            self._check_execution_active(run_id)
            raise RuntimeError("planner checkpoint rejected because execution ownership was lost")
        if request.require_approval:
            waiting = self.store.transition_run(
                run_id,
                expected_statuses={"running"},
                expected_worker_id=self.worker_id,
                status="waiting_approval",
                current_stage="approval",
            )
            if waiting is None:
                self._check_execution_active(run_id)
                raise RuntimeError("approval transition rejected because execution ownership was lost")
            self._event(
                run_id,
                "approval",
                "waiting_approval",
                self.approval_payload_data(plan_state),
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
            current_stage = self.store.require_run(run.run_id).current_stage
            failure_stage = (
                current_stage
                if current_stage in {"researcher", "synthesizer", "verifier"}
                else "researcher"
            )
            return self._fail_run(
                run.run_id,
                failure_stage,
                exc,
                retry_count=retry_count,
            )

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
        search_service = build_search_service(
            self.settings,
            request.search_provider,
            orchestrator._fallback_policy(request),
        )
        max_researchers = min(request.max_researchers, self.settings.max_researchers)

        researcher_checkpoint = (
            self._load_researcher_checkpoint(run_id) if retry_count > 0 else None
        )
        if researcher_checkpoint is None:
            self._raise_if_cancelled(run_id)
            researcher_output = await self._run_researcher_stage(
                run_id,
                request,
                plan,
                orchestrator,
                search_service,
                trace,
                max_researchers,
                llm,
                cost,
                retry_count,
            )
            if len(researcher_output) == 5:
                # Backward compatibility for custom stage implementations written
                # before degraded_count became a first-class metric.
                (
                    findings,
                    raw_search_count,
                    fallback_count,
                    all_sources,
                    sources,
                ) = researcher_output
                degraded_count = 0
            else:
                (
                    findings,
                    raw_search_count,
                    fallback_count,
                    degraded_count,
                    all_sources,
                    sources,
                ) = researcher_output
        else:
            self._heartbeat_execution_lease(run_id)
            self._set_owned_stage(run_id, "researcher")
            self._restore_researcher_cost(run_id, cost)
            (
                findings,
                raw_search_count,
                fallback_count,
                degraded_count,
                all_sources,
                sources,
            ) = researcher_checkpoint
            self._event(
                run_id,
                "researcher",
                "checkpoint_reused",
                {
                    "finding_count": len(findings),
                    "source_count": len(sources),
                    "retry_count": retry_count,
                },
            )

        self._raise_if_cancelled(run_id)
        self._set_owned_stage(run_id, "synthesizer")
        answer, claims, _synthesis_cost = await self._run_synthesizer_stage(
            run_id,
            brief,
            plan,
            findings,
            sources,
            orchestrator,
            llm,
            cost,
            retry_count,
        )

        self._raise_if_cancelled(run_id)
        self._set_owned_stage(run_id, "verifier")
        citation_report = await self._run_verifier_stage(
            run_id,
            request,
            claims,
            sources,
            orchestrator,
            cost,
            retry_count,
        )
        self._check_execution_active(run_id)

        total_cost = _merge_costs(planner_cost, cost.summary())
        latency_ms = sum(step.latency_ms or 0.0 for step in self.store.list_steps(run_id))
        metrics = build_execution_metrics(
            latency_ms=latency_ms,
            raw_search_result_count=raw_search_count,
            verified_source_count=len(all_sources),
            deduped_source_count=len(sources),
            fallback_count=fallback_count,
            degraded_count=degraded_count,
            sources=sources,
            citation_report=citation_report,
        )
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
        completed = self.store.complete_run_if_owned(
            run_id,
            worker_id=self.worker_id,
            result_json=report.model_dump(mode="json"),
            total_tokens=total_cost.total_tokens,
            total_cost=total_cost.total_estimated_cost_usd,
        )
        if completed is None:
            current = self.store.require_run(run_id)
            if current.status == "cancelled":
                raise RunCancelledError("run cancelled before final commit")
            raise RuntimeError("run final commit rejected because execution ownership was lost")
        self._event(
            run_id,
            "run",
            "succeeded",
            {**metrics, "retry_count": retry_count},
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
        llm,
        cost: CostTracker,
        retry_count: int,
    ) -> tuple[list[Finding], int, int, int, list[Source], list[Source]]:
        self._heartbeat_execution_lease(run_id)
        self._set_owned_stage(run_id, "researcher")
        self._event(
            run_id,
            "researcher",
            "started",
            {"subquestion_count": len(plan), "retry_count": retry_count},
        )
        started_at = time.perf_counter()
        self._step(
            run_id,
            "researcher",
            "started",
            input_json={"subquestions": [item.model_dump(mode="json") for item in plan]},
            retry_count=retry_count,
        )
        before_summary = cost.summary()
        before_record_count = len(before_summary.records)
        research = await ResearchExecutionEngine(orchestrator).run_research_stage(
            plan=plan,
            request=request,
            search_service=search_service,
            trace=trace,
            llm=llm,
            cost=cost,
            max_researchers=max_researchers,
            emit=None,
            cancel_check=lambda: self._check_execution_active(run_id),
        )
        plan_state = self._plan_state(self.store.require_run(run_id))
        plan_state["subquestions"] = [item.model_dump(mode="json") for item in plan]
        updated_plan = self.store.transition_run(
            run_id,
            expected_statuses={"running"},
            expected_worker_id=self.worker_id,
            plan_json=plan_state,
        )
        if updated_plan is None:
            self._check_execution_active(run_id)
            raise RuntimeError("researcher plan checkpoint rejected because ownership was lost")
        findings = research.findings
        raw_search_count = research.raw_search_count
        fallback_count = research.fallback_count
        degraded_count = research.degraded_count
        all_sources = research.all_sources
        sources = research.sources
        after_summary = cost.summary()
        researcher_cost_records = after_summary.records[before_record_count:]
        researcher_tokens = after_summary.total_tokens - before_summary.total_tokens
        researcher_cost = round(
            after_summary.total_estimated_cost_usd
            - before_summary.total_estimated_cost_usd,
            8,
        )
        self._step(
            run_id,
            "researcher",
            "succeeded",
            output_json={
                "raw_search_result_count": raw_search_count,
                "fallback_count": fallback_count,
                "deduped_source_count": len(sources),
                "checkpoint": {
                    "findings": [finding.model_dump(mode="json") for finding in findings],
                    "all_sources": [source.model_dump(mode="json") for source in all_sources],
                    "sources": [source.model_dump(mode="json") for source in sources],
                    "raw_search_result_count": raw_search_count,
                    "fallback_count": fallback_count,
                    "degraded_count": degraded_count,
                    "plan": [item.model_dump(mode="json") for item in plan],
                    "cost_records": [
                        record.model_dump(mode="json")
                        for record in researcher_cost_records
                    ],
                },
            },
            latency_ms=_elapsed_ms(started_at),
            token_usage=researcher_tokens,
            cost=researcher_cost,
            retry_count=retry_count,
        )
        self._event(
            run_id,
            "researcher",
            "succeeded",
            {
                "finding_count": len(findings),
                "source_count": len(sources),
                "fallback_count": fallback_count,
                "degraded_count": degraded_count,
                "retry_count": retry_count,
            },
        )
        return (
            findings,
            raw_search_count,
            fallback_count,
            degraded_count,
            all_sources,
            sources,
        )

    async def _run_synthesizer_stage(
        self,
        run_id: str,
        brief: ResearchBrief,
        plan: list[SubQuestion],
        findings: list[Finding],
        sources: list[Source],
        orchestrator: DeepResearchOrchestrator,
        llm,
        cost: CostTracker,
        retry_count: int,
    ) -> tuple[str, list[str], CostSummary]:
        self._heartbeat_execution_lease(run_id)
        self._set_owned_stage(run_id, "synthesizer")
        self._event(
            run_id,
            "synthesizer",
            "started",
            {"source_count": len(sources), "retry_count": retry_count},
        )
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
        answer, claims = await ResearchExecutionEngine(orchestrator).run_synthesizer_stage(
            brief=brief,
            plan=plan,
            findings=findings,
            sources=sources,
            llm=llm,
            cost=cost,
        )
        self._check_execution_active(run_id)
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
        self._event(
            run_id,
            "synthesizer",
            "succeeded",
            {"claim_count": len(claims), "retry_count": retry_count},
        )
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
        self._set_owned_stage(run_id, "verifier")
        self._event(
            run_id,
            "verifier",
            "started",
            {"claim_count": len(claims), "retry_count": retry_count},
        )
        started_at = time.perf_counter()
        self._step(
            run_id,
            "verifier",
            "started",
            input_json={"claim_count": len(claims), "source_count": len(sources)},
            retry_count=retry_count,
        )
        before_summary = cost.summary()
        citation_report = ResearchExecutionEngine(orchestrator).run_verifier_stage(
            request=request,
            claims=claims,
            sources=sources,
            cost=cost,
        )
        self._check_execution_active(run_id)
        after_summary = cost.summary()
        self._step(
            run_id,
            "verifier",
            "succeeded",
            output_json=citation_report.model_dump(mode="json"),
            latency_ms=_elapsed_ms(started_at),
            token_usage=after_summary.total_tokens - before_summary.total_tokens,
            cost=round(
                after_summary.total_estimated_cost_usd
                - before_summary.total_estimated_cost_usd,
                8,
            ),
            retry_count=retry_count,
        )
        self._event(
            run_id,
            "verifier",
            "succeeded",
            {
                "retention_rate": citation_report.retention_rate,
                "retry_count": retry_count,
            },
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
        event_payload = dict(payload or {})
        retry_count = event_payload.get("retry_count", 0)
        if not isinstance(retry_count, int) or retry_count < 0:
            retry_count = 0
        event_payload.setdefault("attempt", retry_count + 1)
        event_payload.setdefault(
            "retryable",
            status == "failed"
            and stage in {"planner", "researcher", "synthesizer", "verifier"}
            and is_retryable_error(event_payload.get("error")),
        )
        event_payload.setdefault(
            "degraded",
            status == "fallback"
            or bool(event_payload.get("fallback_used"))
            or bool(event_payload.get("fallback_count", 0))
            or bool(event_payload.get("degraded_count", 0)),
        )
        return self.store.add_event(
            run_id=run_id,
            stage=stage,
            status=status,
            payload=event_payload,
        )

    def _next_retry_count(self, run_id: str) -> int:
        step_counts = [step.retry_count for step in self.store.list_steps(run_id)]
        event_counts = [
            int(event.payload.get("retry_count", 0))
            for event in self.store.list_events(run_id)
            if isinstance(event.payload.get("retry_count", 0), int)
        ]
        return max([0, *step_counts, *event_counts]) + 1

    def _plan_state(self, run: AgentRun) -> dict[str, Any]:
        if run.plan_json is None:
            raise ValueError(f"run has no planner checkpoint: {run.run_id}")
        return run.plan_json

    def _load_researcher_checkpoint(
        self,
        run_id: str,
    ) -> tuple[list[Finding], int, int, int, list[Source], list[Source]] | None:
        for step in reversed(self.store.list_steps(run_id)):
            if step.stage != "researcher" or step.status != "succeeded":
                continue
            output = step.output_json or {}
            checkpoint = output.get("checkpoint")
            if not isinstance(checkpoint, dict):
                return None
            try:
                findings = [
                    Finding.model_validate(item)
                    for item in checkpoint.get("findings", [])
                ]
                all_sources = [
                    Source.model_validate(item)
                    for item in checkpoint.get("all_sources", [])
                ]
                sources = [
                    Source.model_validate(item)
                    for item in checkpoint.get("sources", [])
                ]
                raw_search_count = int(checkpoint.get("raw_search_result_count", 0))
                fallback_count = int(checkpoint.get("fallback_count", 0))
                degraded_count = int(checkpoint.get("degraded_count", 0))
            except Exception:  # noqa: BLE001
                return None
            if not findings or "sources" not in checkpoint:
                return None
            return (
                findings,
                raw_search_count,
                fallback_count,
                degraded_count,
                all_sources,
                sources,
            )
        return None

    def _restore_researcher_cost(self, run_id: str, cost: CostTracker) -> None:
        for step in reversed(self.store.list_steps(run_id)):
            if step.stage != "researcher" or step.status != "succeeded":
                continue
            checkpoint = (step.output_json or {}).get("checkpoint") or {}
            for raw_record in checkpoint.get("cost_records", []):
                record = CostRecord.model_validate(raw_record)
                cost.add_usage(
                    stage=record.stage,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    estimated_cost_usd=record.estimated_cost_usd,
                    provider=record.provider,
                    model=record.model,
                )
            return

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

    def _check_execution_active(self, run_id: str) -> None:
        self._raise_if_cancelled(run_id)
        self._heartbeat_execution_lease(run_id)

    def _set_owned_stage(self, run_id: str, stage: str) -> AgentRun:
        updated = self.store.transition_run(
            run_id,
            expected_statuses={"running", "queued"},
            expected_worker_id=self.worker_id,
            status="running",
            current_stage=stage,
        )
        if updated is not None:
            return updated
        current = self.store.require_run(run_id)
        if current.status == "cancelled":
            raise RunCancelledError("run cancelled")
        raise RuntimeError(
            f"run {run_id} stage transition to {stage} rejected because ownership was lost"
        )

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
        if run.status == "cancelled":
            return run
        if run.status in TERMINAL_STATUSES:
            return run
        cancelled = self.store.transition_run(
            run_id,
            expected_statuses={run.status},
            expected_worker_id=self.worker_id if run.leased_by == self.worker_id else None,
            require_unleased=run.leased_by is None,
            status="cancelled",
            current_stage="completed",
            error_message="run cancelled",
            leased_by=None,
            heartbeat_at=None,
            lease_expires_at=None,
        )
        if cancelled is None:
            return self.store.require_run(run_id)
        self._event(run_id, "run", "cancelled", {"previous_status": run.status})
        return cancelled

    def _fail_run(
        self,
        run_id: str,
        stage: str,
        exc: Exception,
        retry_count: int = 0,
    ) -> AgentRun:
        current = self.store.require_run(run_id)
        if current.status in TERMINAL_STATUSES:
            return current
        if current.leased_by != self.worker_id:
            return current
        message = str(exc)
        failed = self.store.transition_run(
            run_id,
            expected_statuses={"queued", "running"},
            expected_worker_id=self.worker_id,
            status="failed",
            current_stage=(
                stage
                if stage in {"planner", "researcher", "synthesizer", "verifier"}
                else "completed"
            ),
            error_message=message,
            leased_by=None,
            heartbeat_at=None,
            lease_expires_at=None,
        )
        if failed is None:
            return self.store.require_run(run_id)
        self._step(
            run_id,
            stage,
            "failed",
            error=message,
            retry_count=retry_count,
        )
        self._event(
            run_id,
            stage,
            "failed",
            {"error": message, "retry_count": retry_count},
        )
        return failed

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
