from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import uvicorn
from fastapi import Header, HTTPException
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.run_control import TERMINAL_STATUSES, RunController
from deepresearch_agent.run_models import (
    AgentRun,
    AgentStep,
    CreateRunRequest,
    EditPlanRequest,
    RejectRunRequest,
    RunTrace,
)
from deepresearch_agent.schemas import ResearchRequest, StructuredReport

app = FastAPI(title="DeepResearch Agent", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=StructuredReport)
async def research(request: ResearchRequest) -> StructuredReport:
    orchestrator = DeepResearchOrchestrator()
    return await orchestrator.run(request)


@app.post("/research/stream")
async def research_stream(request: ResearchRequest) -> StreamingResponse:
    async def generate():
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def emit(event: dict[str, Any]) -> None:
            await queue.put(event)

        async def run() -> None:
            try:
                report = await DeepResearchOrchestrator().run(request, emit=emit)
                await queue.put({"event": "final", "data": report.model_dump(mode="json")})
            except Exception as exc:
                await queue.put({"event": "error", "data": {"message": str(exc)}})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse(item["event"], item["data"])
        finally:
            await task

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/runs", response_model=AgentRun)
async def create_run(request: CreateRunRequest) -> AgentRun:
    return await RunController().create_run(request)


@app.get("/runs/{run_id}", response_model=AgentRun)
async def get_run(run_id: str) -> AgentRun:
    return _get_controller_run(run_id)


@app.get("/runs/{run_id}/steps", response_model=list[AgentStep])
async def get_run_steps(run_id: str) -> list[AgentStep]:
    return _controller().steps(run_id)


@app.get("/runs/{run_id}/trace", response_model=RunTrace)
async def get_run_trace(run_id: str) -> RunTrace:
    return _controller().trace(run_id)


@app.post("/runs/{run_id}/approve", response_model=AgentRun)
async def approve_run(run_id: str) -> AgentRun:
    return await _run_action(RunController().approve(run_id))


@app.post("/runs/{run_id}/edit", response_model=AgentRun)
async def edit_run(run_id: str, request: EditPlanRequest) -> AgentRun:
    return await _run_action(RunController().edit(run_id, request))


@app.post("/runs/{run_id}/reject", response_model=AgentRun)
async def reject_run(run_id: str, request: RejectRunRequest) -> AgentRun:
    return _sync_run_action(lambda: RunController().reject(run_id, request))


@app.post("/runs/{run_id}/cancel", response_model=AgentRun)
async def cancel_run(run_id: str) -> AgentRun:
    return _sync_run_action(lambda: RunController().cancel(run_id))


@app.post("/runs/{run_id}/retry", response_model=AgentRun)
async def retry_run(run_id: str) -> AgentRun:
    return await _run_action(RunController().retry(run_id))


@app.get("/runs/{run_id}/events")
async def run_events(
    run_id: str,
    last_event_id: int | None = None,
    last_event_id_header: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    after_event_id = last_event_id
    if after_event_id is None and last_event_id_header:
        try:
            after_event_id = int(last_event_id_header)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid Last-Event-ID") from exc

    async def generate():
        controller = RunController()
        seen_event_id = after_event_id
        while True:
            events = controller.events(run_id, after_event_id=seen_event_id)
            for event in events:
                seen_event_id = event.event_id
                yield _run_sse(event.model_dump(mode="json"))
            run = controller.get_run(run_id)
            if run.status in TERMINAL_STATUSES or run.status == "waiting_approval":
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_sse(event: dict[str, Any]) -> str:
    event_name = f"{event['stage']}.{event['status']}"
    return (
        f"id: {event['event_id']}\n"
        f"event: {event_name}\n"
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    )


def _controller() -> RunController:
    return RunController()


def _get_controller_run(run_id: str) -> AgentRun:
    try:
        return _controller().get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _run_action(coro) -> AgentRun:
    try:
        return await coro
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _sync_run_action(fn) -> AgentRun:
    try:
        return fn()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepResearch Agent API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run("deepresearch_agent.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
