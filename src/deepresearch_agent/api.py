from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from deepresearch_agent.orchestrator import DeepResearchOrchestrator
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


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepResearch Agent API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run("deepresearch_agent.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
