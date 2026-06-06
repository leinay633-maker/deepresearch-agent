from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from deepresearch_agent.schemas import TraceEvent


class TraceLogger:
    def __init__(self, run_id: str, trace_dir: str = "logs") -> None:
        self.run_id = run_id
        self.events: list[TraceEvent] = []
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"research-{run_id}.jsonl"

    def now(self) -> float:
        return time.perf_counter()

    def record(
        self,
        stage: str,
        status: str,
        payload: dict[str, Any] | None = None,
        start: float | None = None,
    ) -> TraceEvent:
        duration_ms = None
        if start is not None:
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
        event = TraceEvent(
            run_id=self.run_id,
            stage=stage,
            status=status,  # type: ignore[arg-type]
            duration_ms=duration_ms,
            payload=payload or {},
        )
        self.events.append(event)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return event
