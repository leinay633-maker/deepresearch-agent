from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Protocol

from deepresearch_agent.config import Settings
from deepresearch_agent.schemas import TraceEvent


class TraceExporter(Protocol):
    def export(self, event: TraceEvent) -> None:
        """Export one trace event to an external sink."""


@dataclass(frozen=True)
class OtlpHttpTraceExporter:
    endpoint: str
    service_name: str = "deepresearch-agent"
    headers: dict[str, str] | None = None
    timeout_seconds: float = 2.0

    def export(self, event: TraceEvent) -> None:
        payload = json.dumps(self._payload(event), ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **(self.headers or {}),
        }
        request = urllib.request.Request(
            _normalize_otlp_trace_endpoint(self.endpoint),
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OTLP trace export failed: {exc}") from exc

    def _payload(self, event: TraceEvent) -> dict[str, Any]:
        end_ns = _unix_nanos(event)
        duration_ns = int((event.duration_ms or 0.0) * 1_000_000)
        start_ns = max(0, end_ns - duration_ns)
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            _string_attr("service.name", self.service_name),
                            _string_attr("deepresearch.run_id", event.run_id),
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "deepresearch_agent.tracing"},
                            "spans": [
                                {
                                    "traceId": _hex_digest(event.run_id, 32),
                                    "spanId": _hex_digest(
                                        f"{event.run_id}:{event.stage}:{event.timestamp.isoformat()}",
                                        16,
                                    ),
                                    "name": event.stage,
                                    "kind": "SPAN_KIND_INTERNAL",
                                    "startTimeUnixNano": str(start_ns),
                                    "endTimeUnixNano": str(end_ns),
                                    "attributes": [
                                        _string_attr("deepresearch.status", event.status),
                                        _string_attr(
                                            "deepresearch.payload",
                                            json.dumps(
                                                event.payload,
                                                ensure_ascii=False,
                                                sort_keys=True,
                                            ),
                                        ),
                                    ],
                                    "status": _span_status(event.status),
                                }
                            ],
                        }
                    ],
                }
            ]
        }


class TraceLogger:
    def __init__(
        self,
        run_id: str,
        trace_dir: str = "logs",
        exporter: TraceExporter | None = None,
        write_enabled: bool = True,
    ) -> None:
        self.run_id = run_id
        self.events: list[TraceEvent] = []
        self.write_enabled = write_enabled
        self.trace_dir = Path(trace_dir)
        self.path = self.trace_dir / f"research-{run_id}.jsonl"
        if self.write_enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.exporter = exporter

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
        self._write_event(event)
        self._export_event(event)
        return event

    def _write_event(self, event: TraceEvent) -> None:
        if not self.write_enabled:
            return
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def _export_event(self, event: TraceEvent) -> None:
        if self.exporter is None:
            return
        try:
            self.exporter.export(event)
        except Exception as exc:  # noqa: BLE001
            error_event = TraceEvent(
                run_id=self.run_id,
                stage="trace_exporter",
                status="error",
                payload={
                    "exporter": self.exporter.__class__.__name__,
                    "error": str(exc),
                    "dropped_stage": event.stage,
                },
            )
            self.events.append(error_event)
            self._write_event(error_event)


def build_trace_exporter(settings: Settings) -> TraceExporter | None:
    exporter = settings.trace_exporter.strip().lower()
    if exporter in {"", "jsonl", "none"}:
        return None
    if exporter == "otlp_http":
        if not settings.otel_exporter_otlp_endpoint:
            return None
        return OtlpHttpTraceExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            service_name=settings.otel_service_name,
            headers=_parse_otlp_headers(settings.otel_exporter_otlp_headers),
            timeout_seconds=settings.otel_exporter_otlp_timeout_seconds,
        )
    raise ValueError(f"unknown trace exporter: {settings.trace_exporter}")


def _parse_otlp_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            headers[key] = value.strip()
    return headers


def _normalize_otlp_trace_endpoint(endpoint: str) -> str:
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/v1/traces"):
        return trimmed
    return f"{trimmed}/v1/traces"


def _unix_nanos(event: TraceEvent) -> int:
    timestamp = event.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return int(timestamp.timestamp() * 1_000_000_000)


def _hex_digest(value: str, length: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _string_attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _span_status(status: str) -> dict[str, str]:
    if status in {"error", "fallback"}:
        return {"code": "STATUS_CODE_ERROR", "message": status}
    return {"code": "STATUS_CODE_OK"}
