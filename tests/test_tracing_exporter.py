from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from deepresearch_agent.config import Settings
from deepresearch_agent.tracing import (
    OtlpHttpTraceExporter,
    TraceLogger,
    build_trace_exporter,
)


class _CaptureHandler(BaseHTTPRequestHandler):
    payloads: list[dict] = []
    paths: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.__class__.paths.append(self.path)
        self.__class__.payloads.append(json.loads(body.decode("utf-8")))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


class _BrokenExporter:
    def export(self, event) -> None:
        raise RuntimeError(f"cannot export {event.stage}")


def test_trace_logger_exports_otlp_http_event(tmp_path: Path) -> None:
    _CaptureHandler.payloads = []
    _CaptureHandler.paths = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        exporter = OtlpHttpTraceExporter(
            endpoint=f"http://127.0.0.1:{server.server_port}",
            service_name="test-deepresearch-agent",
            timeout_seconds=1.0,
        )
        trace = TraceLogger("run-otel", trace_dir=tmp_path, exporter=exporter)

        trace.record("planner", "success", {"subquestion_count": 3})
    finally:
        server.shutdown()
        server.server_close()

    assert _CaptureHandler.paths == ["/v1/traces"]
    payload = _CaptureHandler.payloads[0]
    span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    resource_attrs = payload["resourceSpans"][0]["resource"]["attributes"]
    span_attrs = span["attributes"]
    assert span["name"] == "planner"
    assert {"key": "service.name", "value": {"stringValue": "test-deepresearch-agent"}} in resource_attrs
    assert {"key": "deepresearch.status", "value": {"stringValue": "success"}} in span_attrs
    assert "subquestion_count" in span_attrs[1]["value"]["stringValue"]


def test_trace_logger_keeps_jsonl_when_exporter_fails(tmp_path: Path) -> None:
    trace = TraceLogger("run-broken", trace_dir=tmp_path, exporter=_BrokenExporter())

    event = trace.record("run", "start", {"query": "trace"})

    assert event.stage == "run"
    assert [item.stage for item in trace.events] == ["run", "trace_exporter"]
    assert trace.events[-1].status == "error"
    assert "cannot export run" in trace.events[-1].payload["error"]
    lines = (tmp_path / "research-run-broken.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_build_trace_exporter_uses_otlp_settings() -> None:
    exporter = build_trace_exporter(
        Settings(
            trace_exporter="otlp_http",
            otel_exporter_otlp_endpoint="http://collector:4318",
            otel_exporter_otlp_headers="X-Test=ok",
            otel_service_name="configured-service",
        )
    )

    assert isinstance(exporter, OtlpHttpTraceExporter)
    assert exporter.endpoint == "http://collector:4318"
    assert exporter.headers == {"X-Test": "ok"}
    assert exporter.service_name == "configured-service"
