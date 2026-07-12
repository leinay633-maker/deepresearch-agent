from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from deepresearch_agent.api import _with_stream_compatibility_fields, app
from deepresearch_agent.orchestrator import DeepResearchOrchestrator


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint_returns_structured_report() -> None:
    client = TestClient(app)

    response = client.post(
        "/research",
        json={
            "query": "How should agent tools fail safely?",
            "search_provider": "mock",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["success"] is True
    assert payload["citation_check"]["retention_rate"] >= 0.8
    assert payload["sources"]


def test_stream_stage_payload_has_compatible_delivery_fields() -> None:
    event = _with_stream_compatibility_fields(
        {
            "event": "stage",
            "data": {
                "stage": "researcher.Q1",
                "status": "fallback",
                "payload": {"fallback_used": True},
            },
        }
    )

    payload = event["data"]["payload"]
    assert payload["attempt"] == 1
    assert payload["retryable"] is False
    assert payload["degraded"] is True


@pytest.mark.parametrize(
    ("error", "expected_retryable"),
    [
        (ValueError("unknown provider: missing"), False),
        (TimeoutError("search connection timed out"), True),
    ],
)
def test_research_stream_classifies_top_level_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_retryable: bool,
) -> None:
    async def fail_run(
        self: DeepResearchOrchestrator,
        request: object,
        emit: object | None = None,
    ) -> None:
        raise error

    monkeypatch.setattr(DeepResearchOrchestrator, "run", fail_run)
    client = TestClient(app)

    response = client.post(
        "/research/stream",
        json={"query": "Classify this failure", "search_provider": "mock"},
    )

    assert response.status_code == 200
    event_data = next(
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    )
    assert event_data["retryable"] is expected_retryable
