from __future__ import annotations

from fastapi.testclient import TestClient

from deepresearch_agent.api import _with_stream_compatibility_fields, app


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
