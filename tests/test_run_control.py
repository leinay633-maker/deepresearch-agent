from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from deepresearch_agent.api import app
from deepresearch_agent.run_store import RunStore


def test_create_run_persists_waiting_approval(tmp_path, monkeypatch) -> None:
    client, store = _client(tmp_path, monkeypatch)

    response = client.post("/runs", json=_run_body("How should planner approval work?"))

    assert response.status_code == 200
    payload = response.json()
    run_id = payload["run_id"]
    assert payload["status"] == "waiting_approval"
    assert payload["current_stage"] == "approval"
    assert store.get_run(run_id) is not None
    steps = store.list_steps(run_id)
    events = store.list_events(run_id)
    assert any(step.stage == "planner" and step.status == "succeeded" for step in steps)
    assert any(event.status == "planner_done" for event in events)
    assert any(event.status == "waiting_approval" for event in events)


def test_run_review_ui_and_run_list(tmp_path, monkeypatch) -> None:
    client, _store = _client(tmp_path, monkeypatch)
    created = client.post("/runs", json=_run_body("How should UI review work?"))

    ui = client.get("/ui")
    runs = client.get("/runs")

    assert created.status_code == 200
    assert ui.status_code == 200
    assert "DeepResearch Run Review" in ui.text
    assert "planEditor" in ui.text
    assert runs.status_code == 200
    assert runs.json()[0]["run_id"] == created.json()["run_id"]


def test_approve_continues_to_succeeded(tmp_path, monkeypatch) -> None:
    client, store = _client(tmp_path, monkeypatch)
    run_id = client.post("/runs", json=_run_body("How should approval resume?")).json()["run_id"]

    response = client.post(f"/runs/{run_id}/approve")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["current_stage"] == "completed"
    stages = {step.stage for step in store.list_steps(run_id)}
    assert {"planner", "researcher", "synthesizer", "verifier"} <= stages
    assert store.require_run(run_id).result_json is not None


def test_approve_run_can_execute_reflection_loop(tmp_path, monkeypatch) -> None:
    client, store = _client(tmp_path, monkeypatch)
    body = _run_body("How should reflection extend weak evidence?")
    body.update(
        {
            "reflection_enabled": True,
            "max_reflection_rounds": 1,
            "reflection_min_sources": 4,
        }
    )
    run_id = client.post("/runs", json=body).json()["run_id"]

    response = client.post(f"/runs/{run_id}/approve")

    assert response.status_code == 200
    result = store.require_run(run_id).result_json
    assert result is not None
    assert any(item["id"] == "R1" for item in result["plan"])
    assert any(event["stage"] == "reflection.round1" for event in result["trace_events"])


def test_edit_plan_saves_subquestions_and_continues(tmp_path, monkeypatch) -> None:
    client, store = _client(tmp_path, monkeypatch)
    run_id = client.post("/runs", json=_run_body("How should plan editing work?")).json()["run_id"]
    edited = {
        "subquestions": [
            {
                "id": "Q1",
                "question": "What checkpoint should be reused after edited approval?",
                "rationale": "Verify the edited plan is persisted before researcher execution.",
            }
        ]
    }

    response = client.post(f"/runs/{run_id}/edit", json=edited)

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    run = store.require_run(run_id)
    assert run.plan_json is not None
    assert run.plan_json["subquestions"][0]["question"] == edited["subquestions"][0]["question"]
    assert run.result_json["plan"][0]["question"] == edited["subquestions"][0]["question"]


def test_reject_cancels_without_researcher(tmp_path, monkeypatch) -> None:
    client, store = _client(tmp_path, monkeypatch)
    run_id = client.post("/runs", json=_run_body("How should reject work?")).json()["run_id"]

    response = client.post(f"/runs/{run_id}/reject", json={"reason": "wrong direction"})

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["error_message"] == "wrong direction"
    assert "researcher" not in {step.stage for step in store.list_steps(run_id)}


def test_cancel_prevents_later_approval(tmp_path, monkeypatch) -> None:
    client, store = _client(tmp_path, monkeypatch)
    run_id = client.post("/runs", json=_run_body("How should cancellation work?")).json()["run_id"]

    cancelled = client.post(f"/runs/{run_id}/cancel")
    approved = client.post(f"/runs/{run_id}/approve")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert approved.status_code == 409
    assert "researcher" not in {step.stage for step in store.list_steps(run_id)}


def test_retry_failed_run_reuses_planner_checkpoint(tmp_path, monkeypatch) -> None:
    client, store = _client(tmp_path, monkeypatch)
    run_id = client.post("/runs", json=_run_body("How should retry work?")).json()["run_id"]
    store.update_run(
        run_id,
        status="failed",
        current_stage="researcher",
        error_message="synthetic failure",
    )

    response = client.post(f"/runs/{run_id}/retry")

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    assert any(step.retry_count == 1 for step in store.list_steps(run_id))
    assert any(event.status == "retrying" for event in store.list_events(run_id))


def test_sse_replay_uses_last_event_id(tmp_path, monkeypatch) -> None:
    client, _store = _client(tmp_path, monkeypatch)
    run_id = client.post("/runs", json=_run_body("How should SSE replay work?")).json()["run_id"]

    first = client.get(f"/runs/{run_id}/events")
    first_ids = _sse_ids(first.text)
    replay = client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": str(first_ids[1])})
    replay_ids = _sse_ids(replay.text)

    assert first.status_code == 200
    assert first_ids == sorted(first_ids)
    assert replay.status_code == 200
    assert replay_ids
    assert min(replay_ids) > first_ids[1]


def _client(tmp_path: Path, monkeypatch) -> tuple[TestClient, RunStore]:
    db_path = tmp_path / "runs.sqlite"
    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("LOCAL_RETRIEVAL_MODE", "keyword")
    return TestClient(app), RunStore(db_path)


def _run_body(query: str) -> dict:
    return {
        "query": query,
        "search_provider": "mock",
        "llm_provider": "mock",
        "max_researchers": 1,
        "max_results_per_researcher": 1,
        "require_approval": True,
    }


def _sse_ids(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"^id: (\d+)$", text, flags=re.MULTILINE)]
