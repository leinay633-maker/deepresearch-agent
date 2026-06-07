from __future__ import annotations

import asyncio
import re
import sqlite3
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from deepresearch_agent.config import load_settings
from deepresearch_agent.api import app
from deepresearch_agent.run_control import RunController
from deepresearch_agent.run_models import CreateRunRequest
from deepresearch_agent.run_store import RunStore
from deepresearch_agent.schemas import utc_now


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


def test_deferred_run_waits_for_worker_next(tmp_path, monkeypatch) -> None:
    client, store = _client(tmp_path, monkeypatch)
    body = _run_body("How should deferred worker execution work?")
    body.update({"require_approval": False, "defer_execution": True})

    created = client.post("/runs", json=body)
    run_id = created.json()["run_id"]
    queued = store.require_run(run_id)

    assert created.status_code == 200
    assert created.json()["status"] == "queued"
    assert created.json()["current_stage"] == "planner"
    assert queued.request_json is not None
    assert queued.request_json["defer_execution"] is True
    assert queued.request_json["max_researchers"] == 1
    assert store.list_steps(run_id) == []

    processed = client.post("/runs/worker/next")

    assert processed.status_code == 200
    assert processed.json()["run_id"] == run_id
    assert processed.json()["status"] == "succeeded"
    assert processed.json()["leased_by"] is None
    result = store.require_run(run_id).result_json
    assert result is not None
    assert result["metrics"]["source_provider_count"] >= 1
    assert result["metrics"]["source_domain_count"] >= 1
    assert {"planner", "researcher", "synthesizer", "verifier"} <= {
        step.stage for step in store.list_steps(run_id)
    }
    assert any(event.stage == "worker" and event.status == "claimed" for event in store.list_events(run_id))


def test_worker_next_returns_null_when_no_queued_run(tmp_path, monkeypatch) -> None:
    client, _store = _client(tmp_path, monkeypatch)

    response = client.post("/runs/worker/next")

    assert response.status_code == 200
    assert response.json() is None


def test_run_store_lease_heartbeat_and_release(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.sqlite")
    store.create_run(
        run_id="lease-run",
        query="How should worker leases behave?",
        require_approval=True,
    )

    acquired = store.acquire_lease(
        "lease-run",
        worker_id="worker-a",
        lease_seconds=30,
    )
    blocked = store.acquire_lease(
        "lease-run",
        worker_id="worker-b",
        lease_seconds=30,
    )
    heartbeat = store.heartbeat_lease(
        "lease-run",
        worker_id="worker-a",
        lease_seconds=60,
    )
    released = store.release_lease("lease-run", worker_id="worker-a")
    reacquired = store.acquire_lease(
        "lease-run",
        worker_id="worker-b",
        lease_seconds=30,
    )
    expired_same_worker = store.heartbeat_lease(
        "lease-run",
        worker_id="worker-b",
        lease_seconds=30,
        now=utc_now() + timedelta(seconds=120),
    )

    assert acquired is not None
    assert acquired.leased_by == "worker-a"
    assert acquired.lease_expires_at is not None
    assert blocked is None
    assert heartbeat is not None
    assert heartbeat.leased_by == "worker-a"
    assert released.leased_by is None
    assert reacquired is not None
    assert reacquired.leased_by == "worker-b"
    assert expired_same_worker is not None
    assert expired_same_worker.leased_by == "worker-b"


def test_run_store_migrates_existing_runs_table_for_leases(tmp_path) -> None:
    db_path = tmp_path / "runs.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE agent_runs (
                run_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                status TEXT NOT NULL,
                current_stage TEXT NOT NULL,
                require_approval INTEGER NOT NULL,
                plan_json TEXT,
                result_json TEXT,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                total_cost REAL NOT NULL DEFAULT 0.0,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    store = RunStore(db_path)
    store.create_run(
        run_id="migrated-run",
        query="How should existing run tables migrate?",
        require_approval=True,
    )
    run = store.acquire_lease(
        "migrated-run",
        worker_id="worker-a",
        lease_seconds=30,
    )

    assert run is not None
    assert run.leased_by == "worker-a"
    assert store.require_run("migrated-run").request_json is None


def test_run_lease_api_and_stale_recovery(tmp_path, monkeypatch) -> None:
    client, store = _client(tmp_path, monkeypatch)
    run_id = client.post("/runs", json=_run_body("How should worker leases be exposed?")).json()["run_id"]

    lease = client.post(
        f"/runs/{run_id}/lease",
        json={"worker_id": "worker-a", "lease_seconds": 60},
    )
    conflict = client.post(
        f"/runs/{run_id}/lease",
        json={"worker_id": "worker-b", "lease_seconds": 60},
    )
    heartbeat = client.post(
        f"/runs/{run_id}/heartbeat",
        json={"worker_id": "worker-a", "lease_seconds": 60},
    )

    stale_id = "stale-run"
    store.create_run(
        run_id=stale_id,
        query="How should stale worker runs recover?",
        require_approval=False,
    )
    store.update_run(stale_id, status="running", current_stage="researcher")
    store.acquire_lease(
        stale_id,
        worker_id="stale-worker",
        lease_seconds=1,
        now=utc_now() - timedelta(seconds=10),
    )
    stale = client.get("/runs/stale")
    recovered = client.post(
        "/runs/recover-stale",
        json={"reason": "lease expired in test"},
    )

    assert lease.status_code == 200
    assert lease.json()["leased_by"] == "worker-a"
    assert conflict.status_code == 409
    assert heartbeat.status_code == 200
    assert heartbeat.json()["heartbeat_at"] is not None
    assert stale.status_code == 200
    assert any(item["run_id"] == stale_id for item in stale.json())
    assert recovered.status_code == 200
    assert recovered.json()[0]["run_id"] == stale_id
    assert recovered.json()[0]["status"] == "failed"
    assert recovered.json()[0]["leased_by"] is None
    assert store.require_run(stale_id).error_message == "lease expired in test"
    assert any(event.status == "stale_recovered" for event in store.list_events(stale_id))


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


def test_running_cancel_does_not_become_failed(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "runs.sqlite"
    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("LOCAL_RETRIEVAL_MODE", "keyword")
    store = RunStore(db_path)

    async def cancelling_researcher(self, run_id, *args, **kwargs):
        self.cancel(run_id)
        return [], 0, 0, [], []

    monkeypatch.setattr(RunController, "_run_researcher_stage", cancelling_researcher)
    controller = RunController(store=store, settings=load_settings())

    run = asyncio.run(
        controller.create_run(
            CreateRunRequest(
                query="How should running cancellation preserve terminal state?",
                search_provider="mock",
                llm_provider="mock",
                max_researchers=1,
                max_results_per_researcher=1,
                require_approval=False,
            )
        )
    )

    assert run.status == "cancelled"
    assert run.error_message == "run cancelled"
    assert run.leased_by is None
    assert not any(event.status == "failed" for event in store.list_events(run.run_id))
    assert store.require_run(run.run_id).status == "cancelled"


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


def test_retry_reuses_researcher_checkpoint_after_later_stage_failure(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "runs.sqlite"
    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("LOCAL_RETRIEVAL_MODE", "keyword")
    store = RunStore(db_path)
    calls = {"researcher": 0, "synthesizer": 0}
    original_researcher = RunController._run_researcher_stage
    original_synthesizer = RunController._run_synthesizer_stage

    async def counting_researcher(self, *args, **kwargs):
        calls["researcher"] += 1
        return await original_researcher(self, *args, **kwargs)

    async def flaky_synthesizer(self, *args, **kwargs):
        calls["synthesizer"] += 1
        if calls["synthesizer"] == 1:
            raise RuntimeError("synthetic synthesis failure")
        return await original_synthesizer(self, *args, **kwargs)

    monkeypatch.setattr(RunController, "_run_researcher_stage", counting_researcher)
    monkeypatch.setattr(RunController, "_run_synthesizer_stage", flaky_synthesizer)
    controller = RunController(store=store, settings=load_settings())

    failed = asyncio.run(
        controller.create_run(
            CreateRunRequest(
                query="How should retry reuse researcher checkpoints?",
                search_provider="mock",
                llm_provider="mock",
                max_researchers=1,
                max_results_per_researcher=1,
                require_approval=False,
            )
        )
    )
    retried = asyncio.run(controller.retry(failed.run_id))

    assert failed.status == "failed"
    assert retried.status == "succeeded"
    assert calls == {"researcher": 1, "synthesizer": 2}
    researcher_steps = [
        step for step in store.list_steps(failed.run_id) if step.stage == "researcher"
    ]
    assert any((step.output_json or {}).get("checkpoint") for step in researcher_steps)
    assert any(
        event.stage == "researcher" and event.status == "checkpoint_reused"
        for event in store.list_events(failed.run_id)
    )


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
