from __future__ import annotations

import asyncio
from pathlib import Path

from deepresearch_agent.config import load_settings
from deepresearch_agent.run_control import RunController
from deepresearch_agent.run_models import CreateRunRequest
from deepresearch_agent.run_store import RunStore
from deepresearch_agent.run_worker import run_worker_loop


def test_worker_loop_processes_deferred_run(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)
    controller = RunController(store=store, settings=load_settings())
    created = asyncio.run(
        controller.create_run(
            CreateRunRequest(
                query="How should local worker loops process queued runs?",
                search_provider="mock",
                llm_provider="mock",
                max_researchers=1,
                max_results_per_researcher=1,
                require_approval=False,
                defer_execution=True,
            )
        )
    )

    summary = asyncio.run(
        run_worker_loop(
            controller=RunController(store=store, settings=load_settings()),
            poll_interval_seconds=0,
            max_runs=1,
        )
    )
    run = store.require_run(created.run_id)

    assert summary.processed_count == 1
    assert summary.last_run_id == created.run_id
    assert summary.stopped_reason == "max_runs"
    assert run.status == "succeeded"
    assert any(
        event.stage == "worker" and event.status == "claimed"
        for event in store.list_events(run.run_id)
    )


def test_worker_loop_idle_exit_when_queue_empty(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)

    summary = asyncio.run(
        run_worker_loop(
            controller=RunController(store=store, settings=load_settings()),
            poll_interval_seconds=0,
            max_runs=1,
            idle_exit=True,
        )
    )

    assert summary.processed_count == 0
    assert summary.idle_polls == 1
    assert summary.last_run_id is None
    assert summary.stopped_reason == "idle"


def _store(tmp_path: Path, monkeypatch) -> RunStore:
    db_path = tmp_path / "runs.sqlite"
    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("LOCAL_RETRIEVAL_MODE", "keyword")
    return RunStore(db_path)
