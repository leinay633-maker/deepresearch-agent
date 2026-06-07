from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.run_models import AgentEvent, AgentRun, AgentStep
from deepresearch_agent.schemas import utc_now


class RunStore:
    def __init__(self, path: str | Path | None = None, settings: Settings | None = None) -> None:
        selected = path or (settings or load_settings()).run_store_path
        self.path = Path(selected)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def create_run(
        self,
        *,
        run_id: str,
        query: str,
        require_approval: bool,
        current_stage: str = "planner",
    ) -> AgentRun:
        now = utc_now()
        run = AgentRun(
            run_id=run_id,
            query=query,
            status="queued",
            current_stage=current_stage,  # type: ignore[arg-type]
            require_approval=require_approval,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, query, status, current_stage, require_approval,
                    plan_json, result_json, total_tokens, total_cost, error_message,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.query,
                    run.status,
                    run.current_stage,
                    int(run.require_approval),
                    _json_dumps(run.plan_json),
                    _json_dumps(run.result_json),
                    run.total_tokens,
                    run.total_cost,
                    run.error_message,
                    _dt(run.created_at),
                    _dt(run.updated_at),
                ),
            )
        return run

    def get_run(self, run_id: str) -> AgentRun | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def require_run(self, run_id: str) -> AgentRun:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"run not found: {run_id}")
        return run

    def update_run(self, run_id: str, **fields: Any) -> AgentRun:
        allowed = {
            "status",
            "current_stage",
            "require_approval",
            "plan_json",
            "result_json",
            "total_tokens",
            "total_cost",
            "error_message",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = [_db_value(key, value) for key, value in updates.items()]
        values.append(run_id)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE agent_runs SET {assignments} WHERE run_id = ?",
                values,
            )
        return self.require_run(run_id)

    def add_step(
        self,
        *,
        step_id: str,
        run_id: str,
        stage: str,
        status: str,
        input_json: dict[str, Any] | None = None,
        output_json: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        token_usage: int = 0,
        cost: float = 0.0,
        error: str | None = None,
        retry_count: int = 0,
    ) -> AgentStep:
        step = AgentStep(
            step_id=step_id,
            run_id=run_id,
            stage=stage,
            status=status,  # type: ignore[arg-type]
            input_json=input_json,
            output_json=output_json,
            latency_ms=latency_ms,
            token_usage=token_usage,
            cost=cost,
            error=error,
            retry_count=retry_count,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_steps (
                    step_id, run_id, stage, status, input_json, output_json,
                    latency_ms, token_usage, cost, error, retry_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.step_id,
                    step.run_id,
                    step.stage,
                    step.status,
                    _json_dumps(step.input_json),
                    _json_dumps(step.output_json),
                    step.latency_ms,
                    step.token_usage,
                    step.cost,
                    step.error,
                    step.retry_count,
                    _dt(step.created_at),
                ),
            )
        return step

    def list_steps(self, run_id: str) -> list[AgentStep]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_steps
                WHERE run_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (run_id,),
            ).fetchall()
        return [_step_from_row(row) for row in rows]

    def add_event(
        self,
        *,
        run_id: str,
        stage: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_events (run_id, stage, status, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, stage, status, _json_dumps(payload or {}), _dt(now)),
            )
            event_id = int(cursor.lastrowid)
        return AgentEvent(
            event_id=event_id,
            run_id=run_id,
            stage=stage,
            status=status,
            payload=payload or {},
            created_at=now,
        )

    def list_events(self, run_id: str, after_event_id: int | None = None) -> list[AgentEvent]:
        sql = "SELECT * FROM agent_events WHERE run_id = ?"
        params: list[Any] = [run_id]
        if after_event_id is not None:
            sql += " AND event_id > ?"
            params.append(after_event_id)
        sql += " ORDER BY event_id ASC"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_event_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
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
                );

                CREATE TABLE IF NOT EXISTS agent_steps (
                    step_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT,
                    output_json TEXT,
                    latency_ms REAL,
                    token_usage INTEGER NOT NULL DEFAULT 0,
                    cost REAL NOT NULL DEFAULT 0.0,
                    error TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS agent_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
                );
                """
            )


def _run_from_row(row: sqlite3.Row) -> AgentRun:
    return AgentRun(
        run_id=row["run_id"],
        query=row["query"],
        status=row["status"],
        current_stage=row["current_stage"],
        require_approval=bool(row["require_approval"]),
        plan_json=_json_loads(row["plan_json"]),
        result_json=_json_loads(row["result_json"]),
        total_tokens=int(row["total_tokens"]),
        total_cost=float(row["total_cost"]),
        error_message=row["error_message"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _step_from_row(row: sqlite3.Row) -> AgentStep:
    return AgentStep(
        step_id=row["step_id"],
        run_id=row["run_id"],
        stage=row["stage"],
        status=row["status"],
        input_json=_json_loads(row["input_json"]),
        output_json=_json_loads(row["output_json"]),
        latency_ms=row["latency_ms"],
        token_usage=int(row["token_usage"]),
        cost=float(row["cost"]),
        error=row["error"],
        retry_count=int(row["retry_count"]),
        created_at=_parse_dt(row["created_at"]),
    )


def _event_from_row(row: sqlite3.Row) -> AgentEvent:
    return AgentEvent(
        event_id=int(row["event_id"]),
        run_id=row["run_id"],
        stage=row["stage"],
        status=row["status"],
        payload=_json_loads(row["payload_json"]) or {},
        created_at=_parse_dt(row["created_at"]),
    )


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return json.loads(value)


def _db_value(key: str, value: Any) -> Any:
    if key in {"plan_json", "result_json"}:
        return _json_dumps(value)
    if key == "require_approval":
        return int(value)
    if key == "updated_at":
        return _dt(value)
    return value


def _dt(value: datetime) -> str:
    return value.isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)
