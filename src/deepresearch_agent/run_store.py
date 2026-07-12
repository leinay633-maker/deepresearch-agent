from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
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
        request_json: dict[str, Any] | None = None,
        current_stage: str = "planner",
    ) -> AgentRun:
        now = utc_now()
        run = AgentRun(
            run_id=run_id,
            query=query,
            status="queued",
            current_stage=current_stage,  # type: ignore[arg-type]
            require_approval=require_approval,
            request_json=request_json,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, query, status, current_stage, require_approval,
                    request_json, plan_json, result_json, total_tokens, total_cost,
                    error_message, leased_by, heartbeat_at, lease_expires_at, created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.query,
                    run.status,
                    run.current_stage,
                    int(run.require_approval),
                    _json_dumps(run.request_json),
                    _json_dumps(run.plan_json),
                    _json_dumps(run.result_json),
                    run.total_tokens,
                    run.total_cost,
                    run.error_message,
                    run.leased_by,
                    _dt_or_none(run.heartbeat_at),
                    _dt_or_none(run.lease_expires_at),
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

    def list_runs(self, limit: int = 20) -> list[AgentRun]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_runs
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

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
            "request_json",
            "plan_json",
            "result_json",
            "total_tokens",
            "total_cost",
            "error_message",
            "leased_by",
            "heartbeat_at",
            "lease_expires_at",
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

    def transition_run(
        self,
        run_id: str,
        *,
        expected_statuses: set[str],
        expected_worker_id: str | None = None,
        require_unleased: bool = False,
        **fields: Any,
    ) -> AgentRun | None:
        """Conditionally update one run and return ``None`` when the CAS loses.

        Status transitions must not be implemented as a read followed by an
        unconditional ``update_run``: another API request may cancel, recover,
        approve, or retry the run between those two operations.
        """

        allowed = {
            "status",
            "current_stage",
            "require_approval",
            "request_json",
            "plan_json",
            "result_json",
            "total_tokens",
            "total_cost",
            "error_message",
            "leased_by",
            "heartbeat_at",
            "lease_expires_at",
        }
        if not expected_statuses:
            raise ValueError("expected_statuses must not be empty")
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            raise ValueError("transition_run requires at least one update field")
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        placeholders = ", ".join("?" for _ in expected_statuses)
        where = [f"status IN ({placeholders})"]
        where_values: list[Any] = sorted(expected_statuses)
        if expected_worker_id is not None:
            where.append("leased_by = ?")
            where_values.append(expected_worker_id)
        elif require_unleased:
            where.append("leased_by IS NULL")
        values = [_db_value(key, value) for key, value in updates.items()]
        values.extend([run_id, *where_values])
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE agent_runs SET {assignments} "
                f"WHERE run_id = ? AND {' AND '.join(where)}",
                values,
            )
        if cursor.rowcount != 1:
            return None
        return self.require_run(run_id)

    def recover_stale_run(
        self,
        run_id: str,
        *,
        expected_worker_id: str,
        expected_lease_expires_at: datetime,
        reason: str,
        now: datetime | None = None,
    ) -> AgentRun | None:
        """Fence one exact expired lease so a concurrent heartbeat wins safely."""

        current_time = now or utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_runs
                SET status = 'failed', error_message = ?, leased_by = NULL,
                    heartbeat_at = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE run_id = ? AND status = 'running' AND leased_by = ?
                  AND lease_expires_at = ? AND lease_expires_at <= ?
                """,
                (
                    reason,
                    _dt(current_time),
                    run_id,
                    expected_worker_id,
                    _dt(expected_lease_expires_at),
                    _dt(current_time),
                ),
            )
        if cursor.rowcount != 1:
            return None
        return self.require_run(run_id)

    def complete_run_if_owned(
        self,
        run_id: str,
        *,
        worker_id: str,
        result_json: dict[str, Any],
        total_tokens: int,
        total_cost: float,
    ) -> AgentRun | None:
        """Commit success only while the run is active and still owned by this worker."""

        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_runs
                SET status = 'succeeded', current_stage = 'completed', result_json = ?,
                    total_tokens = ?, total_cost = ?, error_message = NULL,
                    leased_by = NULL, heartbeat_at = NULL, lease_expires_at = NULL,
                    updated_at = ?
                WHERE run_id = ? AND status = 'running' AND leased_by = ?
                """,
                (
                    _json_dumps(result_json),
                    total_tokens,
                    total_cost,
                    _dt(now),
                    run_id,
                    worker_id,
                ),
            )
        if cursor.rowcount != 1:
            return None
        return self.require_run(run_id)

    def acquire_lease(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> AgentRun | None:
        current_time = now or utc_now()
        expires_at = current_time + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_runs
                SET leased_by = ?, heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE run_id = ?
                  AND status NOT IN ('succeeded', 'failed', 'cancelled')
                  AND (
                    leased_by IS NULL
                    OR lease_expires_at IS NULL
                    OR lease_expires_at <= ?
                    OR leased_by = ?
                  )
                """,
                (
                    worker_id,
                    _dt(current_time),
                    _dt(expires_at),
                    _dt(current_time),
                    run_id,
                    _dt(current_time),
                    worker_id,
                ),
            )
        if cursor.rowcount != 1:
            return None
        return self.require_run(run_id)

    def claim_next_queued_run(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> AgentRun | None:
        current_time = now or utc_now()
        expires_at = current_time + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_runs
                SET leased_by = ?, heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE run_id = (
                    SELECT run_id FROM agent_runs
                    WHERE status = 'queued'
                      AND current_stage = 'planner'
                      AND (
                        leased_by IS NULL
                        OR lease_expires_at IS NULL
                        OR lease_expires_at <= ?
                      )
                    ORDER BY created_at ASC, updated_at ASC
                    LIMIT 1
                )
                RETURNING *
                """,
                (
                    worker_id,
                    _dt(current_time),
                    _dt(expires_at),
                    _dt(current_time),
                    _dt(current_time),
                ),
            )
            row = cursor.fetchone()
        return _run_from_row(row) if row is not None else None

    def heartbeat_lease(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> AgentRun | None:
        current_time = now or utc_now()
        expires_at = current_time + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_runs
                SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE run_id = ?
                  AND leased_by = ?
                  AND status NOT IN ('succeeded', 'failed', 'cancelled')
                """,
                (
                    _dt(current_time),
                    _dt(expires_at),
                    _dt(current_time),
                    run_id,
                    worker_id,
                ),
            )
        if cursor.rowcount != 1:
            return None
        return self.require_run(run_id)

    def release_lease(self, run_id: str, *, worker_id: str) -> AgentRun:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET leased_by = NULL, heartbeat_at = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE run_id = ? AND leased_by = ?
                """,
                (_dt(utc_now()), run_id, worker_id),
            )
        return self.require_run(run_id)

    def clear_lease(self, run_id: str) -> AgentRun:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET leased_by = NULL, heartbeat_at = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE run_id = ?
                """,
                (_dt(utc_now()), run_id),
            )
        return self.require_run(run_id)

    def list_stale_runs(self, now: datetime | None = None) -> list[AgentRun]:
        current_time = now or utc_now()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= ?
                ORDER BY lease_expires_at ASC, updated_at ASC
                """,
                (_dt(current_time),),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

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
                    request_json TEXT,
                    plan_json TEXT,
                    result_json TEXT,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    total_cost REAL NOT NULL DEFAULT 0.0,
                    error_message TEXT,
                    leased_by TEXT,
                    heartbeat_at TEXT,
                    lease_expires_at TEXT,
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
            self._ensure_agent_run_columns(connection)

    def _ensure_agent_run_columns(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(agent_runs)").fetchall()
        }
        additions = {
            "request_json": "ALTER TABLE agent_runs ADD COLUMN request_json TEXT",
            "leased_by": "ALTER TABLE agent_runs ADD COLUMN leased_by TEXT",
            "heartbeat_at": "ALTER TABLE agent_runs ADD COLUMN heartbeat_at TEXT",
            "lease_expires_at": "ALTER TABLE agent_runs ADD COLUMN lease_expires_at TEXT",
        }
        for name, statement in additions.items():
            if name not in columns:
                connection.execute(statement)


def _run_from_row(row: sqlite3.Row) -> AgentRun:
    return AgentRun(
        run_id=row["run_id"],
        query=row["query"],
        status=row["status"],
        current_stage=row["current_stage"],
        require_approval=bool(row["require_approval"]),
        request_json=_json_loads(row["request_json"]),
        plan_json=_json_loads(row["plan_json"]),
        result_json=_json_loads(row["result_json"]),
        total_tokens=int(row["total_tokens"]),
        total_cost=float(row["total_cost"]),
        error_message=row["error_message"],
        leased_by=row["leased_by"],
        heartbeat_at=_parse_dt_or_none(row["heartbeat_at"]),
        lease_expires_at=_parse_dt_or_none(row["lease_expires_at"]),
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
    if key in {"request_json", "plan_json", "result_json"}:
        return _json_dumps(value)
    if key == "require_approval":
        return int(value)
    if key in {"updated_at", "heartbeat_at", "lease_expires_at"}:
        return _dt_or_none(value)
    return value


def _dt(value: datetime) -> str:
    return value.isoformat()


def _dt_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _parse_dt_or_none(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)
