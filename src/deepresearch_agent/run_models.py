from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from deepresearch_agent.schemas import ResearchRequest, SubQuestion, utc_now


RunStatus = Literal[
    "queued",
    "running",
    "waiting_approval",
    "paused",
    "succeeded",
    "failed",
    "cancelled",
]
RunStage = Literal[
    "planner",
    "approval",
    "researcher",
    "synthesizer",
    "verifier",
    "completed",
]
StepStatus = Literal["started", "succeeded", "failed", "cancelled"]


class AgentRun(BaseModel):
    run_id: str
    query: str
    status: RunStatus
    current_stage: RunStage
    require_approval: bool = True
    plan_json: dict[str, Any] | None = None
    result_json: dict[str, Any] | None = None
    total_tokens: int = 0
    total_cost: float = 0.0
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AgentStep(BaseModel):
    step_id: str
    run_id: str
    stage: str
    status: StepStatus
    input_json: dict[str, Any] | None = None
    output_json: dict[str, Any] | None = None
    latency_ms: float | None = None
    token_usage: int = 0
    cost: float = 0.0
    error: str | None = None
    retry_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class AgentEvent(BaseModel):
    event_id: int
    run_id: str
    stage: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class CreateRunRequest(ResearchRequest):
    require_approval: bool = True


class RunActionResponse(BaseModel):
    run_id: str
    status: RunStatus
    current_stage: RunStage


class ApprovalPayload(BaseModel):
    plan: dict[str, Any]
    subquestions: list[SubQuestion]
    estimated_researcher_count: int
    risk_note: str


class EditPlanRequest(BaseModel):
    subquestions: list[SubQuestion]


class RejectRunRequest(BaseModel):
    reason: str = "planner output rejected"


class RunTrace(BaseModel):
    run: AgentRun
    steps: list[AgentStep]
    events: list[AgentEvent]
    total_tokens: int
    total_cost: float
    latency_ms: float | None = None
