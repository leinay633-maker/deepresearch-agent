from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    max_researchers: int = Field(default=3, ge=1, le=5)
    max_results_per_researcher: int = Field(default=4, ge=1, le=10)
    search_provider: str | None = None
    seed: int = 20260606


class ResearchBrief(BaseModel):
    original_query: str
    normalized_query: str
    scope: str
    constraints: list[str]
    assumptions: list[str]
    generated_at: datetime = Field(default_factory=utc_now)


class SubQuestion(BaseModel):
    id: str
    question: str
    rationale: str


class Source(BaseModel):
    id: str = ""
    title: str
    url: str
    content: str
    provider: str
    query: str
    score: float = 0.0
    quality_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    subquestion_id: str
    subquestion: str
    summary: str
    source_ids: list[str]
    sources: list[Source]


class CitationAssessment(BaseModel):
    claim: str
    citation_ids: list[str]
    supported: bool
    reason: str
    overlap_score: float


class CitationCheckReport(BaseModel):
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    retention_rate: float
    assessments: list[CitationAssessment]


class CostRecord(BaseModel):
    stage: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class CostSummary(BaseModel):
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_estimated_cost_usd: float
    records: list[CostRecord]


class TraceEvent(BaseModel):
    run_id: str
    stage: str
    status: Literal["start", "success", "error", "fallback"]
    timestamp: datetime = Field(default_factory=utc_now)
    duration_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class StructuredReport(BaseModel):
    run_id: str
    query: str
    brief: ResearchBrief
    plan: list[SubQuestion]
    answer: str
    claims: list[str]
    findings: list[Finding]
    sources: list[Source]
    citation_check: CitationCheckReport
    cost: CostSummary
    metrics: dict[str, Any]
    trace_events: list[TraceEvent]
