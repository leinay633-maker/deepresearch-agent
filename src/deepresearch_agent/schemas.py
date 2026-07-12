from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchBudget(BaseModel):
    max_rounds: int = Field(default=1, ge=1, le=10)
    max_tool_calls: int = Field(default=1, ge=1, le=50)
    deadline_seconds: float | None = Field(default=None, gt=0, le=3600)
    min_evidence_items: int = Field(default=1, ge=1, le=50)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    max_researchers: int = Field(default=3, ge=1, le=5)
    max_results_per_researcher: int = Field(default=4, ge=1, le=10)
    llm_provider: str | None = None
    llm_model: str | None = None
    brief_model: str | None = None
    planner_model: str | None = None
    synthesis_model: str | None = None
    search_provider: str | None = None
    seed: int = 20260606
    reflection_enabled: bool = False
    max_reflection_rounds: int = Field(default=1, ge=0, le=3)
    reflection_min_sources: int = Field(default=4, ge=1, le=20)
    citation_judge_provider: str | None = None
    citation_judge_model: str | None = None
    max_rounds: int = Field(default=1, ge=1, le=10)
    max_tool_calls: int = Field(default=1, ge=1, le=50)
    deadline_seconds: float | None = Field(default=None, gt=0, le=3600)
    min_evidence_items: int = Field(default=1, ge=1, le=50)
    fallback_policy: Literal["mock", "degraded", "fail"] | None = None

    def research_budget(self) -> ResearchBudget:
        return ResearchBudget(
            max_rounds=self.max_rounds,
            max_tool_calls=self.max_tool_calls,
            deadline_seconds=self.deadline_seconds,
            min_evidence_items=self.min_evidence_items,
        )


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


class EvidenceItem(BaseModel):
    source_id: str = ""
    source_title: str
    source_url: str
    quote: str
    query: str
    overlap_score: float = 0.0
    retrieved_at: str | None = None


class ResearchDecision(BaseModel):
    action: Literal["continue", "stop", "need_follow_up", "conflict_found"]
    reason: str
    evidence_gap: str | None = None
    follow_up_query: str | None = None


class ResearchRoundResult(BaseModel):
    round_index: int
    query: str
    source_count: int
    evidence_count: int
    tool_attempts: int = 0
    fallback_used: bool = False
    decision: ResearchDecision


class ResearchResult(BaseModel):
    rounds: list[ResearchRoundResult] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    termination_reason: str = "completed"


class Finding(BaseModel):
    subquestion_id: str
    subquestion: str
    summary: str
    source_ids: list[str]
    sources: list[Source]
    research: ResearchResult | None = None


class EvidenceQuote(BaseModel):
    source_id: str
    source_title: str
    quote: str
    overlap_score: float
    source_url: str | None = None
    retrieved_at: str | None = None
    extract_status: str | None = None
    snippet_only: bool = False


class CitationAssessment(BaseModel):
    claim: str
    citation_ids: list[str]
    missing_citation_ids: list[str] = Field(default_factory=list)
    supported: bool
    support_level: Literal["supported", "partial", "unsupported", "unverifiable"] = "unsupported"
    reason: str
    overlap_score: float
    evidence_quotes: list[EvidenceQuote] = Field(default_factory=list)
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_confidence: float | None = None
    judge_reason: str | None = None


class CitationCheckReport(BaseModel):
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    retention_rate: float
    assessments: list[CitationAssessment]
    citation_grounding: float = 0.0
    citation_coverage: float = 0.0
    unsupported_claim_rate: float = 0.0
    citation_precision: float = 0.0


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
