from __future__ import annotations

from typing import Any

from deepresearch_agent.schemas import CitationCheckReport, Source
from deepresearch_agent.source_metrics import source_diversity_metrics


SUCCESS_SEMANTICS = (
    "deprecated alias of execution_success; answer quality requires an explicit judge"
)


def build_execution_metrics(
    *,
    latency_ms: float,
    raw_search_result_count: int,
    verified_source_count: int,
    deduped_source_count: int,
    fallback_count: int,
    degraded_count: int,
    sources: list[Source],
    citation_report: CitationCheckReport,
) -> dict[str, Any]:
    legacy_success = citation_report.retention_rate >= 0.8 and bool(sources)
    source_quality = (
        round(sum(source.quality_score for source in sources) / len(sources), 4)
        if sources
        else None
    )
    failure_recovery = 1.0 if fallback_count or degraded_count else None
    return {
        "latency_ms": round(latency_ms, 3),
        "raw_search_result_count": raw_search_result_count,
        "verified_source_count": verified_source_count,
        "deduped_source_count": deduped_source_count,
        "fallback_count": fallback_count,
        "degraded_count": degraded_count,
        "citation_retention_rate": citation_report.retention_rate,
        "execution_success": True,
        "task_format_valid": None,
        "answer_quality": None,
        "citation_grounding": citation_report.citation_grounding,
        "citation_precision": citation_report.citation_precision,
        "citation_coverage": citation_report.citation_coverage,
        "unsupported_claim_rate": citation_report.unsupported_claim_rate,
        "source_quality": source_quality,
        "tool_failure_recovery": failure_recovery,
        # Kept for existing clients; this is no longer a quality verdict.
        "success": True,
        "legacy_report_success": legacy_success,
        "success_semantics": SUCCESS_SEMANTICS,
        **source_diversity_metrics(sources),
    }
