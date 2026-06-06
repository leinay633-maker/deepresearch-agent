from __future__ import annotations

import re

from deepresearch_agent.schemas import CitationAssessment, CitationCheckReport, Source


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "or",
    "the",
    "this",
    "to",
    "with",
}


class CitationChecker:
    def __init__(self, min_overlap: float = 0.08) -> None:
        self.min_overlap = min_overlap

    def check(self, claims: list[str], sources: list[Source]) -> CitationCheckReport:
        source_by_id = {source.id: source for source in sources}
        assessments: list[CitationAssessment] = []
        for claim in claims:
            citation_ids = re.findall(r"\[([SR]\d+)\]", claim)
            overlap_scores = []
            for citation_id in citation_ids:
                source = source_by_id.get(citation_id)
                if source is None:
                    overlap_scores.append(0.0)
                else:
                    overlap_scores.append(_overlap(claim, source.content))
            best_overlap = max(overlap_scores) if overlap_scores else 0.0
            supported = bool(citation_ids) and best_overlap >= self.min_overlap
            reason = (
                "citation text overlaps with source content"
                if supported
                else "citation missing or not enough lexical support in cited source"
            )
            assessments.append(
                CitationAssessment(
                    claim=claim,
                    citation_ids=citation_ids,
                    supported=supported,
                    reason=reason,
                    overlap_score=round(best_overlap, 3),
                )
            )
        supported_count = sum(1 for item in assessments if item.supported)
        total = len(assessments)
        retention = supported_count / total if total else 1.0
        return CitationCheckReport(
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=total - supported_count,
            retention_rate=round(retention, 4),
            assessments=assessments,
        )


def _overlap(claim: str, source_text: str) -> float:
    claim_tokens = _tokens(claim) - STOPWORDS
    source_tokens = _tokens(source_text) - STOPWORDS
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", text)}
