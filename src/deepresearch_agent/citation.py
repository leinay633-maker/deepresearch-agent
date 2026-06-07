from __future__ import annotations

import re

from deepresearch_agent.schemas import CitationAssessment, CitationCheckReport, EvidenceQuote, Source


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
            evidence_quotes: list[EvidenceQuote] = []
            missing_citations = []
            for citation_id in citation_ids:
                source = source_by_id.get(citation_id)
                if source is None:
                    overlap_scores.append(0.0)
                    missing_citations.append(citation_id)
                else:
                    quote, quote_overlap = _best_evidence_quote(claim, source)
                    overlap_scores.append(quote_overlap)
                    if quote:
                        evidence_quotes.append(
                            EvidenceQuote(
                                source_id=source.id,
                                source_title=source.title,
                                quote=quote,
                                overlap_score=round(quote_overlap, 3),
                            )
                        )
            best_overlap = max(overlap_scores) if overlap_scores else 0.0
            support_level, reason = _support_level(
                citation_ids=citation_ids,
                missing_citations=missing_citations,
                best_overlap=best_overlap,
                min_overlap=self.min_overlap,
            )
            supported = support_level == "supported"
            assessments.append(
                CitationAssessment(
                    claim=claim,
                    citation_ids=citation_ids,
                    supported=supported,
                    support_level=support_level,
                    reason=reason,
                    overlap_score=round(best_overlap, 3),
                    evidence_quotes=evidence_quotes[:3],
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
    claim_tokens = _claim_tokens(claim)
    source_tokens = _tokens(source_text) - STOPWORDS
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def _best_evidence_quote(claim: str, source: Source, max_chars: int = 280) -> tuple[str, float]:
    claim_tokens = _claim_tokens(claim)
    if not claim_tokens:
        return "", 0.0
    best_quote = ""
    best_score = 0.0
    for sentence in _sentences(source.content):
        sentence_tokens = _tokens(sentence) - STOPWORDS
        if not sentence_tokens:
            continue
        score = len(claim_tokens & sentence_tokens) / len(claim_tokens)
        if score > best_score:
            best_score = score
            best_quote = sentence
    if not best_quote and source.content:
        best_quote = source.content.strip()
    return _truncate_quote(best_quote, max_chars), best_score


def _support_level(
    *,
    citation_ids: list[str],
    missing_citations: list[str],
    best_overlap: float,
    min_overlap: float,
) -> tuple[str, str]:
    if not citation_ids:
        return "unsupported", "citation missing"
    if len(missing_citations) == len(citation_ids):
        return "unverifiable", "all cited source IDs are missing"
    if best_overlap >= min_overlap:
        return "supported", "citation evidence quote overlaps with the claim"
    if best_overlap > 0:
        return "partial", "citation has only partial lexical grounding in the cited source"
    if missing_citations:
        return "unverifiable", "some cited source IDs are missing and no cited text grounds the claim"
    return "unsupported", "citation is present but cited source text does not ground the claim"


def _claim_tokens(claim: str) -> set[str]:
    return _tokens(re.sub(r"\[[SR]\d+\]", " ", claim)) - STOPWORDS


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _truncate_quote(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip()


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", text)}
