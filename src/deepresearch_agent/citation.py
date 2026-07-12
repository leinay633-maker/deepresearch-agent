from __future__ import annotations

import re

from deepresearch_agent.citation_judge import CitationJudgeProvider
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.schemas import CitationAssessment, CitationCheckReport, EvidenceQuote, Source
from deepresearch_agent.text_utils import split_sentences, tokenize


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

    def check(
        self,
        claims: list[str],
        sources: list[Source],
        judge_provider: CitationJudgeProvider | None = None,
        cost: CostTracker | None = None,
    ) -> CitationCheckReport:
        source_by_id = {source.id: source for source in sources}
        assessments: list[CitationAssessment] = []
        for claim in claims:
            citation_ids = re.findall(r"\[([SR]\d+)\]", claim)
            overlap_scores = []
            evidence_quotes: list[EvidenceQuote] = []
            missing_citations = []
            snippet_only_citations = []
            for citation_id in citation_ids:
                source = source_by_id.get(citation_id)
                if source is None:
                    overlap_scores.append(0.0)
                    missing_citations.append(citation_id)
                else:
                    quote, quote_overlap = _best_evidence_quote(claim, source)
                    overlap_scores.append(quote_overlap)
                    if source.metadata.get("snippet_only") or source.metadata.get(
                        "extract_status"
                    ) in {"snippet", "crawl_failed", "empty"}:
                        snippet_only_citations.append(citation_id)
                    if quote:
                        evidence_quotes.append(
                            EvidenceQuote(
                                source_id=source.id,
                                source_title=source.title,
                                quote=quote,
                                overlap_score=round(quote_overlap, 3),
                                source_url=source.url,
                                retrieved_at=source.metadata.get("retrieved_at"),
                                extract_status=source.metadata.get("extract_status"),
                                snippet_only=bool(source.metadata.get("snippet_only")),
                            )
                        )
            best_overlap = max(overlap_scores) if overlap_scores else 0.0
            support_level, reason = _support_level(
                citation_ids=citation_ids,
                missing_citations=missing_citations,
                snippet_only_citations=snippet_only_citations,
                best_overlap=best_overlap,
                min_overlap=self.minimum_overlap_for_claim(claim),
            )
            supported = support_level == "supported"
            assessment = CitationAssessment(
                claim=claim,
                citation_ids=citation_ids,
                missing_citation_ids=missing_citations,
                supported=supported,
                support_level=support_level,
                reason=reason,
                overlap_score=round(best_overlap, 3),
                evidence_quotes=evidence_quotes[:3],
            )
            if judge_provider is not None:
                assessment = _apply_judge(assessment, judge_provider, cost)
            assessments.append(assessment)
        supported_count = sum(1 for item in assessments if item.supported)
        total = len(assessments)
        retention = supported_count / total if total else 0.0
        claims_with_valid_citation = sum(
            1
            for item in assessments
            if any(citation_id in source_by_id for citation_id in item.citation_ids)
        )
        citation_reference_count = 0
        supported_reference_count = 0
        for item in assessments:
            threshold = self.minimum_overlap_for_claim(item.claim)
            for citation_id in item.citation_ids:
                source = source_by_id.get(citation_id)
                if source is None:
                    citation_reference_count += 1
                    continue
                citation_reference_count += 1
                quote, overlap = _best_evidence_quote(item.claim, source)
                if (
                    quote
                    and overlap >= threshold
                    and not source.metadata.get("snippet_only")
                    and source.metadata.get("extract_status")
                    not in {"snippet", "crawl_failed", "empty"}
                ):
                    supported_reference_count += 1
        return CitationCheckReport(
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=total - supported_count,
            retention_rate=round(retention, 4),
            assessments=assessments,
            citation_grounding=round(supported_count / total, 4) if total else 0.0,
            citation_coverage=round(claims_with_valid_citation / total, 4) if total else 0.0,
            unsupported_claim_rate=round((total - supported_count) / total, 4)
            if total
            else 0.0,
            citation_precision=round(
                supported_reference_count / citation_reference_count, 4
            )
            if citation_reference_count
            else 0.0,
            claim_extraction_valid=total > 0,
        )

    def minimum_overlap_for_claim(self, claim: str) -> float:
        if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", claim):
            return max(self.min_overlap, 0.18)
        return self.min_overlap


def _overlap(claim: str, source_text: str) -> float:
    claim_tokens = _claim_tokens(claim)
    source_tokens = _tokens(source_text) - STOPWORDS
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def best_evidence_quote(claim: str, source: Source, max_chars: int = 280) -> tuple[str, float]:
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
    return _truncate_quote(best_quote, max_chars), best_score


_best_evidence_quote = best_evidence_quote


def _support_level(
    *,
    citation_ids: list[str],
    missing_citations: list[str],
    snippet_only_citations: list[str],
    best_overlap: float,
    min_overlap: float,
) -> tuple[str, str]:
    if not citation_ids:
        return "unsupported", "citation missing"
    if len(missing_citations) == len(citation_ids):
        return "unverifiable", "all cited source IDs are missing"
    if missing_citations:
        if best_overlap >= min_overlap:
            return "partial", "some cited source IDs are missing"
        return "unverifiable", "some cited source IDs are missing and no cited text grounds the claim"
    if len(snippet_only_citations) == len(citation_ids):
        return "unverifiable", "citation only points to search snippets without extracted body text"
    if snippet_only_citations and best_overlap >= min_overlap:
        return "partial", "some citations only contain search snippets without extracted body text"
    if best_overlap >= min_overlap:
        return "supported", "citation evidence quote overlaps with the claim"
    if best_overlap > 0:
        return "partial", "citation has only partial lexical grounding in the cited source"
    return "unsupported", "citation is present but cited source text does not ground the claim"


def _apply_judge(
    assessment: CitationAssessment,
    judge_provider: CitationJudgeProvider,
    cost: CostTracker | None,
) -> CitationAssessment:
    judgment = judge_provider.judge(assessment.claim, assessment.evidence_quotes)
    if cost is not None and judgment.input_tokens + judgment.output_tokens > 0:
        cost.add_usage(
            stage="citation_judge",
            input_tokens=judgment.input_tokens,
            output_tokens=judgment.output_tokens,
            estimated_cost_usd=judgment.estimated_cost_usd,
            provider=judgment.provider,
            model=judgment.model,
        )
    structural_block = bool(assessment.missing_citation_ids) or not assessment.evidence_quotes
    structural_block = structural_block or all(
        quote.snippet_only for quote in assessment.evidence_quotes
    )
    support_level = assessment.support_level if structural_block else judgment.verdict
    supported = judgment.verdict == "supported" and not structural_block
    return assessment.model_copy(
        update={
            "supported": supported,
            "support_level": support_level,
            "reason": f"{assessment.reason}; judge verdict: {judgment.reason}",
            "judge_provider": judgment.provider,
            "judge_model": judgment.model,
            "judge_confidence": judgment.confidence,
            "judge_reason": judgment.reason,
        }
    )


def _claim_tokens(claim: str) -> set[str]:
    return _tokens(re.sub(r"\[[SR]\d+\]", " ", claim)) - STOPWORDS


def _sentences(text: str) -> list[str]:
    return split_sentences(text)


def _truncate_quote(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip()


def _tokens(text: str) -> set[str]:
    return tokenize(text)
