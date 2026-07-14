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

_QUERY_GENERIC_TOKENS = {
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "which",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "as",
    "and",
    "or",
    "year",
    "date",
    "officially",
    "recorded",
    "founded",
    "founding",
    "foundation",
    "established",
    "municipality",
    "city",
    "town",
    "country",
    "answer",
    "source",
    "sources",
    "evidence",
    "历史",
    "资料",
    "来源",
    "证据",
    "什么",
    "何时",
    "哪年",
    "哪一年",
    "年份",
    "日期",
    "成立",
    "创立",
    "建于",
    "城市",
    "市",
    "镇",
    "国家",
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
    document_anchored = _has_required_anchor_overlap(
        claim, f"{source.title} {source.content}"
    )
    best_quote = ""
    best_score = 0.0
    for sentence in _sentences(source.content):
        for candidate in _evidence_windows(sentence, max_chars=max_chars):
            score = _evidence_relevance_score(
                claim,
                claim_tokens,
                candidate,
                document_anchored=document_anchored,
            )
            if score > best_score:
                best_score = score
                best_quote = candidate
    return _truncate_quote(best_quote, max_chars), best_score


def _evidence_relevance_score(
    claim: str,
    claim_tokens: set[str],
    candidate: str,
    *,
    document_anchored: bool = False,
) -> float:
    candidate_tokens = _tokens(candidate) - STOPWORDS
    if not candidate_tokens:
        return 0.0
    score = len(claim_tokens & candidate_tokens) / len(claim_tokens)
    claim_lower = claim.lower()
    candidate_lower = candidate.lower()
    requires_anchor = _requires_entity_anchor(claim_lower)
    anchored = _has_required_anchor_overlap(claim, candidate)
    # A date or a generic founding verb from another entity is not evidence.
    # This hard gate prevents a page about Python in 1991 (or San Diego) from
    # satisfying a question about when San Carlos was founded.
    if requires_anchor and not (anchored or document_anchored):
        return 0.0
    version_pattern = r"\b\d+\.\d+(?:\.\d+)?(?:[-_a-z0-9.]*)\b"
    claim_numbers = set(re.findall(version_pattern, claim_lower))
    candidate_numbers = set(re.findall(version_pattern, candidate_lower))
    if claim_numbers:
        if claim_numbers <= candidate_numbers:
            score += 0.4
        elif candidate_numbers:
            score -= 0.15
    if "latest" in claim_lower and "latest" in candidate_lower:
        score += 0.3
    if any(term in claim_lower for term in ("version", "release", "版本", "发布")):
        if candidate_numbers:
            score += 0.25
    if (
        (anchored or document_anchored)
        and _asks_for_time_or_year(claim_lower)
        and _has_date_or_year(candidate_lower)
    ):
        # A question such as "when was X founded?" often has little lexical
        # overlap with an answer phrased as "officially started on 1786".
        # Preserve answer-bearing date sentences for the downstream verifier.
        score += 0.45
    if (
        (anchored or document_anchored)
        and _asks_about_founding(claim_lower)
        and _mentions_founding_alias(candidate_lower)
    ):
        score += 0.3
    if any(marker in candidate for marker in (">>>", "Traceback (most recent call last)")):
        score -= 0.2
    return max(0.0, min(score, 1.0))


def source_is_relevant_to_claim(claim: str, source: Source) -> bool:
    """Whether a source is eligible to become research evidence for a claim."""

    quote, score = best_evidence_quote(claim, source)
    if not quote or score < CitationChecker().minimum_overlap_for_claim(claim):
        return False
    if _requires_entity_anchor(claim.lower()) and not _has_required_anchor_overlap(
        claim, f"{source.title} {quote}"
    ):
        return False
    if _proper_name_anchors(claim) and not _has_entity_anchor_overlap(
        claim, f"{source.title} {source.content}"
    ):
        return False
    return True


def entity_anchor_coverage(claim: str, candidate: str) -> dict[str, int | bool]:
    """Return aggregate-safe named-entity coverage without retaining page text."""

    anchors = _proper_name_anchors(claim)
    candidate_tokens = _tokens(candidate)
    matched = anchors.intersection(candidate_tokens)
    normalized_candidate = " ".join(
        token.lower() for token in re.findall(r"[A-Za-z0-9-]+", candidate)
    )
    complete_multiword_entity = any(
        " ".join(phrase) in normalized_candidate
        for phrase in _multiword_entity_phrases(claim)
    )
    return {
        "anchor_count": len(anchors),
        "matched_anchor_count": len(matched),
        "complete_multiword_entity": complete_multiword_entity,
    }


def _requires_entity_anchor(claim_lower: str) -> bool:
    return _asks_for_time_or_year(claim_lower) or _asks_about_founding(claim_lower)


def _has_required_anchor_overlap(claim: str, candidate: str) -> bool:
    anchors = _meaningful_query_anchors(claim)
    if not anchors:
        return True
    overlap = anchors.intersection(_tokens(candidate))
    # For a multi-token proper-name query, a shared single token such as
    # "San" is too weak. A single specific anchor (e.g. Python) is enough.
    required = 2 if len(anchors) >= 2 else 1
    return len(overlap) >= required


def _has_entity_anchor_overlap(claim: str, candidate: str) -> bool:
    """Reject one-token collisions in prompts containing multiple named anchors."""

    coverage = entity_anchor_coverage(claim, candidate)
    anchor_count = int(coverage["anchor_count"])
    if not anchor_count:
        return True
    matched = int(coverage["matched_anchor_count"])
    required = 2 if anchor_count >= 2 else 1
    return matched >= required or bool(coverage["complete_multiword_entity"])


def _proper_name_anchors(claim: str) -> set[str]:
    """Extract likely English proper-name tokens and all-caps acronyms."""

    return {
        token.lower()
        for token in re.findall(r"\b(?:[A-Z][a-z]{2,}|[A-Z][A-Z0-9-]{1,})\b", claim)
        if token.lower() not in STOPWORDS and token.lower() not in _QUERY_GENERIC_TOKENS
    }


def _multiword_entity_phrases(claim: str) -> list[tuple[str, ...]]:
    phrases = re.findall(
        r"\b(?:[A-Z][a-z]{2,}|[A-Z][A-Z0-9-]{1,})"
        r"(?:\s+(?:[A-Z][a-z]{2,}|[A-Z][A-Z0-9-]{1,}))+\b",
        claim,
    )
    return [
        tuple(token.lower() for token in re.findall(r"[A-Za-z0-9-]+", phrase))
        for phrase in phrases
    ]


def _meaningful_query_anchors(claim: str) -> set[str]:
    anchors = {
        token
        for token in _tokens(claim)
        if token not in STOPWORDS
        and token not in _QUERY_GENERIC_TOKENS
        and (len(token) > 2 or any("\u3400" <= character <= "\u9fff" for character in token))
    }
    return anchors


def _asks_for_time_or_year(claim_lower: str) -> bool:
    return bool(
        re.search(
            r"\b(?:when|what\s+(?:year|date|month)|which\s+year|year|date)\b|"
            r"(?:何时|哪年|哪一年|年份|日期|时间|何月|几月)",
            claim_lower,
        )
    )


def _has_date_or_year(candidate_lower: str) -> bool:
    return bool(
        re.search(r"\b(?:1[5-9]\d{2}|20\d{2})\b", candidate_lower)
        or re.search(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
            r"dec(?:ember)?)\.?\s+\d{1,2}",
            candidate_lower,
        )
    )


def _asks_about_founding(claim_lower: str) -> bool:
    return bool(
        re.search(r"\b(?:founded|founding|foundation|established)\b|(?:建市|成立|创立|建于)", claim_lower)
        or "fundaci" in claim_lower
    )


def _mentions_founding_alias(candidate_lower: str) -> bool:
    return bool(
        re.search(r"\b(?:founded|founding|established|officially\s+started|was\s+started|created)\b", candidate_lower)
        or "fundaci" in candidate_lower
        or any(term in candidate_lower for term in ("成立", "创立", "建于"))
    )


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
    try:
        judgment = judge_provider.judge(assessment.claim, assessment.evidence_quotes)
    except Exception as exc:  # noqa: BLE001 - a judge outage must fail closed, not crash a run.
        return assessment.model_copy(
            update={
                "supported": False,
                "support_level": "unverifiable",
                "reason": f"{assessment.reason}; citation judge failed closed",
                "judge_provider": getattr(judge_provider, "name", "unknown"),
                "judge_model": getattr(judge_provider, "model", None),
                "judge_confidence": 0.0,
                "judge_reason": f"judge error: {type(exc).__name__}: {str(exc)[:180]}",
            }
        )
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


def _evidence_windows(text: str, *, max_chars: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return [normalized] if normalized else []
    overlap = max(40, max_chars // 3)
    step = max(1, max_chars - overlap)
    return [
        normalized[start : start + max_chars]
        for start in range(0, len(normalized), step)
        if normalized[start : start + max_chars].strip()
    ]


def _tokens(text: str) -> set[str]:
    return tokenize(text)
