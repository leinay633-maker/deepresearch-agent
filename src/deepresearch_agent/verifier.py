from __future__ import annotations

import re

from deepresearch_agent.schemas import Source


LOW_QUALITY_PATTERNS = [
    "sponsored content",
    "click here",
    "subscribe to read",
    "unverified rumor",
]


class SourceVerifier:
    def __init__(self, min_quality_score: float = 0.35) -> None:
        self.min_quality_score = min_quality_score

    def verify(self, sources: list[Source]) -> list[Source]:
        verified: list[Source] = []
        for source in sources:
            score, reasons = self.score(source)
            updated = source.model_copy(
                update={
                    "quality_score": score,
                    "metadata": {**source.metadata, "quality_reasons": reasons},
                }
            )
            if score >= self.min_quality_score:
                verified.append(updated)
        return sorted(verified, key=lambda item: (item.quality_score, item.score), reverse=True)

    def score(self, source: Source) -> tuple[float, list[str]]:
        reasons = []
        score = 0.2
        if source.title and len(source.title.strip()) >= 4:
            score += 0.15
            reasons.append("has_title")
        if len(source.content.strip()) >= 60:
            score += 0.25
            reasons.append("substantial_content")
        if source.url.startswith("https://") or source.url.startswith("file://"):
            score += 0.15
            reasons.append("stable_url")
        if source.provider in {"wikipedia", "local_rag", "mock"}:
            score += 0.15
            reasons.append("known_adapter")
        if re.search(r"\[[0-9]+\]", source.content):
            score += 0.05
            reasons.append("contains_reference_marker")
        lowered = source.content.lower()
        if any(pattern in lowered for pattern in LOW_QUALITY_PATTERNS):
            score -= 0.35
            reasons.append("low_quality_pattern")
        return round(max(0.0, min(score, 1.0)), 3), reasons
