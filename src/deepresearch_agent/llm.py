from __future__ import annotations

import json
import re
from typing import Protocol

from deepresearch_agent.cost import CostTracker
from deepresearch_agent.schemas import Finding, ResearchBrief, ResearchRequest, Source, SubQuestion


class LLMProvider(Protocol):
    name: str
    model: str
    supports_structured_output: bool
    supports_tool_calling: bool

    async def create_brief(self, request: ResearchRequest, cost: CostTracker) -> ResearchBrief:
        ...

    async def plan(
        self, brief: ResearchBrief, max_researchers: int, cost: CostTracker
    ) -> list[SubQuestion]:
        ...

    async def synthesize(
        self,
        brief: ResearchBrief,
        plan: list[SubQuestion],
        findings: list[Finding],
        sources: list[Source],
        cost: CostTracker,
    ) -> tuple[str, list[str]]:
        ...


class MockLLMProvider:
    name = "mock"
    supports_structured_output = True
    supports_tool_calling = True

    def __init__(self, model: str = "mock-structured-tool-model") -> None:
        self.model = model

    async def create_brief(self, request: ResearchRequest, cost: CostTracker) -> ResearchBrief:
        normalized = _normalize_query(request.query)
        brief = ResearchBrief(
            original_query=request.query,
            normalized_query=normalized,
            scope=(
                "Answer with implementation-oriented evidence, cite sources for concrete "
                "claims, and call out limitations instead of over-claiming."
            ),
            constraints=[
                "Use at most the configured researcher concurrency.",
                "Prefer sources with concrete implementation or operational detail.",
                "Return a structured report with citation IDs.",
            ],
            assumptions=[
                "No interactive clarification is required in this MVP; ambiguous scope is normalized into a brief.",
                "Mock model output is deterministic and designed for repeatable local tests.",
            ],
        )
        cost.add("brief_generation", request.query, brief.model_dump_json())
        return brief

    async def plan(
        self, brief: ResearchBrief, max_researchers: int, cost: CostTracker
    ) -> list[SubQuestion]:
        topic = brief.normalized_query.rstrip("?")
        templates = [
            (
                "scope",
                "What background and definitions are needed to answer: {topic}?",
                "Establish the frame before comparing designs.",
            ),
            (
                "evidence",
                "What implementation evidence or operational signals matter for: {topic}?",
                "Ground the answer in concrete engineering details.",
            ),
            (
                "tradeoffs",
                "What tradeoffs, risks, and limitations should be considered for: {topic}?",
                "Expose the cost of the chosen design instead of presenting it as free.",
            ),
            (
                "evaluation",
                "How can the claims about {topic} be evaluated reproducibly?",
                "Connect the research answer to benchmark and observability signals.",
            ),
            (
                "fallback",
                "How should fallback behavior be designed for: {topic}?",
                "Check reliability and failure handling for agent tools.",
            ),
        ]
        questions = [
            SubQuestion(id=f"Q{i + 1}", question=text.format(topic=topic), rationale=rationale)
            for i, (_, text, rationale) in enumerate(templates[:max_researchers])
        ]
        cost.add("planning", brief.model_dump_json(), json.dumps([q.model_dump() for q in questions]))
        return questions

    async def synthesize(
        self,
        brief: ResearchBrief,
        plan: list[SubQuestion],
        findings: list[Finding],
        sources: list[Source],
        cost: CostTracker,
    ) -> tuple[str, list[str]]:
        del plan
        claims: list[str] = []
        sections = [
            f"# Research Report: {brief.normalized_query}",
            "",
            "## Executive Summary",
        ]
        for finding in findings:
            citation = f"[{finding.source_ids[0]}]" if finding.source_ids else "[missing]"
            claim = f"{finding.subquestion} {finding.summary} {citation}"
            claims.append(claim)
            sections.append(f"- {claim}")

        sections.extend(["", "## Sources"])
        for source in sources:
            sections.append(f"- [{source.id}] {source.title} - {source.url}")

        answer = "\n".join(sections)
        cost.add(
            "synthesis",
            json.dumps([finding.model_dump() for finding in findings]),
            answer,
        )
        return answer, claims


def _normalize_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query.strip())
    if normalized and normalized[-1] not in ".?!":
        normalized += "?"
    return normalized


def summarize_sources(subquestion: str, sources: list[Source]) -> str:
    if not sources:
        return "No verified source survived filtering for this subquestion."
    lead = sources[0]
    sentence = first_sentence(lead.content)
    return (
        f"The strongest retrieved source is '{lead.title}', which says: {sentence}. "
        f"This supports the subquestion '{subquestion}'."
    )


def first_sentence(text: str, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    match = re.search(r"(?<=[.!?])\s+", cleaned)
    sentence = cleaned[: match.start()] if match else cleaned
    if len(sentence) > max_chars:
        return sentence[: max_chars - 3].rstrip() + "..."
    return sentence
