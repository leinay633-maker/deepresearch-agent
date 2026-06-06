from __future__ import annotations

import json
import os
import re
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


class DeepSeekLLMProvider:
    name = "deepseek"
    supports_structured_output = True
    supports_tool_calling = True

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def create_brief(self, request: ResearchRequest, cost: CostTracker) -> ResearchBrief:
        raise NotImplementedError("DeepSeek brief generation is introduced after step 1.")

    async def plan(
        self, brief: ResearchBrief, max_researchers: int, cost: CostTracker
    ) -> list[SubQuestion]:
        del cost
        payload = await self._chat_json(
            stage="planning",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research planner. Return strict json only. "
                        "The json object must match this schema: "
                        '{"subquestions":[{"id":"Q1","question":"...","rationale":"..."}]}. '
                        "Do not include markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Create json for a deep research plan. "
                        f"Research brief: {brief.model_dump_json()}. "
                        f"Return exactly {max_researchers} subquestions. "
                        "Each id must be Q1, Q2, Q3... in order. "
                        "Each question must be concrete and searchable. "
                        "Each rationale must explain why this subquestion matters."
                    ),
                },
            ],
            max_tokens=1200,
        )
        items = payload.get("subquestions")
        if not isinstance(items, list):
            raise ValueError("DeepSeek JSON response missing list field: subquestions")
        subquestions = [SubQuestion.model_validate(item) for item in items]
        if len(subquestions) != max_researchers:
            raise ValueError(
                f"DeepSeek returned {len(subquestions)} subquestions; expected {max_researchers}"
            )
        return subquestions

    async def synthesize(
        self,
        brief: ResearchBrief,
        plan: list[SubQuestion],
        findings: list[Finding],
        sources: list[Source],
        cost: CostTracker,
    ) -> tuple[str, list[str]]:
        raise NotImplementedError("DeepSeek synthesis is introduced after step 1 passes.")

    async def _chat_json(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> dict:
        import asyncio

        return await asyncio.to_thread(self._chat_json_sync, stage, messages, max_tokens)

    def _chat_json_sync(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                content = self._post_chat_completions(messages, max_tokens=max_tokens)
                if not content.strip():
                    raise ValueError("DeepSeek returned empty content")
                return _parse_json_object(content)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"DeepSeek {stage} JSON validation failed: {last_error}") from last_error

    def _post_chat_completions(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is required")
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek HTTP {exc.code}: {_redact(error_body)}") from exc
        except URLError as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc

        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("DeepSeek response missing choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("DeepSeek response missing message.content")
        return content


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


def _parse_json_object(content: str) -> dict:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek JSON response is not an object")
    return parsed


def _redact(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", text)
