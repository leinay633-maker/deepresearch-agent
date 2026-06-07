from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepresearch_agent.cost import CostTracker, deepseek_usage_cost_usd
from deepresearch_agent.schemas import Finding, ResearchBrief, ResearchRequest, Source, SubQuestion

DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"


@dataclass(frozen=True)
class LLMJsonResult:
    parsed: dict
    content: str
    usage: dict
    model: str


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

    def __init__(
        self,
        model: str = "mock-structured-tool-model",
        stage_models: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.stage_models = stage_models or {}

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
        cost.add(
            "brief_generation",
            request.query,
            brief.model_dump_json(),
            model=self._model_for_stage("brief_generation"),
        )
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
        cost.add(
            "planning",
            brief.model_dump_json(),
            json.dumps([q.model_dump() for q in questions]),
            model=self._model_for_stage("planning"),
        )
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
            model=self._model_for_stage("synthesis"),
        )
        return answer, claims

    def _model_for_stage(self, stage: str) -> str:
        return self.stage_models.get(stage) or self.model


class DeepSeekLLMProvider:
    name = "deepseek"
    supports_structured_output = True
    supports_tool_calling = True

    def __init__(
        self,
        model: str = DEEPSEEK_DEFAULT_MODEL,
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        stage_models: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.stage_models = stage_models or {}

    async def create_brief(self, request: ResearchRequest, cost: CostTracker) -> ResearchBrief:
        result = await self._chat_json_result(
            stage="brief_generation",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You normalize research requests. Return strict json only. "
                        "The json object must match this schema: "
                        '{"normalized_query":"...","scope":"...","constraints":["..."],"assumptions":["..."]}. '
                        "Do not include markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Create json for a DeepResearch brief. "
                        f"User query: {request.query!r}. "
                        "Keep the normalized query close to the user intent, make constraints actionable, "
                        "and include assumptions only when needed."
                    ),
                },
            ],
            max_tokens=1200,
        )
        payload = result.parsed
        brief = ResearchBrief(
            original_query=request.query,
            normalized_query=str(payload["normalized_query"]).strip(),
            scope=str(payload["scope"]).strip(),
            constraints=[str(item).strip() for item in payload.get("constraints", [])],
            assumptions=[str(item).strip() for item in payload.get("assumptions", [])],
        )
        self._add_usage_cost(cost, "brief_generation", result)
        return brief

    async def plan(
        self, brief: ResearchBrief, max_researchers: int, cost: CostTracker
    ) -> list[SubQuestion]:
        result = await self._chat_json_result(
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
        payload = result.parsed
        items = payload.get("subquestions")
        if not isinstance(items, list):
            raise ValueError("DeepSeek JSON response missing list field: subquestions")
        subquestions = [SubQuestion.model_validate(item) for item in items]
        if len(subquestions) != max_researchers:
            raise ValueError(
                f"DeepSeek returned {len(subquestions)} subquestions; expected {max_researchers}"
            )
        self._add_usage_cost(cost, "planning", result)
        return subquestions

    async def synthesize(
        self,
        brief: ResearchBrief,
        plan: list[SubQuestion],
        findings: list[Finding],
        sources: list[Source],
        cost: CostTracker,
    ) -> tuple[str, list[str]]:
        compact_findings = [
            {
                "subquestion_id": finding.subquestion_id,
                "subquestion": finding.subquestion,
                "summary": finding.summary,
                "source_ids": finding.source_ids,
            }
            for finding in findings
        ]
        compact_sources = [
            {
                "id": source.id,
                "title": source.title,
                "url": source.url,
                "content": source.content[:900],
                "provider": source.provider,
                "quality_score": source.quality_score,
            }
            for source in sources
        ]
        synthesis_input = {
            "brief": brief.model_dump(mode="json"),
            "plan": [item.model_dump() for item in plan],
            "findings": compact_findings,
            "sources": compact_sources,
        }
        result = await self._chat_json_result(
            stage="synthesis",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise DeepResearch reports. Return strict json only. "
                        "The json object must match this schema: "
                        '{"answer":"markdown report with citations like [S1]","claims":["claim text [S1]"]}. '
                        "Every factual claim must cite one or more supplied source IDs. "
                        "Use only source IDs present in the input json. Do not invent citations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Write json for the final report from this research context. "
                        "Answer in English unless the query is Chinese. "
                        "Make the answer specific to the evidence and call out limitations. "
                        f"Research context json: {json.dumps(synthesis_input, ensure_ascii=False)}"
                    ),
                },
            ],
            max_tokens=2600,
        )
        payload = result.parsed
        answer = str(payload.get("answer", "")).strip()
        claims_raw = payload.get("claims")
        if not answer:
            raise ValueError("DeepSeek synthesis response missing answer")
        if not isinstance(claims_raw, list) or not claims_raw:
            raise ValueError("DeepSeek synthesis response missing non-empty claims list")
        claims = [str(item).strip() for item in claims_raw if str(item).strip()]
        self._add_usage_cost(cost, "synthesis", result)
        return answer, claims

    async def _chat_json(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> dict:
        import asyncio

        result = await self._chat_json_result(stage, messages, max_tokens)
        return result.parsed

    async def _chat_json_result(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> LLMJsonResult:
        import asyncio

        return await asyncio.to_thread(self._chat_json_sync, stage, messages, max_tokens)

    def _chat_json_sync(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> LLMJsonResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                model = self._model_for_stage(stage)
                payload = self._post_chat_completions(
                    messages,
                    max_tokens=max_tokens,
                    model=model,
                )
                content = _extract_content(payload)
                if not content.strip():
                    raise ValueError("DeepSeek returned empty content")
                return LLMJsonResult(
                    parsed=_parse_json_object(content),
                    content=content,
                    usage=payload.get("usage") or {},
                    model=model,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"DeepSeek {stage} JSON validation failed: {last_error}") from last_error

    def _post_chat_completions(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        model: str,
    ) -> dict:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is required")
        body = json.dumps(
            {
                "model": model,
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

        _extract_content(payload)
        return payload

    def _model_for_stage(self, stage: str) -> str:
        return self.stage_models.get(stage) or self.model

    def _add_usage_cost(
        self, cost: CostTracker, stage: str, result: LLMJsonResult
    ):
        input_tokens, output_tokens, estimated_cost = deepseek_usage_cost_usd(
            result.model, result.usage
        )
        return cost.add_usage(
            stage=stage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            model=result.model,
        )


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


def _extract_content(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("DeepSeek response missing choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("DeepSeek response missing message.content")
    return content
