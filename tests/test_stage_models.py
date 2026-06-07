from __future__ import annotations

import asyncio

from deepresearch_agent.cost import CostTracker
from deepresearch_agent.config import Settings
from deepresearch_agent.llm import DeepSeekLLMProvider, OpenAICompatibleLLMProvider
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import Finding, ResearchBrief, ResearchRequest, Source, SubQuestion


def test_mock_orchestrator_records_stage_specific_models() -> None:
    report = asyncio.run(
        DeepResearchOrchestrator(settings=Settings(local_retrieval_mode="keyword")).run(
            ResearchRequest(
                query="How should stage specific models be recorded?",
                search_provider="mock",
                llm_provider="mock",
                llm_model="mock-default",
                brief_model="mock-brief",
                planner_model="mock-planner",
                synthesis_model="mock-synthesis",
                max_researchers=1,
                max_results_per_researcher=1,
            )
        )
    )
    models_by_stage = {record.stage: record.model for record in report.cost.records}

    assert models_by_stage["brief_generation"] == "mock-brief"
    assert models_by_stage["planning"] == "mock-planner"
    assert models_by_stage["synthesis"] == "mock-synthesis"


def test_deepseek_provider_sends_stage_specific_models(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = RecordingDeepSeekProvider(
        model="deepseek-v4-flash",
        stage_models={
            "brief_generation": "deepseek-chat",
            "planning": "deepseek-reasoner",
            "synthesis": "deepseek-v4-flash",
        },
    )
    cost = CostTracker(provider=provider.name, model=provider.model)
    brief = asyncio.run(provider.create_brief(ResearchRequest(query="What is staged model routing?"), cost))
    plan = asyncio.run(provider.plan(brief, max_researchers=1, cost=cost))
    source = Source(
        id="S1",
        title="Stage source",
        url="https://example.com/stage",
        content="Stage-specific model routing keeps role choices visible in cost records.",
        provider="mock",
        query="stage routing",
    )
    finding = Finding(
        subquestion_id="Q1",
        subquestion=plan[0].question,
        summary="Stage-specific model routing is visible.",
        source_ids=["S1"],
        sources=[source],
    )
    asyncio.run(provider.synthesize(brief, plan, [finding], [source], cost))

    assert provider.requested_models == [
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-v4-flash",
    ]
    assert [record.model for record in cost.records] == provider.requested_models


def test_openai_compatible_provider_records_usage_with_configured_cost() -> None:
    provider = RecordingOpenAICompatibleProvider(
        model="local-model",
        base_url="http://localhost:11434/v1",
        stage_models={
            "brief_generation": "local-brief",
            "planning": "local-planner",
            "synthesis": "local-synthesis",
        },
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)
    brief = asyncio.run(provider.create_brief(ResearchRequest(query="What is local model routing?"), cost))
    plan = asyncio.run(provider.plan(brief, max_researchers=1, cost=cost))
    source = Source(
        id="S1",
        title="Local source",
        url="https://example.com/local",
        content="OpenAI-compatible providers can be backed by local or hosted endpoints.",
        provider="mock",
        query="local routing",
    )
    finding = Finding(
        subquestion_id="Q1",
        subquestion=plan[0].question,
        summary="OpenAI-compatible providers use the configured endpoint.",
        source_ids=["S1"],
        sources=[source],
    )
    asyncio.run(provider.synthesize(brief, plan, [finding], [source], cost))

    assert provider.requested_models == ["local-brief", "local-planner", "local-synthesis"]
    assert [record.provider for record in cost.records] == ["openai_compatible"] * 3
    assert [record.model for record in cost.records] == provider.requested_models
    assert cost.summary().total_estimated_cost_usd == 0.00042


def test_orchestrator_builds_openai_compatible_provider() -> None:
    orchestrator = DeepResearchOrchestrator(
        settings=Settings(
            llm_provider="openai-compatible",
            openai_compatible_model="local-default",
            openai_compatible_base_url="http://localhost:11434/v1",
        )
    )

    provider = orchestrator._build_llm_provider(
        ResearchRequest(query="How should generic providers work?")
    )

    assert isinstance(provider, OpenAICompatibleLLMProvider)
    assert provider.model == "local-default"
    assert provider.base_url == "http://localhost:11434/v1"


class RecordingDeepSeekProvider(DeepSeekLLMProvider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.requested_models: list[str] = []

    def _post_chat_completions(self, messages, max_tokens: int, model: str) -> dict:
        del messages, max_tokens
        self.requested_models.append(model)
        if len(self.requested_models) == 1:
            content = (
                '{"normalized_query":"What is staged model routing?",'
                '"scope":"Verify stage routing.",'
                '"constraints":["Use configured stage models."],'
                '"assumptions":[]}'
            )
        elif len(self.requested_models) == 2:
            content = (
                '{"subquestions":[{"id":"Q1",'
                '"question":"How is stage model routing represented?",'
                '"rationale":"Check planner record fidelity."}]}'
            )
        else:
            content = (
                '{"answer":"Stage-specific routing is represented in cost records [S1]",'
                '"claims":["Stage-specific routing is represented in cost records [S1]"]}'
            )
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
        }


class RecordingOpenAICompatibleProvider(OpenAICompatibleLLMProvider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.requested_models: list[str] = []

    def _post_chat_completions(self, messages, max_tokens: int, model: str) -> dict:
        del messages, max_tokens
        self.requested_models.append(model)
        if len(self.requested_models) == 1:
            content = (
                '{"normalized_query":"What is local model routing?",'
                '"scope":"Verify OpenAI-compatible routing.",'
                '"constraints":["Use configured endpoint."],'
                '"assumptions":[]}'
            )
        elif len(self.requested_models) == 2:
            content = (
                '{"subquestions":[{"id":"Q1",'
                '"question":"How is OpenAI-compatible routing represented?",'
                '"rationale":"Check generic provider record fidelity."}]}'
            )
        else:
            content = (
                '{"answer":"OpenAI-compatible routing uses configured endpoints [S1]",'
                '"claims":["OpenAI-compatible routing uses configured endpoints [S1]"]}'
            )
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
