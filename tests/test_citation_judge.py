from __future__ import annotations

import asyncio

from deepresearch_agent.citation import CitationChecker
from deepresearch_agent.citation_judge import (
    CitationJudgeResult,
    DeepSeekCitationJudgeProvider,
    HeuristicCitationJudgeProvider,
    build_citation_judge_provider,
)
from deepresearch_agent.config import Settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest, Source


class _FixedJudge:
    name = "fixed"
    model = "fixed-judge"

    def judge(self, claim, evidence_quotes):
        assert claim == "Redis queue recovery is implemented [S1]"
        assert evidence_quotes
        return CitationJudgeResult(
            verdict="unsupported",
            reason="the evidence mentions SQLite, not Redis",
            confidence=0.92,
            provider=self.name,
            model=self.model,
            input_tokens=11,
            output_tokens=3,
            estimated_cost_usd=0.000004,
        )


def test_citation_checker_applies_optional_judge_and_records_cost() -> None:
    source = Source(
        id="S1",
        title="Run control source",
        url="https://example.com/run",
        content="The run control plane uses SQLite leases for local worker recovery.",
        provider="mock",
        query="run control",
    )
    cost = CostTracker(provider="mock", model="mock")

    report = CitationChecker(min_overlap=0.01).check(
        ["Redis queue recovery is implemented [S1]"],
        [source],
        judge_provider=_FixedJudge(),
        cost=cost,
    )

    assessment = report.assessments[0]
    assert report.supported_claims == 0
    assert assessment.support_level == "unsupported"
    assert assessment.judge_provider == "fixed"
    assert assessment.judge_model == "fixed-judge"
    assert assessment.judge_confidence == 0.92
    assert cost.records[-1].stage == "citation_judge"
    assert cost.records[-1].provider == "fixed"
    assert cost.records[-1].model == "fixed-judge"


def test_heuristic_judge_provider_has_no_key_dependency() -> None:
    provider = build_citation_judge_provider(
        Settings(citation_judge_provider="heuristic")
    )

    assert isinstance(provider, HeuristicCitationJudgeProvider)
    result = provider.judge("No cited evidence [S1]", [])
    assert result.verdict == "unverifiable"
    assert result.input_tokens == 0
    assert result.estimated_cost_usd == 0.0


def test_deepseek_citation_judge_parses_verdict_and_usage(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = _RecordingDeepSeekCitationJudgeProvider(model="deepseek-v4-flash")
    source = Source(
        id="S1",
        title="Citation source",
        url="https://example.com/citation",
        content="Evidence quotes should directly support cited claims.",
        provider="mock",
        query="citation",
    )
    report = CitationChecker(min_overlap=0.01).check(
        ["Evidence quotes directly support cited claims [S1]"],
        [source],
        judge_provider=provider,
        cost=CostTracker(provider="mock", model="mock"),
    )

    assessment = report.assessments[0]
    assert provider.prompts and "Evidence quotes JSON" in provider.prompts[0]
    assert assessment.support_level == "supported"
    assert assessment.judge_provider == "deepseek"
    assert assessment.judge_model == "deepseek-v4-flash"
    assert assessment.judge_confidence == 0.87


def test_orchestrator_can_enable_heuristic_citation_judge() -> None:
    report = asyncio.run(
        DeepResearchOrchestrator(
            settings=Settings(
                local_retrieval_mode="keyword",
                citation_judge_provider="heuristic",
            )
        ).run(
            ResearchRequest(
                query="How should citation judges work?",
                llm_provider="mock",
                search_provider="mock",
                max_researchers=1,
                max_results_per_researcher=1,
            )
        )
    )

    assert report.metrics["success"] is True
    assert report.citation_check.assessments
    assert {item.judge_provider for item in report.citation_check.assessments} == {"heuristic"}


class _RecordingDeepSeekCitationJudgeProvider(DeepSeekCitationJudgeProvider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prompts: list[str] = []

    def _post_chat_completions(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"verdict":"supported",'
                            '"confidence":0.87,'
                            '"reason":"the quote directly supports the claim"}'
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
        }
