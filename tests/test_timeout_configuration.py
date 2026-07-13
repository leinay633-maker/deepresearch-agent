from __future__ import annotations

import pytest

from deepresearch_agent.citation_judge import build_citation_judge_provider
from deepresearch_agent.cli import _settings_from_args
from deepresearch_agent.config import Settings, with_request_timeout
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest
from deepresearch_agent.search import build_search_adapter


class _CliArgs:
    rerank_enabled = False

    def __init__(self, request_timeout_seconds: float | None) -> None:
        self.request_timeout_seconds = request_timeout_seconds

    def __getattr__(self, name: str):
        del name
        return None


def test_common_request_timeout_updates_all_gateway_and_judge_channels() -> None:
    settings = with_request_timeout(
        Settings(
            request_timeout_seconds=4.0,
            llm_gateway_timeout_seconds=120.0,
            citation_judge_timeout_seconds=30.0,
        ),
        17.5,
    )

    assert settings.request_timeout_seconds == 17.5
    assert settings.llm_gateway_timeout_seconds == 17.5
    assert settings.citation_judge_timeout_seconds == 17.5


def test_common_timeout_reaches_gateway_search_llm_and_citation_judge() -> None:
    settings = with_request_timeout(
        Settings(
            llm_provider="llm-gateway",
            llm_gateway_base_url="https://gateway.example",
            citation_judge_provider="llm-gateway",
        ),
        17.5,
    )

    search = build_search_adapter(settings, "gateway-web")
    llm = DeepResearchOrchestrator(settings=settings)._build_llm_provider(
        ResearchRequest(query="How are timeouts propagated?")
    )
    citation_judge = build_citation_judge_provider(settings)

    assert search.timeout_seconds == 17.5
    assert llm.timeout_seconds == 17.5
    assert llm.client.timeout_seconds == 17.5
    assert citation_judge is not None
    assert citation_judge.client.timeout_seconds == 17.5


def test_common_request_timeout_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        with_request_timeout(Settings(), 0)


def test_cli_common_timeout_uses_the_same_effective_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "deepresearch_agent.cli.load_settings",
        lambda: Settings(
            request_timeout_seconds=4.0,
            llm_gateway_timeout_seconds=120.0,
            citation_judge_timeout_seconds=30.0,
        ),
    )

    settings = _settings_from_args(_CliArgs(23.0))

    assert settings.request_timeout_seconds == 23.0
    assert settings.llm_gateway_timeout_seconds == 23.0
    assert settings.citation_judge_timeout_seconds == 23.0
