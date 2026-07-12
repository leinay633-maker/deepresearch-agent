from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace

from deepresearch_agent.config import load_settings
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest
from deepresearch_agent.search import MockSearchAdapter, SearchError, SearchService


class ForcedFailureSearchAdapter:
    """Deterministic failure fixture for the five-minute reliability demo."""

    name = "demo_failure"

    async def search(self, query: str, max_results: int, timeout: float):
        del query, max_results, timeout
        raise SearchError("demo forced timeout")


async def run_demo(query: str, scenario: str = "normal") -> int:
    settings = replace(
        load_settings(),
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="keyword",
    )
    search_service = None
    fallback_policy = "mock"
    if scenario == "failure":
        fallback_policy = "degraded"
        search_service = SearchService(
            primary=ForcedFailureSearchAdapter(),
            fallback=MockSearchAdapter(),
            settings=settings,
            fallback_policy=fallback_policy,
        )
    report = await DeepResearchOrchestrator(
        settings=settings,
        search_service=search_service,
    ).run(
        ResearchRequest(
            query=query,
            llm_provider="mock",
            search_provider="mock",
            max_researchers=1,
            max_results_per_researcher=1,
            fallback_policy=fallback_policy,
        )
    )
    print(
        json.dumps(
            {
                "scenario": scenario,
                "run_id": report.run_id,
                "execution_success": report.metrics.get("execution_success"),
                "legacy_report_success": report.metrics.get("legacy_report_success"),
                "source_count": len(report.sources),
                "fallback_count": report.metrics.get("fallback_count"),
                "degraded_count": report.metrics.get("degraded_count"),
                "citation_retention_rate": report.citation_check.retention_rate,
                "citation_coverage": report.metrics.get("citation_coverage"),
                "total_tokens": report.cost.total_tokens,
                "trace": [
                    {
                        "stage": event.stage,
                        "status": event.status,
                        "degraded": bool(event.payload.get("degraded")),
                    }
                    for event in report.trace_events
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.metrics.get("execution_success") else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cross-platform offline demo.")
    parser.add_argument(
        "query",
        nargs="?",
        default="How should a research agent recover from tool failures?",
    )
    parser.add_argument(
        "--scenario",
        choices=["normal", "failure"],
        default="normal",
        help="normal report or deterministic forced-search-failure/degraded demo",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_demo(args.query, args.scenario)))


if __name__ == "__main__":
    main()
