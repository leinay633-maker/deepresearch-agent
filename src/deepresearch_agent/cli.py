from __future__ import annotations

import argparse
import asyncio
import json

from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest


async def _run(args: argparse.Namespace) -> int:
    request = ResearchRequest(
        query=args.query,
        max_researchers=args.max_researchers,
        max_results_per_researcher=args.max_results,
        search_provider=args.search_provider,
        seed=args.seed,
    )
    report = await DeepResearchOrchestrator().run(request)
    if args.json:
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print(report.answer)
        print("")
        print("Metrics:")
        print(json.dumps(report.metrics, ensure_ascii=False, indent=2))
        print("")
        print("Cost:")
        print(json.dumps(report.cost.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if report.metrics.get("success") else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local DeepResearch Agent query.")
    parser.add_argument("query")
    parser.add_argument("--search-provider", choices=["mock", "wikipedia"], default=None)
    parser.add_argument("--max-researchers", type=int, default=3)
    parser.add_argument("--max-results", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
