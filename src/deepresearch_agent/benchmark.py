from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from deepresearch_agent.config import load_settings
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cases_path = Path(args.cases) if args.cases else root / "data" / "benchmark_cases.jsonl"
    logs_dir = root / "logs"
    results_dir = root / "results"
    logs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    cases = _load_cases(cases_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = logs_dir / f"benchmark-{timestamp}.jsonl"
    summary_path = results_dir / "benchmark_summary.json"
    records = []

    settings = load_settings()
    config_snapshot = {
        "seed": args.seed,
        "search_provider": args.search_provider,
        "case_count": len(cases),
        "max_researchers": args.max_researchers,
        "max_results": args.max_results,
        "settings": settings.__dict__,
    }

    with raw_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"type": "config", "config": config_snapshot}, ensure_ascii=False) + "\n")
        for case in cases:
            request = ResearchRequest(
                query=case["query"],
                max_researchers=args.max_researchers,
                max_results_per_researcher=args.max_results,
                search_provider=args.search_provider,
                seed=args.seed,
            )
            report = await DeepResearchOrchestrator(settings=settings).run(request)
            record = {
                "type": "case_result",
                "case_id": case["id"],
                "query": case["query"],
                "latency_ms": report.metrics["latency_ms"],
                "total_tokens": report.cost.total_tokens,
                "estimated_cost_usd": report.cost.total_estimated_cost_usd,
                "deduped_source_count": report.metrics["deduped_source_count"],
                "raw_search_result_count": report.metrics["raw_search_result_count"],
                "citation_retention_rate": report.metrics["citation_retention_rate"],
                "success": report.metrics["success"],
                "fallback_count": report.metrics["fallback_count"],
                "output_summary": report.answer[:240],
                "run_id": report.run_id,
            }
            records.append(record)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = _summarize(records, config_snapshot, raw_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_cases(path: Path) -> list[dict[str, str]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def _summarize(
    records: list[dict[str, Any]], config_snapshot: dict[str, Any], raw_path: Path
) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records]
    tokens = [record["total_tokens"] for record in records]
    retentions = [record["citation_retention_rate"] for record in records]
    success_count = sum(1 for record in records if record["success"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_log": str(raw_path),
        "config": config_snapshot,
        "case_count": len(records),
        "success_count": success_count,
        "success_rate": round(success_count / len(records), 4) if records else 0.0,
        "latency_ms": {
            "p50": round(median(latencies), 3) if latencies else 0.0,
            "p90": round(_percentile(latencies, 90), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "tokens": {
            "avg": round(sum(tokens) / len(tokens), 3) if tokens else 0.0,
            "total": sum(tokens),
        },
        "estimated_cost_usd_total": round(
            sum(record["estimated_cost_usd"] for record in records), 8
        ),
        "citation_retention_rate_avg": round(sum(retentions) / len(retentions), 4)
        if retentions
        else 0.0,
        "fallback_count_total": sum(record["fallback_count"] for record in records),
        "records": records,
    }


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepResearch Agent benchmark.")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--search-provider", choices=["mock", "wikipedia"], default="mock")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--max-researchers", type=int, default=3)
    parser.add_argument("--max-results", type=int, default=4)
    args = parser.parse_args()
    summary = asyncio.run(run_benchmark(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
