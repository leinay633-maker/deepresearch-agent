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
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
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
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
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
    benchmark_kind, interpretation, limitations = _benchmark_notes(config_snapshot)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": benchmark_kind,
        "interpretation": interpretation,
        "limitations": limitations,
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


def _benchmark_notes(config_snapshot: dict[str, Any]) -> tuple[str, str, list[str]]:
    llm_provider = config_snapshot.get("llm_provider")
    search_provider = config_snapshot.get("search_provider")
    if llm_provider == "mock" and search_provider == "mock":
        return (
            "mock_plumbing_smoke_test",
            (
                "These numbers validate that the local pipeline can run end to end. "
                "They are not real DeepResearch performance, cost, or answer-quality metrics."
            ),
            [
                "latency_ms measures local Python execution with deterministic mock components",
                "total_tokens is an approximate character-count estimate, not provider tokenizer usage",
                "estimated_cost_usd is 0 because the mock provider price is configured as 0",
                "citation_retention_rate can be 1.0 because mock synthesis cites sources created inside the same pipeline",
            ],
        )
    return (
        "real_llm_live_search_benchmark",
        (
            "These numbers use the configured live LLM/search providers and are suitable "
            "as local benchmark evidence for this exact setup, not as a general product SLA."
        ),
        [
            "latency_ms includes live network/API time and can vary across runs",
            "DeepSeek token usage and cost come from provider usage fields when llm_provider is deepseek",
            "Wikipedia is a real no-key adapter but not a production-grade web search provider",
            "citation_retention_rate is checked by lexical overlap, not semantic entailment",
            "the benchmark set is small and local, so success_rate is not a broad quality score",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepResearch Agent benchmark.")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--search-provider", choices=["mock", "wikipedia"], default="mock")
    parser.add_argument("--llm-provider", choices=["mock", "deepseek"], default="mock")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--max-researchers", type=int, default=3)
    parser.add_argument("--max-results", type=int, default=4)
    args = parser.parse_args()
    summary = asyncio.run(run_benchmark(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
