from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, replace
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
    effective_llm_model = _effective_llm_model(args, settings)
    effective_llm_provider = _normalize_llm_provider(args.llm_provider)
    stage_models = _effective_stage_models(args, settings)
    reflection_enabled = getattr(args, "reflection_enabled", False)
    max_reflection_rounds = getattr(args, "max_reflection_rounds", 1)
    reflection_min_sources = getattr(args, "reflection_min_sources", 4)
    effective_settings = replace(
        settings,
        llm_provider=effective_llm_provider,
        search_provider=args.search_provider,
        max_researchers=args.max_researchers,
        embedding_provider=args.embedding_provider,
        local_retrieval_mode=args.local_retrieval_mode,
        local_keyword_top_k=args.local_keyword_top_k,
        local_vector_top_k=args.local_vector_top_k,
        local_keyword_weight=args.local_keyword_weight,
        local_vector_weight=args.local_vector_weight,
        local_hybrid_rrf_k=args.local_hybrid_rrf_k,
        rerank_enabled=args.rerank_enabled,
        rerank_provider=args.rerank_provider,
        local_rerank_candidate_k=args.local_rerank_candidate_k,
        searxng_base_url=getattr(args, "searxng_base_url", None) or settings.searxng_base_url,
        web_crawler_provider=getattr(args, "web_crawler_provider", None)
        or settings.web_crawler_provider,
        jina_reader_base_url=getattr(args, "jina_reader_base_url", None)
        or settings.jina_reader_base_url,
        jina_search_base_url=getattr(args, "jina_search_base_url", None)
        or settings.jina_search_base_url,
        crawler_max_chars=getattr(args, "crawler_max_chars", None) or settings.crawler_max_chars,
        deepseek_model=effective_llm_model
        if effective_llm_provider == "deepseek"
        else settings.deepseek_model,
        openai_compatible_model=effective_llm_model
        if effective_llm_provider == "openai-compatible"
        else settings.openai_compatible_model,
        llm_brief_model=stage_models["brief_generation"] or settings.llm_brief_model,
        llm_planner_model=stage_models["planning"] or settings.llm_planner_model,
        llm_synthesis_model=stage_models["synthesis"] or settings.llm_synthesis_model,
        citation_judge_provider=getattr(args, "citation_judge_provider", None)
        or settings.citation_judge_provider,
        citation_judge_model=getattr(args, "citation_judge_model", None)
        or settings.citation_judge_model,
    )
    settings_snapshot = asdict(effective_settings)
    settings_snapshot["llm_model"] = effective_llm_model
    settings_snapshot["stage_models"] = stage_models
    settings_snapshot["max_results"] = args.max_results
    config_snapshot = {
        "seed": args.seed,
        "llm_provider": effective_settings.llm_provider,
        "llm_model": effective_llm_model,
        "stage_models": stage_models,
        "search_provider": effective_settings.search_provider,
        "embedding_provider": effective_settings.embedding_provider,
        "local_retrieval_mode": effective_settings.local_retrieval_mode,
        "local_keyword_top_k": effective_settings.local_keyword_top_k,
        "local_vector_top_k": effective_settings.local_vector_top_k,
        "local_keyword_weight": effective_settings.local_keyword_weight,
        "local_vector_weight": effective_settings.local_vector_weight,
        "local_hybrid_rrf_k": effective_settings.local_hybrid_rrf_k,
        "rerank_enabled": effective_settings.rerank_enabled,
        "rerank_provider": effective_settings.rerank_provider,
        "local_rerank_candidate_k": effective_settings.local_rerank_candidate_k,
        "web_crawler_provider": effective_settings.web_crawler_provider,
        "crawler_max_chars": effective_settings.crawler_max_chars,
        "case_count": len(cases),
        "max_researchers": effective_settings.max_researchers,
        "max_results": args.max_results,
        "request_timeout_seconds": effective_settings.request_timeout_seconds,
        "settings": settings_snapshot,
        "reflection_enabled": reflection_enabled,
        "max_reflection_rounds": max_reflection_rounds,
        "reflection_min_sources": reflection_min_sources,
        "citation_judge_provider": effective_settings.citation_judge_provider,
        "citation_judge_model": effective_settings.citation_judge_model,
    }

    with raw_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"type": "config", "config": config_snapshot}, ensure_ascii=False) + "\n")
        for case in cases:
            request = ResearchRequest(
                query=case["query"],
                max_researchers=effective_settings.max_researchers,
                max_results_per_researcher=args.max_results,
                llm_provider=effective_settings.llm_provider,
                llm_model=effective_llm_model,
                brief_model=stage_models["brief_generation"] or None,
                planner_model=stage_models["planning"] or None,
                synthesis_model=stage_models["synthesis"] or None,
                search_provider=effective_settings.search_provider,
                seed=args.seed,
                reflection_enabled=reflection_enabled,
                max_reflection_rounds=max_reflection_rounds,
                reflection_min_sources=reflection_min_sources,
                citation_judge_provider=effective_settings.citation_judge_provider,
                citation_judge_model=effective_settings.citation_judge_model,
            )
            report = await DeepResearchOrchestrator(settings=effective_settings).run(request)
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


def _effective_llm_model(args: argparse.Namespace, settings: Any) -> str:
    if args.llm_model:
        return args.llm_model
    provider = _normalize_llm_provider(args.llm_provider)
    if provider == "deepseek":
        return settings.deepseek_model
    if provider == "openai-compatible":
        return settings.openai_compatible_model
    return settings.mock_model_name


def _normalize_llm_provider(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _effective_stage_models(args: argparse.Namespace, settings: Any) -> dict[str, str]:
    return {
        "brief_generation": getattr(args, "brief_model", None) or settings.llm_brief_model,
        "planning": getattr(args, "planner_model", None) or settings.llm_planner_model,
        "synthesis": getattr(args, "synthesis_model", None) or settings.llm_synthesis_model,
    }


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
    parser.add_argument(
        "--search-provider",
        choices=["mock", "wikipedia", "searxng", "jina", "brave", "tavily", "mcp"],
        default="mock",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["mock", "deepseek", "openai-compatible", "openai_compatible"],
        default="mock",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--brief-model", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--synthesis-model", default=None)
    parser.add_argument("--embedding-provider", choices=["local", "dashscope"], default="local")
    parser.add_argument("--local-retrieval-mode", choices=["keyword", "hybrid"], default="hybrid")
    parser.add_argument("--local-keyword-top-k", type=int, default=4)
    parser.add_argument("--local-vector-top-k", type=int, default=4)
    parser.add_argument("--local-keyword-weight", type=float, default=1.0)
    parser.add_argument("--local-vector-weight", type=float, default=1.0)
    parser.add_argument("--local-hybrid-rrf-k", type=int, default=60)
    parser.add_argument("--rerank-enabled", action="store_true")
    parser.add_argument("--rerank-provider", choices=["local", "dashscope"], default="local")
    parser.add_argument("--local-rerank-candidate-k", type=int, default=6)
    parser.add_argument("--searxng-base-url", default=None)
    parser.add_argument(
        "--web-crawler-provider",
        choices=["none", "jina", "jina_reader", "html"],
        default=None,
    )
    parser.add_argument("--jina-reader-base-url", default=None)
    parser.add_argument("--jina-search-base-url", default=None)
    parser.add_argument("--crawler-max-chars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--max-researchers", type=int, default=3)
    parser.add_argument("--max-results", type=int, default=4)
    parser.add_argument("--reflection-enabled", action="store_true")
    parser.add_argument("--max-reflection-rounds", type=int, default=1)
    parser.add_argument("--reflection-min-sources", type=int, default=4)
    parser.add_argument(
        "--citation-judge-provider",
        choices=["none", "heuristic", "deepseek"],
        default=None,
    )
    parser.add_argument("--citation-judge-model", default=None)
    args = parser.parse_args()
    summary = asyncio.run(run_benchmark(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
