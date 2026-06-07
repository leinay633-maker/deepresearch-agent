from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace

from deepresearch_agent.config import load_settings
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.report_exporter import export_report
from deepresearch_agent.schemas import ResearchRequest


async def _run(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    request = ResearchRequest(
        query=args.query,
        max_researchers=args.max_researchers,
        max_results_per_researcher=args.max_results,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        search_provider=args.search_provider,
        brief_model=args.brief_model,
        planner_model=args.planner_model,
        synthesis_model=args.synthesis_model,
        seed=args.seed,
        reflection_enabled=args.reflection_enabled,
        max_reflection_rounds=args.max_reflection_rounds,
        reflection_min_sources=args.reflection_min_sources,
        citation_judge_provider=args.citation_judge_provider,
        citation_judge_model=args.citation_judge_model,
    )
    report = await DeepResearchOrchestrator(settings=settings).run(request)
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
    if args.export_dir:
        exported = export_report(report, args.export_dir, _export_formats(args.export_formats))
        stream = sys.stderr if args.json else sys.stdout
        print("", file=stream)
        print("Exports:", file=stream)
        print(json.dumps(exported, ensure_ascii=False, indent=2), file=stream)
    return 0 if report.metrics.get("success") else 1


def _settings_from_args(args: argparse.Namespace):
    settings = load_settings()
    overrides = {}
    for argument, field in [
        ("embedding_provider", "embedding_provider"),
        ("local_retrieval_mode", "local_retrieval_mode"),
        ("local_keyword_top_k", "local_keyword_top_k"),
        ("local_vector_top_k", "local_vector_top_k"),
        ("local_keyword_weight", "local_keyword_weight"),
        ("local_vector_weight", "local_vector_weight"),
        ("local_hybrid_rrf_k", "local_hybrid_rrf_k"),
        ("local_vector_index_provider", "local_vector_index_provider"),
        ("qdrant_base_url", "qdrant_base_url"),
        ("qdrant_collection", "qdrant_collection"),
        ("qdrant_api_key_env", "qdrant_api_key_env"),
        ("rerank_provider", "rerank_provider"),
        ("local_rerank_candidate_k", "local_rerank_candidate_k"),
        ("searxng_base_url", "searxng_base_url"),
        ("web_crawler_provider", "web_crawler_provider"),
        ("jina_reader_base_url", "jina_reader_base_url"),
        ("jina_search_base_url", "jina_search_base_url"),
        ("crawler_max_chars", "crawler_max_chars"),
    ]:
        value = getattr(args, argument)
        if value is not None:
            overrides[field] = value
    if args.rerank_enabled:
        overrides["rerank_enabled"] = True
    return replace(settings, **overrides) if overrides else settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local DeepResearch Agent query.")
    parser.add_argument("query")
    parser.add_argument(
        "--search-provider",
        choices=["mock", "wikipedia", "searxng", "jina", "brave", "tavily", "mcp"],
        default=None,
    )
    parser.add_argument(
        "--llm-provider",
        choices=["mock", "deepseek", "openai-compatible", "openai_compatible"],
        default=None,
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--brief-model", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--synthesis-model", default=None)
    parser.add_argument("--embedding-provider", choices=["local", "dashscope"], default=None)
    parser.add_argument("--local-retrieval-mode", choices=["keyword", "hybrid"], default=None)
    parser.add_argument("--local-keyword-top-k", type=int, default=None)
    parser.add_argument("--local-vector-top-k", type=int, default=None)
    parser.add_argument("--local-keyword-weight", type=float, default=None)
    parser.add_argument("--local-vector-weight", type=float, default=None)
    parser.add_argument("--local-hybrid-rrf-k", type=int, default=None)
    parser.add_argument(
        "--local-vector-index-provider",
        choices=["chroma", "qdrant"],
        default=None,
    )
    parser.add_argument("--qdrant-base-url", default=None)
    parser.add_argument("--qdrant-collection", default=None)
    parser.add_argument("--qdrant-api-key-env", default=None)
    parser.add_argument("--rerank-enabled", action="store_true")
    parser.add_argument("--rerank-provider", choices=["local", "dashscope"], default=None)
    parser.add_argument("--local-rerank-candidate-k", type=int, default=None)
    parser.add_argument("--searxng-base-url", default=None)
    parser.add_argument(
        "--web-crawler-provider",
        choices=["none", "jina", "jina_reader", "html"],
        default=None,
    )
    parser.add_argument("--jina-reader-base-url", default=None)
    parser.add_argument("--jina-search-base-url", default=None)
    parser.add_argument("--crawler-max-chars", type=int, default=None)
    parser.add_argument("--max-researchers", type=int, default=3)
    parser.add_argument("--max-results", type=int, default=4)
    parser.add_argument("--reflection-enabled", action="store_true")
    parser.add_argument("--max-reflection-rounds", type=int, default=1)
    parser.add_argument("--reflection-min-sources", type=int, default=4)
    parser.add_argument(
        "--citation-judge-provider",
        default=None,
        help="Optional citation judge provider: none, heuristic, deepseek.",
    )
    parser.add_argument("--citation-judge-model", default=None)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument(
        "--export-formats",
        default="markdown,html,json",
        help="Comma-separated report export formats: markdown, html, json, pdf, docx, pptx, wav.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


def _export_formats(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    main()
