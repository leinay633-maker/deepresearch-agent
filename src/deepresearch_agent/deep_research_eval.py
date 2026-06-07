from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from deepresearch_agent.config import load_settings
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import ResearchRequest, StructuredReport

LIVE_DR_BENCH_DATASET = "microsoft/LiveDRBench"
LIVE_DR_BENCH_CONFIGS = {
    "livedrbench-preview": "preview",
    "livedrbench-v1-full": "v1-full",
}


async def run_public_deep_research_eval(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    logs_dir = root / "logs"
    results_dir = root / "results"
    logs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    cases = load_eval_cases(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_path = Path(args.raw_log) if args.raw_log else logs_dir / f"deep-research-eval-{timestamp}.jsonl"
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else results_dir / "deep_research_eval_summary.json"
    )
    predictions_path = (
        Path(args.predictions_output)
        if args.predictions_output
        else results_dir / "livedrbench_predictions.json"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

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
        request_timeout_seconds=args.request_timeout_seconds,
        embedding_provider=args.embedding_provider,
        local_retrieval_mode=args.local_retrieval_mode,
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
    config_snapshot = {
        "benchmark_name": args.benchmark_name,
        "dataset": args.dataset,
        "dataset_config": _dataset_config_name(args),
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "case_count": len(cases),
        "seed": args.seed,
        "llm_provider": effective_settings.llm_provider,
        "llm_model": effective_llm_model,
        "stage_models": stage_models,
        "search_provider": effective_settings.search_provider,
        "embedding_provider": effective_settings.embedding_provider,
        "local_retrieval_mode": effective_settings.local_retrieval_mode,
        "max_researchers": effective_settings.max_researchers,
        "max_results": args.max_results,
        "request_timeout_seconds": effective_settings.request_timeout_seconds,
        "web_crawler_provider": effective_settings.web_crawler_provider,
        "crawler_max_chars": effective_settings.crawler_max_chars,
        "reflection_enabled": reflection_enabled,
        "max_reflection_rounds": max_reflection_rounds,
        "reflection_min_sources": reflection_min_sources,
        "citation_judge_provider": effective_settings.citation_judge_provider,
        "citation_judge_model": effective_settings.citation_judge_model,
        "judge_provider": args.judge_provider,
        "official_judge_score": "not_run",
        "settings": {
            **asdict(effective_settings),
            "llm_model": effective_llm_model,
            "stage_models": stage_models,
            "max_results": args.max_results,
        },
    }

    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"type": "config", "config": config_snapshot}, ensure_ascii=False) + "\n")
        for case in cases:
            record = await _run_case(
                case,
                args,
                effective_settings,
                effective_llm_model,
                stage_models,
            )
            records.append(record)
            predictions.append(
                {
                    "key": case["id"],
                    "preds": _prediction_payload(record.get("answer", "")),
                }
            )
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summarize(records, config_snapshot, raw_path, predictions_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_eval_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.cases:
        cases = load_eval_cases_from_file(Path(args.cases), args.benchmark_name)
    else:
        cases = load_livedrbench_cases(
            dataset=args.dataset,
            split=args.split,
            offset=args.offset,
            limit=args.limit,
            timeout_seconds=args.request_timeout_seconds,
        )
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("no eval cases loaded")
    return cases


def load_eval_cases_from_file(path: Path, benchmark_name: str) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("rows", raw.get("cases", []))
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        raise ValueError(f"unsupported eval case file: {path}")
    return [_normalize_case(row, index, benchmark_name) for index, row in enumerate(rows)]


def load_livedrbench_cases(
    *,
    dataset: str,
    split: str,
    offset: int,
    limit: int | None,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    if dataset not in LIVE_DR_BENCH_CONFIGS:
        raise ValueError(f"unsupported remote dataset: {dataset}")
    length = limit if limit is not None else 10
    params = urllib.parse.urlencode(
        {
            "dataset": LIVE_DR_BENCH_DATASET,
            "config": LIVE_DR_BENCH_CONFIGS[dataset],
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.load(response)
    rows = [item["row"] for item in payload.get("rows", [])]
    return [_normalize_case(row, index + offset, dataset) for index, row in enumerate(rows)]


async def _run_case(
    case: dict[str, Any],
    args: argparse.Namespace,
    settings: Any,
    llm_model: str,
    stage_models: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    request = ResearchRequest(
        query=case["query"],
        max_researchers=settings.max_researchers,
        max_results_per_researcher=args.max_results,
        llm_provider=settings.llm_provider,
        llm_model=llm_model,
        brief_model=stage_models["brief_generation"] or None,
        planner_model=stage_models["planning"] or None,
        synthesis_model=stage_models["synthesis"] or None,
        search_provider=settings.search_provider,
        seed=args.seed,
        reflection_enabled=getattr(args, "reflection_enabled", False),
        max_reflection_rounds=getattr(args, "max_reflection_rounds", 1),
        reflection_min_sources=getattr(args, "reflection_min_sources", 4),
        citation_judge_provider=settings.citation_judge_provider,
        citation_judge_model=settings.citation_judge_model,
    )
    try:
        report = await DeepResearchOrchestrator(settings=settings).run(request)
        return _case_success_record(case, report)
    except Exception as exc:  # pragma: no cover - exercised by integration failures.
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return {
            "type": "case_result",
            "case_id": case["id"],
            "query": case["query"],
            "category": case.get("category"),
            "benchmark_name": case.get("benchmark_name"),
            "latency_ms": latency_ms,
            "success": False,
            "error": repr(exc),
            "answer": "",
            "claims": [],
            "sources": [],
            "citation_check": None,
            "cost": None,
            "metrics": {"latency_ms": latency_ms, "success": False},
            "trace_events": [],
            "metadata": case.get("metadata", {}),
        }


def _case_success_record(case: dict[str, Any], report: StructuredReport) -> dict[str, Any]:
    return {
        "type": "case_result",
        "case_id": case["id"],
        "query": case["query"],
        "category": case.get("category"),
        "benchmark_name": case.get("benchmark_name"),
        "latency_ms": report.metrics["latency_ms"],
        "total_tokens": report.cost.total_tokens,
        "estimated_cost_usd": report.cost.total_estimated_cost_usd,
        "deduped_source_count": report.metrics["deduped_source_count"],
        "raw_search_result_count": report.metrics["raw_search_result_count"],
        "citation_retention_rate": report.metrics["citation_retention_rate"],
        "fallback_count": report.metrics["fallback_count"],
        "success": report.metrics["success"],
        "run_id": report.run_id,
        "answer": report.answer,
        "claims": report.claims,
        "sources": [source.model_dump(mode="json") for source in report.sources],
        "citation_check": report.citation_check.model_dump(mode="json"),
        "cost": report.cost.model_dump(mode="json"),
        "metrics": report.metrics,
        "trace_events": [event.model_dump(mode="json") for event in report.trace_events],
        "metadata": case.get("metadata", {}),
    }


def _normalize_case(row: dict[str, Any], index: int, benchmark_name: str) -> dict[str, Any]:
    query = row.get("query") or row.get("question") or row.get("prompt") or row.get("input")
    if not query:
        raise ValueError(f"eval row {index} does not contain query/question/prompt/input")
    case_id = str(row.get("id") or row.get("case_id") or row.get("key") or f"case-{index + 1:04d}")
    metadata = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "id",
            "case_id",
            "key",
            "query",
            "question",
            "prompt",
            "input",
            "category",
        }
    }
    return {
        "id": case_id,
        "query": str(query),
        "category": row.get("category"),
        "benchmark_name": benchmark_name,
        "metadata": metadata,
    }


def _prediction_payload(answer: str) -> list[Any]:
    parsed = _extract_json(answer)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return [[answer]]


def _extract_json(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    for candidate in (stripped, *_json_spans(stripped)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_spans(text: str) -> list[str]:
    spans: list[str] = []
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            spans.append(text[start : end + 1])
    return spans


def _summarize(
    records: list[dict[str, Any]],
    config_snapshot: dict[str, Any],
    raw_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records]
    tokens = [record.get("total_tokens", 0) for record in records]
    retentions = [
        record.get("citation_retention_rate", 0.0)
        for record in records
        if record.get("citation_retention_rate") is not None
    ]
    source_counts = [record.get("deduped_source_count", 0) for record in records]
    success_count = sum(1 for record in records if record["success"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "public_deep_research_artifact_eval",
        "interpretation": (
            "This run executes full DeepResearch Agent reports on public deep research "
            "tasks and writes auditable artifacts. Official answer-quality scoring is "
            "not run unless a judge provider is configured."
        ),
        "limitations": [
            "without an official judge score, this is an artifact-producing evaluation run",
            "citation_retention_rate still uses the current lexical checker",
            "live search or LLM providers can vary across runs",
        ],
        "raw_log": str(raw_path),
        "predictions_output": str(predictions_path),
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
            sum(record.get("estimated_cost_usd", 0.0) for record in records), 8
        ),
        "citation_retention_rate_avg": round(sum(retentions) / len(retentions), 4)
        if retentions
        else 0.0,
        "deduped_source_count_avg": round(sum(source_counts) / len(source_counts), 3)
        if source_counts
        else 0.0,
        "fallback_count_total": sum(record.get("fallback_count", 0) for record in records),
        "records": [
            {
                "case_id": record["case_id"],
                "query": record["query"],
                "success": record["success"],
                "latency_ms": record["latency_ms"],
                "total_tokens": record.get("total_tokens", 0),
                "estimated_cost_usd": record.get("estimated_cost_usd", 0.0),
                "deduped_source_count": record.get("deduped_source_count", 0),
                "citation_retention_rate": record.get("citation_retention_rate", 0.0),
                "fallback_count": record.get("fallback_count", 0),
                "error": record.get("error"),
                "run_id": record.get("run_id"),
            }
            for record in records
        ],
    }


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


def _dataset_config_name(args: argparse.Namespace) -> str | None:
    if args.cases:
        return None
    return LIVE_DR_BENCH_CONFIGS.get(args.dataset)


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
    parser = argparse.ArgumentParser(
        description="Run full DeepResearch Agent reports on public deep research eval cases."
    )
    parser.add_argument("--cases", default=None, help="Optional JSONL/JSON/CSV eval cases file.")
    parser.add_argument(
        "--dataset",
        choices=sorted(LIVE_DR_BENCH_CONFIGS),
        default="livedrbench-preview",
        help="Remote public benchmark to fetch when --cases is not provided.",
    )
    parser.add_argument("--benchmark-name", default="public_deep_research_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
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
    parser.add_argument("--local-retrieval-mode", choices=["keyword", "hybrid"], default="keyword")
    parser.add_argument("--max-researchers", type=int, default=2)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--reflection-enabled", action="store_true")
    parser.add_argument("--max-reflection-rounds", type=int, default=1)
    parser.add_argument("--reflection-min-sources", type=int, default=4)
    parser.add_argument(
        "--citation-judge-provider",
        choices=["none", "heuristic", "deepseek"],
        default=None,
    )
    parser.add_argument("--citation-judge-model", default=None)
    parser.add_argument("--searxng-base-url", default=None)
    parser.add_argument("--web-crawler-provider", choices=["none", "jina", "jina_reader"], default=None)
    parser.add_argument("--jina-reader-base-url", default=None)
    parser.add_argument("--jina-search-base-url", default=None)
    parser.add_argument("--crawler-max-chars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--judge-provider",
        choices=["none"],
        default="none",
        help="Reserved for official/LLM judge scoring; currently writes artifacts only.",
    )
    parser.add_argument("--raw-log", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--predictions-output", default=None)
    args = parser.parse_args()
    summary = asyncio.run(run_public_deep_research_eval(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
