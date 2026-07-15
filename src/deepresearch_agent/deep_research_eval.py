from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
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

from deepresearch_agent.benchmark import (
    SUCCESS_SEMANTICS,
    attach_answer_quality,
    build_benchmark_manifest,
    build_case_evaluation_metrics,
    evaluation_summary,
    mark_live_judge_nondeterminism,
    portable_artifact_path,
    refresh_replayed_case_result,
    require_clean_worktree,
    sanitized_settings_snapshot,
)
from deepresearch_agent.config import load_settings, with_request_timeout
from deepresearch_agent.eval_judge import build_eval_judge_provider
from deepresearch_agent.llm_gateway import response_model_matches
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.search import BenchmarkContaminationError
from deepresearch_agent.replay import (
    load_case_result_records,
    replay_case_result,
    validate_replay_case_ids,
)
from deepresearch_agent.schemas import ResearchRequest, StructuredReport

LIVE_DR_BENCH_DATASET = "microsoft/LiveDRBench"
LIVE_DR_BENCH_CONFIGS = {
    "livedrbench-preview": "preview",
    "livedrbench-v1-full": "v1-full",
}

_SEALED_JUDGE_VERDICTS = frozenset(
    {"correct", "incorrect", "not_attempted", "unscored"}
)
_SEALED_FAILURE_CATEGORIES = frozenset(
    {
        "retrieval",
        "ranking_context",
        "planning",
        "evidence_extraction",
        "reasoning",
        "citation_mismatch",
        "source_quality",
        "format",
        "hallucination",
        "abstention",
        "tool_failure",
        "judge_uncertainty",
    }
)
_SEALED_ERROR_CATEGORIES = frozenset({"benchmark_contamination"})


async def run_public_deep_research_eval(args: argparse.Namespace) -> dict[str, Any]:
    sealed_holdout = bool(getattr(args, "sealed_holdout", False))
    _validate_sealed_holdout_args(args, sealed_holdout=sealed_holdout)
    settings = with_request_timeout(
        load_settings(),
        getattr(args, "request_timeout_seconds", None),
    )
    synthesis_timeout_seconds = getattr(args, "synthesis_timeout_seconds", None)
    if synthesis_timeout_seconds is not None:
        synthesis_timeout_seconds = float(synthesis_timeout_seconds)
        if synthesis_timeout_seconds <= 0:
            raise ValueError("synthesis timeout must be positive")
        settings = replace(
            settings,
            llm_synthesis_timeout_seconds=synthesis_timeout_seconds,
        )
    _validate_single_model_run_args(args, settings=settings)
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    logs_dir = root / "logs"
    if not sealed_holdout:
        logs_dir.mkdir(exist_ok=True)

    replay_records = (
        load_case_result_records(args.replay_dir)
        if getattr(args, "replay_dir", None)
        else None
    )
    if replay_records is not None and not args.cases:
        cases = _cases_from_replay_records(replay_records, args.benchmark_name)
    else:
        cases = load_eval_cases(args)
    if replay_records is not None:
        validate_replay_case_ids(cases, replay_records)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_path = (
        None
        if sealed_holdout
        else Path(args.raw_log)
        if args.raw_log
        else logs_dir / f"deep-research-eval-{timestamp}.jsonl"
    )
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else results_dir / "deep_research_eval_summary.json"
    )
    predictions_path = (
        None
        if sealed_holdout
        else (
            Path(args.predictions_output)
            if args.predictions_output
            else results_dir / "livedrbench_predictions.json"
        )
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)

    effective_llm_model = _effective_llm_model(args, settings)
    effective_llm_provider = _normalize_llm_provider(args.llm_provider)
    stage_models = _effective_stage_models(args, settings)
    reflection_enabled = getattr(args, "reflection_enabled", False)
    max_reflection_rounds = getattr(args, "max_reflection_rounds", 1)
    reflection_min_sources = getattr(args, "reflection_min_sources", 4)
    max_rounds = getattr(args, "max_rounds", 1)
    max_tool_calls = getattr(args, "max_tool_calls", 1)
    deadline_seconds = getattr(args, "deadline_seconds", None)
    min_evidence_items = getattr(args, "min_evidence_items", 1)
    fallback_policy = getattr(args, "fallback_policy", "fail")
    effective_settings = replace(
        settings,
        trace_write_enabled=False if sealed_holdout else settings.trace_write_enabled,
        trace_exporter="none" if sealed_holdout else settings.trace_exporter,
        benchmark_source_exclusion=True,
        llm_provider=effective_llm_provider,
        search_provider=args.search_provider,
        max_researchers=args.max_researchers,
        embedding_provider=args.embedding_provider,
        local_retrieval_mode=args.local_retrieval_mode,
        searxng_base_url=getattr(args, "searxng_base_url", None) or settings.searxng_base_url,
        bing_search_base_url=getattr(args, "bing_search_base_url", None)
        or settings.bing_search_base_url,
        gateway_web_search_model=getattr(args, "gateway_web_search_model", None)
        or settings.gateway_web_search_model,
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
        llm_gateway_model=effective_llm_model
        if effective_llm_provider == "llm-gateway"
        else settings.llm_gateway_model,
        llm_gateway_require_response_model_match=(
            bool(getattr(args, "single_model_run", False))
        ),
        llm_brief_model=stage_models["brief_generation"] or settings.llm_brief_model,
        llm_planner_model=stage_models["planning"] or settings.llm_planner_model,
        llm_synthesis_model=stage_models["synthesis"] or settings.llm_synthesis_model,
        citation_judge_provider=getattr(args, "citation_judge_provider", None)
        or settings.citation_judge_provider,
        citation_judge_model=_effective_citation_judge_model(args, settings),
    )
    judge_provider = (getattr(args, "judge_provider", "none") or "none").strip().lower()
    rejudge_replay = bool(getattr(args, "rejudge_replay", False))
    if judge_provider == "deepseek":
        effective_judge_model = (
            getattr(args, "judge_model", None) or effective_settings.deepseek_model
        )
    elif judge_provider in {"llm-gateway", "gateway"}:
        effective_judge_model = (
            getattr(args, "judge_model", None) or "kimi-k2.7-code-highspeed"
        )
    else:
        effective_judge_model = None
    answer_judge = build_eval_judge_provider(
        judge_provider,
        model=effective_judge_model,
        timeout_seconds=effective_settings.request_timeout_seconds,
        gateway_base_url=effective_settings.llm_gateway_base_url,
        gateway_thinking_budget_tokens=effective_settings.llm_gateway_thinking_budget_tokens,
    )
    config_snapshot = {
        "benchmark_name": args.benchmark_name,
        "dataset": "sealed_holdout" if sealed_holdout else args.dataset,
        "dataset_config": None if sealed_holdout else _dataset_config_name(args),
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
        "gateway_web_search_timeout_seconds": (
            effective_settings.llm_gateway_timeout_seconds
        ),
        "llm_gateway_timeout_seconds": effective_settings.llm_gateway_timeout_seconds,
        "llm_synthesis_timeout_seconds": (
            effective_settings.llm_synthesis_timeout_seconds
        ),
        "citation_judge_timeout_seconds": (
            effective_settings.citation_judge_timeout_seconds
        ),
        "answer_judge_timeout_seconds": effective_settings.request_timeout_seconds,
        "web_crawler_provider": effective_settings.web_crawler_provider,
        "crawler_max_chars": effective_settings.crawler_max_chars,
        "reflection_enabled": reflection_enabled,
        "max_reflection_rounds": max_reflection_rounds,
        "reflection_min_sources": reflection_min_sources,
        "citation_judge_provider": effective_settings.citation_judge_provider,
        "citation_judge_model": effective_settings.citation_judge_model,
        "judge_provider": judge_provider,
        "judge_model": effective_judge_model,
        "rejudge_replay": rejudge_replay,
        "max_rounds": max_rounds,
        "max_tool_calls": max_tool_calls,
        "deadline_seconds": deadline_seconds,
        "min_evidence_items": min_evidence_items,
        "fallback_policy": fallback_policy,
        "require_clean_worktree": bool(
            getattr(args, "require_clean_worktree", False)
        ),
        "official_judge_score": "not_run",
        "sealed_holdout": sealed_holdout,
        "single_model_run": bool(getattr(args, "single_model_run", False)),
        "benchmark_source_exclusion": True,
        "settings": {
            **sanitized_settings_snapshot(effective_settings),
            "llm_model": effective_llm_model,
            "stage_models": stage_models,
            "max_results": args.max_results,
        },
    }
    if sealed_holdout:
        # The manifest stays auditable without embedding any private question or
        # answer metadata in an artifact intended for the development session.
        config_snapshot["settings"].pop("trace_dir", None)
    dataset_name = (
        "sealed_holdout"
        if sealed_holdout
        else portable_artifact_path(args.cases, root)
        if args.cases
        else f"replay:{portable_artifact_path(args.replay_dir, root)}"
        if replay_records is not None
        else args.dataset
    )
    manifest = build_benchmark_manifest(
        root=root,
        benchmark_name=args.benchmark_name,
        dataset_name=dataset_name,
        cases=_manifest_cases(cases, sealed_holdout=sealed_holdout),
        config_snapshot=config_snapshot,
        llm_provider=effective_settings.llm_provider,
        llm_model=effective_llm_model,
        search_provider=effective_settings.search_provider,
        seed=args.seed,
        dataset_config=None if sealed_holdout else _dataset_config_name(args),
        dataset_split=args.split,
        replay_dir=getattr(args, "replay_dir", None),
        cassette_id=getattr(args, "cassette_id", None),
    )
    if getattr(args, "require_clean_worktree", False):
        require_clean_worktree(manifest)
    mark_live_judge_nondeterminism(
        manifest,
        citation_judge_provider=effective_settings.citation_judge_provider,
        citation_judge_model=effective_settings.citation_judge_model,
        citation_judge_executed=replay_records is None,
        answer_judge_provider=judge_provider,
        answer_judge_model=effective_judge_model,
        answer_judge_executed=replay_records is None or rejudge_replay,
    )
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    file = raw_path.open("w", encoding="utf-8") if raw_path is not None else None
    try:
        if file is not None:
            file.write(
                json.dumps(
                    {"type": "config", "config": config_snapshot, "manifest": manifest},
                    ensure_ascii=False,
                )
                + "\n"
            )
        for case in cases:
            if replay_records is not None:
                record = replay_case_result(
                    case,
                    replay_records,
                    manifest_id=manifest["manifest_id"],
                )
                record = refresh_replayed_case_result(case, record)
            else:
                record = await _run_case(
                    case,
                    args,
                    effective_settings,
                    effective_llm_model,
                    stage_models,
                )
            record["manifest_id"] = manifest["manifest_id"]
            should_run_judge = (
                not record.get("benchmark_contamination")
                and answer_judge is not None
                and (replay_records is None or rejudge_replay)
            )
            if should_run_judge:
                generation_models = _generation_models(
                    record,
                    configured_model=effective_llm_model,
                )
                try:
                    judgment = asdict(answer_judge.judge(case, record))
                    judgment["self_judge"] = _is_self_judge(
                        judgment,
                        generation_models=generation_models,
                    )
                    record["answer_judgment"] = judgment
                except Exception as exc:  # noqa: BLE001 - judge failure is a scored artifact.
                    record["answer_judgment"] = {
                        "provider": judge_provider,
                        "score": None,
                        "verdict": "unscored",
                        "reason": f"judge error: {type(exc).__name__}: {str(exc)[:240]}",
                        "matched": [],
                        "missing": [],
                        "model": effective_judge_model,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "confidence": 0.0,
                        "critical_errors": [],
                        "failure_categories": ["judge_uncertainty"],
                        "self_judge": _is_self_judge(
                            {
                                "provider": judge_provider,
                                "model": effective_judge_model,
                            },
                            generation_models=generation_models,
                        ),
                    }
                attach_answer_quality(record, record["answer_judgment"])
            elif answer_judge is not None and record.get("answer_judgment"):
                attach_answer_quality(record, record["answer_judgment"])
            records.append(record)
            if predictions_path is not None:
                predictions.append(
                    {
                        "key": case["id"],
                        "preds": _prediction_payload(record.get("answer", "")),
                    }
                )
            if file is not None:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if file is not None:
            file.close()

    if predictions_path is not None:
        predictions_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = _summarize(
        records,
        config_snapshot,
        raw_path=raw_path,
        predictions_path=predictions_path,
        manifest=manifest,
        root=root,
        sealed_holdout=sealed_holdout,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _validate_sealed_holdout_args(
    args: argparse.Namespace,
    *,
    sealed_holdout: bool,
) -> None:
    """Reject artifact options that could spill a private holdout into a shell."""

    if not sealed_holdout:
        return
    if not getattr(args, "cases", None):
        raise ValueError("sealed holdout mode requires an explicit --cases file")
    if getattr(args, "replay_dir", None):
        raise ValueError("sealed holdout mode does not allow replay artifacts")
    if getattr(args, "raw_log", None) or getattr(args, "predictions_output", None):
        raise ValueError(
            "sealed holdout mode never writes raw logs or predictions; use only --summary-output"
        )
    judge = (getattr(args, "judge_provider", "none") or "none").strip().lower()
    if judge in {"", "none"}:
        raise ValueError("sealed holdout mode requires an explicit answer judge")


def _validate_single_model_run_args(
    args: argparse.Namespace,
    *,
    settings: Any | None = None,
) -> None:
    """Require one explicitly named model for every answer-generating stage."""

    if not bool(getattr(args, "single_model_run", False)):
        return
    generation_model = getattr(args, "llm_model", None)
    if not generation_model:
        raise ValueError("single-model run requires an explicit --llm-model")
    model_options = {
        "--brief-model": getattr(args, "brief_model", None),
        "--planner-model": getattr(args, "planner_model", None),
        "--synthesis-model": getattr(args, "synthesis_model", None),
        "--gateway-web-search-model": getattr(args, "gateway_web_search_model", None),
    }
    mismatched = [name for name, value in model_options.items() if value != generation_model]
    if mismatched:
        joined = ", ".join(mismatched)
        raise ValueError(
            "single-model run requires every generation stage to equal --llm-model; "
            f"mismatched or missing options: {joined}"
        )
    citation_judge_provider = (
        getattr(args, "citation_judge_provider", None)
        or getattr(settings, "citation_judge_provider", "none")
        or "none"
    ).strip().lower()
    if citation_judge_provider not in {"", "none", "heuristic"}:
        raise ValueError(
            "single-model run does not allow an LLM citation judge during generation; "
            "use none/heuristic and run independent judges after generation"
        )
    answer_judge_provider = (getattr(args, "judge_provider", "none") or "none").strip().lower()
    if answer_judge_provider not in {"", "none"}:
        raise ValueError(
            "single-model run does not run an answer judge; score generated artifacts "
            "in a separate post-generation pass"
        )
    search_provider = (getattr(args, "search_provider", "") or "").strip().lower()
    crawler_provider = (
        getattr(args, "web_crawler_provider", None)
        or getattr(settings, "web_crawler_provider", "none")
        or "none"
    ).strip().lower()
    if search_provider in {"gateway-web", "gateway_web"} and crawler_provider != "html":
        raise ValueError(
            "single-model Gateway web evaluation requires --web-crawler-provider html; "
            "the Jina Reader server-side fetch path is not an auditable SSRF boundary"
        )


def load_eval_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.cases:
        cases = load_eval_cases_from_file(Path(args.cases), args.benchmark_name)
        if args.offset:
            cases = cases[args.offset :]
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


def _cases_from_replay_records(
    records: dict[str, dict[str, Any]],
    benchmark_name: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id, record in records.items():
        metadata = record.get("metadata") or record.get("case_metadata") or {}
        cases.append(
            {
                "id": case_id,
                "query": str(record.get("query") or ""),
                "category": record.get("category"),
                "benchmark_name": record.get("benchmark_name") or benchmark_name,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        )
    if any(not case["query"] for case in cases):
        raise ValueError("replay artifact contains a case_result without query")
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
        max_rounds=getattr(args, "max_rounds", 1),
        max_tool_calls=getattr(args, "max_tool_calls", 1),
        deadline_seconds=getattr(args, "deadline_seconds", None),
        min_evidence_items=getattr(args, "min_evidence_items", 1),
        fallback_policy=getattr(args, "fallback_policy", "fail"),
        report_depth=(case.get("metadata") or {}).get("report_depth") or "concise",
        expected_format=(case.get("metadata") or {}).get("expected_format") or "markdown",
        blocked_source_urls=(case.get("metadata") or {}).get("blocked_source_urls") or [],
    )
    try:
        report = await DeepResearchOrchestrator(settings=settings).run(request)
        return _case_success_record(case, report)
    except Exception as exc:  # pragma: no cover - exercised by integration failures.
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        case_metrics = build_case_evaluation_metrics(case, None, latency_ms=latency_ms)
        failed_cost = getattr(exc, "deepresearch_cost", None)
        if failed_cost is not None:
            case_metrics["total_tokens"] = int(
                getattr(failed_cost, "total_tokens", 0) or 0
            )
            case_metrics["estimated_cost_usd"] = float(
                getattr(failed_cost, "total_estimated_cost_usd", 0.0) or 0.0
            )
        contaminated = isinstance(exc, BenchmarkContaminationError)
        return {
            "type": "case_result",
            "case_id": case["id"],
            "query": case["query"],
            "category": case.get("category"),
            "benchmark_name": case.get("benchmark_name"),
            **case_metrics,
            "error": repr(exc),
            "error_category": "benchmark_contamination" if contaminated else None,
            "benchmark_contamination": contaminated,
            "answer": "",
            "claims": [],
            "sources": [],
            "citation_check": None,
            "cost": (
                failed_cost.model_dump(mode="json")
                if failed_cost is not None and hasattr(failed_cost, "model_dump")
                else None
            ),
            "metrics": case_metrics,
            "trace_events": [
                event.model_dump(mode="json")
                for event in getattr(exc, "deepresearch_trace_events", [])
            ],
            "run_id": getattr(exc, "deepresearch_run_id", None),
            "metadata": case.get("metadata", {}),
        }


def _case_success_record(case: dict[str, Any], report: StructuredReport) -> dict[str, Any]:
    case_metrics = build_case_evaluation_metrics(case, report)
    return {
        "type": "case_result",
        "case_id": case["id"],
        "query": case["query"],
        "category": case.get("category"),
        "benchmark_name": case.get("benchmark_name"),
        **case_metrics,
        "deduped_source_count": report.metrics["deduped_source_count"],
        "source_provider_count": report.metrics["source_provider_count"],
        "source_domain_count": report.metrics["source_domain_count"],
        "raw_search_result_count": report.metrics["raw_search_result_count"],
        "citation_retention_rate": report.metrics["citation_retention_rate"],
        "fallback_count": report.metrics["fallback_count"],
        "run_id": report.run_id,
        "answer": report.answer,
        "claims": report.claims,
        "sources": [source.model_dump(mode="json") for source in report.sources],
        "citation_check": report.citation_check.model_dump(mode="json"),
        "cost": report.cost.model_dump(mode="json"),
        "metrics": {**report.metrics, **case_metrics},
        "trace_events": [event.model_dump(mode="json") for event in report.trace_events],
        "report": report.model_dump(mode="json"),
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
    report_depth = metadata.get("report_depth") or "concise"
    expected_format = metadata.get("expected_format") or "markdown"
    if report_depth == "deep" and expected_format != "markdown":
        raise ValueError(
            f"eval row {index} report_depth='deep' requires "
            "expected_format='markdown'"
        )
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
    raw_path: Path | None,
    predictions_path: Path | None,
    manifest: dict[str, Any] | None = None,
    *,
    root: Path | None = None,
    sealed_holdout: bool = False,
) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records]
    tokens = [record.get("total_tokens", 0) for record in records]
    retentions = [
        record.get("citation_retention_rate", 0.0)
        for record in records
        if record.get("citation_retention_rate") is not None
    ]
    source_counts = [record.get("deduped_source_count", 0) for record in records]
    provider_counts = [record.get("source_provider_count", 0) for record in records]
    domain_counts = [record.get("source_domain_count", 0) for record in records]
    all_answer_judgments = [
        record["answer_judgment"]
        for record in records
        if isinstance(record.get("answer_judgment"), dict)
    ]
    answer_judgments = [
        record["answer_judgment"]
        for record in records
        if isinstance(record.get("answer_judgment"), dict)
        and record.get("answer_verdict")
        in {"correct", "incorrect", "not_attempted"}
        and record["answer_judgment"].get("score") is not None
    ]
    answer_verdict_counts = {
        verdict: sum(
            1
            for record in records
            if str(record.get("answer_verdict") or "unscored") == verdict
        )
        for verdict in ("correct", "incorrect", "not_attempted", "unscored")
    }
    split_metrics = evaluation_summary(records)
    summary = {
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
            "success_rate is a deprecated execution-success alias, not answer quality",
        ],
        "raw_log": (
            portable_artifact_path(raw_path, root) if root else str(raw_path)
        )
        if raw_path is not None
        else None,
        "predictions_output": (
            portable_artifact_path(predictions_path, root)
            if root
            else str(predictions_path)
        )
        if predictions_path is not None
        else None,
        "config": config_snapshot,
        "manifest": manifest,
        "deterministic": manifest.get("deterministic") if manifest else None,
        "case_count": len(records),
        **split_metrics,
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
        else None,
        "deduped_source_count_avg": round(sum(source_counts) / len(source_counts), 3)
        if source_counts
        else 0.0,
        "source_provider_count_avg": round(sum(provider_counts) / len(provider_counts), 3)
        if provider_counts
        else 0.0,
        "source_domain_count_avg": round(sum(domain_counts) / len(domain_counts), 3)
        if domain_counts
        else 0.0,
        "fallback_count_total": sum(record.get("fallback_count", 0) for record in records),
        "benchmark_contamination_count": sum(
            1 for record in records if record.get("benchmark_contamination")
        ),
        "answer_judge": {
            "provider": config_snapshot.get("judge_provider", "none"),
            "model": config_snapshot.get("judge_model"),
            "scored_count": len(answer_judgments),
            "fixed_denominator": len(records),
            "verdict_counts": answer_verdict_counts,
            "unscored_count": answer_verdict_counts["unscored"],
            "self_judge_count": sum(
                1 for item in all_answer_judgments if item.get("self_judge") is True
            ),
            "score_avg": round(
                sum(item["score"] for item in answer_judgments) / len(answer_judgments),
                4,
            )
            if answer_judgments
            else None,
            "pass_rate": round(
                answer_verdict_counts["correct"] / len(records),
                4,
            )
            if all_answer_judgments
            else None,
            "correct_rate": round(
                answer_verdict_counts["correct"] / len(records), 4
            )
            if records
            else 0.0,
            "tokens_total": sum(
                item.get("input_tokens", 0) + item.get("output_tokens", 0)
                for item in all_answer_judgments
            ),
            "estimated_cost_usd_total": round(
                sum(
                    item.get("estimated_cost_usd", 0.0)
                    for item in all_answer_judgments
                ),
                8,
            ),
            "official_judge_score": "not_run",
        },
        "records": _summary_records(records, sealed_holdout=sealed_holdout),
    }
    if sealed_holdout:
        summary["benchmark_kind"] = "sealed_holdout_aggregate_eval"
        summary["interpretation"] = (
            "Sealed holdout run: per-case questions, answers, gold metadata, raw traces, "
            "and judge rationale are intentionally omitted from this artifact."
        )
        summary["limitations"].append(
            "sealed output is intentionally aggregate-only and cannot be used to inspect individual answers"
        )
    return summary


def _summary_records(
    records: list[dict[str, Any]],
    *,
    sealed_holdout: bool,
) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for record in records:
        judgment = record.get("answer_judgment") or {}
        base = {
            "success": record["success"],
            "execution_success": record.get("execution_success", record["success"]),
            "task_format_valid": record.get("task_format_valid", False),
            "answer_quality": record.get("answer_quality"),
            "answer_verdict": record.get("answer_verdict", "unscored"),
            "grounded_correct": record.get("grounded_correct"),
            "report_emitted": record.get("report_emitted", False),
            "substantive_answer": record.get("substantive_answer", False),
            "grounded_answer": record.get("grounded_answer", False),
            "evidence_abstention": record.get("evidence_abstention", False),
            "citation_grounding": record.get("citation_grounding"),
            "citation_precision": record.get("citation_precision"),
            "citation_coverage": record.get("citation_coverage"),
            "unsupported_claim_rate": record.get("unsupported_claim_rate"),
            "source_quality": record.get("source_quality"),
            "tool_failure_recovery": record.get("tool_failure_recovery"),
            "tool_failure_attempted": record.get("tool_failure_attempted", False),
            "tool_failure_recovered": record.get("tool_failure_recovered"),
            "final_result_usable": record.get("final_result_usable", False),
            "legacy_report_success": record.get("legacy_report_success"),
            "success_semantics": record.get("success_semantics", SUCCESS_SEMANTICS),
            "latency_ms": record["latency_ms"],
            "total_tokens": record.get("total_tokens", 0),
            "estimated_cost_usd": record.get("estimated_cost_usd", 0.0),
            "deduped_source_count": record.get("deduped_source_count", 0),
            "source_provider_count": record.get("source_provider_count", 0),
            "source_domain_count": record.get("source_domain_count", 0),
            "citation_retention_rate": record.get("citation_retention_rate", 0.0),
            "fallback_count": record.get("fallback_count", 0),
            "benchmark_contamination": bool(record.get("benchmark_contamination")),
        }
        if sealed_holdout:
            base.update(
                {
                    "case_ref": _opaque_case_ref(str(record.get("case_id") or "")),
                    "answer_judgment": _sealed_answer_judgment(judgment),
                    "error_type": _safe_error_type(record.get("error")),
                    "error_category": _sealed_error_category(
                        record.get("error_category")
                    ),
                }
            )
        else:
            base.update(
                {
                    "case_id": record["case_id"],
                    "query": record["query"],
                    "answer_judgment": judgment or None,
                    "error": record.get("error"),
                    "error_category": record.get("error_category"),
                    "run_id": record.get("run_id"),
                }
            )
        summarized.append(base)
    return summarized


def _opaque_case_ref(case_id: str) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    return f"case-{digest}"


def _safe_error_type(error: Any) -> str | None:
    if not error:
        return None
    text = str(error).strip()
    match = re.match(r"([A-Za-z_][A-Za-z0-9_.]*)", text)
    candidate = match.group(1).rsplit(".", 1)[-1] if match else ""
    # Exception messages can contain the private question or answer.  Keep only
    # a conventional exception class name; everything else becomes a fixed
    # label instead of copying attacker- or data-controlled text.
    if candidate.endswith(("Error", "Exception")) and len(candidate) <= 80:
        return candidate
    return "runtime_error"


def _sealed_answer_judgment(judgment: dict[str, Any]) -> dict[str, Any]:
    """Keep only bounded numeric values and fixed enums from a holdout judge."""

    verdict = str(judgment.get("verdict") or "").strip().lower()
    if verdict not in _SEALED_JUDGE_VERDICTS:
        verdict = "unscored"
    score = _sealed_unit_interval(judgment.get("score"))
    if verdict == "unscored":
        score = None
    raw_categories = judgment.get("failure_categories")
    categories = raw_categories if isinstance(raw_categories, list) else []
    return {
        "score": score,
        "verdict": verdict,
        "confidence": _sealed_unit_interval(judgment.get("confidence")),
        "self_judge": judgment.get("self_judge") is True,
        "failure_categories": sorted(
            {
                str(category).strip().lower()
                for category in categories
                if str(category).strip().lower() in _SEALED_FAILURE_CATEGORIES
            }
        ),
    }


def _sealed_unit_interval(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= number <= 1.0:
        return None
    return round(number, 4)


def _sealed_error_category(value: Any) -> str | None:
    category = str(value or "").strip().lower()
    return category if category in _SEALED_ERROR_CATEGORIES else None


def _manifest_cases(
    cases: list[dict[str, Any]],
    *,
    sealed_holdout: bool,
) -> list[dict[str, Any]]:
    if not sealed_holdout:
        return cases
    return [
        {
            "id": _opaque_case_ref(str(case.get("id") or "")),
            "benchmark_name": "sealed_holdout",
        }
        for case in cases
    ]


def _effective_llm_model(args: argparse.Namespace, settings: Any) -> str:
    if args.llm_model:
        return args.llm_model
    provider = _normalize_llm_provider(args.llm_provider)
    if provider == "deepseek":
        return settings.deepseek_model
    if provider == "openai-compatible":
        return settings.openai_compatible_model
    if provider == "llm-gateway":
        return settings.llm_gateway_model
    return settings.mock_model_name


def _is_self_judge(
    judgment: dict[str, Any],
    *,
    generation_models: set[str],
) -> bool:
    if str(judgment.get("provider") or "").strip().lower() == "heuristic":
        return False
    judge_model = judgment.get("model")
    return bool(
        isinstance(judge_model, str)
        and judge_model
        and any(
            response_model_matches(generation_model, judge_model)
            for generation_model in generation_models
        )
    )


def _generation_models(
    record: dict[str, Any],
    *,
    configured_model: str,
) -> set[str]:
    """Recover generation models from replayed usage instead of trusting CLI defaults."""

    models = {configured_model} if configured_model else set()
    cost = record.get("cost") if isinstance(record.get("cost"), dict) else {}
    usage_records = cost.get("records") if isinstance(cost.get("records"), list) else []
    generation_stages = {
        "brief_generation",
        "planning",
        "research_decision",
        "synthesis",
    }
    for usage in usage_records:
        if not isinstance(usage, dict) or usage.get("stage") not in generation_stages:
            continue
        model = usage.get("model")
        if isinstance(model, str) and model.strip():
            models.add(model.strip())
    return models


def _normalize_llm_provider(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _effective_stage_models(args: argparse.Namespace, settings: Any) -> dict[str, str]:
    return {
        "brief_generation": getattr(args, "brief_model", None) or settings.llm_brief_model,
        "planning": getattr(args, "planner_model", None) or settings.llm_planner_model,
        "synthesis": getattr(args, "synthesis_model", None) or settings.llm_synthesis_model,
    }


def _effective_citation_judge_model(args: argparse.Namespace, settings: Any) -> str:
    explicit = getattr(args, "citation_judge_model", None)
    if explicit:
        return explicit
    provider = (
        getattr(args, "citation_judge_provider", None)
        or settings.citation_judge_provider
        or "none"
    ).strip().lower()
    if provider in {"llm-gateway", "gateway"}:
        return settings.citation_judge_gateway_model
    return settings.citation_judge_model


def _dataset_config_name(args: argparse.Namespace) -> str | None:
    if args.cases or getattr(args, "replay_dir", None):
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
        choices=[
            "mock",
            "wikipedia",
            "bing",
            "searxng",
            "jina",
            "brave",
            "tavily",
            "gateway-web",
            "mcp",
        ],
        default="mock",
    )
    parser.add_argument(
        "--llm-provider",
        choices=[
            "mock",
            "deepseek",
            "openai-compatible",
            "openai_compatible",
            "llm-gateway",
        ],
        default="mock",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--brief-model", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--synthesis-model", default=None)
    parser.add_argument(
        "--single-model-run",
        action="store_true",
        help=(
            "Fail before evaluation unless llm, brief, planner, synthesis, and Gateway "
            "web-search models are all explicitly set to the same model."
        ),
    )
    parser.add_argument("--embedding-provider", choices=["local", "dashscope"], default="local")
    parser.add_argument(
        "--local-retrieval-mode",
        choices=["none", "keyword", "hybrid"],
        default="none",
    )
    parser.add_argument("--max-researchers", type=int, default=2)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--max-tool-calls", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=float, default=None)
    parser.add_argument("--min-evidence-items", type=int, default=1)
    parser.add_argument(
        "--fallback-policy",
        choices=["mock", "degraded", "fail"],
        default="fail",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=8.0,
        help=(
            "Common timeout for search/crawling, Gateway web search, non-synthesis "
            "LLM calls, citation judging, answer judging, and remote dataset loading."
        ),
    )
    parser.add_argument(
        "--synthesis-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Socket timeout for long synthesis calls; defaults to "
            "LLM_SYNTHESIS_TIMEOUT_SECONDS (360 seconds)."
        ),
    )
    parser.add_argument("--reflection-enabled", action="store_true")
    parser.add_argument("--max-reflection-rounds", type=int, default=1)
    parser.add_argument("--reflection-min-sources", type=int, default=4)
    parser.add_argument(
        "--citation-judge-provider",
        choices=["none", "heuristic", "deepseek", "llm-gateway"],
        default=None,
    )
    parser.add_argument("--citation-judge-model", default=None)
    parser.add_argument("--searxng-base-url", default=None)
    parser.add_argument("--bing-search-base-url", default=None)
    parser.add_argument("--gateway-web-search-model", default=None)
    parser.add_argument(
        "--web-crawler-provider",
        choices=["none", "jina", "jina_reader", "html"],
        default=None,
    )
    parser.add_argument("--jina-reader-base-url", default=None)
    parser.add_argument("--jina-search-base-url", default=None)
    parser.add_argument("--crawler-max-chars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--judge-provider",
        choices=["none", "heuristic", "deepseek", "llm-gateway"],
        default="none",
        help=(
            "Optional answer scoring provider. 'heuristic' checks normalized "
            "ground-truth strings in the generated answer; 'deepseek' calls DeepSeek "
            "JSON mode with DEEPSEEK_API_KEY; 'llm-gateway' uses the internal "
            "Messages endpoint. Official scoring is not run."
        ),
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--raw-log", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument(
        "--sealed-holdout",
        action="store_true",
        help=(
            "Run a private holdout without raw logs, predictions, persistent traces, "
            "or question/answer-bearing stdout output. Requires an explicit answer judge."
        ),
    )
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="Optional case-result JSONL file or single-artifact directory for offline replay.",
    )
    parser.add_argument("--cassette-id", default=None)
    parser.add_argument(
        "--require-clean-worktree",
        action="store_true",
        help="Fail before running cases unless git reports a clean worktree.",
    )
    parser.add_argument(
        "--rejudge-replay",
        action="store_true",
        help="Explicitly call the selected live/local answer judge on replayed artifacts.",
    )
    args = parser.parse_args()
    summary = asyncio.run(run_public_deep_research_eval(args))
    if args.sealed_holdout:
        print(json.dumps(_sealed_stdout_summary(summary), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def _sealed_stdout_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the strictly aggregate subset allowed to reach a shell transcript."""

    keys = (
        "benchmark_kind",
        "case_count",
        "execution_success_count",
        "execution_success_rate",
        "task_format_valid_count",
        "task_format_valid_rate",
        "answer_quality_scored_count",
        "answer_quality_avg",
        "answer_verdict_counts",
        "answer_correct_count",
        "answer_correct_rate",
        "answer_incorrect_count",
        "answer_not_attempted_count",
        "answer_unscored_count",
        "answer_fixed_denominator",
        "grounded_correct_count",
        "grounded_correct_rate",
        "self_judge_count",
        "report_emitted_count",
        "substantive_answer_count",
        "grounded_answer_count",
        "evidence_abstention_count",
        "citation_grounding_avg",
        "citation_precision_avg",
        "citation_coverage_avg",
        "unsupported_claim_rate_avg",
        "claim_extraction_valid_count",
        "claim_extraction_valid_rate",
        "source_quality_avg",
        "tool_failure_recovery_applicable_count",
        "tool_failure_recovery_avg",
        "latency_ms",
        "tokens",
        "estimated_cost_usd_total",
        "citation_retention_rate_avg",
        "deduped_source_count_avg",
        "source_provider_count_avg",
        "source_domain_count_avg",
        "fallback_count_total",
        "answer_judge",
        "deterministic",
    )
    return {key: summary.get(key) for key in keys}


if __name__ == "__main__":
    main()
