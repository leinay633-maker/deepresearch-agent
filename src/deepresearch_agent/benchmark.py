from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from deepresearch_agent.config import load_settings
from deepresearch_agent.citation import CitationChecker
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.replay import (
    case_result_artifact_id,
    load_case_result_records,
    replay_case_result,
    validate_replay_case_ids,
)
from deepresearch_agent.schemas import ResearchRequest, StructuredReport
from deepresearch_agent.search import MockSearchAdapter, SearchError, SearchService


class BenchmarkFailureAdapter:
    """Deterministic fault injection used only by offline regression cases."""

    name = "benchmark_failure"

    def __init__(self, failure: str) -> None:
        self.failure = failure

    async def search(self, query: str, max_results: int, timeout: float):
        del query, max_results, timeout
        if self.failure == "empty_search_results":
            return []
        if self.failure == "search_rate_limit":
            raise SearchError("simulated HTTP 429 rate limit")
        if self.failure == "search_timeout":
            raise SearchError("simulated search timeout")
        raise SearchError(f"unknown benchmark failure injection: {self.failure}")


SUCCESS_SEMANTICS = (
    "deprecated alias of execution_success; answer quality requires an explicit judge"
)
PROMPT_VERSION = "source-bundle-v1"
_SECRET_SETTING_NAMES = {"otel_exporter_otlp_headers", "mcp_args"}
_SECRET_SETTING_SUFFIXES = ("_password", "_secret", "_api_token", "_access_token")


def sanitized_settings_snapshot(settings: Any) -> dict[str, Any]:
    """Serialize settings without persisting credential-bearing values."""

    snapshot = asdict(settings)
    for key in list(snapshot):
        lowered = key.lower()
        if lowered in _SECRET_SETTING_NAMES or lowered.endswith(
            _SECRET_SETTING_SUFFIXES
        ):
            snapshot[key] = "<redacted>" if snapshot[key] else ""
    return snapshot


def build_benchmark_manifest(
    *,
    root: Path,
    benchmark_name: str,
    dataset_name: str,
    cases: list[dict[str, Any]],
    config_snapshot: dict[str, Any],
    llm_provider: str,
    llm_model: str,
    search_provider: str,
    seed: int,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    replay_dir: str | None = None,
    cassette_id: str | None = None,
) -> dict[str, Any]:
    """Build a self-contained, secret-free identity for one benchmark run."""

    git_commit_sha, git_dirty, git_worktree_hash = _git_metadata(root)
    prompt_bundle_hash = _prompt_bundle_hash(root)
    dataset_version = f"sha256:{_sha256_json(cases)}"
    normalized_replay_dir = portable_artifact_path(replay_dir, root) if replay_dir else None
    if normalized_replay_dir:
        execution_mode = "replay"
        deterministic = True
        determinism_reason = (
            "benchmark snapshot replay reuses recorded case artifacts and reruns "
            "the current deterministic evaluators"
        )
    elif llm_provider == "mock" and search_provider == "mock":
        execution_mode = "mock"
        deterministic = True
        determinism_reason = "both LLM and search providers are deterministic mocks"
    elif llm_provider == "mock" or search_provider == "mock":
        execution_mode = "mixed"
        deterministic = False
        determinism_reason = (
            "the run mixes a deterministic mock with a live provider; seed and config are "
            "recorded but live output is not guaranteed deterministic"
        )
    else:
        execution_mode = "live"
        deterministic = False
        determinism_reason = (
            "the run uses at least one live provider; seed and config are recorded but "
            "provider outputs are not guaranteed deterministic"
        )
    replay_artifact_id = case_result_artifact_id(replay_dir) if replay_dir else None
    identity = {
        "schema_version": "1.0",
        "git_commit_sha": git_commit_sha,
        "git_dirty": git_dirty,
        "git_worktree_hash": git_worktree_hash,
        "benchmark_name": benchmark_name,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "case_count": len(cases),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "search_provider": search_provider,
        "seed": seed,
        "prompt_version": PROMPT_VERSION,
        "prompt_bundle_hash": prompt_bundle_hash,
        "execution_mode": execution_mode,
        "replay_kind": "benchmark_snapshot" if replay_dir else None,
        "replay_dir": normalized_replay_dir,
        "replay_artifact_id": replay_artifact_id,
        # Backward-compatible field retained for older result consumers.
        "cassette_id": cassette_id or replay_artifact_id,
        "config_snapshot": config_snapshot,
    }
    return {
        **identity,
        "manifest_id": _sha256_json(identity)[:16],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_dirty": git_dirty,
        "deterministic": deterministic,
        "determinism_reason": determinism_reason,
    }


def build_case_evaluation_metrics(
    case: dict[str, Any],
    report: StructuredReport | None,
    *,
    latency_ms: float = 0.0,
    answer_quality: float | None = None,
) -> dict[str, Any]:
    """Derive honest, orthogonal metrics without treating citations as answer quality."""

    if report is None:
        failure_attempted = _tool_failure_case(case)
        return {
            "execution_success": False,
            "task_format_valid": False,
            "answer_quality": None,
            "citation_grounding": None,
            "citation_coverage": None,
            "unsupported_claim_rate": None,
            "claim_extraction_valid": False,
            "source_quality": None,
            "tool_failure_attempted": failure_attempted,
            "tool_failure_recovered": 0.0 if failure_attempted else None,
            "final_result_usable": False,
            "tool_failure_recovery": 0.0 if failure_attempted else None,
            "latency_ms": round(max(latency_ms, 0.0), 3),
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "success": False,
            "legacy_report_success": None,
            "success_semantics": SUCCESS_SEMANTICS,
        }

    total_claims = report.citation_check.total_claims
    citation_coverage = report.citation_check.citation_coverage
    citation_grounding = report.citation_check.citation_grounding
    unsupported_claim_rate = (
        report.citation_check.unsupported_claims / total_claims if total_claims else None
    )
    source_quality = (
        sum(source.quality_score for source in report.sources) / len(report.sources)
        if report.sources
        else None
    )
    fallback_count = int(report.metrics.get("fallback_count", 0) or 0)
    degraded_count = int(report.metrics.get("degraded_count", 0) or 0)
    task_format_valid = _task_format_valid(case, report.answer)
    failure_attempted = _tool_failure_case(case)
    final_result_usable = bool(report.answer.strip()) and task_format_valid
    failure_recovered = (
        1.0
        if failure_attempted and final_result_usable and (fallback_count or degraded_count)
        else 0.0
        if failure_attempted
        else None
    )
    return {
        "execution_success": True,
        "task_format_valid": task_format_valid,
        "answer_quality": answer_quality,
        "citation_grounding": _round_optional(citation_grounding),
        "citation_precision": _round_optional(report.citation_check.citation_precision),
        "citation_coverage": _round_optional(citation_coverage),
        "unsupported_claim_rate": _round_optional(unsupported_claim_rate),
        "claim_extraction_valid": report.citation_check.claim_extraction_valid,
        "source_quality": _round_optional(source_quality),
        "tool_failure_attempted": failure_attempted,
        "tool_failure_recovered": failure_recovered,
        "final_result_usable": final_result_usable,
        "tool_failure_recovery": failure_recovered,
        "latency_ms": float(report.metrics.get("latency_ms", latency_ms) or 0.0),
        "total_tokens": report.cost.total_tokens,
        "estimated_cost_usd": report.cost.total_estimated_cost_usd,
        "success": True,
        "legacy_report_success": report.metrics.get("legacy_report_success"),
        "success_semantics": SUCCESS_SEMANTICS,
    }


def attach_answer_quality(record: dict[str, Any], judgment: dict[str, Any] | None) -> None:
    """Attach answer quality only when a configured judge returned a real score."""

    score = judgment.get("score") if judgment else None
    record["answer_quality"] = score
    record.setdefault("metrics", {})["answer_quality"] = score


def refresh_replayed_case_result(
    case: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Re-evaluate a stored report with the current citation/format code."""

    payload = record.get("report")
    if not isinstance(payload, dict):
        return record
    recorded_metrics = json.loads(
        json.dumps(record.get("metrics") or {}, ensure_ascii=False)
    )
    record["recorded_metrics"] = recorded_metrics
    report = StructuredReport.model_validate(payload)
    recorded_quality = record.get("answer_quality")
    if recorded_quality is not None:
        record["recorded_answer_quality"] = recorded_quality
    citation_check = CitationChecker().check(report.claims, report.sources)
    report_metrics = {
        **report.metrics,
        "citation_grounding": citation_check.citation_grounding,
        "citation_precision": citation_check.citation_precision,
        "citation_coverage": citation_check.citation_coverage,
        "unsupported_claim_rate": citation_check.unsupported_claim_rate,
        "citation_retention_rate": citation_check.retention_rate,
        "claim_extraction_valid": citation_check.claim_extraction_valid,
    }
    report = report.model_copy(
        update={"citation_check": citation_check, "metrics": report_metrics}
    )
    record["report"] = report.model_dump(mode="json")
    record["citation_check"] = citation_check.model_dump(mode="json")
    refreshed = build_case_evaluation_metrics(case, report)
    for key, value in refreshed.items():
        if key in {"latency_ms", "total_tokens", "estimated_cost_usd"}:
            continue
        record[key] = value
    record["metrics"] = {**(record.get("metrics") or {}), **refreshed}
    record["answer_quality"] = None
    return record


def evaluation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    execution_success_count = sum(1 for record in records if record.get("execution_success"))
    task_format_valid_count = sum(1 for record in records if record.get("task_format_valid"))
    answer_quality = _present_numbers(records, "answer_quality")
    citation_grounding = _present_numbers(records, "citation_grounding")
    citation_precision = _present_numbers(records, "citation_precision")
    citation_coverage = _present_numbers(records, "citation_coverage")
    unsupported_claim_rate = _present_numbers(records, "unsupported_claim_rate")
    source_quality = _present_numbers(records, "source_quality")
    tool_failure_recovery = _present_numbers(records, "tool_failure_recovery")
    claim_extraction_valid_count = sum(
        1 for record in records if record.get("claim_extraction_valid") is True
    )
    case_count = len(records)
    return {
        "execution_success_count": execution_success_count,
        "execution_success_rate": round(execution_success_count / case_count, 4)
        if case_count
        else 0.0,
        "task_format_valid_count": task_format_valid_count,
        "task_format_valid_rate": round(task_format_valid_count / case_count, 4)
        if case_count
        else 0.0,
        "answer_quality_scored_count": len(answer_quality),
        "answer_quality_avg": _average(answer_quality),
        "citation_grounding_avg": _average(citation_grounding),
        "citation_precision_avg": _average(citation_precision),
        "citation_coverage_avg": _average(citation_coverage),
        "unsupported_claim_rate_avg": _average(unsupported_claim_rate),
        "claim_extraction_valid_count": claim_extraction_valid_count,
        "claim_extraction_valid_rate": round(
            claim_extraction_valid_count / case_count, 4
        )
        if case_count
        else 0.0,
        "source_quality_avg": _average(source_quality),
        "tool_failure_recovery_applicable_count": len(tool_failure_recovery),
        "tool_failure_recovery_avg": _average(tool_failure_recovery),
        "success_count": execution_success_count,
        "success_rate": round(execution_success_count / case_count, 4) if case_count else 0.0,
        "success_semantics": SUCCESS_SEMANTICS,
    }


def _task_format_valid(case: dict[str, Any], answer: str) -> bool:
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    expected = str(
        case.get("expected_format")
        or metadata.get("expected_format")
        or metadata.get("output_format")
        or "text"
    ).strip().lower()
    if not answer or not answer.strip():
        return False
    if expected in {"json", "structured_json", "application/json"}:
        try:
            json.loads(_strip_json_fence(answer))
        except (json.JSONDecodeError, TypeError):
            return False
    return True


def _tool_failure_case(case: dict[str, Any]) -> bool:
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    return bool(case.get("failure_injection") or metadata.get("failure_injection"))


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _present_numbers(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def portable_artifact_path(path: str | Path, root: Path) -> str:
    """Use repository-relative paths in committed artifacts when possible."""

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _prompt_bundle_hash(root: Path) -> str:
    digest = hashlib.sha256()
    prompt_files = [
        root / "src" / "deepresearch_agent" / "llm.py",
        root / "src" / "deepresearch_agent" / "citation_judge.py",
        root / "src" / "deepresearch_agent" / "eval_judge.py",
    ]
    for path in prompt_files:
        if path.exists():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _git_metadata(root: Path) -> tuple[str | None, bool | None, str | None]:
    """Return commit and a content hash that distinguishes dirty worktrees.

    The hash covers staged/unstaged tracked changes plus untracked, non-ignored
    files. Only the digest is stored, so local source or secret contents never
    enter benchmark artifacts.
    """

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None, None
    if not status:
        return sha or None, False, None

    try:
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return sha or None, True, None

    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    for raw_path in sorted(path for path in untracked_output.split(b"\0") if path):
        digest.update(b"untracked\0")
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        path = root / os.fsdecode(raw_path)
        try:
            content = path.read_bytes()
        except OSError:
            content = b"<unreadable-or-removed>"
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return sha or None, True, f"sha256:{digest.hexdigest()}"


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cases_path = Path(args.cases) if args.cases else root / "data" / "benchmark_cases.jsonl"
    logs_dir = root / "logs"
    results_dir = root / "results"
    logs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    cases = _load_cases(cases_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path: Path | None = None
    summary_path: Path | None = None
    records = []

    settings = load_settings()
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
    fallback_policy = _effective_fallback_policy(
        getattr(args, "fallback_policy", None),
        search_provider=args.search_provider,
    )
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
    settings_snapshot = sanitized_settings_snapshot(effective_settings)
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
        "max_rounds": max_rounds,
        "max_tool_calls": max_tool_calls,
        "deadline_seconds": deadline_seconds,
        "min_evidence_items": min_evidence_items,
        "fallback_policy": fallback_policy,
    }
    manifest = build_benchmark_manifest(
        root=root,
        benchmark_name=getattr(args, "benchmark_name", "local_benchmark"),
        dataset_name=portable_artifact_path(cases_path, root),
        cases=cases,
        config_snapshot=config_snapshot,
        llm_provider=effective_settings.llm_provider,
        llm_model=effective_llm_model,
        search_provider=effective_settings.search_provider,
        seed=args.seed,
        replay_dir=getattr(args, "replay_dir", None),
        cassette_id=getattr(args, "cassette_id", None),
    )
    mark_live_judge_nondeterminism(
        manifest,
        citation_judge_provider=effective_settings.citation_judge_provider,
    )
    raw_path = logs_dir / f"benchmark-{timestamp}-{manifest['manifest_id']}.jsonl"
    summary_path = results_dir / "benchmarks" / manifest["manifest_id"] / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    replay_records = (
        load_case_result_records(args.replay_dir)
        if getattr(args, "replay_dir", None)
        else None
    )
    if replay_records is not None:
        validate_replay_case_ids(cases, replay_records)

    with raw_path.open("w", encoding="utf-8") as file:
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
                records.append(record)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            started = time.perf_counter()
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
                max_rounds=max_rounds,
                max_tool_calls=max_tool_calls,
                deadline_seconds=deadline_seconds,
                min_evidence_items=min_evidence_items,
                fallback_policy=fallback_policy,
            )
            try:
                orchestrator = _benchmark_orchestrator(
                    case,
                    settings=effective_settings,
                    fallback_policy=fallback_policy,
                )
                report = await orchestrator.run(request)
                case_metrics = build_case_evaluation_metrics(case, report)
                record = {
                    "type": "case_result",
                    "case_id": case["id"],
                    "query": case["query"],
                    "category": case.get("category"),
                    "language": case.get("language"),
                    "expected_format": case.get("expected_format"),
                    **case_metrics,
                    "deduped_source_count": report.metrics["deduped_source_count"],
                    "source_provider_count": report.metrics["source_provider_count"],
                    "source_domain_count": report.metrics["source_domain_count"],
                    "raw_search_result_count": report.metrics["raw_search_result_count"],
                    "citation_retention_rate": report.metrics["citation_retention_rate"],
                    "fallback_count": report.metrics["fallback_count"],
                    "output_summary": report.answer[:240],
                    "run_id": report.run_id,
                    "answer": report.answer,
                    "claims": report.claims,
                    "sources": [source.model_dump(mode="json") for source in report.sources],
                    "citation_check": report.citation_check.model_dump(mode="json"),
                    "cost": report.cost.model_dump(mode="json"),
                    "metrics": {**report.metrics, **case_metrics},
                    "trace_events": [
                        event.model_dump(mode="json") for event in report.trace_events
                    ],
                    "report": report.model_dump(mode="json"),
                    "case_metadata": {
                        key: value
                        for key, value in case.items()
                        if key not in {"id", "query"}
                    },
                    "manifest_id": manifest["manifest_id"],
                }
            except Exception as exc:  # pragma: no cover - integration/provider failures.
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                case_metrics = build_case_evaluation_metrics(
                    case,
                    None,
                    latency_ms=latency_ms,
                )
                record = {
                    "type": "case_result",
                    "case_id": case["id"],
                    "query": case["query"],
                    "category": case.get("category"),
                    "language": case.get("language"),
                    "expected_format": case.get("expected_format"),
                    **case_metrics,
                    "deduped_source_count": 0,
                    "source_provider_count": 0,
                    "source_domain_count": 0,
                    "raw_search_result_count": 0,
                    "citation_retention_rate": None,
                    "fallback_count": 0,
                    "output_summary": "",
                    "run_id": None,
                    "answer": "",
                    "claims": [],
                    "sources": [],
                    "citation_check": None,
                    "cost": None,
                    "metrics": case_metrics,
                    "trace_events": [],
                    "error": repr(exc),
                    "case_metadata": {
                        key: value
                        for key, value in case.items()
                        if key not in {"id", "query"}
                    },
                    "manifest_id": manifest["manifest_id"],
                }
            records.append(record)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = _summarize(records, config_snapshot, raw_path, manifest, root=root)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    summary_path.write_text(summary_text, encoding="utf-8")
    (results_dir / "benchmark_summary.json").write_text(summary_text, encoding="utf-8")
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


def _effective_fallback_policy(requested: str | None, *, search_provider: str) -> str:
    """Keep offline mock runs convenient, but never mix mock into live benchmarks by default."""

    if requested:
        return requested
    return "mock" if search_provider.strip().lower() == "mock" else "fail"


def mark_live_judge_nondeterminism(
    manifest: dict[str, Any],
    *,
    citation_judge_provider: str | None = None,
    answer_judge_provider: str | None = None,
) -> None:
    if manifest.get("execution_mode") == "replay":
        return
    live_judges = [
        name
        for name, provider in (
            ("citation judge", citation_judge_provider),
            ("answer judge", answer_judge_provider),
        )
        if (provider or "").strip().lower() in {"deepseek", "openai", "anthropic"}
    ]
    if not live_judges:
        return
    manifest["deterministic"] = False
    manifest["determinism_reason"] = (
        f"live {' and '.join(live_judges)} output is not guaranteed deterministic"
    )


def _effective_stage_models(args: argparse.Namespace, settings: Any) -> dict[str, str]:
    return {
        "brief_generation": getattr(args, "brief_model", None) or settings.llm_brief_model,
        "planning": getattr(args, "planner_model", None) or settings.llm_planner_model,
        "synthesis": getattr(args, "synthesis_model", None) or settings.llm_synthesis_model,
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def _benchmark_orchestrator(
    case: dict[str, Any],
    *,
    settings: Any,
    fallback_policy: str,
) -> DeepResearchOrchestrator:
    failure = str(case.get("failure_injection") or "").strip()
    if not failure or settings.search_provider != "mock":
        return DeepResearchOrchestrator(settings=settings)
    service = SearchService(
        primary=BenchmarkFailureAdapter(failure),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy=fallback_policy,
    )
    return DeepResearchOrchestrator(settings=settings, search_service=service)


def _summarize(
    records: list[dict[str, Any]],
    config_snapshot: dict[str, Any],
    raw_path: Path,
    manifest: dict[str, Any] | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    latencies = [float(record.get("latency_ms", 0.0) or 0.0) for record in records]
    tokens = [int(record.get("total_tokens", 0) or 0) for record in records]
    retentions = [
        float(record["citation_retention_rate"])
        for record in records
        if record.get("citation_retention_rate") is not None
    ]
    provider_counts = [record.get("source_provider_count", 0) for record in records]
    domain_counts = [record.get("source_domain_count", 0) for record in records]
    split_metrics = evaluation_summary(records)
    benchmark_kind, interpretation, limitations = _benchmark_notes(config_snapshot)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": benchmark_kind,
        "interpretation": interpretation,
        "limitations": limitations,
        "raw_log": portable_artifact_path(raw_path, root) if root else str(raw_path),
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
            sum(float(record.get("estimated_cost_usd", 0.0) or 0.0) for record in records), 8
        ),
        "citation_retention_rate_avg": round(sum(retentions) / len(retentions), 4)
        if retentions
        else None,
        "fallback_count_total": sum(record.get("fallback_count", 0) for record in records),
        "source_provider_count_avg": round(sum(provider_counts) / len(provider_counts), 3)
        if provider_counts
        else 0.0,
        "source_domain_count_avg": round(sum(domain_counts) / len(domain_counts), 3)
        if domain_counts
        else 0.0,
        # The raw JSONL keeps full answer/source/trace artifacts.  Keep the
        # summary compact so `deepresearch-benchmark` remains readable and the
        # tracked summary does not duplicate megabytes of per-case payloads.
        "records": [
            {
                "case_id": record.get("case_id"),
                "query": record.get("query"),
                "success": record.get("success", record.get("execution_success", False)),
                "execution_success": record.get(
                    "execution_success", record.get("success", False)
                ),
                "task_format_valid": record.get("task_format_valid", False),
                "answer_quality": record.get("answer_quality"),
                "citation_grounding": record.get("citation_grounding"),
                "citation_precision": record.get("citation_precision"),
                "citation_coverage": record.get("citation_coverage"),
                "unsupported_claim_rate": record.get("unsupported_claim_rate"),
                "claim_extraction_valid": record.get("claim_extraction_valid", False),
                "source_quality": record.get("source_quality"),
                "tool_failure_recovery": record.get("tool_failure_recovery"),
                "tool_failure_attempted": record.get("tool_failure_attempted", False),
                "tool_failure_recovered": record.get("tool_failure_recovered"),
                "final_result_usable": record.get("final_result_usable", False),
                "legacy_report_success": record.get("legacy_report_success"),
                "success_semantics": record.get("success_semantics", SUCCESS_SEMANTICS),
                "latency_ms": record.get("latency_ms", 0.0),
                "total_tokens": record.get("total_tokens", 0),
                "estimated_cost_usd": record.get("estimated_cost_usd", 0.0),
                "deduped_source_count": record.get("deduped_source_count", 0),
                "source_provider_count": record.get("source_provider_count", 0),
                "source_domain_count": record.get("source_domain_count", 0),
                "raw_search_result_count": record.get("raw_search_result_count", 0),
                "citation_retention_rate": record.get("citation_retention_rate"),
                "fallback_count": record.get("fallback_count", 0),
                "degraded_count": record.get("degraded_count", 0),
                "output_summary": record.get("output_summary", ""),
                "answer_judgment": record.get("answer_judgment"),
                "error": record.get("error"),
                "run_id": record.get("run_id"),
            }
            for record in records
        ],
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
                "success_rate is a deprecated execution-success alias, not an answer-quality score",
            ],
        )
    if llm_provider == "mock" or search_provider == "mock":
        return (
            "mixed_provider_benchmark",
            (
                "This run mixes one live provider with one deterministic mock. It validates "
                "the configured integration boundary, not fully live answer quality or cost."
            ),
            [
                "live provider output and latency can vary across runs",
                "mock components are fixtures and must not be presented as live evidence",
                "citation metrics use lexical grounding unless an explicit judge is configured",
                "success_rate is a deprecated execution-success alias, not an answer-quality score",
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
            "success_rate is a deprecated execution-success alias, not an answer-quality score",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepResearch Agent benchmark.")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--benchmark-name", default="local_benchmark")
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
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--max-tool-calls", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=float, default=None)
    parser.add_argument("--min-evidence-items", type=int, default=1)
    parser.add_argument(
        "--fallback-policy",
        choices=["mock", "degraded", "fail"],
        default=None,
        help="Fallback policy; defaults to mock for mock search and fail for live search.",
    )
    parser.add_argument("--reflection-enabled", action="store_true")
    parser.add_argument("--max-reflection-rounds", type=int, default=1)
    parser.add_argument("--reflection-min-sources", type=int, default=4)
    parser.add_argument(
        "--citation-judge-provider",
        choices=["none", "heuristic", "deepseek"],
        default=None,
    )
    parser.add_argument("--citation-judge-model", default=None)
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="Optional case-result JSONL file or single-artifact directory for offline replay.",
    )
    parser.add_argument("--cassette-id", default=None)
    args = parser.parse_args()
    summary = asyncio.run(run_benchmark(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
