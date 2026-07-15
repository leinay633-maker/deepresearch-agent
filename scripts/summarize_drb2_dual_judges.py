#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from deepresearch_agent.drb2_rubric_eval import (
    DRB2_PROTOCOL_NAME,
    DRB2_RUBRIC_CATEGORIES,
    DRB2RubricSpec,
    load_drb2_judge_artifact,
    load_drb2_rubric_specs,
    load_generation_artifact,
)
from deepresearch_agent.llm_gateway import response_model_matches
from deepresearch_agent.replay import case_result_artifact_id, citation_judge_identities


def summarize_drb2_dual_judges(
    *,
    generation_path: str | Path,
    rubrics_path: str | Path,
    kimi_judge_path: str | Path,
    opus_judge_path: str | Path,
) -> dict[str, Any]:
    generation_source = Path(generation_path).expanduser().resolve()
    rubric_source = Path(rubrics_path).expanduser().resolve()
    generation_config, generation_records = load_generation_artifact(generation_source)
    specs = load_drb2_rubric_specs(rubric_source)
    expected_case_ids = {spec.case_id for spec in specs}
    if expected_case_ids != set(generation_records):
        raise ValueError("generation and rubric artifacts have different case ID sets")
    generation_manifest = generation_config["manifest"]
    generation_manifest_id = str(generation_manifest.get("manifest_id") or "")
    if not generation_manifest_id:
        raise ValueError("generation manifest_id is missing")
    expected_generation_id = case_result_artifact_id(generation_source)
    expected_rubrics_sha = hashlib.sha256(rubric_source.read_bytes()).hexdigest()
    artifacts = {
        "kimi": _load_and_validate_judge_artifact(
            Path(kimi_judge_path),
            expected_role="kimi",
            expected_generation_id=expected_generation_id,
            expected_generation_manifest_id=generation_manifest_id,
            expected_rubrics_sha=expected_rubrics_sha,
            expected_rubric_count=len(specs),
            expected_rubric_ids={spec.rubric_id for spec in specs},
        ),
        "opus": _load_and_validate_judge_artifact(
            Path(opus_judge_path),
            expected_role="opus",
            expected_generation_id=expected_generation_id,
            expected_generation_manifest_id=generation_manifest_id,
            expected_rubrics_sha=expected_rubrics_sha,
            expected_rubric_count=len(specs),
            expected_rubric_ids={spec.rubric_id for spec in specs},
        ),
    }
    audits: list[dict[str, Any]] = []
    for spec in specs:
        report = str(generation_records[spec.case_id].get("answer") or "")
        judgments = {
            role: _validated_judgment(
                artifact["records"].get(spec.rubric_id),
                spec=spec,
                report=report,
                header=artifact["header"],
            )
            for role, artifact in artifacts.items()
        }
        scores = [judgments["kimi"]["score"], judgments["opus"]["score"]]
        all_scored = all(score in {0, 1} for score in scores)
        agreement = all_scored and scores[0] == scores[1]
        disagreement = all_scored and scores[0] != scores[1]
        conservative_pass = scores == [1, 1]
        self_values = [item.get("self_judge") for item in judgments.values()]
        any_self_judge = any(value is True for value in self_values)
        audits.append(
            {
                "rubric_id": spec.rubric_id,
                "case_id": spec.case_id,
                "category": spec.category,
                "rubric_index": spec.rubric_index,
                "rubric_sha256": spec.rubric_sha256,
                "rubric": spec.text,
                "judges": judgments,
                "agreement": agreement,
                "disagreement": disagreement,
                "unscored": not all_scored,
                "conservative_pass_including_self": conservative_pass,
                "independent_conservative_pass": conservative_pass
                and not any_self_judge
                and all(value is False for value in self_values),
                "any_self_judge": any_self_judge,
            }
        )

    grouped = {
        category: _aggregate_audits([item for item in audits if item["category"] == category])
        for category in DRB2_RUBRIC_CATEGORIES
    }
    case_order = list(dict.fromkeys(spec.case_id for spec in specs))
    cases = {
        case_id: {
            **_aggregate_audits([item for item in audits if item["case_id"] == case_id]),
            "generation_status": _case_generation_status(generation_records[case_id]),
            "citation_grounding": _case_citation_grounding(generation_records[case_id]),
        }
        for case_id in case_order
    }
    return {
        "schema_version": "1.0",
        "protocol_name": DRB2_PROTOCOL_NAME,
        "benchmark_kind": "local_public_drb2_rubric_dual_judge",
        "official_benchmark_score": False,
        "interpretation": (
            "This is a local Kimi/Opus dual-judge protocol based on public DRB II rubrics, "
            "not an official DRB II leaderboard result."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lineage": {
            "generation_artifact_id": expected_generation_id,
            "generation_manifest_id": generation_manifest_id,
            "rubrics_sha256": expected_rubrics_sha,
            "judge_artifacts": {
                role: _sha256_path(artifact["path"]) for role, artifact in artifacts.items()
            },
        },
        **_aggregate_audits(audits),
        "generation_status": _aggregate_generation_status(generation_records),
        "groups": grouped,
        "cases": cases,
        "citation_grounding": _aggregate_citation_grounding(generation_records),
        "rubrics": audits,
    }


def _load_and_validate_judge_artifact(
    path: Path,
    *,
    expected_role: str,
    expected_generation_id: str,
    expected_generation_manifest_id: str,
    expected_rubrics_sha: str,
    expected_rubric_count: int,
    expected_rubric_ids: set[str],
) -> dict[str, Any]:
    source = path.expanduser().resolve()
    header, records = load_drb2_judge_artifact(source)
    expected = {
        "protocol_name": DRB2_PROTOCOL_NAME,
        "generation_artifact_id": expected_generation_id,
        "generation_manifest_id": expected_generation_manifest_id,
        "rubrics_sha256": expected_rubrics_sha,
        "judge_role": expected_role,
        "expected_rubric_count": expected_rubric_count,
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise ValueError(f"{expected_role} judge artifact lineage mismatch: {key}")
    if set(records) - expected_rubric_ids:
        raise ValueError(f"{expected_role} judge artifact contains unknown rubric IDs")
    requested = str(header.get("requested_judge_model") or "").strip()
    if not requested:
        raise ValueError(f"{expected_role} judge artifact has no requested model")
    if expected_role == "kimi" and "kimi" not in requested.lower():
        raise ValueError("kimi judge artifact does not use a Kimi model")
    if expected_role == "opus" and not response_model_matches("claude-opus-4-8", requested):
        raise ValueError("opus judge artifact does not use claude-opus-4-8")
    return {"path": source, "header": header, "records": records}


def _validated_judgment(
    row: dict[str, Any] | None,
    *,
    spec: DRB2RubricSpec,
    report: str,
    header: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(row, dict):
        return _unscored_judgment("missing_judge_record")
    required = {
        "rubric_id",
        "rubric_sha256",
        "score",
        "reason",
        "report_evidence_quote",
        "provider",
        "judge_role",
        "requested_judge_model",
        "actual_judge_model",
        "self_judge",
        "attempt",
        "protocol_violation",
    }
    missing = sorted(required - row.keys())
    if missing:
        return _unscored_judgment("missing fields: " + ", ".join(missing), raw=row)
    if row.get("rubric_id") != spec.rubric_id or row.get("rubric_sha256") != spec.rubric_sha256:
        return _unscored_judgment("rubric identity mismatch", raw=row)
    if row.get("judge_role") != header.get("judge_role"):
        return _unscored_judgment("judge role mismatch", raw=row)
    if str(row.get("provider") or "").strip().lower() != "llm-gateway":
        return _unscored_judgment("judge provider mismatch", raw=row)
    requested = str(row.get("requested_judge_model") or "").strip()
    if requested != str(header.get("requested_judge_model") or "").strip():
        return _unscored_judgment("requested judge model mismatch", raw=row)
    score = row.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or score not in {-1, 0, 1}:
        return _unscored_judgment("invalid score", raw=row)
    reason = row.get("reason")
    quote = row.get("report_evidence_quote")
    if not isinstance(reason, str) or not reason.strip():
        return _unscored_judgment("reason is missing", raw=row)
    if not isinstance(quote, str):
        return _unscored_judgment("report evidence quote is invalid", raw=row)
    if quote and quote not in report:
        return _unscored_judgment("report evidence quote is not exact", raw=row)
    if score == 1 and not quote:
        return _unscored_judgment("passing score has no report evidence quote", raw=row)
    actual = row.get("actual_judge_model")
    if score in {0, 1} and (
        not isinstance(actual, str)
        or not actual.strip()
        or not response_model_matches(requested, actual)
    ):
        return _unscored_judgment("actual judge model mismatch", raw=row)
    self_judge = row.get("self_judge")
    if self_judge is not None and not isinstance(self_judge, bool):
        return _unscored_judgment("self_judge is invalid", raw=row)
    if score in {0, 1} and row.get("protocol_violation") is not None:
        return _unscored_judgment("scored result contains a protocol violation", raw=row)
    return {
        "score": score,
        "reason": reason.strip(),
        "report_evidence_quote": quote,
        "actual_judge_model": actual,
        "self_judge": row.get("self_judge"),
        "attempt": row.get("attempt"),
        "protocol_violation": row.get("protocol_violation"),
    }


def _unscored_judgment(
    reason: str,
    *,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = raw or {}
    return {
        "score": -1,
        "reason": f"invalid judge artifact: {reason}",
        "report_evidence_quote": "",
        "actual_judge_model": source.get("actual_judge_model"),
        "self_judge": source.get("self_judge")
        if isinstance(source.get("self_judge"), bool)
        else None,
        "attempt": source.get("attempt"),
        "protocol_violation": reason,
    }


def _aggregate_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = len(audits)
    judge_summaries: dict[str, Any] = {}
    for role in ("kimi", "opus"):
        counts = Counter(item["judges"][role]["score"] for item in audits)
        scored = counts[0] + counts[1]
        judge_summaries[role] = {
            "fixed_denominator": denominator,
            "scored_count": scored,
            "score_1_count": counts[1],
            "score_0_count": counts[0],
            "unscored_count": counts[-1],
            "score_1_rate_fixed": round(counts[1] / denominator, 4) if denominator else 0.0,
            "score_1_rate_scored": round(counts[1] / scored, 4) if scored else None,
            "self_judge_count": sum(
                1 for item in audits if item["judges"][role].get("self_judge") is True
            ),
        }
    conservative = sum(1 for item in audits if item["conservative_pass_including_self"])
    independent = sum(1 for item in audits if item["independent_conservative_pass"])
    return {
        "rubric_count": denominator,
        "judge_summaries": judge_summaries,
        "agreement_count": sum(1 for item in audits if item["agreement"]),
        "agreement_rate": round(sum(1 for item in audits if item["agreement"]) / denominator, 4)
        if denominator
        else 0.0,
        "disagreement_count": sum(1 for item in audits if item["disagreement"]),
        "disagreement_rate": round(
            sum(1 for item in audits if item["disagreement"]) / denominator, 4
        )
        if denominator
        else 0.0,
        "unscored_count": sum(1 for item in audits if item["unscored"]),
        "unscored_rate": round(sum(1 for item in audits if item["unscored"]) / denominator, 4)
        if denominator
        else 0.0,
        "self_judged_rubric_count": sum(1 for item in audits if item["any_self_judge"]),
        "conservative_pass_including_self_count": conservative,
        "conservative_pass_including_self_rate": round(conservative / denominator, 4)
        if denominator
        else 0.0,
        "independent_conservative_pass_count": independent,
        "independent_conservative_pass_rate": round(independent / denominator, 4)
        if denominator
        else 0.0,
    }


def _case_citation_grounding(record: dict[str, Any]) -> dict[str, Any]:
    citation_check = record.get("citation_check")
    check = citation_check if isinstance(citation_check, dict) else {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    return {
        "citation_grounding": _optional_number(
            _first_not_none(
                record.get("citation_grounding"),
                metrics.get("citation_grounding"),
                check.get("citation_grounding"),
            )
        ),
        "citation_precision": _optional_number(
            _first_not_none(
                record.get("citation_precision"),
                metrics.get("citation_precision"),
                check.get("citation_precision"),
            )
        ),
        "citation_coverage": _optional_number(
            _first_not_none(
                record.get("citation_coverage"),
                metrics.get("citation_coverage"),
                check.get("citation_coverage"),
            )
        ),
        "unsupported_claim_rate": _optional_number(
            _first_not_none(
                record.get("unsupported_claim_rate"),
                metrics.get("unsupported_claim_rate"),
                check.get("unsupported_claim_rate"),
            )
        ),
        "judge_identities": citation_judge_identities(check),
        "mixed_into_content_rubric_score": False,
    }


def _case_generation_status(record: dict[str, Any]) -> dict[str, Any]:
    execution_success = record.get("execution_success") is True
    report_emitted = record.get("report_emitted") is True
    contaminated = record.get("benchmark_contamination") is True
    status = (
        "benchmark_contamination"
        if contaminated
        else "success"
        if execution_success
        else "execution_failure"
    )
    return {
        "status": status,
        "execution_success": execution_success,
        "report_emitted": report_emitted,
        "benchmark_contamination": contaminated,
        "error": record.get("error"),
        "error_category": record.get("error_category"),
    }


def _aggregate_generation_status(
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = {case_id: _case_generation_status(record) for case_id, record in records.items()}
    return {
        "case_count": len(cases),
        "execution_success_count": sum(1 for item in cases.values() if item["execution_success"]),
        "execution_failure_count": sum(
            1 for item in cases.values() if not item["execution_success"]
        ),
        "report_emitted_count": sum(1 for item in cases.values() if item["report_emitted"]),
        "report_not_emitted_count": sum(1 for item in cases.values() if not item["report_emitted"]),
        "benchmark_contamination_count": sum(
            1 for item in cases.values() if item["benchmark_contamination"]
        ),
        "error_count": sum(1 for item in cases.values() if item["error"]),
        "cases": cases,
        "mixed_into_rubric_scores": False,
    }


def _aggregate_citation_grounding(
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = {case_id: _case_citation_grounding(record) for case_id, record in records.items()}
    output: dict[str, Any] = {
        "policy": "recorded generation-artifact citation check",
        "mixed_into_content_rubric_score": False,
        "cases": cases,
    }
    for metric in (
        "citation_grounding",
        "citation_precision",
        "citation_coverage",
        "unsupported_claim_rate",
    ):
        values = [case[metric] for case in cases.values() if case[metric] is not None]
        output[f"{metric}_avg"] = round(mean(values), 4) if values else None
        output[f"{metric}_scored_count"] = len(values)
    return output


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _sha256_path(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize independent Kimi/Opus DRB II judges.")
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--kimi-judge", type=Path, required=True)
    parser.add_argument("--opus-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = summarize_drb2_dual_judges(
        generation_path=args.generation,
        rubrics_path=args.rubrics,
        kimi_judge_path=args.kimi_judge,
        opus_judge_path=args.opus_judge,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
