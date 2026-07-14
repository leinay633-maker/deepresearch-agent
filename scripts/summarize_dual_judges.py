#!/usr/bin/env python3
"""Merge independent SimpleQA replay judgments into one auditable ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from deepresearch_agent.llm_gateway import response_model_matches
from deepresearch_agent.replay import case_result_artifact_id


VERDICTS = ("correct", "incorrect", "not_attempted", "unscored")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _case_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["case_id"]): row
        for row in rows
        if row.get("type") == "case_result" and row.get("case_id")
    }


def _artifact_manifest(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    configs = [row for row in rows if row.get("type") == "config"]
    if len(configs) != 1 or not isinstance(configs[0].get("manifest"), dict):
        raise ValueError(f"{label} artifact requires exactly one config manifest")
    return configs[0]["manifest"]


def _invalid_judgment(reason: str, judgment: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = judgment or {}
    return {
        "verdict": "unscored",
        "confidence": 0.0,
        "reason": f"invalid judge artifact: {reason}",
        "actual_judge_model": raw.get("model"),
        "self_judge": raw.get("self_judge") is True,
        "grounded_correct": False,
    }


def _validated_judgment(
    record: dict[str, Any],
    *,
    expected_provider: str,
    expected_model: str,
) -> dict[str, Any]:
    judgment = record.get("answer_judgment")
    if not isinstance(judgment, dict):
        return _invalid_judgment("answer_judgment is missing")
    required = {
        "provider",
        "score",
        "verdict",
        "reason",
        "matched",
        "missing",
        "model",
        "confidence",
        "critical_errors",
        "failure_categories",
        "self_judge",
    }
    missing = sorted(required - judgment.keys())
    if missing:
        return _invalid_judgment("missing fields: " + ", ".join(missing), judgment)
    verdict = str(judgment.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        return _invalid_judgment("invalid verdict", judgment)
    if str(record.get("answer_verdict") or "").strip().lower() != verdict:
        return _invalid_judgment("top-level verdict contradicts answer_judgment", judgment)
    if not isinstance(judgment.get("provider"), str) or not judgment["provider"].strip():
        return _invalid_judgment("provider is missing", judgment)
    if judgment["provider"].strip().lower() != expected_provider.lower():
        return _invalid_judgment("provider contradicts replay manifest", judgment)
    if not isinstance(judgment.get("model"), str) or not judgment["model"].strip():
        return _invalid_judgment("actual judge model is missing", judgment)
    if not response_model_matches(expected_model, judgment["model"]):
        return _invalid_judgment("actual judge model contradicts replay manifest", judgment)
    if not isinstance(judgment.get("reason"), str) or not judgment["reason"].strip():
        return _invalid_judgment("reason is missing", judgment)
    confidence = judgment.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return _invalid_judgment("confidence is invalid", judgment)
    if not isinstance(judgment.get("self_judge"), bool):
        return _invalid_judgment("self_judge must be explicit", judgment)
    for field in ("matched", "missing", "critical_errors", "failure_categories"):
        value = judgment.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return _invalid_judgment(f"{field} must be an array of strings", judgment)
    score = judgment.get("score")
    expected_score = None if verdict == "unscored" else 1.0 if verdict == "correct" else 0.0
    if isinstance(score, bool) or score != expected_score:
        return _invalid_judgment("score contradicts verdict", judgment)
    answer_quality = record.get("answer_quality")
    if answer_quality != expected_score:
        return _invalid_judgment("answer_quality contradicts verdict", judgment)
    if record.get("grounded_correct") is True and verdict != "correct":
        return _invalid_judgment("grounded_correct contradicts verdict", judgment)
    return {
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "reason": judgment["reason"].strip(),
        "actual_judge_model": judgment["model"].strip(),
        "self_judge": judgment["self_judge"],
        "grounded_correct": record.get("grounded_correct") is True,
    }


def summarize_dual_judges(
    cases: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    judge_rows: dict[str, list[dict[str, Any]]],
    *,
    generation_artifact_id: str,
) -> dict[str, Any]:
    if len(judge_rows) < 2:
        raise ValueError("at least two judge artifacts are required")
    generation_manifest = _artifact_manifest(generation_rows, label="generation")
    generation_manifest_id = str(generation_manifest.get("manifest_id") or "")
    if not generation_manifest_id:
        raise ValueError("generation manifest_id is missing")
    generation = _case_records(generation_rows)
    judges = {name: _case_records(rows) for name, rows in judge_rows.items()}
    judge_contracts: dict[str, tuple[str, str]] = {}
    for name, rows in judge_rows.items():
        manifest = _artifact_manifest(rows, label=f"judge {name}")
        if manifest.get("replay_kind") != "benchmark_snapshot":
            raise ValueError(f"judge {name} is not a benchmark snapshot replay")
        if manifest.get("replay_artifact_id") != generation_artifact_id:
            raise ValueError(f"judge {name} did not replay the generation artifact")
        config = manifest.get("config_snapshot")
        if not isinstance(config, dict):
            raise ValueError(f"judge {name} replay manifest has no config snapshot")
        provider = str(config.get("judge_provider") or "").strip()
        model = str(config.get("judge_model") or "").strip()
        if not provider or not model:
            raise ValueError(f"judge {name} replay manifest has no judge contract")
        judge_contracts[name] = (provider, model)
    audits: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id") or case.get("case_id") or "")
        if case_id not in generation:
            raise ValueError(f"generation artifact is missing case {case_id}")
        generated = generation[case_id]
        if generated.get("manifest_id") != generation_manifest_id:
            raise ValueError(f"generation manifest mismatch for case {case_id}")
        answer = str(generated.get("answer") or "")
        judgments: dict[str, dict[str, Any]] = {}
        for name, records in judges.items():
            if case_id not in records:
                raise ValueError(f"judge {name} is missing case {case_id}")
            record = records[case_id]
            if str(record.get("answer") or "") != answer:
                raise ValueError(f"judge {name} replay answer differs for case {case_id}")
            replay = record.get("generation_replay")
            if not isinstance(replay, dict) or replay.get(
                "source_manifest_id"
            ) != generated.get("manifest_id"):
                raise ValueError(f"judge {name} replay lineage differs for case {case_id}")
            provider, model = judge_contracts[name]
            judgments[name] = _validated_judgment(
                record,
                expected_provider=provider,
                expected_model=model,
            )
        verdicts = [item["verdict"] for item in judgments.values()]
        all_scored = all(verdict != "unscored" for verdict in verdicts)
        unanimous = len(set(verdicts)) == 1
        dual_correct_including_self = (
            all_scored and unanimous and verdicts[0] == "correct"
        )
        any_self_judge = any(item["self_judge"] for item in judgments.values())
        audits.append(
            {
                "case_id": case_id,
                "query": str(case.get("query") or case.get("question") or ""),
                "reference_answer": str(
                    case.get("answer") or case.get("expected_answer") or ""
                ),
                "generated_answer": answer,
                "execution_success": generated.get("execution_success") is True,
                "judges": judgments,
                "all_judges_scored": all_scored,
                "unanimous_verdict": verdicts[0] if all_scored and unanimous else None,
                "dual_judge_correct_including_self": dual_correct_including_self,
                "independent_dual_correct": dual_correct_including_self
                and not any_self_judge,
                "any_self_judge": any_self_judge,
                "manual_review_required": not all_scored
                or not unanimous
                or any_self_judge,
            }
        )

    judge_summaries: dict[str, dict[str, Any]] = {}
    for name in judges:
        verdict_counts = Counter(item["judges"][name]["verdict"] for item in audits)
        judge_summaries[name] = {
            "verdict_counts": {verdict: verdict_counts[verdict] for verdict in VERDICTS},
            "fixed_denominator": len(audits),
            "correct_rate": round(verdict_counts["correct"] / len(audits), 4)
            if audits
            else 0.0,
            "self_judge_count": sum(
                1 for item in audits if item["judges"][name]["self_judge"]
            ),
        }
    return {
        "schema_version": "1.0",
        "case_count": len(audits),
        "judge_summaries": judge_summaries,
        "independent_dual_correct_count": sum(
            1 for item in audits if item["independent_dual_correct"]
        ),
        "independent_dual_correct_rate": round(
            sum(1 for item in audits if item["independent_dual_correct"])
            / len(audits),
            4,
        )
        if audits
        else 0.0,
        "dual_judge_correct_including_self_count": sum(
            1 for item in audits if item["dual_judge_correct_including_self"]
        ),
        "dual_judge_correct_including_self_rate": round(
            sum(1 for item in audits if item["dual_judge_correct_including_self"])
            / len(audits),
            4,
        )
        if audits
        else 0.0,
        "unscored_case_count": sum(
            1 for item in audits if not item["all_judges_scored"]
        ),
        "disagreement_case_count": sum(
            1
            for item in audits
            if item["all_judges_scored"] and item["unanimous_verdict"] is None
        ),
        "self_judged_case_count": sum(1 for item in audits if item["any_self_judge"]),
        "manual_review_case_ids": [
            item["case_id"] for item in audits if item["manual_review_required"]
        ],
        "cases": audits,
    }


def _judge_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("judge must use NAME=/path/to/raw.jsonl")
    return name.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two or more answer-judge replays.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--judge", action="append", type=_judge_argument, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    judge_paths = dict(args.judge)
    if len(judge_paths) != len(args.judge):
        raise ValueError("judge names must be unique")
    result = summarize_dual_judges(
        _load_jsonl(args.cases),
        _load_jsonl(args.generation),
        {name: _load_jsonl(path) for name, path in judge_paths.items()},
        generation_artifact_id=case_result_artifact_id(args.generation),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
