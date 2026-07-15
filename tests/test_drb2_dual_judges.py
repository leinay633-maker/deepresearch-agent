from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deepresearch_agent.drb2_rubric_eval import (
    DRB2_PROTOCOL_NAME,
    load_drb2_rubric_specs,
)
from deepresearch_agent.replay import case_result_artifact_id
from scripts.summarize_drb2_dual_judges import summarize_drb2_dual_judges


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _generation_and_rubrics(tmp_path: Path) -> tuple[Path, Path]:
    generation = tmp_path / "generation.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    _write_jsonl(
        generation,
        [
            {
                "type": "config",
                "manifest": {
                    "manifest_id": "m1",
                    "llm_model": "generation-model",
                },
            },
            {
                "type": "case_result",
                "case_id": "task-1",
                "manifest_id": "m1",
                "query": "Explain the report.",
                "answer": "Alpha fact. Beta analysis.\n\n## Presentation",
                "execution_success": True,
                "report_emitted": True,
                "benchmark_contamination": False,
                "error": None,
                "error_category": None,
                "citation_grounding": 0.75,
                "citation_precision": 0.8,
                "citation_coverage": 1.0,
                "unsupported_claim_rate": 0.25,
                "citation_check": {
                    "citation_grounding": 0.75,
                    "citation_precision": 0.8,
                    "citation_coverage": 1.0,
                    "unsupported_claim_rate": 0.25,
                    "assessments": [
                        {
                            "judge_provider": "heuristic",
                            "judge_model": "local-overlap-judge",
                        }
                    ],
                },
            },
        ],
    )
    _write_jsonl(
        rubrics,
        [
            {
                "id": "task-1",
                "info_recall": ["Include alpha."],
                "analysis": ["Analyze beta."],
                "presentation": ["Use a heading."],
            }
        ],
    )
    return generation, rubrics


def _judge_artifact(
    path: Path,
    *,
    role: str,
    model: str,
    generation: Path,
    rubrics: Path,
    scores: dict[str, int],
    self_judge: bool = False,
    actual_model: str | None = None,
    omit: set[str] | None = None,
    quote_override: dict[str, str] | None = None,
) -> None:
    specs = load_drb2_rubric_specs(rubrics)
    quote_by_category = {
        "info_recall": "Alpha fact.",
        "analysis": "Beta analysis.",
        "presentation": "## Presentation",
    }
    rows = [
        {
            "type": "drb2_judge_config",
            "schema_version": "1.0",
            "protocol_name": DRB2_PROTOCOL_NAME,
            "generation_artifact_id": case_result_artifact_id(generation),
            "generation_manifest_id": "m1",
            "rubrics_sha256": hashlib.sha256(rubrics.read_bytes()).hexdigest(),
            "judge_role": role,
            "requested_judge_model": model,
            "expected_rubric_count": len(specs),
        }
    ]
    for spec in specs:
        if spec.rubric_id in (omit or set()):
            continue
        score = scores[spec.category]
        quote = (quote_override or {}).get(
            spec.rubric_id,
            quote_by_category[spec.category] if score == 1 else "",
        )
        rows.append(
            {
                "type": "drb2_rubric_result",
                "rubric_id": spec.rubric_id,
                "rubric_sha256": spec.rubric_sha256,
                "score": score,
                "reason": "fixed test judgment",
                "report_evidence_quote": quote,
                "provider": "llm-gateway",
                "judge_role": role,
                "requested_judge_model": model,
                "actual_judge_model": actual_model or model,
                "self_judge": self_judge,
                "attempt": 1,
                "protocol_violation": None if score in {0, 1} else "unscored_test",
            }
        )
    _write_jsonl(path, rows)


def test_dual_summary_reports_pass_disagreement_unscored_groups_and_citations(
    tmp_path: Path,
) -> None:
    generation, rubrics = _generation_and_rubrics(tmp_path)
    kimi = tmp_path / "kimi.jsonl"
    opus = tmp_path / "opus.jsonl"
    _judge_artifact(
        kimi,
        role="kimi",
        model="kimi-k2.7-code-highspeed",
        generation=generation,
        rubrics=rubrics,
        scores={"info_recall": 1, "analysis": 1, "presentation": -1},
    )
    _judge_artifact(
        opus,
        role="opus",
        model="claude-opus-4-8",
        generation=generation,
        rubrics=rubrics,
        scores={"info_recall": 1, "analysis": 0, "presentation": 0},
    )

    result = summarize_drb2_dual_judges(
        generation_path=generation,
        rubrics_path=rubrics,
        kimi_judge_path=kimi,
        opus_judge_path=opus,
    )

    assert result["protocol_name"] == DRB2_PROTOCOL_NAME
    assert result["official_benchmark_score"] is False
    assert "not an official" in result["interpretation"]
    assert result["rubric_count"] == 3
    assert result["conservative_pass_including_self_count"] == 1
    assert result["agreement_count"] == 1
    assert result["disagreement_count"] == 1
    assert result["unscored_count"] == 1
    assert result["groups"]["info_recall"]["conservative_pass_including_self_count"] == 1
    assert result["groups"]["analysis"]["disagreement_count"] == 1
    assert result["groups"]["presentation"]["unscored_count"] == 1
    assert result["judge_summaries"]["kimi"] == {
        "fixed_denominator": 3,
        "scored_count": 2,
        "score_1_count": 2,
        "score_0_count": 0,
        "unscored_count": 1,
        "score_1_rate_fixed": 0.6667,
        "score_1_rate_scored": 1.0,
        "self_judge_count": 0,
    }
    assert result["citation_grounding"]["citation_grounding_avg"] == 0.75
    assert result["citation_grounding"]["mixed_into_content_rubric_score"] is False
    assert result["generation_status"] == {
        "case_count": 1,
        "execution_success_count": 1,
        "execution_failure_count": 0,
        "report_emitted_count": 1,
        "report_not_emitted_count": 0,
        "benchmark_contamination_count": 0,
        "error_count": 0,
        "cases": {
            "task-1": {
                "status": "success",
                "execution_success": True,
                "report_emitted": True,
                "benchmark_contamination": False,
                "error": None,
                "error_category": None,
            }
        },
        "mixed_into_rubric_scores": False,
    }
    assert result["cases"]["task-1"]["generation_status"]["status"] == "success"
    assert result["cases"]["task-1"]["citation_grounding"]["judge_identities"] == [
        {"provider": "heuristic", "model": "local-overlap-judge"}
    ]


def test_both_zero_is_agreement_but_not_a_pass(tmp_path: Path) -> None:
    generation, rubrics = _generation_and_rubrics(tmp_path)
    kimi = tmp_path / "kimi.jsonl"
    opus = tmp_path / "opus.jsonl"
    scores = {category: 0 for category in ("info_recall", "analysis", "presentation")}
    _judge_artifact(
        kimi,
        role="kimi",
        model="kimi-k2.7-code-highspeed",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
    )
    _judge_artifact(
        opus,
        role="opus",
        model="claude-opus-4-8",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
    )

    result = summarize_drb2_dual_judges(
        generation_path=generation,
        rubrics_path=rubrics,
        kimi_judge_path=kimi,
        opus_judge_path=opus,
    )

    assert result["agreement_count"] == 3
    assert result["conservative_pass_including_self_count"] == 0


def test_self_judged_dual_one_is_not_independent(tmp_path: Path) -> None:
    generation, rubrics = _generation_and_rubrics(tmp_path)
    kimi = tmp_path / "kimi.jsonl"
    opus = tmp_path / "opus.jsonl"
    scores = {category: 1 for category in ("info_recall", "analysis", "presentation")}
    _judge_artifact(
        kimi,
        role="kimi",
        model="kimi-k2.7-code-highspeed",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
        self_judge=True,
    )
    _judge_artifact(
        opus,
        role="opus",
        model="claude-opus-4-8",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
    )

    result = summarize_drb2_dual_judges(
        generation_path=generation,
        rubrics_path=rubrics,
        kimi_judge_path=kimi,
        opus_judge_path=opus,
    )

    assert result["conservative_pass_including_self_count"] == 3
    assert result["independent_conservative_pass_count"] == 0
    assert result["self_judged_rubric_count"] == 3


def test_missing_judge_row_is_unscored_without_shrinking_denominator(
    tmp_path: Path,
) -> None:
    generation, rubrics = _generation_and_rubrics(tmp_path)
    specs = load_drb2_rubric_specs(rubrics)
    kimi = tmp_path / "kimi.jsonl"
    opus = tmp_path / "opus.jsonl"
    scores = {category: 1 for category in ("info_recall", "analysis", "presentation")}
    _judge_artifact(
        kimi,
        role="kimi",
        model="kimi-k2.7-code-highspeed",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
        omit={specs[0].rubric_id},
    )
    _judge_artifact(
        opus,
        role="opus",
        model="claude-opus-4-8",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
    )

    result = summarize_drb2_dual_judges(
        generation_path=generation,
        rubrics_path=rubrics,
        kimi_judge_path=kimi,
        opus_judge_path=opus,
    )

    assert result["rubric_count"] == 3
    assert result["unscored_count"] == 1
    assert result["rubrics"][0]["judges"]["kimi"]["score"] == -1
    assert "missing_judge_record" in result["rubrics"][0]["judges"]["kimi"]["reason"]


def test_tampered_quote_and_actual_model_are_unscored(tmp_path: Path) -> None:
    generation, rubrics = _generation_and_rubrics(tmp_path)
    specs = load_drb2_rubric_specs(rubrics)
    kimi = tmp_path / "kimi.jsonl"
    opus = tmp_path / "opus.jsonl"
    scores = {category: 1 for category in ("info_recall", "analysis", "presentation")}
    _judge_artifact(
        kimi,
        role="kimi",
        model="kimi-k2.7-code-highspeed",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
        quote_override={specs[0].rubric_id: "invented evidence"},
    )
    _judge_artifact(
        opus,
        role="opus",
        model="claude-opus-4-8",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
        actual_model="glm-5.2",
    )

    result = summarize_drb2_dual_judges(
        generation_path=generation,
        rubrics_path=rubrics,
        kimi_judge_path=kimi,
        opus_judge_path=opus,
    )

    assert result["unscored_count"] == 3
    assert result["rubrics"][0]["judges"]["kimi"]["score"] == -1
    assert all(item["judges"]["opus"]["score"] == -1 for item in result["rubrics"])


def test_wrong_generation_lineage_is_rejected(tmp_path: Path) -> None:
    generation, rubrics = _generation_and_rubrics(tmp_path)
    kimi = tmp_path / "kimi.jsonl"
    opus = tmp_path / "opus.jsonl"
    scores = {category: 1 for category in ("info_recall", "analysis", "presentation")}
    _judge_artifact(
        kimi,
        role="kimi",
        model="kimi-k2.7-code-highspeed",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
    )
    _judge_artifact(
        opus,
        role="opus",
        model="claude-opus-4-8",
        generation=generation,
        rubrics=rubrics,
        scores=scores,
    )
    rows = [json.loads(line) for line in kimi.read_text().splitlines() if line.strip()]
    rows[0]["generation_artifact_id"] = "sha256:wrong"
    _write_jsonl(kimi, rows)

    with pytest.raises(ValueError, match="lineage mismatch"):
        summarize_drb2_dual_judges(
            generation_path=generation,
            rubrics_path=rubrics,
            kimi_judge_path=kimi,
            opus_judge_path=opus,
        )


def test_generation_failure_and_judge_unscored_remain_separate_ledgers(
    tmp_path: Path,
) -> None:
    generation, rubrics = _generation_and_rubrics(tmp_path)
    rows = [json.loads(line) for line in generation.read_text().splitlines() if line.strip()]
    rows[1].update(
        {
            "answer": "",
            "execution_success": False,
            "report_emitted": False,
            "benchmark_contamination": True,
            "error": "BenchmarkContaminationError('blocked source')",
            "error_category": "benchmark_contamination",
        }
    )
    _write_jsonl(generation, rows)
    kimi = tmp_path / "kimi.jsonl"
    opus = tmp_path / "opus.jsonl"
    unscored = {category: -1 for category in ("info_recall", "analysis", "presentation")}
    _judge_artifact(
        kimi,
        role="kimi",
        model="kimi-k2.7-code-highspeed",
        generation=generation,
        rubrics=rubrics,
        scores=unscored,
    )
    _judge_artifact(
        opus,
        role="opus",
        model="claude-opus-4-8",
        generation=generation,
        rubrics=rubrics,
        scores=unscored,
    )

    result = summarize_drb2_dual_judges(
        generation_path=generation,
        rubrics_path=rubrics,
        kimi_judge_path=kimi,
        opus_judge_path=opus,
    )

    assert result["generation_status"]["execution_failure_count"] == 1
    assert result["generation_status"]["benchmark_contamination_count"] == 1
    assert result["generation_status"]["error_count"] == 1
    assert result["cases"]["task-1"]["generation_status"] == {
        "status": "benchmark_contamination",
        "execution_success": False,
        "report_emitted": False,
        "benchmark_contamination": True,
        "error": "BenchmarkContaminationError('blocked source')",
        "error_category": "benchmark_contamination",
    }
    assert result["unscored_count"] == 3
    assert result["judge_summaries"]["kimi"]["score_0_count"] == 0
    assert result["judge_summaries"]["opus"]["score_0_count"] == 0
