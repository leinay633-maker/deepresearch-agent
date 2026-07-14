from __future__ import annotations

import asyncio
import csv
import io
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.analyze_eval_snapshot import analyze_snapshot
from scripts.build_simpleqa_public32 import (
    _candidate_rows,
    _metadata_urls,
    _public_url,
    build_manifest,
    select_cases,
)
from scripts.probe_gateway_web_search import probe_models
from scripts.summarize_dual_judges import summarize_dual_judges


def _csv(rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["metadata", "problem", "answer"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def test_public_simpleqa_selection_is_balanced_reproducible_and_safe() -> None:
    rows = []
    for topic in ("History", "Science"):
        for answer_type in ("Person", "Date"):
            for index in range(3):
                rows.append(
                    {
                        "metadata": (
                            "{'topic': '"
                            + topic
                            + "', 'answer_type': '"
                            + answer_type
                            + "', 'urls': ['https://example.com/item-"
                            + str(index)
                            + "']}"
                        ),
                        "problem": f"{topic} {answer_type} question {index}?",
                        "answer": f"answer-{topic}-{answer_type}-{index}",
                    }
                )
    candidates = _candidate_rows(_csv(rows), excluded_queries={"history person question 0"})

    first = select_cases(candidates, count=8, seed=17)
    second = select_cases(candidates, count=8, seed=17)

    assert first == second
    assert len(first) == 8
    assert {row["category"] for row in first} == {"History", "Science"}
    assert {row["answer_type"] for row in first} == {"Person", "Date"}
    assert Counter(row["category"] for row in first) == {"History": 4, "Science": 4}
    assert Counter(row["answer_type"] for row in first) == {"Person": 4, "Date": 4}
    assert all(row["expected_format"] == "text" for row in first)
    assert all("answer" not in row["query"].lower() for row in first)
    assert all(row["gold_urls"] for row in first)


def test_public_simpleqa_selection_balances_types_globally_not_per_topic() -> None:
    rows = []
    for topic in ("Art", "History", "Science"):
        for answer_type in ("Date", "Number", "Other", "Person", "Place"):
            for index in range(3):
                rows.append(
                    {
                        "metadata": (
                            f"{{'topic': '{topic}', 'answer_type': '{answer_type}', "
                            f"'urls': ['https://example.com/{topic}/{answer_type}/{index}']}}"
                        ),
                        "problem": f"{topic} {answer_type} question {index}?",
                        "answer": f"{topic}-{answer_type}-{index}",
                    }
                )

    selected = select_cases(_candidate_rows(_csv(rows), excluded_queries=set()), count=12, seed=9)

    assert sorted(Counter(row["category"] for row in selected).values()) == [4, 4, 4]
    assert sorted(Counter(row["answer_type"] for row in selected).values()) == [2, 2, 2, 3, 3]
    for topic in ("Art", "History", "Science"):
        assert len({row["answer_type"] for row in selected if row["category"] == topic}) >= 3


def test_metadata_url_extraction_never_evaluates_upstream_text() -> None:
    metadata = (
        "{'topic': 'TV shows', 'answer_type': 'Person', "
        "'urls': [\"https://example.com/George_O'Malley\", "
        "'https://example.org/wiki/Betulia_(Antioquia)']}"
    )

    urls = _metadata_urls(metadata)

    assert urls
    assert "https://example.com/George_O'Malley" in urls
    assert "https://example.org/wiki/Betulia_(Antioquia)" in urls


def test_metadata_url_extraction_splits_escaped_newlines_and_rejects_junk() -> None:
    metadata = (
        "{'urls': ['https://example.com/first\\n\\nhttps://example.org/second', "
        "'https://example.net/third', 'https://bad.example/path\\junk']}"
    )

    assert _metadata_urls(metadata) == [
        "https://example.com/first",
        "https://example.org/second",
        "https://example.net/third",
    ]


def test_manifest_records_source_and_selection_identity() -> None:
    cases = [
        {
            "id": "c1",
            "source_index": 7,
            "query": "Question?",
            "answer": "Answer",
            "category": "History",
            "answer_type": "Other",
            "gold_urls": ["https://example.com/source"],
            "expected_format": "text",
        }
    ]

    manifest = build_manifest(
        source_url="https://example.com/simpleqa.csv",
        source_sha256="a" * 64,
        seed=5,
        count=1,
        cases=cases,
        excluded_queries={"excluded question"},
        candidate_count=10,
    )

    assert manifest["selection_seed"] == 5
    assert manifest["source_sha256"] == "a" * 64
    assert manifest["source_indices"] == [7]
    assert manifest["candidate_count_after_exclusions"] == 10
    assert manifest["selection_distribution"] == {
        "topic": {"History": 1},
        "answer_type": {"Other": 1},
    }
    assert manifest["excluded_query_count"] == 1
    assert len(manifest["excluded_query_sha256"]) == 64
    assert len(manifest["case_sha256"]) == 64


def test_frozen_public_cases_only_contain_lexically_valid_gold_urls() -> None:
    cases_path = Path("evals/simpleqa_public32_v1.jsonl")
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]

    assert len(cases) == 32
    for case in cases:
        assert case["gold_urls"]
        assert case["gold_urls"] == list(dict.fromkeys(case["gold_urls"]))
        assert all(_public_url(url) == url for url in case["gold_urls"])


def test_capability_probe_artifact_is_secret_free_without_key(
    monkeypatch,
) -> None:
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)

    result = asyncio.run(
        probe_models(
            ["glm-5.2"],
            base_url="https://gateway.example",
            timeout_seconds=1.0,
        )
    )

    assert result["probe_retains_response_text"] is False
    assert result["probes"][0]["status"] == "missing_api_key"
    assert "response" not in result["probes"][0]


def test_snapshot_analysis_compares_gold_url_and_context_budgets() -> None:
    cases = [
        {
            "id": "c1",
            "query": "What year was San Carlos founded?",
            "answer": "1786",
            "gold_urls": ["https://example.com/san-carlos#history"],
        }
    ]
    records = [
        {
            "type": "case_result",
            "case_id": "c1",
            "execution_success": True,
            "sources": [
                {
                    "id": "S1",
                    "title": "San Carlos history",
                    "url": "https://example.com/san-carlos?ref=search",
                    "content": "The municipality of San Carlos was founded in 1786.",
                    "provider": "gateway-web",
                    "query": "San Carlos founding year",
                    "score": 3.0,
                    "quality_score": 0.8,
                    "metadata": {"extract_status": "ok", "snippet_only": False},
                },
                {
                    "id": "S2",
                    "title": "Unverified search candidate",
                    "url": "https://example.org/search-only",
                    "content": "A search snippet repeats 1786 but was never crawled.",
                    "provider": "gateway-web",
                    "query": "San Carlos founding year",
                    "score": 2.0,
                    "quality_score": 0.2,
                    "metadata": {
                        "extract_status": "crawl_failed",
                        "snippet_only": True,
                    },
                }
            ],
        }
    ]

    result = analyze_snapshot(cases, records)

    assert result["gold_url_retrieved_count"] == 1
    assert result["gold_url_citable_count"] == 1
    assert result["answer_in_source_text_count"] == 1
    assert result["answer_in_citable_source_count"] == 1
    assert result["answer_in_snippet_or_failed_source_count"] == 1
    assert result["answer_in_packed_650_count"] == 1
    assert result["answer_in_packed_1200_count"] == 1
    assert result["answer_in_citable_packed_650_count"] == 1
    assert result["answer_in_citable_packed_1200_count"] == 1
    assert result["cases"][0]["source_count"] == 2
    assert result["cases"][0]["citable_source_count"] == 1
    assert result["cases"][0]["snippet_or_failed_source_count"] == 1


def test_dual_judge_summary_keeps_unscored_disagreement_and_self_judge_visible() -> None:
    cases = [{"id": "c1", "query": "Question?", "answer": "Reference"}]
    generation_manifest_id = "generation-manifest"
    generation_artifact_id = "sha256:generation"
    generation = [
        {
            "type": "config",
            "manifest": {"manifest_id": generation_manifest_id},
        },
        {
            "type": "case_result",
            "case_id": "c1",
            "answer": "Candidate",
            "execution_success": True,
            "manifest_id": generation_manifest_id,
        }
    ]

    def judgment_record(
        *, verdict: str, model: str, self_judge: bool, reason: str
    ) -> dict:
        score = 1.0 if verdict == "correct" else 0.0
        return {
            **generation[1],
            "replayed": True,
            "generation_replay": {"source_manifest_id": generation_manifest_id},
            "answer_verdict": verdict,
            "answer_quality": score,
            "grounded_correct": verdict == "correct",
            "answer_judgment": {
                "provider": "llm-gateway",
                "score": score,
                "verdict": verdict,
                "confidence": 0.9,
                "reason": reason,
                "matched": [],
                "missing": [],
                "model": model,
                "critical_errors": [],
                "failure_categories": [],
                "self_judge": self_judge,
            },
        }

    judge_config = {
        "type": "config",
        "manifest": {
            "replay_kind": "benchmark_snapshot",
            "replay_artifact_id": generation_artifact_id,
            "config_snapshot": {
                "judge_provider": "llm-gateway",
                "judge_model": "kimi-k2.7-code-highspeed",
            },
        },
    }
    judge_rows = {
        "kimi": [
            judge_config,
            judgment_record(
                verdict="correct",
                model="kimi-k2.7-code-highspeed",
                self_judge=True,
                reason="matches",
            ),
        ],
        "opus": [
            {
                "type": "config",
                "manifest": {
                    "replay_kind": "benchmark_snapshot",
                    "replay_artifact_id": generation_artifact_id,
                    "config_snapshot": {
                        "judge_provider": "llm-gateway",
                        "judge_model": "claude-opus-4-8",
                    },
                },
            },
            judgment_record(
                verdict="incorrect",
                model="claude-opus-4-8",
                self_judge=False,
                reason="wrong",
            ),
        ],
    }

    result = summarize_dual_judges(
        cases,
        generation,
        judge_rows,
        generation_artifact_id=generation_artifact_id,
    )

    assert result["case_count"] == 1
    assert result["independent_dual_correct_count"] == 0
    assert result["dual_judge_correct_including_self_count"] == 0
    assert result["disagreement_case_count"] == 1
    assert result["unscored_case_count"] == 0
    assert result["self_judged_case_count"] == 1
    assert result["manual_review_case_ids"] == ["c1"]


def test_dual_judge_summary_marks_incomplete_optimistic_record_unscored() -> None:
    cases = [{"id": "c1", "query": "Question?", "answer": "Reference"}]
    generation = [
        {"type": "config", "manifest": {"manifest_id": "m1"}},
        {
            "type": "case_result",
            "case_id": "c1",
            "answer": "Candidate",
            "manifest_id": "m1",
        },
    ]
    config = {
        "type": "config",
        "manifest": {
            "replay_kind": "benchmark_snapshot",
            "replay_artifact_id": "sha256:generation",
            "config_snapshot": {
                "judge_provider": "llm-gateway",
                "judge_model": "claude-opus-4-8",
            },
        },
    }
    incomplete = {
        **generation[1],
        "answer_verdict": "correct",
        "answer_quality": 1.0,
        "generation_replay": {"source_manifest_id": "m1"},
        "answer_judgment": {"verdict": "correct"},
    }

    result = summarize_dual_judges(
        cases,
        generation,
        {"judge1": [config, incomplete], "judge2": [config, incomplete]},
        generation_artifact_id="sha256:generation",
    )

    assert result["unscored_case_count"] == 1
    assert result["independent_dual_correct_count"] == 0
    assert result["dual_judge_correct_including_self_count"] == 0


def test_dual_judge_summary_does_not_call_self_judged_agreement_independent() -> None:
    cases = [{"id": "c1", "query": "Question?", "answer": "Reference"}]
    generation = [
        {"type": "config", "manifest": {"manifest_id": "m1"}},
        {
            "type": "case_result",
            "case_id": "c1",
            "answer": "Reference",
            "manifest_id": "m1",
        },
    ]
    config = {
        "type": "config",
        "manifest": {
            "replay_kind": "benchmark_snapshot",
            "replay_artifact_id": "sha256:generation",
            "config_snapshot": {
                "judge_provider": "llm-gateway",
                "judge_model": "kimi-k2.7-code-highspeed",
            },
        },
    }

    def correct_record(model: str, self_judge: bool) -> dict:
        return {
            **generation[1],
            "generation_replay": {"source_manifest_id": "m1"},
            "answer_verdict": "correct",
            "answer_quality": 1.0,
            "grounded_correct": True,
            "answer_judgment": {
                "provider": "llm-gateway",
                "score": 1.0,
                "verdict": "correct",
                "confidence": 0.9,
                "reason": "matches",
                "matched": ["Reference"],
                "missing": [],
                "model": model,
                "critical_errors": [],
                "failure_categories": [],
                "self_judge": self_judge,
            },
        }

    result = summarize_dual_judges(
        cases,
        generation,
        {
            "kimi": [config, correct_record("kimi-k2.7-code-highspeed", True)],
            "opus": [
                {
                    "type": "config",
                    "manifest": {
                        "replay_kind": "benchmark_snapshot",
                        "replay_artifact_id": "sha256:generation",
                        "config_snapshot": {
                            "judge_provider": "llm-gateway",
                            "judge_model": "claude-opus-4-8",
                        },
                    },
                },
                correct_record("claude-opus-4-8", False),
            ],
        },
        generation_artifact_id="sha256:generation",
    )

    assert result["dual_judge_correct_including_self_count"] == 1
    assert result["independent_dual_correct_count"] == 0
    assert result["manual_review_case_ids"] == ["c1"]


def test_dual_judge_summary_rejects_wrong_replay_artifact_identity() -> None:
    cases = [{"id": "c1", "query": "Question?", "answer": "Reference"}]
    generation = [
        {"type": "config", "manifest": {"manifest_id": "m1"}},
        {
            "type": "case_result",
            "case_id": "c1",
            "answer": "Candidate",
            "manifest_id": "m1",
        },
    ]
    wrong_config = {
        "type": "config",
        "manifest": {
            "replay_kind": "benchmark_snapshot",
            "replay_artifact_id": "sha256:wrong",
            "config_snapshot": {
                "judge_provider": "llm-gateway",
                "judge_model": "claude-opus-4-8",
            },
        },
    }

    with pytest.raises(ValueError, match="did not replay"):
        summarize_dual_judges(
            cases,
            generation,
            {"judge1": [wrong_config], "judge2": [wrong_config]},
            generation_artifact_id="sha256:generation",
        )
