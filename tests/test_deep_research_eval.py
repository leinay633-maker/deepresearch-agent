from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from deepresearch_agent.deep_research_eval import (
    _prediction_payload,
    load_eval_cases_from_file,
    run_public_deep_research_eval,
)
from deepresearch_agent.eval_judge import HeuristicAnswerJudgeProvider


def test_load_eval_cases_from_jsonl_normalizes_livedrbench_shape(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "key": "live-001",
                "question": "Find the paper that introduced a benchmark.",
                "category": "academic",
                "ground_truths": [["paper title"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_eval_cases_from_file(path, "livedrbench-preview")

    assert cases == [
        {
            "id": "live-001",
            "query": "Find the paper that introduced a benchmark.",
            "category": "academic",
            "benchmark_name": "livedrbench-preview",
            "metadata": {"ground_truths": [["paper title"]]},
        }
    ]


def test_load_eval_cases_from_csv_accepts_query_column(tmp_path: Path) -> None:
    path = tmp_path / "cases.csv"
    path.write_text("id,query,category\nc1,What is agentic RAG?,systems\n", encoding="utf-8")

    cases = load_eval_cases_from_file(path, "custom")

    assert cases[0]["id"] == "c1"
    assert cases[0]["query"] == "What is agentic RAG?"
    assert cases[0]["category"] == "systems"


def test_prediction_payload_keeps_structured_json_when_present() -> None:
    assert _prediction_payload('```json\n[{"answer": "x"}]\n```') == [{"answer": "x"}]
    assert _prediction_payload('{"answer": "x"}') == [{"answer": "x"}]
    assert _prediction_payload("plain markdown answer") == [["plain markdown answer"]]


def test_heuristic_answer_judge_scores_ground_truth_groups() -> None:
    provider = HeuristicAnswerJudgeProvider()

    judgment = provider.judge(
        {"metadata": {"ground_truths": [["paper title", "alternate"], ["second fact"]]}},
        {"answer": "The generated answer names the alternate and the second fact."},
    )

    assert judgment.score == 1.0
    assert judgment.verdict == "pass"
    assert judgment.missing == []


def test_run_public_eval_with_mock_writes_artifacts(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    raw_log = tmp_path / "raw.jsonl"
    summary_output = tmp_path / "summary.json"
    predictions_output = tmp_path / "predictions.json"
    cases_path.write_text(
        json.dumps(
            {
                "id": "case-001",
                "query": "How does citation checking reduce hallucination in agentic RAG?",
                "ground_truths": [["citation checking"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        cases=str(cases_path),
        dataset="livedrbench-preview",
        benchmark_name="local-smoke",
        split="test",
        offset=0,
        limit=None,
        search_provider="mock",
        llm_provider="mock",
        llm_model=None,
        embedding_provider="local",
        local_retrieval_mode="keyword",
        max_researchers=1,
        max_results=1,
        request_timeout_seconds=4.0,
        seed=20260607,
        judge_provider="heuristic",
        raw_log=str(raw_log),
        summary_output=str(summary_output),
        predictions_output=str(predictions_output),
    )

    summary = asyncio.run(run_public_deep_research_eval(args))

    assert summary["case_count"] == 1
    assert summary["config"]["benchmark_name"] == "local-smoke"
    assert summary["config"]["llm_provider"] == "mock"
    assert summary["config"]["judge_provider"] == "heuristic"
    assert summary["answer_judge"]["scored_count"] == 1
    assert summary["answer_judge"]["score_avg"] == 1.0
    assert summary["config"]["settings"]["local_retrieval_mode"] == "keyword"
    assert raw_log.exists()
    assert summary_output.exists()
    predictions = json.loads(predictions_output.read_text(encoding="utf-8"))
    assert predictions[0]["key"] == "case-001"
    raw_lines = [
        json.loads(line)
        for line in raw_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert raw_lines[0]["type"] == "config"
    assert raw_lines[1]["type"] == "case_result"
    assert raw_lines[1]["sources"]
    assert raw_lines[1]["answer_judgment"]["verdict"] == "pass"
