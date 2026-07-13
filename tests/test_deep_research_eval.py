from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import deepresearch_agent.deep_research_eval as eval_module
import pytest
from deepresearch_agent.config import Settings
from deepresearch_agent.deep_research_eval import (
    _effective_citation_judge_model,
    _prediction_payload,
    _sealed_stdout_summary,
    _validate_single_model_run_args,
    load_eval_cases,
    load_eval_cases_from_file,
    run_public_deep_research_eval,
)
from deepresearch_agent.eval_judge import (
    AnswerJudgment,
    DeepSeekAnswerJudgeProvider,
    HeuristicAnswerJudgeProvider,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


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


def test_load_eval_cases_applies_offset_to_local_case_files(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"c{index}", "question": f"Question {index}?"})
            for index in range(1, 6)
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        cases=str(path),
        benchmark_name="custom",
        dataset="livedrbench-preview",
        split="test",
        offset=2,
        limit=2,
        request_timeout_seconds=8.0,
    )

    cases = load_eval_cases(args)

    assert [case["id"] for case in cases] == ["c3", "c4"]


def test_prediction_payload_keeps_structured_json_when_present() -> None:
    assert _prediction_payload('```json\n[{"answer": "x"}]\n```') == [{"answer": "x"}]
    assert _prediction_payload('{"answer": "x"}') == [{"answer": "x"}]
    assert _prediction_payload("plain markdown answer") == [["plain markdown answer"]]


def test_gateway_citation_judge_uses_gateway_model_default() -> None:
    args = argparse.Namespace(
        citation_judge_provider="llm-gateway",
        citation_judge_model=None,
    )

    assert _effective_citation_judge_model(
        args,
        Settings(
            citation_judge_model="deepseek-v4-flash",
            citation_judge_gateway_model="glm-5.2",
        ),
    ) == "glm-5.2"


def test_single_model_run_requires_all_generation_models_to_match() -> None:
    model = "claude-opus-4-8"
    matching = argparse.Namespace(
        single_model_run=True,
        llm_model=model,
        brief_model=model,
        planner_model=model,
        synthesis_model=model,
        gateway_web_search_model=model,
    )

    _validate_single_model_run_args(matching)

    matching.gateway_web_search_model = "glm-5.2"
    with pytest.raises(ValueError, match="--gateway-web-search-model"):
        _validate_single_model_run_args(matching)


def test_single_model_run_rejects_in_generation_llm_judges() -> None:
    model = "claude-opus-4-8"
    args = argparse.Namespace(
        single_model_run=True,
        llm_model=model,
        brief_model=model,
        planner_model=model,
        synthesis_model=model,
        gateway_web_search_model=model,
        citation_judge_provider="llm-gateway",
        judge_provider="none",
    )

    with pytest.raises(ValueError, match="citation judge"):
        _validate_single_model_run_args(args)

    args.citation_judge_provider = "none"
    args.judge_provider = "llm-gateway"
    with pytest.raises(ValueError, match="answer judge"):
        _validate_single_model_run_args(args)


def test_single_model_run_rejects_llm_citation_judge_from_effective_settings() -> None:
    model = "claude-opus-4-8"
    args = argparse.Namespace(
        single_model_run=True,
        llm_model=model,
        brief_model=model,
        planner_model=model,
        synthesis_model=model,
        gateway_web_search_model=model,
        citation_judge_provider=None,
        judge_provider="none",
    )

    with pytest.raises(ValueError, match="citation judge"):
        _validate_single_model_run_args(
            args,
            settings=Settings(citation_judge_provider="llm-gateway"),
        )


def test_single_model_gateway_eval_requires_auditable_html_crawler() -> None:
    model = "claude-opus-4-8"
    args = argparse.Namespace(
        single_model_run=True,
        llm_model=model,
        brief_model=model,
        planner_model=model,
        synthesis_model=model,
        gateway_web_search_model=model,
        citation_judge_provider="none",
        judge_provider="none",
        search_provider="gateway-web",
        web_crawler_provider="jina_reader",
    )

    with pytest.raises(ValueError, match="web-crawler-provider html"):
        _validate_single_model_run_args(args)

    args.web_crawler_provider = "html"
    _validate_single_model_run_args(args)


def test_single_model_run_requires_explicit_generation_model() -> None:
    args = argparse.Namespace(single_model_run=True, llm_model=None)

    with pytest.raises(ValueError, match="explicit --llm-model"):
        _validate_single_model_run_args(args)


def test_heuristic_answer_judge_scores_ground_truth_groups() -> None:
    provider = HeuristicAnswerJudgeProvider()

    judgment = provider.judge(
        {"metadata": {"ground_truths": [["paper title", "alternate"], ["second fact"]]}},
        {"answer": "The generated answer names the alternate and the second fact."},
    )

    assert judgment.score == 1.0
    assert judgment.verdict == "pass"
    assert judgment.missing == []


def test_deepseek_answer_judge_parses_json_response(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    requested = {}

    def fake_urlopen(request, timeout):
        requested["url"] = request.full_url
        requested["timeout"] = timeout
        requested["authorization"] = request.get_header("Authorization")
        requested["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "score": 0.75,
                                    "verdict": "partial",
                                    "reason": "answer missed one expected group",
                                    "matched": ["paper title"],
                                    "missing": ["second fact"],
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_cache_hit_tokens": 10,
                    "prompt_cache_miss_tokens": 90,
                },
            }
        )

    monkeypatch.setattr("deepresearch_agent.eval_judge.urlopen", fake_urlopen)
    provider = DeepSeekAnswerJudgeProvider(model="deepseek-v4-flash", timeout_seconds=2.0)

    judgment = provider.judge(
        {
            "query": "Find the benchmark paper.",
            "metadata": {"ground_truths": [["paper title"], ["second fact"]]},
        },
        {"answer": "The answer mentions only the paper title."},
    )

    assert requested["url"] == "https://api.deepseek.com/chat/completions"
    assert requested["timeout"] == 2.0
    assert requested["authorization"] == "Bearer sk-test-key"
    assert requested["payload"]["model"] == "deepseek-v4-flash"
    assert requested["payload"]["response_format"] == {"type": "json_object"}
    user_payload = json.loads(requested["payload"]["messages"][1]["content"])
    assert user_payload["ground_truth_groups"] == [["paper title"], ["second fact"]]
    assert judgment.provider == "deepseek"
    assert judgment.model == "deepseek-v4-flash"
    assert judgment.score == 0.75
    assert judgment.verdict == "partial"
    assert judgment.matched == ["paper title"]
    assert judgment.missing == ["second fact"]
    assert judgment.input_tokens == 100
    assert judgment.output_tokens == 20
    assert judgment.estimated_cost_usd > 0


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
    assert summary["answer_quality_scored_count"] == 1
    assert summary["answer_quality_avg"] == 1.0
    assert summary["execution_success_rate"] == 1.0
    assert summary["success_semantics"].startswith("deprecated alias")
    assert summary["config"]["settings"]["local_retrieval_mode"] == "keyword"
    assert summary["config"]["request_timeout_seconds"] == 4.0
    assert summary["config"]["gateway_web_search_timeout_seconds"] == 4.0
    assert summary["config"]["llm_gateway_timeout_seconds"] == 4.0
    assert summary["config"]["citation_judge_timeout_seconds"] == 4.0
    assert summary["config"]["answer_judge_timeout_seconds"] == 4.0
    assert summary["config"]["settings"]["request_timeout_seconds"] == 4.0
    assert summary["config"]["settings"]["llm_gateway_timeout_seconds"] == 4.0
    assert summary["config"]["settings"]["citation_judge_timeout_seconds"] == 4.0
    assert summary["manifest"]["git_commit_sha"]
    assert summary["manifest"]["dataset_version"].startswith("sha256:")
    assert summary["manifest"]["prompt_bundle_hash"].startswith("sha256:")
    assert summary["manifest"]["llm_provider"] == "mock"
    assert summary["manifest"]["config_snapshot"]["llm_gateway_timeout_seconds"] == 4.0
    assert summary["manifest"]["deterministic"] is True
    assert summary["deterministic"] is True
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
    assert raw_lines[0]["manifest"]["manifest_id"] == summary["manifest"]["manifest_id"]
    assert raw_lines[1]["type"] == "case_result"
    assert raw_lines[1]["sources"]
    assert raw_lines[1]["answer_judgment"]["verdict"] == "pass"
    assert raw_lines[1]["answer_quality"] == 1.0


def test_mock_eval_without_judge_never_infers_answer_quality_from_citations(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    raw_log = tmp_path / "raw.jsonl"
    summary_output = tmp_path / "summary.json"
    predictions_output = tmp_path / "predictions.json"
    cases_path.write_text(
        json.dumps(
            {
                "id": "case-no-judge",
                "query": "How should citation checking reduce unsupported claims?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        cases=str(cases_path),
        dataset="livedrbench-preview",
        benchmark_name="no-judge-smoke",
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
        judge_provider="none",
        raw_log=str(raw_log),
        summary_output=str(summary_output),
        predictions_output=str(predictions_output),
    )

    summary = asyncio.run(run_public_deep_research_eval(args))
    raw_lines = [
        json.loads(line)
        for line in raw_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = raw_lines[1]

    assert record["citation_retention_rate"] >= 0.8
    assert record["answer_quality"] is None
    assert record["metrics"]["answer_quality"] is None
    assert record["success"] is record["execution_success"] is True
    assert summary["answer_quality_scored_count"] == 0
    assert summary["answer_quality_avg"] is None
    assert summary["answer_judge"]["scored_count"] == 0
    assert summary["answer_judge"]["pass_rate"] is None
    assert "answer_quality_success_rate" not in summary


def test_sealed_holdout_omits_question_answer_gold_and_raw_artifacts(tmp_path: Path) -> None:
    query_sentinel = "PRIVATE_QUESTION_SENTINEL_93ac"
    answer_sentinel = "PRIVATE_ANSWER_SENTINEL_d2f1"
    cases_path = tmp_path / "sealed.jsonl"
    summary_output = tmp_path / "sealed-summary.json"
    cases_path.write_text(
        json.dumps(
            {
                "id": "private-case-1",
                "query": query_sentinel,
                "answer": answer_sentinel,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        cases=str(cases_path),
        dataset="livedrbench-preview",
        benchmark_name="sealed-smoke",
        split="test",
        offset=0,
        limit=None,
        search_provider="mock",
        llm_provider="mock",
        llm_model=None,
        brief_model=None,
        planner_model=None,
        synthesis_model=None,
        embedding_provider="local",
        local_retrieval_mode="keyword",
        max_researchers=1,
        max_results=1,
        max_rounds=1,
        max_tool_calls=1,
        deadline_seconds=None,
        min_evidence_items=1,
        fallback_policy="fail",
        request_timeout_seconds=4.0,
        reflection_enabled=False,
        max_reflection_rounds=1,
        reflection_min_sources=4,
        citation_judge_provider=None,
        citation_judge_model=None,
        searxng_base_url=None,
        bing_search_base_url=None,
        gateway_web_search_model=None,
        web_crawler_provider=None,
        jina_reader_base_url=None,
        jina_search_base_url=None,
        crawler_max_chars=None,
        seed=20260607,
        judge_provider="heuristic",
        judge_model=None,
        raw_log=None,
        summary_output=str(summary_output),
        predictions_output=None,
        replay_dir=None,
        cassette_id=None,
        rejudge_replay=False,
        sealed_holdout=True,
    )

    summary = asyncio.run(run_public_deep_research_eval(args))
    serialized = json.dumps(summary, ensure_ascii=False)
    stdout_payload = json.dumps(_sealed_stdout_summary(summary), ensure_ascii=False)

    assert summary["benchmark_kind"] == "sealed_holdout_aggregate_eval"
    assert summary["raw_log"] is None
    assert summary["predictions_output"] is None
    assert query_sentinel not in serialized
    assert answer_sentinel not in serialized
    assert query_sentinel not in stdout_payload
    assert answer_sentinel not in stdout_payload
    assert "query" not in summary["records"][0]
    assert "reason" not in summary["records"][0]["answer_judgment"]
    assert "matched" not in summary["records"][0]["answer_judgment"]
    assert summary_output.exists()
    assert query_sentinel not in summary_output.read_text(encoding="utf-8")
    assert answer_sentinel not in summary_output.read_text(encoding="utf-8")


def test_sealed_main_emits_only_aggregate_and_creates_no_private_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    query_sentinel = "PRIVATE_QUERY_SENTINEL_47f8"
    answer_sentinel = "PRIVATE_ANSWER_SENTINEL_a6c3"
    cases_path = tmp_path / "sealed-cases.jsonl"
    summary_output = tmp_path / "sealed-summary.json"
    trace_dir = tmp_path / "sealed-traces"
    cases_path.write_text(
        json.dumps(
            {
                "id": answer_sentinel,
                "query": query_sentinel,
                "category": answer_sentinel,
                "answer": answer_sentinel,
                "ground_truths": [[answer_sentinel]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _EchoingJudge:
        def judge(self, case, record) -> AnswerJudgment:
            return AnswerJudgment(
                provider=query_sentinel,
                model=answer_sentinel,
                score=0.5,
                verdict=query_sentinel,
                confidence=0.75,
                reason=answer_sentinel,
                matched=[query_sentinel],
                missing=[answer_sentinel],
                critical_errors=[query_sentinel],
                failure_categories=[answer_sentinel, "format"],
            )

    monkeypatch.setattr(
        eval_module,
        "build_eval_judge_provider",
        lambda *args, **kwargs: _EchoingJudge(),
    )
    # Redirect root-relative default artifacts into the isolated test directory.
    monkeypatch.setattr(
        eval_module,
        "__file__",
        str(tmp_path / "src" / "deepresearch_agent" / "deep_research_eval.py"),
    )
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("TRACE_WRITE_ENABLED", "true")
    monkeypatch.setenv("TRACE_EXPORTER", "otlp_http")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deep-research-eval",
            "--cases",
            str(cases_path),
            "--benchmark-name",
            "sealed-sentinel-smoke",
            "--search-provider",
            "mock",
            "--llm-provider",
            "mock",
            "--local-retrieval-mode",
            "keyword",
            "--max-researchers",
            "1",
            "--max-results",
            "1",
            "--judge-provider",
            "heuristic",
            "--summary-output",
            str(summary_output),
            "--sealed-holdout",
        ],
    )

    eval_module.main()

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    summary_text = summary_output.read_text(encoding="utf-8")
    assert stdout_payload["benchmark_kind"] == "sealed_holdout_aggregate_eval"
    assert stdout_payload["case_count"] == 1
    assert stdout_payload["execution_success_count"] == 1
    assert query_sentinel not in captured.out
    assert answer_sentinel not in captured.out
    assert query_sentinel not in captured.err
    assert answer_sentinel not in captured.err
    assert query_sentinel not in summary_text
    assert answer_sentinel not in summary_text

    summary = json.loads(summary_text)
    assert summary["raw_log"] is None
    assert summary["predictions_output"] is None
    assert summary["records"][0]["answer_judgment"] == {
        "score": None,
        "verdict": "unscored",
        "confidence": 0.75,
        "failure_categories": ["format"],
    }
    assert not trace_dir.exists()
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "results" / "livedrbench_predictions.json").exists()


def test_fixed_benchmark_cases_cover_required_offline_scenarios() -> None:
    cases_path = Path(__file__).resolve().parents[1] / "data" / "benchmark_cases.jsonl"
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(cases) == 24
    assert {case["language"] for case in cases} == {"en", "zh-CN"}
    assert {"text", "markdown", "json"} <= {case["expected_format"] for case in cases}
    assert {
        "single_fact",
        "comparison",
        "multi_hop",
        "citation_conflict",
        "tool_failure",
        "structured_output",
    } <= {case["category"] for case in cases}
    assert all("answer_quality" not in case for case in cases)


def test_public_eval_replay_is_offline_and_does_not_reinvoke_live_judge(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    first_raw = tmp_path / "first.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "replay-case",
                "query": "How should replay preserve evaluation artifacts?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    common = {
        "dataset": "livedrbench-preview",
        "benchmark_name": "replay-smoke",
        "split": "test",
        "offset": 0,
        "limit": None,
        "search_provider": "mock",
        "llm_provider": "mock",
        "llm_model": None,
        "embedding_provider": "local",
        "local_retrieval_mode": "keyword",
        "max_researchers": 1,
        "max_results": 1,
        "request_timeout_seconds": 4.0,
        "seed": 20260607,
        "judge_model": None,
    }
    first_args = argparse.Namespace(
        **common,
        cases=str(cases_path),
        judge_provider="none",
        raw_log=str(first_raw),
        summary_output=str(tmp_path / "first-summary.json"),
        predictions_output=str(tmp_path / "first-predictions.json"),
        replay_dir=None,
        cassette_id=None,
    )
    asyncio.run(run_public_deep_research_eval(first_args))

    replay_raw = tmp_path / "replay.jsonl"
    replay_args = argparse.Namespace(
        **common,
        cases=None,
        judge_provider="deepseek",
        raw_log=str(replay_raw),
        summary_output=str(tmp_path / "replay-summary.json"),
        predictions_output=str(tmp_path / "replay-predictions.json"),
        replay_dir=str(first_raw),
        cassette_id=None,
    )
    summary = asyncio.run(run_public_deep_research_eval(replay_args))
    rows = [
        json.loads(line)
        for line in replay_raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["case_count"] == 1
    assert summary["deterministic"] is True
    assert summary["manifest"]["replay_kind"] == "benchmark_snapshot"
    assert summary["manifest"]["replay_artifact_id"].startswith("sha256:")
    assert summary["manifest"]["cassette_id"].startswith("sha256:")
    assert rows[1]["replayed"] is True
    assert rows[1].get("answer_judgment") is None


def test_public_eval_tracks_local_and_live_rejudge_determinism(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    first_raw = tmp_path / "first.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "rejudge-case",
                "query": "Which marker must appear?",
                "answer": "citation marker",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    common = {
        "dataset": "livedrbench-preview",
        "benchmark_name": "rejudge-smoke",
        "split": "test",
        "offset": 0,
        "limit": None,
        "search_provider": "mock",
        "llm_provider": "mock",
        "llm_model": None,
        "embedding_provider": "local",
        "local_retrieval_mode": "keyword",
        "max_researchers": 1,
        "max_results": 1,
        "request_timeout_seconds": 4.0,
        "seed": 20260607,
        "judge_model": None,
    }
    first_args = argparse.Namespace(
        **common,
        cases=str(cases_path),
        judge_provider="none",
        raw_log=str(first_raw),
        summary_output=str(tmp_path / "first-summary.json"),
        predictions_output=str(tmp_path / "first-predictions.json"),
        replay_dir=None,
        cassette_id=None,
        rejudge_replay=False,
    )
    asyncio.run(run_public_deep_research_eval(first_args))

    replay_args = argparse.Namespace(
        **common,
        cases=None,
        judge_provider="heuristic",
        raw_log=str(tmp_path / "rejudged.jsonl"),
        summary_output=str(tmp_path / "rejudged-summary.json"),
        predictions_output=str(tmp_path / "rejudged-predictions.json"),
        replay_dir=str(first_raw),
        cassette_id=None,
        rejudge_replay=True,
    )
    summary = asyncio.run(run_public_deep_research_eval(replay_args))

    assert summary["answer_judge"]["scored_count"] == 1
    assert summary["records"][0]["answer_judgment"] is not None
    assert summary["config"]["rejudge_replay"] is True
    assert summary["manifest"]["generation_deterministic"] is True
    assert summary["manifest"]["evaluation_deterministic"] is True
    assert summary["deterministic"] is True

    class _RecordedLiveJudge:
        def judge(self, case, record) -> AnswerJudgment:
            del case, record
            return AnswerJudgment(
                provider="llm-gateway",
                model="kimi-k2.7-code-highspeed",
                score=1.0,
                verdict="pass",
                reason="test live rejudge",
                matched=["citation marker"],
                missing=[],
            )

    monkeypatch.setattr(
        eval_module,
        "build_eval_judge_provider",
        lambda *args, **kwargs: _RecordedLiveJudge(),
    )
    live_args = argparse.Namespace(
        **{**common, "judge_model": "kimi-k2.7-code-highspeed"},
        cases=None,
        judge_provider="llm-gateway",
        raw_log=str(tmp_path / "live-rejudged.jsonl"),
        summary_output=str(tmp_path / "live-rejudged-summary.json"),
        predictions_output=str(tmp_path / "live-rejudged-predictions.json"),
        replay_dir=str(first_raw),
        cassette_id=None,
        rejudge_replay=True,
    )
    live_summary = asyncio.run(run_public_deep_research_eval(live_args))

    assert live_summary["manifest"]["generation_deterministic"] is True
    assert live_summary["manifest"]["evaluation_deterministic"] is False
    assert live_summary["deterministic"] is False
    assert live_summary["manifest"]["evaluation_judges"][-1] == {
        "kind": "answer",
        "provider": "llm-gateway",
        "model": "kimi-k2.7-code-highspeed",
        "executed": True,
        "deterministic": False,
    }
