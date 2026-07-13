from __future__ import annotations

import asyncio
import json

import pytest

from deepresearch_agent.benchmark import refresh_replayed_case_result
from deepresearch_agent.citation import CitationChecker
from deepresearch_agent.citation_judge import HeuristicCitationJudgeProvider
from deepresearch_agent.config import Settings
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.replay import (
    CassetteEntry,
    CassetteLLMProvider,
    CassetteMismatchError,
    CassetteReplayer,
    CassetteSearchAdapter,
    case_result_artifact_id,
    load_case_result_records,
    read_cassette,
    replay_case_result,
    validate_replay_case_ids,
    write_cassette,
)
from deepresearch_agent.schemas import ResearchRequest, Source
from deepresearch_agent.search import MockSearchAdapter, SearchService


def test_jsonl_cassette_round_trip_and_strict_replay(tmp_path) -> None:
    path = tmp_path / "provider.jsonl"
    entries = [
        CassetteEntry(
            sequence=1,
            kind="search",
            operation="search",
            request={"query": "citation grounding", "max_results": 2},
            response={"sources": [{"url": "https://example.com/evidence"}]},
            metadata={"provider": "fixture"},
        )
    ]

    write_cassette(path, entries)
    loaded = read_cassette(path)
    replayer = CassetteReplayer(loaded)
    response = replayer.next_response(
        kind="search",
        operation="search",
        request={"query": "citation grounding", "max_results": 2},
    )

    assert loaded == entries
    assert response == entries[0].response
    assert replayer.remaining == 0
    replayer.assert_exhausted()


def test_cassette_replay_rejects_unconsumed_entries() -> None:
    replayer = CassetteReplayer(
        [
            CassetteEntry(
                sequence=1,
                kind="llm",
                operation="brief",
                request={},
                response={},
            ),
            CassetteEntry(
                sequence=2,
                kind="llm",
                operation="plan",
                request={},
                response={},
            ),
        ]
    )
    replayer.next_response(kind="llm", operation="brief", request={})
    with pytest.raises(CassetteMismatchError, match="unconsumed"):
        replayer.assert_exhausted()


def test_cassette_replay_rejects_request_drift(tmp_path) -> None:
    path = write_cassette(
        tmp_path / "provider.jsonl",
        [
            CassetteEntry(
                sequence=1,
                kind="llm",
                operation="plan",
                request={"query": "original"},
                response={"subquestions": []},
            )
        ],
    )

    with pytest.raises(CassetteMismatchError, match="cassette mismatch"):
        CassetteReplayer.from_path(path).next_response(
            kind="llm",
            operation="plan",
            request={"query": "changed"},
        )


def test_provider_cassette_reexecutes_the_orchestrator() -> None:
    entries = [
        CassetteEntry(
            sequence=1,
            kind="llm",
            operation="brief",
            request={"query": "How should replay work?"},
            response={
                "normalized_query": "How should replay work?",
                "scope": "Replay one deterministic research path.",
                "constraints": [],
                "assumptions": [],
            },
        ),
        CassetteEntry(
            sequence=2,
            kind="llm",
            operation="plan",
            request={
                "normalized_query": "How should replay work?",
                "max_researchers": 1,
            },
            response={
                "subquestions": [
                    {
                        "id": "Q1",
                        "question": "What makes provider replay deterministic?",
                        "rationale": "Verify recorded provider responses.",
                    }
                ]
            },
        ),
        CassetteEntry(
            sequence=3,
            kind="search",
            operation="search",
            request={
                "query": "What makes provider replay deterministic?",
                "max_results": 1,
            },
            response={
                "sources": [
                    {
                        "title": "Replay evidence",
                        "url": "https://example.com/replay",
                        "content": "Recorded provider responses make offline replay deterministic and auditable.",
                        "provider": "cassette",
                        "query": "What makes provider replay deterministic?",
                        "score": 100.0,
                        "metadata": {"extract_status": "ok", "snippet_only": False},
                    }
                ]
            },
        ),
        CassetteEntry(
            sequence=4,
            kind="llm",
            operation="research_decision",
            request={
                "subquestion_id": "Q1",
                "evidence_count": 1,
                "min_evidence_items": 1,
                "round_index": 1,
            },
            response={"action": "stop", "reason": "recorded evidence is sufficient"},
        ),
        CassetteEntry(
            sequence=5,
            kind="llm",
            operation="synthesis",
            request={},
            response={
                "answer": "Recorded provider responses support deterministic replay [S1].",
                "claims": [
                    "Recorded provider responses support deterministic replay [S1]"
                ],
            },
        ),
    ]
    replayer = CassetteReplayer(entries)
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        local_retrieval_mode="keyword",
        max_retries=0,
    )
    service = SearchService(
        primary=CassetteSearchAdapter(replayer),
        fallback=MockSearchAdapter(),
        settings=settings,
        fallback_policy="fail",
    )
    orchestrator = DeepResearchOrchestrator(
        settings=settings,
        search_service=service,
        llm_provider=CassetteLLMProvider(replayer),
    )

    report = asyncio.run(
        orchestrator.run(
            ResearchRequest(
                query="How should replay work?",
                max_researchers=1,
                max_results_per_researcher=1,
                fallback_policy="fail",
            )
        )
    )

    replayer.assert_exhausted()
    assert report.plan[0].id == "Q1"
    assert report.sources[0].id == "S1"
    assert report.citation_check.supported_claims == 1


def test_case_result_artifact_can_be_replayed_without_providers(tmp_path) -> None:
    artifact = tmp_path / "benchmark.jsonl"
    artifact.write_text(
        "\n".join(
            [
                '{"type":"config","config":{}}',
                (
                    '{"type":"case_result","case_id":"case-1","query":"q",'
                    '"answer":"a","success":true,"answer_judgment":'
                    '{"provider":"llm-gateway","model":"glm-5.2","score":1.0}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_case_result_records(artifact)
    replayed = replay_case_result(
        {"id": "case-1", "query": "q"},
        records,
        manifest_id="manifest-new",
    )

    assert replayed["answer"] == "a"
    assert replayed["replayed"] is True
    assert replayed["manifest_id"] == "manifest-new"
    assert replayed["generation_replay"] == {
        "deterministic": True,
        "mode": "recorded_case_artifact",
        "source_manifest_id": None,
    }
    assert replayed["recorded_answer_judgment"] == {
        "provider": "llm-gateway",
        "model": "glm-5.2",
        "score": 1.0,
    }
    assert case_result_artifact_id(artifact).startswith("sha256:")


def test_snapshot_replay_rejects_case_contract_drift() -> None:
    records = {
        "case-1": {
            "type": "case_result",
            "case_id": "case-1",
            "query": "q",
            "expected_format": "text",
        }
    }

    with pytest.raises(CassetteMismatchError, match="contract mismatch"):
        replay_case_result(
            {"id": "case-1", "query": "q", "expected_format": "json"},
            records,
            manifest_id="manifest-new",
        )


def test_snapshot_replay_requires_exact_case_id_set() -> None:
    records = {
        "case-1": {"type": "case_result", "case_id": "case-1", "query": "q1"},
        "case-extra": {
            "type": "case_result",
            "case_id": "case-extra",
            "query": "q2",
        },
    }

    with pytest.raises(CassetteMismatchError, match="case ID set mismatch"):
        validate_replay_case_ids([{"id": "case-1", "query": "q1"}], records)


def test_refresh_replay_preserves_metrics_snapshot() -> None:
    entries = [
        CassetteEntry(
            sequence=1,
            kind="llm",
            operation="brief",
            request={"query": "How should metrics snapshots work?"},
            response={
                "normalized_query": "How should metrics snapshots work?",
                "scope": "test",
                "constraints": [],
                "assumptions": [],
            },
        ),
        CassetteEntry(
            sequence=2,
            kind="llm",
            operation="plan",
            request={
                "normalized_query": "How should metrics snapshots work?",
                "max_researchers": 1,
            },
            response={
                "subquestions": [
                    {"id": "Q1", "question": "metrics evidence", "rationale": "test"}
                ]
            },
        ),
        CassetteEntry(
            sequence=3,
            kind="search",
            operation="search",
            request={"query": "metrics evidence", "max_results": 1},
            response={
                "sources": [
                    {
                        "title": "Metrics evidence",
                        "url": "https://example.com/metrics",
                        "content": "Metrics snapshots preserve recorded evaluation evidence.",
                        "provider": "cassette",
                        "query": "metrics evidence",
                        "metadata": {"extract_status": "ok", "snippet_only": False},
                    }
                ]
            },
        ),
        CassetteEntry(
            sequence=4,
            kind="llm",
            operation="research_decision",
            request={
                "subquestion_id": "Q1",
                "evidence_count": 2,
                "min_evidence_items": 1,
                "round_index": 1,
            },
            response={"action": "stop", "reason": "enough"},
        ),
        CassetteEntry(
            sequence=5,
            kind="llm",
            operation="synthesis",
            request={},
            response={
                "answer": "Metrics snapshots preserve evidence [S1].",
                "claims": ["Metrics snapshots preserve evidence [S1]"],
            },
        ),
    ]
    replayer = CassetteReplayer(entries)
    settings = Settings(local_retrieval_mode="keyword", max_retries=0)
    report = asyncio.run(
        DeepResearchOrchestrator(
            settings=settings,
            search_service=SearchService(
                CassetteSearchAdapter(replayer),
                MockSearchAdapter(),
                settings,
                fallback_policy="fail",
            ),
            llm_provider=CassetteLLMProvider(replayer),
        ).run(
            ResearchRequest(
                query="How should metrics snapshots work?",
                max_researchers=1,
                max_results_per_researcher=1,
                fallback_policy="fail",
            )
        )
    )
    record = {
        "metrics": {"citation_grounding": 0.123, "custom_recorded": 7},
        "report": report.model_dump(mode="json"),
    }

    refreshed = refresh_replayed_case_result(
        {"id": "case-1", "query": report.query}, record
    )

    assert refreshed["recorded_metrics"] == {
        "citation_grounding": 0.123,
        "custom_recorded": 7,
    }
    assert refreshed["recorded_metrics"] is not refreshed["metrics"]
    assert (
        refreshed["report"]["metrics"]["claim_extraction_valid"]
        is refreshed["citation_check"]["claim_extraction_valid"]
    )


def _record_with_citation_judge(provider: str, model: str) -> dict:
    claim = "Recorded evidence supports citation replay [S1]."
    source = Source(
        id="S1",
        title="Recorded evidence",
        url="https://example.com/recorded-evidence",
        content="Recorded evidence supports citation replay.",
        provider="fixture",
        query="citation replay",
        metadata={"extract_status": "ok", "snippet_only": False},
    )
    if provider == "heuristic":
        check = CitationChecker().check(
            [claim],
            [source],
            judge_provider=HeuristicCitationJudgeProvider(),
        )
    else:
        check = CitationChecker().check([claim], [source])
        check = check.model_copy(
            update={
                "assessments": [
                    check.assessments[0].model_copy(
                        update={
                            "judge_provider": provider,
                            "judge_model": model,
                            "judge_confidence": 0.91,
                            "judge_reason": "recorded live judgment",
                        }
                    )
                ]
            }
        )
    check_payload = check.model_dump(mode="json")
    report = {
        "run_id": "recorded-run",
        "query": "How should citation replay work?",
        "brief": {
            "original_query": "How should citation replay work?",
            "normalized_query": "How should citation replay work?",
            "scope": "test",
            "constraints": [],
            "assumptions": [],
        },
        "plan": [],
        "answer": claim,
        "claims": [claim],
        "findings": [],
        "sources": [source.model_dump(mode="json")],
        "citation_check": check_payload,
        "cost": {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_estimated_cost_usd": 0.0,
            "records": [],
        },
        "metrics": {
            "citation_grounding": check.citation_grounding,
            "citation_precision": check.citation_precision,
            "citation_coverage": check.citation_coverage,
            "citation_retention_rate": check.retention_rate,
            "latency_ms": 1.0,
        },
        "trace_events": [],
    }
    return {
        "citation_check": check_payload,
        "metrics": dict(report["metrics"]),
        "report": report,
    }


def test_refresh_replay_marks_live_citation_judge_as_unscored() -> None:
    record = _record_with_citation_judge("llm-gateway", "glm-5.2")
    recorded_check = json.loads(json.dumps(record["citation_check"]))

    refreshed = refresh_replayed_case_result(
        {"id": "case-1", "query": "How should citation replay work?"},
        record,
    )

    assert refreshed["recorded_citation_check"] == recorded_check
    assert refreshed["recorded_citation_check"]["assessments"][0][
        "judge_provider"
    ] == "llm-gateway"
    assert refreshed["recorded_citation_check"]["assessments"][0][
        "judge_model"
    ] == "glm-5.2"
    assert refreshed["replayed_citation_check"]["assessments"][0][
        "judge_provider"
    ] is None
    assert refreshed["citation_replay"] == {
        "status": "unscored",
        "reason": (
            "the recorded citation check used a non-replayable judge; the current "
            "lexical check is diagnostic only and does not replace its scores"
        ),
        "recorded_judges": [{"provider": "llm-gateway", "model": "glm-5.2"}],
        "replayed_judge": {"provider": "lexical", "model": None},
    }
    assert refreshed["citation_check"] is None
    assert refreshed["citation_grounding"] is None
    assert refreshed["citation_precision"] is None
    assert refreshed["citation_coverage"] is None
    assert refreshed["metrics"]["citation_retention_rate"] is None
    assert refreshed["report"]["citation_check"] == recorded_check


def test_refresh_replay_reuses_recorded_deterministic_citation_judge() -> None:
    record = _record_with_citation_judge("heuristic", "local-overlap-judge")

    refreshed = refresh_replayed_case_result(
        {"id": "case-1", "query": "How should citation replay work?"},
        record,
    )

    assert refreshed["citation_replay"]["status"] == "scored"
    assert refreshed["citation_replay"]["recorded_judges"] == [
        {"provider": "heuristic", "model": "local-overlap-judge"}
    ]
    assert refreshed["citation_replay"]["replayed_judge"] == {
        "provider": "heuristic",
        "model": "local-overlap-judge",
    }
    assert refreshed["citation_check"]["assessments"][0][
        "judge_provider"
    ] == "heuristic"
