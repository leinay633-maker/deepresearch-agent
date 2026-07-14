from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from deepresearch_agent.benchmark import (
    build_benchmark_manifest,
    mark_live_judge_nondeterminism,
    require_clean_worktree,
    run_benchmark,
    sanitized_settings_snapshot,
)
from deepresearch_agent.config import Settings


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _manifest(root: Path) -> dict:
    return build_benchmark_manifest(
        root=root,
        benchmark_name="manifest-test",
        dataset_name="cases.jsonl",
        cases=[{"id": "case-1", "query": "q"}],
        config_snapshot={"provider": "mock"},
        llm_provider="mock",
        llm_model="mock-model",
        search_provider="mock",
        seed=7,
    )


def test_settings_snapshot_redacts_credential_bearing_values() -> None:
    sentinel = "NEVER_WRITE_THIS_BEARER_SECRET"
    snapshot = sanitized_settings_snapshot(
        Settings(
            otel_exporter_otlp_headers=f"Authorization=Bearer {sentinel}",
            qdrant_api_key_env="QDRANT_REAL_KEY_NAME",
            openai_compatible_api_key_env="OPENAI_REAL_KEY_NAME",
        )
    )

    serialized = repr(snapshot)
    assert sentinel not in serialized
    assert snapshot["otel_exporter_otlp_headers"] == "<redacted>"
    assert snapshot["qdrant_api_key_env"] == "QDRANT_REAL_KEY_NAME"
    assert snapshot["openai_compatible_api_key_env"] == "OPENAI_REAL_KEY_NAME"
    assert snapshot["mock_input_cost_per_1m_tokens"] == 0.0


def test_dataset_content_hash_excludes_benchmark_run_label(tmp_path: Path) -> None:
    common = {
        "root": tmp_path,
        "dataset_name": "cases.jsonl",
        "cases": [
            {
                "id": "case-1",
                "query": "Who won?",
                "benchmark_name": "label-inside-case",
                "metadata": {"answer": "Ada"},
            }
        ],
        "config_snapshot": {"provider": "mock"},
        "llm_provider": "mock",
        "llm_model": "mock-model",
        "search_provider": "mock",
        "seed": 7,
    }

    first = build_benchmark_manifest(benchmark_name="run-a", **common)
    common["cases"][0]["benchmark_name"] = "different-case-label"
    second = build_benchmark_manifest(benchmark_name="run-b", **common)

    assert first["dataset_version"] == second["dataset_version"]
    assert first["dataset_content_hash"] == second["dataset_content_hash"]
    assert first["manifest_id"] != second["manifest_id"]


def test_require_clean_worktree_fails_closed() -> None:
    require_clean_worktree({"git_dirty": False})

    with pytest.raises(RuntimeError, match="verified clean git worktree"):
        require_clean_worktree({"git_dirty": True})
    with pytest.raises(RuntimeError, match="verified clean git worktree"):
        require_clean_worktree({"git_dirty": None})


def test_replay_manifest_separates_generation_from_live_rejudge(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "recorded.jsonl"
    artifact.write_text(
        '{"type":"case_result","case_id":"case-1","query":"q"}\n',
        encoding="utf-8",
    )

    def build() -> dict:
        return build_benchmark_manifest(
            root=tmp_path,
            benchmark_name="rejudge-test",
            dataset_name="cases.jsonl",
            cases=[{"id": "case-1", "query": "q"}],
            config_snapshot={"rejudge_replay": True},
            llm_provider="mock",
            llm_model="mock-model",
            search_provider="mock",
            seed=7,
            replay_dir=str(artifact),
        )

    offline = build()
    mark_live_judge_nondeterminism(
        offline,
        answer_judge_provider="llm-gateway",
        answer_judge_model="kimi-k2.7-code-highspeed",
        answer_judge_executed=False,
    )
    assert offline["generation_deterministic"] is True
    assert offline["evaluation_deterministic"] is True
    assert offline["deterministic"] is True
    assert offline["evaluation_judges"] == [
        {
            "kind": "answer",
            "provider": "llm-gateway",
            "model": "kimi-k2.7-code-highspeed",
            "executed": False,
            "deterministic": None,
        }
    ]

    live_rejudge = build()
    mark_live_judge_nondeterminism(
        live_rejudge,
        answer_judge_provider="llm-gateway",
        answer_judge_model="kimi-k2.7-code-highspeed",
        answer_judge_executed=True,
    )
    assert live_rejudge["generation_deterministic"] is True
    assert live_rejudge["evaluation_deterministic"] is False
    assert live_rejudge["deterministic"] is False
    assert "live answer judge" in live_rejudge["determinism_reason"]


def test_benchmark_manifest_records_the_timeout_used_by_gateway_channels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "timeout-case",
                "query": "How should a timeout be recorded?",
                "category": "single_fact",
                "language": "en",
                "expected_format": "text",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "deepresearch_agent.benchmark.__file__",
        str(tmp_path / "src" / "deepresearch_agent" / "benchmark.py"),
    )
    monkeypatch.setattr(
        "deepresearch_agent.benchmark.load_settings",
        lambda: Settings(trace_dir=str(tmp_path / "traces")),
    )
    args = argparse.Namespace(
        cases=str(cases_path),
        benchmark_name="timeout-smoke",
        search_provider="mock",
        llm_provider="mock",
        llm_model=None,
        brief_model=None,
        planner_model=None,
        synthesis_model=None,
        embedding_provider="local",
        local_retrieval_mode="keyword",
        local_keyword_top_k=4,
        local_vector_top_k=4,
        local_keyword_weight=1.0,
        local_vector_weight=1.0,
        local_hybrid_rrf_k=60,
        rerank_enabled=False,
        rerank_provider="local",
        local_rerank_candidate_k=6,
        searxng_base_url=None,
        bing_search_base_url=None,
        gateway_web_search_model=None,
        web_crawler_provider="none",
        jina_reader_base_url=None,
        jina_search_base_url=None,
        crawler_max_chars=None,
        seed=20260606,
        max_researchers=1,
        max_results=1,
        request_timeout_seconds=6.5,
        max_rounds=1,
        max_tool_calls=1,
        deadline_seconds=None,
        min_evidence_items=1,
        fallback_policy="mock",
        reflection_enabled=False,
        max_reflection_rounds=1,
        reflection_min_sources=4,
        citation_judge_provider="none",
        citation_judge_model=None,
        replay_dir=None,
        cassette_id=None,
    )

    summary = asyncio.run(run_benchmark(args))
    config = summary["config"]

    assert config["request_timeout_seconds"] == 6.5
    assert config["gateway_web_search_timeout_seconds"] == 6.5
    assert config["llm_gateway_timeout_seconds"] == 6.5
    assert config["citation_judge_timeout_seconds"] == 6.5
    assert config["settings"]["request_timeout_seconds"] == 6.5
    assert config["settings"]["llm_gateway_timeout_seconds"] == 6.5
    assert config["settings"]["citation_judge_timeout_seconds"] == 6.5
    assert summary["manifest"]["config_snapshot"] == config


def test_dirty_worktree_hash_prevents_manifest_collisions(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Manifest Test",
        "-c",
        "user.email=manifest@example.com",
        "commit",
        "-q",
        "-m",
        "initial",
    )

    clean = _manifest(tmp_path)
    assert clean["git_dirty"] is False
    assert clean["git_worktree_hash"] is None

    tracked.write_text("first change\n", encoding="utf-8")
    first_dirty = _manifest(tmp_path)
    tracked.write_text("second change\n", encoding="utf-8")
    second_dirty = _manifest(tmp_path)

    assert first_dirty["git_dirty"] is True
    assert first_dirty["git_worktree_hash"].startswith("sha256:")
    assert first_dirty["git_worktree_hash"] != second_dirty["git_worktree_hash"]
    assert first_dirty["manifest_id"] != second_dirty["manifest_id"]


def test_untracked_file_content_is_part_of_dirty_manifest_identity(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Manifest Test",
        "-c",
        "user.email=manifest@example.com",
        "commit",
        "-q",
        "-m",
        "initial",
    )

    untracked = tmp_path / "new-source.py"
    untracked.write_text("value = 1\n", encoding="utf-8")
    first_dirty = _manifest(tmp_path)
    untracked.write_text("value = 2\n", encoding="utf-8")
    second_dirty = _manifest(tmp_path)

    assert first_dirty["git_worktree_hash"] != second_dirty["git_worktree_hash"]
    assert first_dirty["manifest_id"] != second_dirty["manifest_id"]


def test_untracked_manifest_hash_uses_unambiguous_path_content_boundaries(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Manifest Test",
        "-c",
        "user.email=manifest@example.com",
        "commit",
        "-q",
        "-m",
        "initial",
    )

    (tmp_path / "ab").write_text("c", encoding="utf-8")
    first = _manifest(tmp_path)
    (tmp_path / "ab").unlink()
    (tmp_path / "a").write_text("bc", encoding="utf-8")
    second = _manifest(tmp_path)

    assert first["git_worktree_hash"] != second["git_worktree_hash"]


def test_untracked_symlink_hashes_link_target_without_following_it(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Manifest Test",
        "-c",
        "user.email=manifest@example.com",
        "commit",
        "-q",
        "-m",
        "initial",
    )
    (tmp_path / "target-a").write_text("same", encoding="utf-8")
    (tmp_path / "target-b").write_text("same", encoding="utf-8")
    link = tmp_path / "artifact-link"
    link.symlink_to("target-a")
    first = _manifest(tmp_path)
    link.unlink()
    link.symlink_to("target-b")
    second = _manifest(tmp_path)

    assert first["git_worktree_hash"] != second["git_worktree_hash"]
