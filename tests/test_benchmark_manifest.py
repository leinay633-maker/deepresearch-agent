from __future__ import annotations

import subprocess
from pathlib import Path

from deepresearch_agent.benchmark import (
    build_benchmark_manifest,
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
