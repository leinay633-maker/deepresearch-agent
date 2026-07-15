from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_drb2_public12 import (
    LICENSE,
    RUBRIC_CATEGORIES,
    SELECTED_SOURCE_INDICES,
    SOURCE_SHA256,
    UPSTREAM_ROW_COUNT,
    UPSTREAM_RUBRIC_COUNTS,
    UPSTREAM_COMMIT,
    _jsonl_bytes,
    _normalize_public_url,
    build_subset,
    count_upstream_rubrics,
    extract_blocked_source_urls,
)


TASKS_PATH = Path("evals/drb2_public12_v1.tasks.jsonl")
RUBRICS_PATH = Path("evals/drb2_public12_v1.rubrics.jsonl")
MANIFEST_PATH = Path("evals/drb2_public12_v1.manifest.json")

TASK_KEYS = {
    "id",
    "source_idx",
    "query",
    "language",
    "theme",
    "description",
    "expected_format",
    "report_depth",
    "blocked_source_urls",
    "license",
}
RUBRIC_KEYS = {"id", *RUBRIC_CATEGORIES}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_frozen_drb2_subset_has_locked_balanced_public_distribution() -> None:
    tasks = _jsonl(TASKS_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert len(tasks) == 12
    assert [task["source_idx"] for task in tasks] == list(SELECTED_SOURCE_INDICES)
    assert Counter(task["language"] for task in tasks) == {"en": 6, "zh": 6}
    assert Counter(task["theme"] for task in tasks) == {
        "Finance & Business": 2,
        "Science & Technology": 2,
        "Software Development": 2,
        "Education & Jobs": 2,
        "Health": 2,
        "History": 2,
    }
    assert {task["license"] for task in tasks} == {LICENSE}
    assert all(task["expected_format"] == "markdown" for task in tasks)
    assert all(task["report_depth"] == "deep" for task in tasks)
    assert manifest["selection_count"] == 12
    assert manifest["selection_distribution"] == {
        "language": {"en": 6, "zh": 6},
        "theme": dict(sorted(Counter(task["theme"] for task in tasks).items())),
    }


def test_generation_tasks_do_not_leak_prompts_content_or_judge_rubrics() -> None:
    tasks = _jsonl(TASKS_PATH)
    rubrics = _jsonl(RUBRICS_PATH)

    assert all(set(task) == TASK_KEYS for task in tasks)
    assert all(set(rubric) == RUBRIC_KEYS for rubric in rubrics)
    assert [task["id"] for task in tasks] == [rubric["id"] for rubric in rubrics]
    for task in tasks:
        assert task["query"].strip()
        assert "**important**" not in task["query"]
        assert "not allowed to view" not in task["query"].lower()
        assert not ({"prompt", "content", "rubric"} & set(task))
    for rubric in rubrics:
        for category in RUBRIC_CATEGORIES:
            assert rubric[category]
            assert all(isinstance(item, str) and item.strip() for item in rubric[category])


def test_blocked_urls_are_lexically_valid_normalized_and_unique() -> None:
    tasks = _jsonl(TASKS_PATH)

    for task in tasks:
        urls = task["blocked_source_urls"]
        assert urls
        assert urls == list(dict.fromkeys(urls))
        assert all(_normalize_public_url(url) == url for url in urls)


def test_manifest_locks_upstream_and_exact_artifact_hashes_and_rubric_counts() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rubrics = _jsonl(RUBRICS_PATH)
    counts = {
        category: sum(len(row[category]) for row in rubrics)
        for category in RUBRIC_CATEGORIES
    }

    assert manifest["upstream_commit"] == UPSTREAM_COMMIT
    assert manifest["source_sha256"] == SOURCE_SHA256
    assert manifest["upstream_row_count"] == UPSTREAM_ROW_COUNT
    assert manifest["upstream_rubric_counts"] == UPSTREAM_RUBRIC_COUNTS
    assert manifest["upstream_rubric_counts"]["total"] == 9415
    assert manifest["source_url"].endswith(f"/{UPSTREAM_COMMIT}/tasks_and_rubrics.jsonl")
    assert manifest["tasks_sha256"] == hashlib.sha256(TASKS_PATH.read_bytes()).hexdigest()
    assert manifest["rubrics_sha256"] == hashlib.sha256(RUBRICS_PATH.read_bytes()).hexdigest()
    assert manifest["rubric_counts"] == {**counts, "total": sum(counts.values())}
    assert manifest["rubric_counts"] == {
        "info_recall": 560,
        "analysis": 169,
        "presentation": 78,
        "total": 807,
    }


def test_upstream_snapshot_rubric_count_is_derived_not_readme_copied() -> None:
    frozen_tasks = _jsonl(TASKS_PATH)
    frozen_rubrics = _jsonl(RUBRICS_PATH)
    synthetic_rows = []
    for task, rubrics in zip(frozen_tasks, frozen_rubrics, strict=True):
        synthetic_rows.append(
            {
                "content": {
                    "rubric": {
                        category: rubrics[category] for category in RUBRIC_CATEGORIES
                    }
                }
            }
        )

    counts = count_upstream_rubrics(synthetic_rows)

    assert counts == {
        "info_recall": 560,
        "analysis": 169,
        "presentation": 78,
        "total": 807,
    }


def test_fixed_selection_and_serialization_are_reproducible_from_reordered_rows() -> None:
    frozen_tasks = _jsonl(TASKS_PATH)
    frozen_rubrics = _jsonl(RUBRICS_PATH)
    upstream_rows = []
    for task, rubrics in zip(frozen_tasks, frozen_rubrics, strict=True):
        blocked = {"urls": task["blocked_source_urls"]}
        suffix = "\n**important** forbidden source audit: " + json.dumps(
            blocked,
            ensure_ascii=False,
        )
        upstream_rows.append(
            {
                "id": task["id"],
                "idx": task["source_idx"],
                "language": task["language"],
                "theme": task["theme"],
                "description": task["description"],
                "prompt": task["query"] + suffix,
                "content": {
                    "task": task["query"],
                    "rubric": {
                        category: rubrics[category] for category in RUBRIC_CATEGORIES
                    },
                    "blocked": blocked,
                },
                "license": task["license"],
            }
        )

    rebuilt_tasks, rebuilt_rubrics = build_subset(list(reversed(upstream_rows)))

    assert rebuilt_tasks == frozen_tasks
    assert rebuilt_rubrics == frozen_rubrics
    assert _jsonl_bytes(rebuilt_tasks) == TASKS_PATH.read_bytes()
    assert _jsonl_bytes(rebuilt_rubrics) == RUBRICS_PATH.read_bytes()


def test_forbidden_url_parser_is_lexical_deduplicated_and_scoped_to_suffix() -> None:
    task = "Research https://allowed.example/question without leaking it."
    prompt = task + (
        "\n**important** {'urls': ['HTTPS://EXAMPLE.COM/O'Brien#section', "
        "'https://example.com/O'Brien#other', "
        "'https://example.org/source?q=one']}; "
        "__import__('os').system('this text must remain inert')"
    )

    assert extract_blocked_source_urls(prompt=prompt, task=task) == [
        "https://example.com/O'Brien",
        "https://example.org/source?q=one",
    ]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://user:password@example.com/source",
        "https://example.com/bad path",
        "https://example.com/bad\\path",
        "https://example.com:invalid/source",
    ],
)
def test_forbidden_url_parser_rejects_lexical_junk(url: str) -> None:
    task = "Research task"
    prompt = task + f"\n**important** {{'urls': ['{url}']}}"

    with pytest.raises(ValueError, match="forbidden"):
        extract_blocked_source_urls(prompt=prompt, task=task)
