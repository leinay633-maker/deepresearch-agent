#!/usr/bin/env python3
"""Build the frozen public 12-task DeepResearch Bench II subset.

The generation tasks and judge rubrics are deliberately written to separate
JSONL files.  The upstream prompt suffix is treated as plain text: forbidden
source URLs are extracted with a small lexical parser, never with ``eval`` or
``ast.literal_eval``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


UPSTREAM_REPOSITORY = "https://github.com/imlrz/DeepResearch-Bench-II"
UPSTREAM_COMMIT = "11d87de486ba7a9e10190be0afd66a9a0fc5d5da"
SOURCE_URL = (
    "https://raw.githubusercontent.com/imlrz/DeepResearch-Bench-II/"
    f"{UPSTREAM_COMMIT}/tasks_and_rubrics.jsonl"
)
SOURCE_SHA256 = "263aaabb8c279fb16cbe7c9499afe82d657a8ab3ccfb07ace084387e367d921a"
UPSTREAM_ROW_COUNT = 132
UPSTREAM_RUBRIC_COUNTS = {
    "info_recall": 6983,
    "analysis": 1686,
    "presentation": 746,
    "total": 9415,
}
SELECTED_SOURCE_INDICES = (4, 9, 23, 30, 43, 48, 54, 57, 63, 66, 82, 83)
LICENSE = "CC BY 4.0"
RUBRIC_CATEGORIES = ("info_recall", "analysis", "presentation")

DEFAULT_TASKS_OUTPUT = Path("evals/drb2_public12_v1.tasks.jsonl")
DEFAULT_RUBRICS_OUTPUT = Path("evals/drb2_public12_v1.rubrics.jsonl")
DEFAULT_MANIFEST_OUTPUT = Path("evals/drb2_public12_v1.manifest.json")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _normalize_public_url(raw_url: str) -> str:
    """Return a canonical HTTP(S) URL, or an empty string for lexical junk."""

    raw = raw_url.strip()
    if (
        not raw
        or "\\" in raw
        or any(character.isspace() or ord(character) < 32 for character in raw)
    ):
        return ""
    try:
        parsed = urlsplit(raw)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return ""
        _ = parsed.port
        parsed.hostname.encode("idna")
    except (UnicodeError, ValueError):
        return ""
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.query,
            "",
        )
    )


def _quoted_list_values(text: str, start: int) -> list[str]:
    """Lexically read a quoted string list beginning immediately after ``[``."""

    values: list[str] = []
    cursor = start
    while True:
        while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ","):
            cursor += 1
        if cursor >= len(text):
            raise ValueError("unterminated forbidden URL list")
        if text[cursor] == "]":
            return values
        quote = text[cursor]
        if quote not in {"'", '"'}:
            raise ValueError("forbidden URL list contains an unquoted value")
        value_start = cursor + 1
        cursor = value_start
        while cursor < len(text):
            if text[cursor] == "\\":
                raise ValueError("escaped forbidden URLs are not accepted")
            if text[cursor] == quote:
                following = cursor + 1
                while following < len(text) and text[following].isspace():
                    following += 1
                if following < len(text) and text[following] in {",", "]"}:
                    values.append(text[value_start:cursor])
                    cursor = following
                    break
            cursor += 1
        else:
            raise ValueError("unterminated quoted forbidden URL")


def extract_blocked_source_urls(*, prompt: str, task: str) -> list[str]:
    """Extract and validate the forbidden URL list from the prompt suffix."""

    if not task or not prompt.startswith(task):
        raise ValueError("upstream prompt must begin with content.task")
    suffix = prompt[len(task) :]
    match = re.search(r"['\"]urls['\"]\s*:\s*\[", suffix)
    if match is None:
        raise ValueError("upstream prompt has no forbidden urls list")

    normalized: list[str] = []
    for raw_url in _quoted_list_values(suffix, match.end()):
        url = _normalize_public_url(raw_url)
        if not url:
            raise ValueError(f"invalid forbidden source URL: {raw_url!r}")
        if url not in normalized:
            normalized.append(url)
    if not normalized:
        raise ValueError("upstream forbidden urls list is empty")
    return normalized


def parse_upstream_rows(source: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"upstream line {line_number} is not a JSON object")
        rows.append(row)
    return rows


def count_upstream_rubrics(
    upstream_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Count every atomic rubric in the pinned upstream snapshot."""

    counts = dict.fromkeys(RUBRIC_CATEGORIES, 0)
    for row_number, row in enumerate(upstream_rows, 1):
        content = row.get("content")
        rubrics = content.get("rubric") if isinstance(content, dict) else None
        if not isinstance(rubrics, dict):
            raise ValueError(f"upstream row {row_number} has no content.rubric")
        for category in RUBRIC_CATEGORIES:
            value = rubrics.get(category)
            if not isinstance(value, list):
                raise ValueError(
                    f"upstream row {row_number} has invalid {category} rubrics"
                )
            counts[category] += len(value)
    return {**counts, "total": sum(counts.values())}


def _atomic_rubrics(value: Any, *, category: str, source_idx: int) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"source idx {source_idx} has no {category} rubrics")
    rubrics: list[str] = []
    for rubric in value:
        if not isinstance(rubric, str) or not rubric.strip():
            raise ValueError(
                f"source idx {source_idx} contains a non-atomic {category} rubric"
            )
        rubrics.append(rubric.strip())
    return rubrics


def build_subset(
    upstream_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the fixed source indices and split tasks from private judge rubrics."""

    by_index: dict[int, dict[str, Any]] = {}
    for row in upstream_rows:
        source_idx = row.get("idx")
        if not isinstance(source_idx, int):
            continue
        if source_idx in by_index:
            raise ValueError(f"duplicate upstream source idx: {source_idx}")
        by_index[source_idx] = row

    missing = [index for index in SELECTED_SOURCE_INDICES if index not in by_index]
    if missing:
        raise ValueError(f"missing selected upstream source indices: {missing}")

    tasks: list[dict[str, Any]] = []
    rubric_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source_idx in SELECTED_SOURCE_INDICES:
        row = by_index[source_idx]
        task_id = row.get("id")
        content = row.get("content")
        if not isinstance(task_id, str) or not task_id or task_id in seen_ids:
            raise ValueError(f"source idx {source_idx} has an invalid or duplicate id")
        if not isinstance(content, dict):
            raise ValueError(f"source idx {source_idx} has no content object")
        query = content.get("task")
        upstream_rubrics = content.get("rubric")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"source idx {source_idx} has no content.task")
        if not isinstance(upstream_rubrics, dict):
            raise ValueError(f"source idx {source_idx} has no content.rubric")
        if row.get("license") != LICENSE:
            raise ValueError(f"source idx {source_idx} is not licensed {LICENSE}")

        blocked_urls = extract_blocked_source_urls(
            prompt=str(row.get("prompt") or ""),
            task=query,
        )
        blocked = content.get("blocked")
        if not isinstance(blocked, dict) or not isinstance(blocked.get("urls"), list):
            raise ValueError(f"source idx {source_idx} has no blocked URL audit field")
        audit_urls = list(
            dict.fromkeys(_normalize_public_url(str(url)) for url in blocked["urls"])
        )
        if not audit_urls or "" in audit_urls or blocked_urls != audit_urls:
            raise ValueError(
                f"source idx {source_idx} prompt URLs differ from the upstream audit field"
            )

        language = row.get("language")
        theme = row.get("theme")
        description = row.get("description")
        if language not in {"en", "zh"}:
            raise ValueError(f"source idx {source_idx} has unsupported language")
        if not isinstance(theme, str) or not theme.strip():
            raise ValueError(f"source idx {source_idx} has no theme")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"source idx {source_idx} has no description")

        tasks.append(
            {
                "id": task_id,
                "source_idx": source_idx,
                "query": query.strip(),
                "language": language,
                "theme": theme.strip(),
                "description": description.strip(),
                "expected_format": "markdown",
                "report_depth": "deep",
                "blocked_source_urls": blocked_urls,
                "license": LICENSE,
            }
        )
        rubric_rows.append(
            {
                "id": task_id,
                **{
                    category: _atomic_rubrics(
                        upstream_rubrics.get(category),
                        category=category,
                        source_idx=source_idx,
                    )
                    for category in RUBRIC_CATEGORIES
                },
            }
        )
        seen_ids.add(task_id)
    return tasks, rubric_rows


def build_manifest(
    *,
    source_sha256: str,
    tasks: list[dict[str, Any]],
    rubrics: list[dict[str, Any]],
    tasks_bytes: bytes,
    rubrics_bytes: bytes,
    upstream_row_count: int,
    upstream_rubric_counts: dict[str, int],
) -> dict[str, Any]:
    rubric_counts = {
        category: sum(len(row[category]) for row in rubrics)
        for category in RUBRIC_CATEGORIES
    }
    return {
        "schema_version": "1.0",
        "name": "drb2_public12_v1",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_url": SOURCE_URL,
        "source_sha256": source_sha256,
        "upstream_row_count": upstream_row_count,
        "upstream_rubric_counts": upstream_rubric_counts,
        "selection_count": len(tasks),
        "source_indices": [task["source_idx"] for task in tasks],
        "task_ids": [task["id"] for task in tasks],
        "license": LICENSE,
        "selection_algorithm": (
            "fixed curated source indices, ordered exactly as recorded; selected to provide "
            "6 English and 6 Chinese tasks across 6 themes with 2 tasks per theme; no random "
            "sampling"
        ),
        "selection_distribution": {
            "language": dict(sorted(Counter(task["language"] for task in tasks).items())),
            "theme": dict(sorted(Counter(task["theme"] for task in tasks).items())),
        },
        "rubric_counts": {**rubric_counts, "total": sum(rubric_counts.values())},
        "tasks_sha256": _sha256(tasks_bytes),
        "rubrics_sha256": _sha256(rubrics_bytes),
    }


def _load_source(source_file: Path | None) -> bytes:
    if source_file is not None:
        return source_file.read_bytes()
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "deepresearch-agent-public-eval-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen DeepResearch Bench II public 12-task subset."
    )
    parser.add_argument("--source-file", type=Path, default=None)
    parser.add_argument("--tasks-output", type=Path, default=DEFAULT_TASKS_OUTPUT)
    parser.add_argument("--rubrics-output", type=Path, default=DEFAULT_RUBRICS_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    source = _load_source(args.source_file)
    source_sha256 = _sha256(source)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            "pinned upstream source SHA-256 mismatch: "
            f"expected {SOURCE_SHA256}, got {source_sha256}"
        )
    upstream_rows = parse_upstream_rows(source)
    upstream_rubric_counts = count_upstream_rubrics(upstream_rows)
    if len(upstream_rows) != UPSTREAM_ROW_COUNT:
        raise ValueError(
            "pinned upstream row count mismatch: "
            f"expected {UPSTREAM_ROW_COUNT}, got {len(upstream_rows)}"
        )
    if upstream_rubric_counts != UPSTREAM_RUBRIC_COUNTS:
        raise ValueError(
            "pinned upstream rubric counts mismatch: "
            f"expected {UPSTREAM_RUBRIC_COUNTS}, got {upstream_rubric_counts}"
        )
    tasks, rubrics = build_subset(upstream_rows)
    tasks_bytes = _jsonl_bytes(tasks)
    rubrics_bytes = _jsonl_bytes(rubrics)
    manifest = build_manifest(
        source_sha256=source_sha256,
        tasks=tasks,
        rubrics=rubrics,
        tasks_bytes=tasks_bytes,
        rubrics_bytes=rubrics_bytes,
        upstream_row_count=len(upstream_rows),
        upstream_rubric_counts=upstream_rubric_counts,
    )

    args.tasks_output.parent.mkdir(parents=True, exist_ok=True)
    args.rubrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.tasks_output.write_bytes(tasks_bytes)
    args.rubrics_output.write_bytes(rubrics_bytes)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "rubric_count": manifest["rubric_counts"]["total"],
                "manifest": str(args.manifest_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
