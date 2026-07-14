#!/usr/bin/env python3
"""Build a reproducible, public 32-case SimpleQA evaluation sample.

This script intentionally treats the upstream ``metadata`` column as opaque
text.  A few public rows contain URLs that make it unsafe to parse the Python
repr with ``ast.literal_eval``; the small metadata fields needed for sampling
are extracted without executing or evaluating it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DEFAULT_SOURCE_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
)
DEFAULT_UPSTREAM_COMMIT = "652c89d0ca9df547706735883097e9537d40dc47"
DEFAULT_SEED = 20260714
DEFAULT_COUNT = 32


def _normalized_query(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _metadata_value(metadata: str, key: str) -> str:
    match = re.search(rf"['\"]{re.escape(key)}['\"]\s*:\s*['\"]([^'\"]*)", metadata)
    return match.group(1).strip() if match else "Unknown"


def _metadata_urls(metadata: str) -> list[str]:
    """Extract public HTTP(S) URLs without evaluating upstream metadata."""

    urls: list[str] = []
    cursor = 0
    while match := re.search(r"https?://", metadata[cursor:]):
        start = cursor + match.start()
        quote = metadata[start - 1] if start > 0 and metadata[start - 1] in "'\"" else ""
        if quote:
            end = metadata.find(quote, start)
            if end < 0:
                break
            value = metadata[start:end]
            cursor = end + 1
        else:
            end_match = re.search(r"[\s\]\[,'\"]", metadata[start:])
            end = start + end_match.start() if end_match else len(metadata)
            value = metadata[start:end]
            cursor = max(end + 1, start + 1)
        for part in re.split(r"(?:\\n)+|[\r\n]+", value):
            normalized = _public_url(part)
            if normalized and normalized not in urls:
                urls.append(normalized)
    return urls


def _stable_rank(seed: int, *parts: object) -> str:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_excluded_queries(path: Path | None) -> set[str]:
    if path is None:
        return set()
    excluded: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query = str(row.get("query") or row.get("question") or "")
        if query:
            excluded.add(_normalized_query(query))
    return excluded


def _public_url(url: str) -> str:
    raw = url.strip().strip("'\"").rstrip(".;")
    if not raw or "\\" in raw or any(character.isspace() for character in raw):
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
    except ValueError:
        return ""
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", "")
    )


def _candidate_rows(csv_text: str, *, excluded_queries: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, raw in enumerate(csv.DictReader(io.StringIO(csv_text))):
        query = str(raw.get("problem") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        metadata = str(raw.get("metadata") or "")
        if not query or not answer or _normalized_query(query) in excluded_queries:
            continue
        urls = list(dict.fromkeys(filter(None, (_public_url(url) for url in _metadata_urls(metadata)))))
        if not urls:
            continue
        rows.append(
            {
                "source_index": source_index,
                "query": query,
                "answer": answer,
                "category": _metadata_value(metadata, "topic"),
                "answer_type": _metadata_value(metadata, "answer_type"),
                "gold_urls": urls,
            }
        )
    return rows


def _balanced_quotas(values: list[str], *, count: int, seed: int, label: str) -> dict[str, int]:
    if not values:
        raise ValueError(f"cannot allocate {label} quotas without values")
    base, remainder = divmod(count, len(values))
    extra_values = set(
        sorted(values, key=lambda value: _stable_rank(seed, label, value))[:remainder]
    )
    return {value: base + int(value in extra_values) for value in values}


def select_cases(candidates: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    """Balance both topics and answer types with deterministic global quotas."""

    if count <= 0:
        raise ValueError("count must be positive")
    by_topic: dict[str, dict[str, deque[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for candidate in candidates:
        by_topic[candidate["category"]][candidate["answer_type"]].append(candidate)
    for by_type in by_topic.values():
        for answer_type, values in by_type.items():
            ordered = sorted(
                values,
                key=lambda item: _stable_rank(seed, answer_type, item["source_index"]),
            )
            by_type[answer_type] = deque(ordered)

    topics = sorted(by_topic)
    if not topics:
        raise ValueError("no public SimpleQA candidates remain after exclusions")
    answer_types = sorted(
        {answer_type for by_type in by_topic.values() for answer_type in by_type}
    )
    topic_quotas = _balanced_quotas(topics, count=count, seed=seed, label="topic")
    type_quotas = _balanced_quotas(
        answer_types,
        count=count,
        seed=seed,
        label="answer_type",
    )
    topic_remaining = dict(topic_quotas)
    type_remaining = dict(type_quotas)
    pair_uses: dict[tuple[str, str], int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        progressed = False
        for topic in topics:
            if topic_remaining[topic] <= 0:
                continue
            available_types = [
                answer_type
                for answer_type in answer_types
                if type_remaining[answer_type] > 0
                and by_topic[topic].get(answer_type)
            ]
            if not available_types:
                continue
            answer_type = min(
                available_types,
                key=lambda value: (
                    pair_uses[(topic, value)],
                    -type_remaining[value],
                    _stable_rank(seed, "pair", round_index, topic, value),
                ),
            )
            selected.append(by_topic[topic][answer_type].popleft())
            pair_uses[(topic, answer_type)] += 1
            topic_remaining[topic] -= 1
            type_remaining[answer_type] -= 1
            progressed = True
        round_index += 1
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(
            f"requested {count} cases but only selected {len(selected)}; "
            f"topic_remaining={topic_remaining}, type_remaining={type_remaining}"
        )

    rows: list[dict[str, Any]] = []
    for selected_index, item in enumerate(selected, 1):
        rows.append(
            {
                "id": f"simpleqa-public32-v1-{selected_index:02d}",
                "source_index": item["source_index"],
                "query": item["query"],
                "answer": item["answer"],
                "category": item["category"],
                "answer_type": item["answer_type"],
                "gold_urls": item["gold_urls"],
                "expected_format": "text",
            }
        )
    return rows


def build_manifest(
    *,
    source_url: str,
    source_sha256: str,
    seed: int,
    count: int,
    cases: list[dict[str, Any]],
    excluded_queries: set[str] | None = None,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    normalized_exclusions = sorted(excluded_queries or set())
    return {
        "schema_version": "1.0",
        "name": "simpleqa_public32_v1",
        "source_url": source_url,
        "upstream_simple_evals_commit": DEFAULT_UPSTREAM_COMMIT,
        "source_sha256": source_sha256,
        "selection_seed": seed,
        "selection_count": count,
        "candidate_count_after_exclusions": candidate_count,
        "selection_distribution": {
            "topic": dict(sorted(Counter(case["category"] for case in cases).items())),
            "answer_type": dict(
                sorted(Counter(case["answer_type"] for case in cases).items())
            ),
        },
        "excluded_query_count": len(normalized_exclusions),
        "excluded_query_sha256": hashlib.sha256(
            json.dumps(
                normalized_exclusions,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "selection_algorithm": (
            "deterministic round-robin with near-equal global quotas for metadata.topic "
            "and metadata.answer_type; prefer unused topic/type pairs, then the largest "
            "remaining answer-type quota; rank candidates by sha256(seed:answer_type:source_index)"
        ),
        "case_ids": [case["id"] for case in cases],
        "source_indices": [case["source_index"] for case in cases],
        "case_sha256": hashlib.sha256(
            json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _load_source(source_url: str, source_file: Path | None) -> bytes:
    if source_file is not None:
        return source_file.read_bytes()
    with urllib.request.urlopen(source_url, timeout=30) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen public SimpleQA 32-case sample.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-cases", type=Path, default=None)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--source-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args()

    source = _load_source(args.source_url, args.source_file)
    source_text = source.decode("utf-8")
    excluded_queries = _read_excluded_queries(args.exclude_cases)
    candidates = _candidate_rows(source_text, excluded_queries=excluded_queries)
    cases = select_cases(
        candidates,
        count=args.count,
        seed=args.seed,
    )
    manifest = build_manifest(
        source_url=args.source_url,
        source_sha256=hashlib.sha256(source).hexdigest(),
        seed=args.seed,
        count=args.count,
        cases=cases,
        excluded_queries=excluded_queries,
        candidate_count=len(candidates),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(cases), "manifest": str(args.manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
