#!/usr/bin/env python3
"""Audit frozen evaluation artifacts without calling search or language models."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from deepresearch_agent.context_packer import pack_sources_for_synthesis
from deepresearch_agent.schemas import Finding, Source, SubQuestion


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", value.lower()).strip()


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _contains_answer(text: str, answer: str) -> bool:
    normalized_answer = _normalize_text(answer)
    return bool(normalized_answer) and normalized_answer in _normalize_text(text)


def _is_citable_source(source: Source) -> bool:
    return bool(
        source.content.strip()
        and not source.metadata.get("snippet_only")
        and source.metadata.get("extract_status")
        not in {"snippet", "crawl_failed", "empty"}
    )


def _packed_text(
    *,
    query: str,
    plan: list[SubQuestion],
    findings: list[Finding],
    sources: list[Source],
    per_source_tokens: int,
) -> tuple[str, int]:
    packed = pack_sources_for_synthesis(
        query=query,
        plan=plan,
        findings=findings,
        sources=sources,
        per_source_tokens=per_source_tokens,
    )
    text = " ".join(str(source.get("excerpt") or "") for source in packed.sources)
    return text, packed.estimated_tokens


def analyze_snapshot(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {
        str(record.get("case_id")): record
        for record in records
        if record.get("type") == "case_result" and record.get("case_id")
    }
    audits: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id") or case.get("case_id") or "")
        record = by_id.get(case_id, {})
        report = record.get("report") if isinstance(record.get("report"), dict) else {}
        query = str(case.get("query") or case.get("question") or "")
        answer = str(case.get("answer") or case.get("expected_answer") or "")
        gold_urls = {
            _canonical_url(str(url))
            for url in (case.get("gold_urls") or [])
            if str(url).startswith(("http://", "https://"))
        }
        sources = [
            Source.model_validate(source)
            for source in (report.get("sources") or record.get("sources") or [])
        ]
        raw_plan = report.get("plan") if isinstance(report.get("plan"), list) else []
        raw_findings = (
            report.get("findings") if isinstance(report.get("findings"), list) else []
        )
        plan = [SubQuestion.model_validate(item) for item in raw_plan] or [
            SubQuestion(id="Q1", question=query, rationale="offline snapshot audit")
        ]
        findings = [Finding.model_validate(item) for item in raw_findings]
        citable_sources = [source for source in sources if _is_citable_source(source)]
        snippet_sources = [source for source in sources if not _is_citable_source(source)]
        packed_650_text, packed_650_tokens = _packed_text(
            query=query,
            plan=plan,
            findings=findings,
            sources=sources,
            per_source_tokens=650,
        )
        packed_1200_text, packed_1200_tokens = _packed_text(
            query=query,
            plan=plan,
            findings=findings,
            sources=sources,
            per_source_tokens=1200,
        )
        citable_packed_650_text, citable_packed_650_tokens = _packed_text(
            query=query,
            plan=plan,
            findings=findings,
            sources=citable_sources,
            per_source_tokens=650,
        )
        citable_packed_1200_text, citable_packed_1200_tokens = _packed_text(
            query=query,
            plan=plan,
            findings=findings,
            sources=citable_sources,
            per_source_tokens=1200,
        )
        audits.append(
            {
                "case_id": case_id,
                "record_found": bool(record),
                "execution_success": bool(record.get("execution_success")),
                "source_count": len(sources),
                "citable_source_count": len(citable_sources),
                "snippet_or_failed_source_count": len(snippet_sources),
                "gold_url_retrieved": any(
                    _canonical_url(source.url) in gold_urls for source in sources
                ),
                "gold_url_citable": any(
                    _canonical_url(source.url) in gold_urls for source in citable_sources
                ),
                "answer_in_source_text": any(
                    _contains_answer(source.content, answer) for source in sources
                ),
                "answer_in_citable_source": any(
                    _contains_answer(source.content, answer) for source in citable_sources
                ),
                "answer_in_snippet_or_failed_source": any(
                    _contains_answer(source.content, answer) for source in snippet_sources
                ),
                "answer_in_packed_650": _contains_answer(packed_650_text, answer),
                "answer_in_packed_1200": _contains_answer(packed_1200_text, answer),
                "answer_in_citable_packed_650": _contains_answer(
                    citable_packed_650_text, answer
                ),
                "answer_in_citable_packed_1200": _contains_answer(
                    citable_packed_1200_text, answer
                ),
                "packed_tokens_650": packed_650_tokens,
                "packed_tokens_1200": packed_1200_tokens,
                "citable_packed_tokens_650": citable_packed_650_tokens,
                "citable_packed_tokens_1200": citable_packed_1200_tokens,
            }
        )
    return {
        "case_count": len(audits),
        "record_count": sum(1 for item in audits if item["record_found"]),
        "gold_url_retrieved_count": sum(1 for item in audits if item["gold_url_retrieved"]),
        "gold_url_citable_count": sum(1 for item in audits if item["gold_url_citable"]),
        "answer_in_source_text_count": sum(
            1 for item in audits if item["answer_in_source_text"]
        ),
        "answer_in_citable_source_count": sum(
            1 for item in audits if item["answer_in_citable_source"]
        ),
        "answer_in_snippet_or_failed_source_count": sum(
            1 for item in audits if item["answer_in_snippet_or_failed_source"]
        ),
        "answer_in_packed_650_count": sum(
            1 for item in audits if item["answer_in_packed_650"]
        ),
        "answer_in_packed_1200_count": sum(
            1 for item in audits if item["answer_in_packed_1200"]
        ),
        "answer_in_citable_packed_650_count": sum(
            1 for item in audits if item["answer_in_citable_packed_650"]
        ),
        "answer_in_citable_packed_1200_count": sum(
            1 for item in audits if item["answer_in_citable_packed_1200"]
        ),
        "cases": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a frozen public-eval artifact offline.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = analyze_snapshot(_load_jsonl(args.cases), _load_jsonl(args.artifact))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
