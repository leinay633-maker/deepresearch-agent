from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

from deepresearch_agent.schemas import StructuredReport


SUPPORTED_EXPORT_FORMATS = {"markdown", "md", "html", "json"}


def export_report(
    report: StructuredReport,
    output_dir: str | Path,
    formats: Iterable[str] = ("markdown", "html", "json"),
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for requested in formats:
        fmt = _normalize_format(requested)
        suffix = "md" if fmt == "markdown" else fmt
        path = target_dir / f"{report.run_id}.{suffix}"
        if fmt == "markdown":
            path.write_text(report_to_markdown(report), encoding="utf-8")
        elif fmt == "html":
            path.write_text(report_to_html(report), encoding="utf-8")
        elif fmt == "json":
            path.write_text(
                json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        paths[fmt] = str(path)
    return paths


def report_to_markdown(report: StructuredReport) -> str:
    lines = [
        f"# DeepResearch Report: {report.query}",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Citation retention: `{report.citation_check.retention_rate}`",
        f"- Total tokens: `{report.cost.total_tokens}`",
        f"- Estimated cost USD: `{report.cost.total_estimated_cost_usd}`",
        "",
        "## Answer",
        "",
        report.answer.strip(),
        "",
        "## Sources",
    ]
    for source in report.sources:
        lines.append(f"- [{source.id}] {source.title} - {source.url}")
    lines.extend(["", "## Citation Assessments"])
    for assessment in report.citation_check.assessments:
        lines.append(
            f"- `{assessment.support_level}` score `{assessment.overlap_score}`: "
            f"{assessment.claim}"
        )
        for quote in assessment.evidence_quotes:
            lines.append(f"  - Evidence [{quote.source_id}]: {quote.quote}")
    return "\n".join(lines).rstrip() + "\n"


def report_to_html(report: StructuredReport) -> str:
    sources = "\n".join(
        f"<li><strong>{html.escape(source.id)}</strong> "
        f"{html.escape(source.title)} - "
        f"<a href=\"{html.escape(source.url)}\">{html.escape(source.url)}</a></li>"
        for source in report.sources
    )
    assessments = []
    for assessment in report.citation_check.assessments:
        quotes = "".join(
            f"<li>Evidence {html.escape(quote.source_id)}: {html.escape(quote.quote)}</li>"
            for quote in assessment.evidence_quotes
        )
        assessments.append(
            "<li>"
            f"<span>{html.escape(assessment.support_level)}</span> "
            f"<code>{assessment.overlap_score}</code> "
            f"{html.escape(assessment.claim)}"
            f"<ul>{quotes}</ul>"
            "</li>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            f"  <title>{html.escape(report.query)}</title>",
            "  <style>body{font-family:Segoe UI,Arial,sans-serif;max-width:960px;margin:32px auto;line-height:1.55;color:#1f2933}pre{white-space:pre-wrap;background:#f4f6f8;padding:16px;border-radius:6px}code{background:#eef1f5;padding:1px 4px;border-radius:4px}</style>",
            "</head>",
            "<body>",
            f"  <h1>{html.escape(report.query)}</h1>",
            f"  <p><strong>Run ID:</strong> <code>{html.escape(report.run_id)}</code></p>",
            f"  <p><strong>Citation retention:</strong> {report.citation_check.retention_rate}</p>",
            f"  <p><strong>Total tokens:</strong> {report.cost.total_tokens}</p>",
            f"  <p><strong>Estimated cost USD:</strong> {report.cost.total_estimated_cost_usd}</p>",
            "  <h2>Answer</h2>",
            f"  <pre>{html.escape(report.answer.strip())}</pre>",
            "  <h2>Sources</h2>",
            f"  <ul>{sources}</ul>",
            "  <h2>Citation Assessments</h2>",
            f"  <ul>{''.join(assessments)}</ul>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _normalize_format(value: str) -> str:
    fmt = value.strip().lower()
    if fmt == "md":
        fmt = "markdown"
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        expected = ", ".join(sorted(SUPPORTED_EXPORT_FORMATS))
        raise ValueError(f"unsupported export format: {value}; expected one of {expected}")
    return fmt
