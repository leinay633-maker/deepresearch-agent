from __future__ import annotations

import json
import zipfile

import pytest

from deepresearch_agent.report_exporter import export_report, report_to_html
from deepresearch_agent.schemas import (
    CitationAssessment,
    CitationCheckReport,
    CostRecord,
    CostSummary,
    EvidenceQuote,
    Finding,
    ResearchBrief,
    Source,
    StructuredReport,
    SubQuestion,
)


def test_export_report_writes_markdown_html_and_json(tmp_path) -> None:
    report = _report()

    paths = export_report(report, tmp_path, formats=["markdown", "html", "json", "pdf", "docx"])

    assert set(paths) == {"markdown", "html", "json", "pdf", "docx"}
    markdown = (tmp_path / "run-export.md").read_text(encoding="utf-8")
    html = (tmp_path / "run-export.html").read_text(encoding="utf-8")
    payload = json.loads((tmp_path / "run-export.json").read_text(encoding="utf-8"))
    pdf = (tmp_path / "run-export.pdf").read_bytes()
    with zipfile.ZipFile(tmp_path / "run-export.docx") as archive:
        docx_xml = archive.read("word/document.xml").decode("utf-8")
    assert "# DeepResearch Report" in markdown
    assert "Evidence [S1]" in markdown
    assert "&lt;unsafe&gt;" in html
    assert payload["run_id"] == "run-export"
    assert pdf.startswith(b"%PDF")
    assert "DeepResearch Report" in docx_xml
    assert "Exported reports preserve cited evidence" in docx_xml


def test_report_export_rejects_unknown_format(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported export format"):
        export_report(_report(), tmp_path, formats=["pptx"])


def test_report_to_html_escapes_answer_content() -> None:
    html = report_to_html(_report())

    assert "<unsafe>" not in html
    assert "&lt;unsafe&gt;" in html


def _report() -> StructuredReport:
    source = Source(
        id="S1",
        title="Export source",
        url="https://example.com/export",
        content="Exported reports should preserve cited evidence.",
        provider="mock",
        query="export",
    )
    claim = "Exported reports preserve cited evidence [S1]"
    return StructuredReport(
        run_id="run-export",
        query="How should reports export?",
        brief=ResearchBrief(
            original_query="How should reports export?",
            normalized_query="How should reports export?",
            scope="Export a report.",
            constraints=["Keep citations."],
            assumptions=[],
        ),
        plan=[
            SubQuestion(
                id="Q1",
                question="What should export include?",
                rationale="Verify artifacts.",
            )
        ],
        answer="Report body with <unsafe> content [S1]",
        claims=[claim],
        findings=[
            Finding(
                subquestion_id="Q1",
                subquestion="What should export include?",
                summary="It should preserve cited evidence.",
                source_ids=["S1"],
                sources=[source],
            )
        ],
        sources=[source],
        citation_check=CitationCheckReport(
            total_claims=1,
            supported_claims=1,
            unsupported_claims=0,
            retention_rate=1.0,
            assessments=[
                CitationAssessment(
                    claim=claim,
                    citation_ids=["S1"],
                    supported=True,
                    support_level="supported",
                    reason="source overlap",
                    overlap_score=1.0,
                    evidence_quotes=[
                        EvidenceQuote(
                            source_id="S1",
                            source_title="Export source",
                            quote="Exported reports should preserve cited evidence.",
                            overlap_score=1.0,
                        )
                    ],
                )
            ],
        ),
        cost=CostSummary(
            total_input_tokens=10,
            total_output_tokens=5,
            total_tokens=15,
            total_estimated_cost_usd=0.0,
            records=[
                CostRecord(
                    stage="synthesis",
                    provider="mock",
                    model="mock",
                    input_tokens=10,
                    output_tokens=5,
                    estimated_cost_usd=0.0,
                )
            ],
        ),
        metrics={"success": True, "latency_ms": 1.0},
        trace_events=[],
    )
