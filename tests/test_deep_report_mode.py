from __future__ import annotations

import asyncio

import pytest

from deepresearch_agent.cost import CostTracker
from deepresearch_agent.llm import (
    LLMJsonResult,
    DeepSeekLLMProvider,
    LLMGatewayLLMProvider,
    MockLLMProvider,
    _brief_from_payload,
    _string_array,
    _synthesis_repair_messages,
    _synthesis_from_payload,
)
from deepresearch_agent.schemas import (
    CitationCheckReport,
    Finding,
    ResearchBrief,
    ResearchRequest,
    Source,
)


def _deep_brief() -> ResearchBrief:
    return ResearchBrief(
        original_query="Compare Alpha and Beta and explain the tradeoffs.",
        normalized_query="Compare Alpha and Beta and explain the tradeoffs?",
        scope="Compare evidence, tradeoffs, and implications.",
        constraints=["Use independent evidence."],
        assumptions=[],
        report_depth="deep",
        expected_format="markdown",
    )


def test_concise_defaults_remain_bounded_while_deep_expands_research_budget() -> None:
    concise = ResearchRequest(query="Compare Alpha and Beta.")
    deep = ResearchRequest(query="Compare Alpha and Beta.", report_depth="deep")

    assert concise.report_depth == "concise"
    assert concise.research_budget().max_rounds == 1
    assert concise.research_budget().max_tool_calls == 1
    assert concise.research_budget().min_evidence_items == 1
    assert concise.search_results_per_researcher() == 4

    assert deep.research_budget().max_rounds == 2
    assert deep.research_budget().max_tool_calls == 2
    assert deep.research_budget().deadline_seconds == 300.0
    assert deep.research_budget().min_evidence_items == 2
    assert deep.search_results_per_researcher() == 8

    explicit = ResearchRequest(
        query="Compare Alpha and Beta.",
        report_depth="deep",
        max_rounds=3,
        max_tool_calls=4,
        deadline_seconds=600,
    )
    assert explicit.research_budget().max_rounds == 3
    assert explicit.research_budget().max_tool_calls == 4
    assert explicit.research_budget().deadline_seconds == 600


@pytest.mark.parametrize("expected_format", ["text", "json"])
def test_deep_requests_and_briefs_require_markdown(expected_format: str) -> None:
    with pytest.raises(
        ValueError,
        match="report_depth='deep' requires expected_format='markdown'",
    ):
        ResearchRequest(
            query="Compare Alpha and Beta.",
            report_depth="deep",
            expected_format=expected_format,
        )

    with pytest.raises(
        ValueError,
        match="report_depth='deep' requires expected_format='markdown'",
    ):
        ResearchBrief(
            original_query="Compare Alpha and Beta.",
            normalized_query="Compare Alpha and Beta?",
            scope="Compare evidence.",
            constraints=[],
            assumptions=[],
            report_depth="deep",
            expected_format=expected_format,
        )


@pytest.mark.parametrize("expected_format", ["text", "markdown", "json"])
def test_concise_requests_keep_all_output_formats(expected_format: str) -> None:
    request = ResearchRequest(
        query="Compare Alpha and Beta.",
        expected_format=expected_format,
    )

    assert request.report_depth == "concise"
    assert request.expected_format == expected_format


def test_brief_accepts_restricted_text_objects_for_constraints_and_assumptions() -> None:
    brief = _brief_from_payload(
        {
            "normalized_query": "Compare Alpha and Beta?",
            "scope": "Compare the supplied evidence.",
            "constraints": [
                {"constraint": "Use independent primary sources."},
                {"description": "Cite every factual claim."},
            ],
            "assumptions": [
                {"assumption": "The requested period is inclusive."},
                {"value": "Currency values are nominal unless stated otherwise."},
            ],
        },
        "Compare Alpha and Beta?",
        report_depth="deep",
    )

    assert brief.constraints == [
        "Use independent primary sources.",
        "Cite every factual claim.",
    ]
    assert brief.assumptions == [
        "The requested period is inclusive.",
        "Currency values are nominal unless stated otherwise.",
    ]
    assert brief.report_depth == "deep"


def test_brief_parser_rejects_non_markdown_deep_contract() -> None:
    with pytest.raises(
        ValueError,
        match="report_depth='deep' requires expected_format='markdown'",
    ):
        _brief_from_payload(
            {
                "normalized_query": "Compare Alpha and Beta?",
                "scope": "Compare evidence.",
                "constraints": [],
                "assumptions": [],
            },
            "Compare Alpha and Beta?",
            expected_format="text",
            report_depth="deep",
        )


@pytest.mark.parametrize(
    "invalid_item, error",
    [
        ({"label": "Unknown field."}, "unknown text fields"),
        ({"text": {"nested": "No."}}, "must be a string"),
        ({"text": 42}, "must be a string"),
        ({"text": "First.", "value": "Second."}, "exactly one"),
        (["nested list"], "string or supported text object"),
    ],
)
def test_string_array_rejects_ambiguous_or_unbounded_object_shapes(
    invalid_item,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _string_array({"items": [invalid_item]}, "items")


def test_mock_deep_plan_uses_five_non_overlapping_report_branches() -> None:
    provider = MockLLMProvider()
    cost = CostTracker(provider=provider.name, model=provider.model)

    plan = asyncio.run(provider.plan(_deep_brief(), max_researchers=5, cost=cost))

    assert [item.id for item in plan] == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    combined = " ".join(
        f"{item.question} {item.rationale}" for item in plan
    ).lower()
    assert "must-answer" in combined
    assert "comparison dimensions" in combined
    assert "table-ready evidence" in combined
    assert "limitations" in combined
    assert "implications" in combined


class _RecordingDeepProvider(DeepSeekLLMProvider):
    def __init__(self) -> None:
        super().__init__(model="deepseek-v4-flash", max_retries=0)
        self.calls: list[dict] = []

    async def _chat_json_result(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        validator=None,
    ) -> LLMJsonResult:
        self.calls.append(
            {"stage": stage, "messages": messages, "max_tokens": max_tokens}
        )
        if stage == "planning":
            payload = {
                "subquestions": [
                    {
                        "id": f"Q{index}",
                        "question": f"How should Alpha and Beta be compared in section {index}?",
                        "search_query": f"Alpha Beta comparison evidence {index}",
                        "rationale": "Supply a distinct report section and table-ready evidence.",
                        "required_entities": ["Alpha", "Beta"],
                        "required_aspects": [f"section {index} comparison"],
                    }
                    for index in range(1, 6)
                ]
            }
        else:
            payload = {
                "answer": (
                    "# Alpha and Beta\n\n"
                    "## Evidence\n\n"
                    "- Alpha has documented property A [S1].\n"
                    "- Beta has documented property B [S2].\n\n"
                    "## Comparison\n\n"
                    "| Option | Evidence-backed difference |\n"
                    "| --- | --- |\n"
                    "| Alpha | Alpha has documented property A [S1] |\n"
                    "| Beta | Beta has documented property B [S2] |"
                ),
                "claims": [],
            }
        if validator is not None:
            validator(payload)
        return LLMJsonResult(
            parsed=payload,
            content="fixture",
            usage={"prompt_tokens": 100, "completion_tokens": 20},
            model=self.model,
        )


class _FailingDeepProvider(DeepSeekLLMProvider):
    async def _chat_json_result(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        validator=None,
    ) -> LLMJsonResult:
        del stage, messages, max_tokens, validator
        error = RuntimeError(
            "LLM synthesis answer contains uncited factual text " + "x" * 2_000
        )
        setattr(error, "validation_output_sha256", "a" * 64)
        raise error


class _EmptyClaimsDeepProvider(DeepSeekLLMProvider):
    async def _chat_json_result(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        validator=None,
    ) -> LLMJsonResult:
        del stage, messages, max_tokens, validator
        error = RuntimeError("LLM synthesis response contains no usable claims")
        setattr(error, "validation_output_sha256", "b" * 64)
        raise error


class _UncitedOutputDeepProvider(DeepSeekLLMProvider):
    async def _chat_json_result(
        self,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        validator=None,
    ) -> LLMJsonResult:
        del stage, messages, max_tokens
        payload = {
            "answer": "# Findings\n\nAlpha is the clear winner.\n\n- Beta is weaker.",
            "claims": [],
        }
        try:
            if validator is not None:
                validator(payload)
        except ValueError as exc:
            error = RuntimeError(f"LLM synthesis JSON validation failed: {exc}")
            setattr(error, "validation_output_sha256", "c" * 64)
            raise error from exc
        raise AssertionError("uncited fixture unexpectedly passed validation")


def test_deep_provider_uses_expanded_plan_context_and_output_limits() -> None:
    provider = _RecordingDeepProvider()
    cost = CostTracker(provider=provider.name, model=provider.model)
    brief = _deep_brief()
    plan = asyncio.run(provider.plan(brief, max_researchers=5, cost=cost))
    sources = [
        Source(
            id=f"S{index}",
            title=f"Evidence source {index}",
            url=f"https://source{index}.example/report",
            content=(
                f"Alpha and Beta comparison evidence {index}. "
                f"Alpha has documented property A and Beta has documented property B. "
            )
            * 500,
            provider="fixture",
            query="Alpha Beta comparison",
            quality_score=0.9,
        )
        for index in range(1, 21)
    ]
    findings = [
        Finding(
            subquestion_id=item.id,
            subquestion=item.question,
            summary="Evidence collected.",
            source_ids=[],
            sources=[],
        )
        for item in plan
    ]

    answer, claims = asyncio.run(
        provider.synthesize(brief, plan, findings, sources, cost)
    )

    planning_call, synthesis_call = provider.calls
    assert planning_call["max_tokens"] == 2400
    assert "comparison dimensions" in planning_call["messages"][1]["content"]
    assert synthesis_call["max_tokens"] == 10_000
    assert "comparison table" in synthesis_call["messages"][0]["content"]
    assert "requested coverage is incomplete" in synthesis_call["messages"][0]["content"]
    assert '"claims":[]' in synthesis_call["messages"][0]["content"]
    assert "claims must be exactly the empty array" in synthesis_call["messages"][0]["content"]
    assert "Do not duplicate any answer prose into claims" in synthesis_call["messages"][0]["content"]
    assert "must appear verbatim in claims" not in synthesis_call["messages"][0]["content"]
    assert "# Alpha and Beta" in answer
    assert "| Option | Evidence-backed difference |" in answer
    assert len(claims) == 4
    assert provider.last_synthesis_context["report_depth"] == "deep"
    assert provider.last_synthesis_context["max_claims"] == 72
    assert provider.last_synthesis_context["estimated_tokens"] <= 36_000
    assert len(provider.last_synthesis_context["kept_source_ids"]) <= 36
    assert provider.last_synthesis_context["synthesis_sanitization"] == {
        "enabled": True,
        "applied": True,
        "dropped_uncited_sentence_count": 0,
        "dropped_uncited_line_count": 0,
        "dropped_uncited_table_row_count": 0,
    }

    targeted_repair = _synthesis_repair_messages(
        synthesis_call["messages"],
        ValueError("fixture citation validation failed"),
    )
    repair_instruction = targeted_repair[-1]["content"]
    assert "Keep claims exactly []" in repair_instruction
    assert "Python extracts cited units" in repair_instruction
    assert "appear verbatim in claims" not in repair_instruction


def test_deep_citation_repair_prompt_keeps_empty_claims_contract() -> None:
    provider = _RecordingDeepProvider()
    cost = CostTracker(provider=provider.name, model=provider.model)
    sources = [
        Source(
            id=source_id,
            title=f"Evidence source {source_id}",
            url=f"https://{source_id.lower()}.example/report",
            content="Supported comparison evidence.",
            provider="fixture",
            query="Alpha Beta comparison",
        )
        for source_id in ("S1", "S2")
    ]
    citation_report = CitationCheckReport(
        total_claims=0,
        supported_claims=0,
        unsupported_claims=0,
        retention_rate=0.0,
        assessments=[],
    )

    answer, claims = asyncio.run(
        provider.repair_synthesis(
            _deep_brief(),
            "# Draft\n\nAlpha has documented property A [S1].",
            citation_report,
            sources,
            cost,
        )
    )

    repair_call = provider.calls[-1]
    system_prompt = repair_call["messages"][0]["content"]
    assert '"claims":[]' in system_prompt
    assert "claims must remain exactly empty" in system_prompt
    assert "Python extracts the cited factual units deterministically" in system_prompt
    assert "appear in claims" not in system_prompt
    assert answer.startswith("# Alpha and Beta")
    assert len(claims) == 4


def test_deep_synthesis_timeout_records_bounded_context_and_attempt_ledger() -> None:
    class TimeoutGatewayClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create_message(
            self,
            *,
            model,
            messages,
            max_tokens,
            timeout_seconds=None,
        ):
            self.calls.append(
                {
                    "model": model,
                    "message_count": len(messages),
                    "max_tokens": max_tokens,
                    "timeout_seconds": timeout_seconds,
                }
            )
            raise TimeoutError("The read operation timed out")

    client = TimeoutGatewayClient()
    provider = LLMGatewayLLMProvider(
        model="claude-4.6-opus",
        base_url="https://gateway.local",
        timeout_seconds=240.0,
        synthesis_timeout_seconds=360.0,
        max_retries=2,
        client=client,
    )
    source = Source(
        id="S1",
        title="Evidence source",
        url="https://source.example/report",
        content="Alpha has documented property A.",
        provider="fixture",
        query="Alpha evidence",
    )

    asyncio.run(
        provider.synthesize(
            _deep_brief(),
            [],
            [],
            [source],
            CostTracker(provider=provider.name, model=provider.model),
        )
    )

    context = provider.last_synthesis_context
    assert len(client.calls) == 1
    assert context["estimated_tokens"] > 0
    assert context["max_claims"] == 72
    assert context["max_output_tokens"] == 10_000
    assert context["socket_timeout_seconds"] == 360.0
    assert len(context["attempt_ledger"]) == 1
    assert context["attempt_ledger"][0]["failure_class"] == "transport_timeout"
    assert "Research context" not in str(context["attempt_ledger"])


def test_deep_synthesis_records_bounded_validation_failure_audit_fields() -> None:
    provider = _FailingDeepProvider(
        model="deepseek-v4-flash",
        max_retries=0,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)
    source = Source(
        id="S1",
        title="Evidence source",
        url="https://source.example/report",
        content="Alpha has documented property A.",
        provider="fixture",
        query="Alpha evidence",
    )

    asyncio.run(provider.synthesize(_deep_brief(), [], [], [source], cost))

    context = provider.last_synthesis_context
    assert context["synthesis_fallback"] is True
    assert context["synthesis_validation_output_sha256"] == "a" * 64
    assert "uncited factual text" in context["synthesis_validation_failure_reason"]
    assert len(context["synthesis_validation_failure_reason"]) <= 500
    assert "x" * 1_000 not in str(context)
    assert context["synthesis_sanitization"]["enabled"] is True
    assert context["synthesis_sanitization"]["applied"] is False


def test_deep_synthesis_failure_records_aggregate_sanitization_counts() -> None:
    provider = _UncitedOutputDeepProvider(model="deepseek-v4-flash", max_retries=0)
    source = Source(
        id="S1",
        title="Evidence source",
        url="https://source.example/report",
        content="Alpha has documented property A.",
        provider="fixture",
        query="Alpha evidence",
    )

    asyncio.run(
        provider.synthesize(
            _deep_brief(),
            [],
            [],
            [source],
            CostTracker(provider=provider.name, model=provider.model),
        )
    )

    audit = provider.last_synthesis_context["synthesis_sanitization"]
    assert audit == {
        "enabled": True,
        "applied": True,
        "dropped_uncited_sentence_count": 2,
        "dropped_uncited_line_count": 2,
        "dropped_uncited_table_row_count": 0,
    }
    assert all(isinstance(value, (bool, int)) for value in audit.values())
    assert "clear winner" not in str(audit)


def test_empty_claims_with_verified_sources_remain_a_validation_fallback() -> None:
    provider = _EmptyClaimsDeepProvider(
        model="deepseek-v4-flash",
        max_retries=0,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)
    source = Source(
        id="S1",
        title="Evidence source",
        url="https://source.example/report",
        content="Alpha has documented property A.",
        provider="fixture",
        query="Alpha evidence",
    )

    answer, claims = asyncio.run(
        provider.synthesize(_deep_brief(), [], [], [source], cost)
    )

    assert claims == ["Alpha has documented property A. [S1]"]
    assert "Alpha has documented property A" in answer
    context = provider.last_synthesis_context
    assert context["synthesis_fallback"] is True
    assert "synthesis_abstained" not in context
    assert context["synthesis_validation_output_sha256"] == "b" * 64
    assert "no usable claims" in context["synthesis_validation_failure_reason"]


def test_source_free_empty_claims_keep_bounded_abstention_audit_fields() -> None:
    provider = _EmptyClaimsDeepProvider(
        model="deepseek-v4-flash",
        max_retries=0,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)

    answer, claims = asyncio.run(
        provider.synthesize(_deep_brief(), [], [], [], cost)
    )

    assert claims == []
    assert "insufficient" in answer
    context = provider.last_synthesis_context
    assert context["synthesis_abstained"] is True
    assert "synthesis_fallback" not in context
    assert context["synthesis_validation_output_sha256"] == "b" * 64
    assert "no usable claims" in context["synthesis_validation_failure_reason"]


def test_deep_markdown_drops_uncited_table_rows_and_keeps_cited_rows() -> None:
    audit: dict[str, int | bool] = {}
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Comparison\n\n"
                "| Option | Result |\n"
                "| --- | --- |\n"
                "| Alpha | Supported result [S1] |\n"
                "| Beta | Unsupported result |"
            ),
            "claims": [],
        },
        allowed_source_ids={"S1"},
        expected_format="markdown",
        query="Compare Alpha and Beta.",
        max_claims=72,
        preserve_markdown_structure=True,
        sanitization_audit=audit,
    )

    assert "| Option | Result |" in answer
    assert "| --- | --- |" in answer
    assert "| Alpha | Supported result [S1] |" in answer
    assert "Beta" not in answer
    assert claims == ["| Alpha | Supported result [S1] |"]
    assert audit == {
        "enabled": True,
        "applied": True,
        "dropped_uncited_sentence_count": 0,
        "dropped_uncited_line_count": 1,
        "dropped_uncited_table_row_count": 1,
    }
    assert "Unsupported result" not in str(audit)


def test_deep_markdown_drops_uncited_factual_heading() -> None:
    audit: dict[str, int | bool] = {}
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Overview\n\n"
                "## Vietnam covers private-sector workers\n\n"
                "Vietnam has a supported scheme detail [S1]."
            ),
            "claims": [],
        },
        allowed_source_ids={"S1"},
        expected_format="markdown",
        query=(
            "This comparison covers Vietnam and private-sector workers in pension systems."
        ),
        max_claims=72,
        preserve_markdown_structure=True,
        sanitization_audit=audit,
    )

    assert "# Overview" in answer
    assert "covers private-sector workers" not in answer
    assert claims == ["Vietnam has a supported scheme detail [S1]."]
    assert audit["dropped_uncited_line_count"] == 1


def test_deep_markdown_positive_structure_authorization_blocks_nominalized_fact() -> None:
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Overview\n\n"
                "## Vietnam: Universal private-sector pension coverage\n\n"
                "Vietnam has a supported scheme detail [S1]."
            ),
            "claims": [],
        },
        allowed_source_ids={"S1"},
        expected_format="markdown",
        query="Compare private-sector pension coverage in Vietnam.",
        max_claims=72,
        preserve_markdown_structure=True,
    )

    assert "Universal private-sector" not in answer
    assert claims == ["Vietnam has a supported scheme detail [S1]."]

    with pytest.raises(ValueError, match="table header contains factual or cited text"):
        _synthesis_from_payload(
            {
                "answer": (
                    "| Country | Universal private-sector pension coverage in Vietnam |\n"
                    "| --- | --- |\n"
                    "| Vietnam | Supported detail [S1] |"
                ),
                "claims": [],
            },
            allowed_source_ids={"S1"},
            expected_format="markdown",
            query="Compare private-sector pension coverage in Vietnam.",
            max_claims=72,
            preserve_markdown_structure=True,
        )


def test_deep_markdown_allows_query_derived_topic_heading_without_assertive_modifier() -> None:
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Overview\n\n"
                "## Vietnam private-sector pension coverage\n\n"
                "Vietnam has a supported scheme detail [S1]."
            ),
            "claims": [],
        },
        allowed_source_ids={"S1"},
        expected_format="markdown",
        query="Compare private-sector pension coverage in Vietnam.",
        max_claims=72,
        preserve_markdown_structure=True,
    )

    assert "## Vietnam private-sector pension coverage" in answer
    assert claims == ["Vietnam has a supported scheme detail [S1]."]


def test_deep_markdown_rejects_factual_table_header() -> None:
    with pytest.raises(ValueError, match="table header contains factual or cited text"):
        _synthesis_from_payload(
            {
                "answer": (
                    "# Overview\n\n"
                    "| Country | Vietnam covers private-sector workers |\n"
                    "| --- | --- |\n"
                    "| Vietnam | Supported detail [S1] |"
                ),
                "claims": [],
            },
            allowed_source_ids={"S1"},
            expected_format="markdown",
            query="Compare pension systems.",
            max_claims=72,
            preserve_markdown_structure=True,
        )


def test_deep_markdown_drops_cited_factual_heading_and_rejects_cited_header() -> None:
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Overview\n\n"
                "## Vietnam covers private-sector workers [S1]\n\n"
                "Vietnam has a supported scheme detail [S1]."
            ),
            "claims": [],
        },
        allowed_source_ids={"S1"},
        expected_format="markdown",
        query="Compare pension systems.",
        max_claims=72,
        preserve_markdown_structure=True,
    )

    assert "covers private-sector workers" not in answer
    assert claims == ["Vietnam has a supported scheme detail [S1]."]

    with pytest.raises(ValueError, match="table header contains factual or cited text"):
        _synthesis_from_payload(
            {
                "answer": (
                    "| Country | Vietnam covers private-sector workers [S1] |\n"
                    "| --- | --- |\n"
                    "| Vietnam | Supported detail [S1] |"
                ),
                "claims": [],
            },
            allowed_source_ids={"S1"},
            expected_format="markdown",
            query="Compare pension systems.",
            max_claims=72,
            preserve_markdown_structure=True,
        )


def test_deep_markdown_keeps_generic_numbered_heading_and_column_labels() -> None:
    query = (
        "Create Table 1: Benefit Formula for Mandatory Defined Benefit (DB) Pension "
        "Schemes with "
        "Country, Plan Name, Sector (Public/Private), Annual Accrual Rate (%), Salary "
        "Base (e.g., Final Salary, Average Salary over X Years), Minimum Benefit, Maximum "
        "Benefit, and Minimum Years of Service Required to Receive Pension. Create Table 2: "
        "Normal Retirement Age with Country, Plan Type, and Retirement Age."
    )
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Executive Summary\n\n"
                "## Table 1: Benefit Formula for Mandatory Defined Benefit (DB) Pension Schemes\n\n"
                "| Country | Plan Name | Sector (Public/Private) | Annual Accrual Rate (%) | Salary Base (e.g., Final Salary, Average Salary over X Years) | Minimum Benefit | Maximum Benefit | Minimum Years of Service Required to Receive Pension |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
                "| Alpha | Plan A | Public | 1.5% | Final Salary | Supported minimum | Supported maximum | 10 years [S1] |\n\n"
                "## Table 2: Normal Retirement Age\n\n"
                "| Country | Plan Type | Retirement Age |\n"
                "| --- | --- | --- |\n"
                "| Alpha | MDB | 65 [S2] |"
            ),
            "claims": [],
        },
        allowed_source_ids={"S1", "S2"},
        expected_format="markdown",
        query=query,
        max_claims=72,
        preserve_markdown_structure=True,
    )

    assert "# Executive Summary" in answer
    assert "## Table 1: Benefit Formula" in answer
    assert "| Country | Plan Name | Sector (Public/Private) | Annual Accrual Rate (%) |" in answer
    assert "| Country | Plan Type | Retirement Age |" in answer
    assert claims == [
        "| Alpha | Plan A | Public | 1.5% | Final Salary | Supported minimum | "
        "Supported maximum | 10 years [S1] |",
        "| Alpha | MDB | 65 [S2] |",
    ]


def test_deep_markdown_keeps_custom_structure_from_another_drb_topic() -> None:
    query = (
        "Provide an AI Scaffolding Approaches section and a Research Summary Table with "
        "Study (Author & Year), AI Technologies Implemented, Sample, and Research Findings."
    )
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# AI Scaffolding Approaches\n\n"
                "## Research Summary Table\n\n"
                "| Study (Author & Year) | AI Technologies Implemented | Sample | Research Findings |\n"
                "| --- | --- | --- | --- |\n"
                "| Study A | Tool A | Students | Supported finding [S1] |"
            ),
            "claims": [],
        },
        allowed_source_ids={"S1"},
        expected_format="markdown",
        query=query,
        max_claims=72,
        preserve_markdown_structure=True,
    )

    assert "# AI Scaffolding Approaches" in answer
    assert "| Study (Author & Year) | AI Technologies Implemented |" in answer
    assert claims == ["| Study A | Tool A | Students | Supported finding [S1] |"]


def test_deep_markdown_allows_only_generic_collection_lead_ins() -> None:
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Comparison\n\n"
                "The following table summarizes the comparison:\n\n"
                "| Option | Result |\n"
                "| --- | --- |\n"
                "| Alpha | Supported result [S1] |\n"
                "| Beta | Supported result [S2] |"
            ),
            "claims": [
                "| Alpha | Supported result [S1] |",
                "| Beta | Supported result [S2] |",
            ],
        },
        allowed_source_ids={"S1", "S2"},
        expected_format="markdown",
        query="Compare Alpha and Beta.",
        max_claims=72,
        preserve_markdown_structure=True,
    )

    assert "The following table summarizes the comparison:" in answer
    assert len(claims) == 2

    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Comparison\n\n"
                "Alpha is the clear winner:\n\n"
                "- Alpha has the stronger result [S1]"
            ),
            "claims": [],
        },
        allowed_source_ids={"S1"},
        expected_format="markdown",
        query="Compare Alpha and Beta.",
        max_claims=72,
        preserve_markdown_structure=True,
    )

    assert "clear winner" not in answer
    assert claims == ["Alpha has the stronger result [S1]"]


def test_deep_markdown_mixed_paragraph_keeps_only_cited_sentences() -> None:
    audit: dict[str, int | bool] = {}
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Findings\n\n"
                "Alpha has documented property A [S1]. "
                "Beta has an unsupported property. "
                "Gamma has documented property C [S2]."
            ),
            "claims": [],
        },
        allowed_source_ids={"S1", "S2"},
        expected_format="markdown",
        query="Compare Alpha, Beta, and Gamma.",
        max_claims=72,
        preserve_markdown_structure=True,
        sanitization_audit=audit,
    )

    assert "Alpha has documented property A [S1]." in answer
    assert "Gamma has documented property C [S2]." in answer
    assert "unsupported property" not in answer
    assert claims == [
        "Alpha has documented property A [S1].",
        "Gamma has documented property C [S2].",
    ]
    assert audit["dropped_uncited_sentence_count"] == 1
    assert audit["dropped_uncited_line_count"] == 0


def test_deep_markdown_recovers_cited_answer_units_missing_from_claims() -> None:
    answer, claims = _synthesis_from_payload(
        {
            "answer": (
                "# Findings\n\n"
                "- **Alpha** has documented property A [S1].\n"
                "- Beta has documented property B [S2]."
            ),
            "claims": ["Alpha has documented property A [S1]."],
        },
        allowed_source_ids={"S1", "S2"},
        expected_format="markdown",
        query="Compare Alpha and Beta.",
        max_claims=72,
        preserve_markdown_structure=True,
    )

    assert answer.startswith("# Findings")
    assert claims == [
        "**Alpha** has documented property A [S1].",
        "Beta has documented property B [S2].",
    ]


def test_deep_markdown_all_uncited_content_still_fails_closed() -> None:
    audit: dict[str, int | bool] = {}
    with pytest.raises(ValueError, match="no usable claims"):
        _synthesis_from_payload(
            {
                "answer": "# Findings\n\nAlpha is better.\n\n- Beta is weaker.",
                "claims": ["Model-supplied claim [S1]"],
            },
            allowed_source_ids={"S1"},
            expected_format="markdown",
            query="Compare Alpha and Beta.",
            max_claims=72,
            preserve_markdown_structure=True,
            sanitization_audit=audit,
        )

    assert audit["applied"] is True
    assert audit["dropped_uncited_sentence_count"] == 2
    assert audit["dropped_uncited_line_count"] == 2


def test_deep_markdown_rejects_unknown_citation_in_recovered_answer_unit() -> None:
    with pytest.raises(ValueError, match="unknown citations"):
        _synthesis_from_payload(
            {
                "answer": "# Findings\n\n- Alpha has documented property A [S99].",
                "claims": [],
            },
            allowed_source_ids={"S1"},
            expected_format="markdown",
            query="Compare Alpha and Beta.",
            max_claims=72,
            preserve_markdown_structure=True,
        )


def test_deep_claim_limit_applies_after_recovering_missing_answer_units() -> None:
    claims = [f"Fact {index} [S1]" for index in range(71)]
    answer_claims = [*claims, "Fact 72 [S1]", "Fact 73 [S1]"]

    with pytest.raises(ValueError, match="maximum is 72"):
        _synthesis_from_payload(
            {"answer": "\n".join(answer_claims), "claims": claims},
            allowed_source_ids={"S1"},
            expected_format="markdown",
            query="Produce a deep report.",
            max_claims=72,
            preserve_markdown_structure=True,
        )


def test_deep_claim_limit_is_large_but_still_bounded() -> None:
    valid_claims = [f"Fact {index} [S1]" for index in range(72)]
    answer, claims = _synthesis_from_payload(
        {"answer": "\n".join(valid_claims), "claims": valid_claims},
        allowed_source_ids={"S1"},
        expected_format="markdown",
        query="Produce a deep report.",
        max_claims=72,
        preserve_markdown_structure=True,
    )

    assert answer.splitlines() == valid_claims
    assert claims == valid_claims

    with pytest.raises(ValueError, match="maximum is 72"):
        _synthesis_from_payload(
            {
                "answer": "\n".join([*valid_claims, "Fact 73 [S1]"]),
                "claims": [*valid_claims, "Fact 73 [S1]"],
            },
            allowed_source_ids={"S1"},
            expected_format="markdown",
            query="Produce a deep report.",
            max_claims=72,
            preserve_markdown_structure=True,
        )
