from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deepresearch_agent.cost import CostTracker
from deepresearch_agent.schemas import (
    EvidenceItem,
    Finding,
    ResearchBrief,
    ResearchDecision,
    ResearchRequest,
    Source,
    SubQuestion,
)


CassetteMode = Literal["record", "replay"]
CassetteKind = Literal["llm", "search", "tool", "stage"]


class CassetteEntry(BaseModel):
    """One deterministic provider interaction stored as a JSONL row."""

    sequence: int = Field(ge=1)
    kind: CassetteKind
    operation: str = Field(min_length=1)
    request: dict[str, Any] = Field(default_factory=dict)
    response: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class CassetteMismatchError(RuntimeError):
    pass


class CassetteSearchAdapter:
    """Search adapter backed by a strict provider cassette."""

    name = "cassette"

    def __init__(self, replayer: CassetteReplayer) -> None:
        self.replayer = replayer

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        del timeout
        response = self.replayer.next_response(
            kind="search",
            operation="search",
            request={"query": query, "max_results": max_results},
        )
        rows = response.get("sources", response) if isinstance(response, dict) else response
        if not isinstance(rows, list):
            raise ValueError("cassette search response must be a source list")
        return [Source.model_validate(row) for row in rows]


class CassetteLLMProvider:
    """Minimal stage provider for offline orchestration replay tests."""

    name = "cassette"
    supports_structured_output = True
    supports_tool_calling = False

    def __init__(self, replayer: CassetteReplayer, model: str = "cassette") -> None:
        self.replayer = replayer
        self.model = model

    async def create_brief(self, request: ResearchRequest, cost: CostTracker) -> ResearchBrief:
        response = self._next("brief", {"query": request.query})
        payload = response.get("brief", response) if isinstance(response, dict) else response
        brief = ResearchBrief(
            original_query=request.query,
            **{key: payload[key] for key in ("normalized_query", "scope", "constraints", "assumptions")},
        )
        self._record_cost(cost, "brief_generation", response)
        return brief

    async def plan(
        self, brief: ResearchBrief, max_researchers: int, cost: CostTracker
    ) -> list[SubQuestion]:
        response = self._next(
            "plan", {"normalized_query": brief.normalized_query, "max_researchers": max_researchers}
        )
        payload = response.get("subquestions", response) if isinstance(response, dict) else response
        plan = [SubQuestion.model_validate(item) for item in payload]
        self._record_cost(cost, "planning", response)
        return plan

    async def synthesize(
        self,
        brief: ResearchBrief,
        plan: list[SubQuestion],
        findings: list[Finding],
        sources: list[Source],
        cost: CostTracker,
    ) -> tuple[str, list[str]]:
        del brief, plan, findings, sources
        response = self._next("synthesis", {})
        payload = response.get("result", response) if isinstance(response, dict) else response
        answer = str(payload["answer"])
        claims = [str(item) for item in payload["claims"]]
        self._record_cost(cost, "synthesis", response)
        return answer, claims

    async def decide_research(
        self,
        subquestion: SubQuestion,
        evidence: list[EvidenceItem],
        min_evidence_items: int,
        round_index: int,
        cost: CostTracker,
    ) -> ResearchDecision:
        response = self._next(
            "research_decision",
            {
                "subquestion_id": subquestion.id,
                "evidence_count": len(evidence),
                "min_evidence_items": min_evidence_items,
                "round_index": round_index,
            },
        )
        payload = response.get("decision", response) if isinstance(response, dict) else response
        self._record_cost(cost, "research_decision", response)
        return ResearchDecision.model_validate(payload)

    def _next(self, operation: str, request: dict[str, Any]) -> Any:
        return self.replayer.next_response(kind="llm", operation=operation, request=request)

    def _record_cost(self, cost: CostTracker, stage: str, response: Any) -> None:
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            cost.add_usage(
                stage=stage,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                estimated_cost_usd=float(usage.get("estimated_cost_usd", 0.0)),
                provider=self.name,
                model=self.model,
                cache_creation_input_tokens=int(
                    usage.get("cache_creation_input_tokens", 0)
                ),
                cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0)),
            )


def write_cassette(path: str | Path, entries: list[CassetteEntry]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for entry in entries
    ]
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return target


def read_cassette(path: str | Path) -> list[CassetteEntry]:
    source = Path(path)
    entries: list[CassetteEntry] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entries.append(CassetteEntry.model_validate_json(raw))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid cassette row {line_number}: {source}") from exc
    sequences = [entry.sequence for entry in entries]
    if sequences != list(range(1, len(entries) + 1)):
        raise ValueError(f"cassette sequence must be contiguous starting at 1: {source}")
    return entries


class CassetteReplayer:
    """Strict sequential replay so request drift fails visibly."""

    def __init__(self, entries: list[CassetteEntry]) -> None:
        self.entries = entries
        self.position = 0

    @classmethod
    def from_path(cls, path: str | Path) -> "CassetteReplayer":
        return cls(read_cassette(path))

    @property
    def remaining(self) -> int:
        return len(self.entries) - self.position

    def next_response(
        self,
        *,
        kind: CassetteKind,
        operation: str,
        request: dict[str, Any],
    ) -> Any:
        if self.position >= len(self.entries):
            raise CassetteMismatchError("cassette exhausted")
        entry = self.entries[self.position]
        expected = (entry.kind, entry.operation, entry.request)
        actual = (kind, operation, request)
        if actual != expected:
            raise CassetteMismatchError(
                f"cassette mismatch at sequence {entry.sequence}: "
                f"expected {expected!r}, got {actual!r}"
            )
        self.position += 1
        return entry.response

    def assert_exhausted(self) -> None:
        if self.remaining:
            raise CassetteMismatchError(
                f"cassette has {self.remaining} unconsumed entries starting at position {self.position}"
            )


def load_case_result_records(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load auditable case-result artifacts for deterministic benchmark replay."""

    source = _resolve_case_result_path(Path(path))
    records: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid replay JSONL row {line_number}: {source}") from exc
        if row.get("type") != "case_result":
            continue
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"replay row {line_number} has no case_id: {source}")
        if case_id in records:
            raise ValueError(f"duplicate replay case_id {case_id}: {source}")
        records[case_id] = row
    if not records:
        raise ValueError(f"no case_result rows found in replay artifact: {source}")
    return records


def validate_replay_case_ids(
    cases: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> None:
    """Require replay artifacts to match the current dataset exactly."""

    case_ids = [str(case.get("id") or "").strip() for case in cases]
    if any(not case_id for case_id in case_ids):
        raise CassetteMismatchError("current replay dataset contains a case without an id")
    if len(case_ids) != len(set(case_ids)):
        raise CassetteMismatchError("current replay dataset contains duplicate case IDs")
    current = set(case_ids)
    recorded = set(records)
    if current == recorded:
        return
    missing = sorted(current - recorded)
    extra = sorted(recorded - current)
    raise CassetteMismatchError(
        "replay case ID set mismatch: "
        f"missing_from_artifact={missing}, extra_in_artifact={extra}"
    )


def case_result_artifact_id(path: str | Path) -> str:
    source = _resolve_case_result_path(Path(path))
    return f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}"


def replay_case_result(
    case: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    manifest_id: str,
) -> dict[str, Any]:
    case_id = str(case["id"])
    if case_id not in records:
        raise CassetteMismatchError(f"replay artifact has no case_id {case_id}")
    recorded = records[case_id]
    if str(recorded.get("query") or "") != str(case["query"]):
        raise CassetteMismatchError(f"replay query mismatch for case_id {case_id}")
    _validate_case_contract(case, recorded)
    replayed = json.loads(json.dumps(recorded, ensure_ascii=False))
    replayed["manifest_id"] = manifest_id
    replayed["replayed"] = True
    replayed["generation_replay"] = {
        "deterministic": True,
        "mode": "recorded_case_artifact",
        "source_manifest_id": recorded.get("manifest_id"),
    }
    if isinstance(recorded.get("answer_judgment"), dict):
        replayed["recorded_answer_judgment"] = json.loads(
            json.dumps(recorded["answer_judgment"], ensure_ascii=False)
        )
    return replayed


def citation_judge_identities(
    citation_check: dict[str, Any] | None,
) -> list[dict[str, str | None]]:
    """Return the distinct recorded citation judges without guessing missing metadata."""

    if not isinstance(citation_check, dict):
        return []
    identities: list[dict[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    assessments = citation_check.get("assessments")
    if not isinstance(assessments, list):
        return identities
    for assessment in assessments:
        if not isinstance(assessment, dict):
            continue
        provider = str(assessment.get("judge_provider") or "").strip().lower()
        if not provider:
            continue
        model_value = assessment.get("judge_model")
        model = str(model_value).strip() if model_value is not None else None
        identity = (provider, model or None)
        if identity in seen:
            continue
        seen.add(identity)
        identities.append({"provider": provider, "model": model or None})
    return identities


def _validate_case_contract(case: dict[str, Any], recorded: dict[str, Any]) -> None:
    """Reject replay when the current case contract drifted from the artifact."""

    recorded_metadata = recorded.get("case_metadata") or recorded.get("metadata") or {}
    if not isinstance(recorded_metadata, dict):
        recorded_metadata = {}
    comparable_keys = {
        "category",
        "language",
        "expected_format",
        "scenario",
        "failure_injection",
    }
    for key in comparable_keys:
        current = case.get(key)
        if current is None and isinstance(case.get("metadata"), dict):
            current = case["metadata"].get(key)
        recorded_value = recorded.get(key)
        if recorded_value is None:
            recorded_value = recorded_metadata.get(key)
        if current is not None and recorded_value is not None and current != recorded_value:
            raise CassetteMismatchError(
                f"replay case contract mismatch for {case.get('id')}: {key} "
                f"expected {recorded_value!r}, got {current!r}"
            )
    current_metadata = case.get("metadata")
    if isinstance(current_metadata, dict):
        for key in ("ground_truths", "ground_truth", "answers", "answer", "expected_answer"):
            if key in current_metadata and key in recorded_metadata:
                if current_metadata[key] != recorded_metadata[key]:
                    raise CassetteMismatchError(
                        f"replay ground-truth metadata mismatch for case_id {case.get('id')}"
                    )


def _resolve_case_result_path(path: Path) -> Path:
    source = path.expanduser().resolve()
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(source)
    candidates = sorted(source.glob("*.jsonl"))
    if len(candidates) != 1:
        raise ValueError(
            f"replay directory must contain exactly one JSONL artifact; found {len(candidates)}"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a DeepResearch JSONL cassette.")
    parser.add_argument("cassette")
    args = parser.parse_args()
    path = Path(args.cassette)
    entries = read_cassette(path)
    summary = {
        "path": str(path),
        "entry_count": len(entries),
        "kinds": dict(sorted(Counter(entry.kind for entry in entries).items())),
        "operations": sorted({entry.operation for entry in entries}),
        "deterministic": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
