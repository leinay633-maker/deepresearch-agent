from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AnswerJudgment:
    provider: str
    score: float | None
    verdict: str
    reason: str
    matched: list[str]
    missing: list[str]


class EvalJudgeProvider(Protocol):
    name: str

    def judge(self, case: dict[str, Any], record: dict[str, Any]) -> AnswerJudgment:
        raise NotImplementedError


class HeuristicAnswerJudgeProvider:
    name = "heuristic"

    def judge(self, case: dict[str, Any], record: dict[str, Any]) -> AnswerJudgment:
        groups = _ground_truth_groups(case)
        if not groups:
            return AnswerJudgment(
                provider=self.name,
                score=None,
                verdict="unscored",
                reason="case did not provide ground_truths/answer/expected_answer metadata",
                matched=[],
                missing=[],
            )
        answer = _normalize_text(str(record.get("answer") or ""))
        matched: list[str] = []
        missing: list[str] = []
        for group in groups:
            normalized_group = [_normalize_text(item) for item in group if item.strip()]
            if any(item and item in answer for item in normalized_group):
                matched.append(group[0])
            else:
                missing.append(group[0])
        score = len(matched) / len(groups)
        if score >= 1.0:
            verdict = "pass"
        elif score > 0:
            verdict = "partial"
        else:
            verdict = "fail"
        return AnswerJudgment(
            provider=self.name,
            score=round(score, 4),
            verdict=verdict,
            reason=(
                "heuristic score is the fraction of ground-truth groups whose normalized "
                "string appears in the generated answer"
            ),
            matched=matched,
            missing=missing,
        )


def build_eval_judge_provider(name: str | None) -> EvalJudgeProvider | None:
    selected = (name or "none").strip().lower()
    if selected in {"", "none"}:
        return None
    if selected == "heuristic":
        return HeuristicAnswerJudgeProvider()
    raise ValueError(f"unknown eval judge provider: {name}")


def _ground_truth_groups(case: dict[str, Any]) -> list[list[str]]:
    metadata = case.get("metadata") or {}
    for key in ("ground_truths", "ground_truth", "answers", "answer", "expected_answer"):
        if key in metadata:
            return _value_to_groups(metadata[key])
        if key in case:
            return _value_to_groups(case[key])
    return []


def _value_to_groups(value: Any) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [[value]]
    if isinstance(value, dict):
        groups: list[list[str]] = []
        for item in value.values():
            groups.extend(_value_to_groups(item))
        return groups
    if isinstance(value, list):
        groups = []
        for item in value:
            if isinstance(item, list):
                alternatives = [str(text) for text in _flatten_strings(item) if str(text).strip()]
                if alternatives:
                    groups.append(alternatives)
            else:
                groups.extend(_value_to_groups(item))
        return groups
    return [[str(value)]]


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        output: list[str] = []
        for item in value.values():
            output.extend(_flatten_strings(item))
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(_flatten_strings(item))
        return output
    return [str(value)]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", text.lower())).strip()
