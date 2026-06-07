from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepresearch_agent.cost import deepseek_usage_cost_usd


@dataclass(frozen=True)
class AnswerJudgment:
    provider: str
    score: float | None
    verdict: str
    reason: str
    matched: list[str]
    missing: list[str]
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class EvalJudgeProvider(Protocol):
    name: str

    def judge(self, case: dict[str, Any], record: dict[str, Any]) -> AnswerJudgment:
        raise NotImplementedError


class HeuristicAnswerJudgeProvider:
    name = "heuristic"
    model = "local-groundtruth-substring"

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
                model=self.model,
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
            model=self.model,
        )


class DeepSeekAnswerJudgeProvider:
    name = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def judge(self, case: dict[str, Any], record: dict[str, Any]) -> AnswerJudgment:
        prompt = _deepseek_answer_judge_prompt(case, record)
        payload = self._post_chat_completions(prompt)
        content = _extract_content(payload)
        parsed = _parse_json_object(content)
        score = _normalize_score(parsed.get("score"))
        verdict = _normalize_answer_verdict(parsed.get("verdict"), score)
        if score is None:
            score = _score_from_verdict(verdict)
        reason = str(parsed.get("reason") or "").strip() or "judge returned no reason"
        input_tokens, output_tokens, estimated_cost = deepseek_usage_cost_usd(
            self.model,
            payload.get("usage") or {},
        )
        return AnswerJudgment(
            provider=self.name,
            score=round(score, 4),
            verdict=verdict,
            reason=reason,
            matched=_string_list(parsed.get("matched")),
            missing=_string_list(parsed.get("missing")),
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(estimated_cost, 8),
        )

    def _post_chat_completions(self, prompt: str) -> dict[str, Any]:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is required")
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an answer-quality judge for Deep Research eval cases. "
                            "Return strict json only. Use the query, ground-truth groups, "
                            "and generated answer provided by the user message. The json "
                            "object must match "
                            '{"score":0.0,"verdict":"pass|partial|fail|unscored",'
                            '"reason":"...","matched":["..."],"missing":["..."]}'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "max_tokens": 700,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek answer judge HTTP {exc.code}: {_redact(error_body)}") from exc
        except URLError as exc:
            raise RuntimeError(f"DeepSeek answer judge request failed: {exc.reason}") from exc


def build_eval_judge_provider(
    name: str | None,
    *,
    model: str | None = None,
    timeout_seconds: float = 30.0,
) -> EvalJudgeProvider | None:
    selected = (name or "none").strip().lower()
    if selected in {"", "none"}:
        return None
    if selected == "heuristic":
        return HeuristicAnswerJudgeProvider()
    if selected == "deepseek":
        return DeepSeekAnswerJudgeProvider(
            model=model or "deepseek-v4-flash",
            timeout_seconds=timeout_seconds,
        )
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


def _deepseek_answer_judge_prompt(case: dict[str, Any], record: dict[str, Any]) -> str:
    payload = {
        "query": case.get("query", ""),
        "ground_truth_groups": _ground_truth_groups(case),
        "generated_answer": record.get("answer", ""),
        "grading_policy": (
            "Score 1.0 only when the answer fully satisfies every ground-truth group. "
            "Use partial scores for incomplete but relevant answers. Use 0.0 when the "
            "answer misses the expected facts or is empty. Ground-truth groups contain "
            "acceptable alternatives; matching one item in a group is enough for that group."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("answer judge response missing choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("answer judge response missing message.content")
    return content


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("answer judge response is not a json object")
    return parsed


def _normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(score, 0.0), 1.0)


def _normalize_answer_verdict(value: Any, score: float | None) -> str:
    verdict = str(value or "").strip().lower()
    if verdict in {"pass", "partial", "fail", "unscored"}:
        return verdict
    if score is None:
        return "unscored"
    if score >= 0.95:
        return "pass"
    if score > 0:
        return "partial"
    return "fail"


def _score_from_verdict(verdict: str) -> float:
    if verdict == "pass":
        return 1.0
    if verdict == "partial":
        return 0.5
    return 0.0


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _redact(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", text)
