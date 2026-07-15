from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit, urlunsplit

from deepresearch_agent.text_utils import split_sentences, tokenize

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_URL = re.compile(r"(?:https?|file|ftp)://", re.IGNORECASE)
_ROLE_OR_TOOL_SYNTAX = re.compile(
    r"(?:^|\s)(?:system|assistant|developer|tool)\s*:|"
    r"<\s*/?\s*(?:system|assistant|developer|tool|function)|"
    r"(?:tool_call|function_call|recipient_name)",
    re.IGNORECASE,
)
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)", re.I),
    re.compile(r"reveal\s+(?:the\s+)?(?:system|developer)\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\b", re.I),
    re.compile(r"忽略(?:以上|之前|前面|所有).{0,12}(?:指令|要求|提示词)"),
    re.compile(r"(?:泄露|输出|展示).{0,12}(?:系统提示词|开发者指令|密钥|api\s*key)", re.I),
    re.compile(r"你现在是.{0,20}(?:助手|智能体|模型)"),
)


def looks_like_prompt_injection(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    return bool(
        _ROLE_OR_TOOL_SYNTAX.search(normalized)
        or any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS)
    )


def sanitize_untrusted_text(text: str) -> tuple[str, bool]:
    """Remove instruction-like sentences while preserving ordinary evidence text."""

    flagged = False
    kept: list[str] = []
    for sentence in split_sentences(text):
        if looks_like_prompt_injection(sentence):
            flagged = True
            continue
        kept.append(_CONTROL_CHARACTERS.sub("", sentence))
    return " ".join(kept).strip(), flagged


def safe_untrusted_source_payload(
    *,
    source_id: str | None = None,
    title: str | None = None,
    url: str | None = None,
    quote: str | None = None,
    query: str | None = None,
) -> dict[str, object]:
    """Create one prompt-safe representation at a model trust boundary."""

    clean_title, title_flagged = sanitize_untrusted_text(title or "")
    clean_quote, quote_flagged = sanitize_untrusted_text(quote or "")
    clean_query, query_flagged = sanitize_untrusted_text(query or "")
    clean_url, url_flagged = safe_display_url(url or "")
    return {
        "source_id": source_id or "",
        "source_title": clean_title,
        "source_url": clean_url,
        "quote": clean_quote,
        "query": clean_query,
        "untrusted_external_content": True,
        "injection_suspected": (
            title_flagged or quote_flagged or query_flagged or url_flagged
        ),
    }


def safe_display_url(raw_url: str) -> tuple[str, bool]:
    """Remove query/fragment and reject instruction-shaped URL metadata."""

    if not isinstance(raw_url, str):
        return "", False
    try:
        decoded = unquote(raw_url)
        parsed = urlsplit(raw_url.strip())
    except ValueError:
        return "", False
    probe = re.sub(r"[^A-Za-z0-9\u3400-\u9fff]+", " ", decoded)
    if looks_like_prompt_injection(probe):
        return "", True
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return "", False
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "metadata.google.internal"} or host.endswith(
        (".localhost", ".metadata.google.internal")
    ):
        return "", False
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path, "", "")), False


def safe_follow_up_query(
    candidate: str | None,
    *,
    original_question: str,
    evidence_gap: str | None = None,
    max_chars: int = 240,
) -> str:
    """Validate model-proposed search actions before they reach a live search adapter."""

    fallback = " ".join(
        part.strip()
        for part in (original_question, evidence_gap or "补充权威一手证据")
        if part and part.strip()
    )
    raw = (candidate or "").strip()
    invalid_shape = (
        not raw
        or "\n" in raw
        or "\r" in raw
        or bool(_CONTROL_CHARACTERS.search(raw))
        or bool(_URL.search(raw))
        or bool(_ROLE_OR_TOOL_SYNTAX.search(raw))
        or looks_like_prompt_injection(raw)
    )
    if invalid_shape:
        return fallback[:max_chars].rstrip()

    normalized = re.sub(r"\s+", " ", raw).strip()
    bounded = normalized[:max_chars].rstrip()
    if len(normalized) > max_chars and not normalized[max_chars].isspace():
        boundary = bounded.rfind(" ")
        if boundary > 0:
            bounded = bounded[:boundary].rstrip()
    if not bounded:
        return fallback[:max_chars].rstrip()

    anchor_terms = tokenize(f"{original_question} {evidence_gap or ''}")
    candidate_terms = tokenize(bounded)
    if anchor_terms and not anchor_terms.intersection(candidate_terms):
        return fallback[:max_chars].rstrip()
    return bounded
