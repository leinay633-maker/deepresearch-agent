from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit, urlunsplit

from deepresearch_agent.guardrails import looks_like_prompt_injection, sanitize_untrusted_text
from deepresearch_agent.schemas import Finding, Source, SubQuestion
from deepresearch_agent.text_utils import split_sentences, tokenize


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_DISPLAY_URL_PROBE_SEPARATORS = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff]+")
_BLOCKED_DISPLAY_HOSTS = frozenset({"localhost", "metadata.google.internal"})
_MAX_DISPLAY_URL_CHARS = 2_048


@dataclass(frozen=True)
class PackedContext:
    sources: list[dict]
    estimated_tokens: int
    kept_source_ids: list[str]
    dropped_source_ids: list[str]
    injection_flagged_source_ids: list[str]


def estimate_tokens(text: str) -> int:
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    return max(1, (ascii_count + 3) // 4 + non_ascii_count)


def pack_sources_for_synthesis(
    *,
    query: str,
    plan: list[SubQuestion],
    findings: list[Finding],
    sources: list[Source],
    max_input_tokens: int = 12_000,
    reserved_tokens: int = 3_500,
    per_source_tokens: int = 650,
    max_sources: int = 18,
    max_sources_per_domain: int = 2,
) -> PackedContext:
    available = max(256, max_input_tokens - reserved_tokens)
    context_terms = tokenize(
        " ".join(
            [query, *[item.question for item in plan], *[item.summary for item in findings]]
        )
    )
    evidence_by_source: dict[str, list[str]] = {}
    for finding in findings:
        if finding.research is None:
            continue
        for evidence in finding.research.evidence:
            if evidence.source_id and evidence.quote:
                evidence_by_source.setdefault(evidence.source_id, []).append(evidence.quote)

    ranked: list[tuple[float, Source, str, str, str, bool]] = []
    for source in sources:
        excerpt, relevance, excerpt_flagged = _best_excerpt(
            source,
            context_terms=context_terms,
            evidence_quotes=evidence_by_source.get(source.id, []),
            token_limit=per_source_tokens,
        )
        if not excerpt:
            continue
        title, title_flagged = sanitize_untrusted_text(source.title)
        display_url, url_flagged = _safe_display_url(source.url)
        evidence_bonus = 20.0 if evidence_by_source.get(source.id) else 0.0
        ranked.append(
            (
                evidence_bonus + relevance + source.quality_score * 4 + source.score * 0.05,
                source,
                title or "Untitled source",
                display_url,
                excerpt,
                excerpt_flagged or title_flagged or url_flagged,
            )
        )
    ranked.sort(key=lambda row: (row[0], row[1].quality_score, row[1].id), reverse=True)

    packed: list[dict] = []
    used_tokens = 0
    domain_counts: dict[str, int] = {}
    flagged_ids: list[str] = []
    for _, source, title, display_url, excerpt, flagged in ranked:
        if len(packed) >= max_sources:
            break
        domain = (urlsplit(display_url).hostname or source.provider or "unknown").lower()
        if domain_counts.get(domain, 0) >= max_sources_per_domain:
            continue
        item = {
            "id": source.id,
            "title": title,
            "url": display_url,
            "provider": source.provider,
            "quality_score": source.quality_score,
            "untrusted_external_content": True,
            "excerpt": excerpt,
        }
        item_tokens = estimate_tokens(str(item))
        if used_tokens + item_tokens > available:
            continue
        packed.append(item)
        used_tokens += item_tokens
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if flagged:
            flagged_ids.append(source.id)

    kept_ids = [item["id"] for item in packed]
    kept_set = set(kept_ids)
    return PackedContext(
        sources=packed,
        estimated_tokens=used_tokens,
        kept_source_ids=kept_ids,
        dropped_source_ids=[source.id for source in sources if source.id not in kept_set],
        injection_flagged_source_ids=flagged_ids,
    )


def _best_excerpt(
    source: Source,
    *,
    context_terms: set[str],
    evidence_quotes: list[str],
    token_limit: int,
) -> tuple[str, float, bool]:
    candidates = [*evidence_quotes, *_passages(source.content)]
    scored: list[tuple[float, str, bool]] = []
    for index, passage in enumerate(candidates):
        cleaned, flagged = sanitize_untrusted_text(passage)
        if not cleaned:
            continue
        overlap = len(context_terms.intersection(tokenize(cleaned)))
        evidence_bonus = 100 if index < len(evidence_quotes) else 0
        scored.append((float(evidence_bonus + overlap), cleaned, flagged))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)

    selected: list[str] = []
    selected_tokens = 0
    flagged = False
    seen: set[str] = set()
    for score, passage, passage_flagged in scored:
        normalized = passage.strip()
        if normalized in seen:
            continue
        passage_tokens = estimate_tokens(normalized)
        if selected and selected_tokens + passage_tokens > token_limit:
            continue
        if not selected and passage_tokens > token_limit:
            normalized = _truncate_to_tokens(normalized, token_limit)
            passage_tokens = estimate_tokens(normalized)
        selected.append(normalized)
        seen.add(normalized)
        selected_tokens += passage_tokens
        flagged = flagged or passage_flagged
        if selected_tokens >= token_limit or len(selected) >= 4:
            break
    relevance = scored[0][0] if scored else 0.0
    return "\n\n".join(selected), relevance, flagged


def _passages(text: str, window_chars: int = 1_200, overlap_chars: int = 160) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) > 1:
        passages: list[str] = []
        current: list[str] = []
        current_chars = 0
        for sentence in sentences:
            if current and current_chars + len(sentence) > window_chars:
                passages.append(" ".join(current))
                current = current[-1:]
                current_chars = sum(len(item) for item in current)
            current.append(sentence)
            current_chars += len(sentence)
        if current:
            passages.append(" ".join(current))
        return passages
    if len(text) <= window_chars:
        return [text]
    step = max(1, window_chars - overlap_chars)
    return [text[start : start + window_chars] for start in range(0, len(text), step)]


def _truncate_to_tokens(text: str, token_limit: int) -> str:
    if estimate_tokens(text) <= token_limit:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= token_limit:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip()


def _safe_display_url(raw_url: str) -> tuple[str, bool]:
    """Return a prompt-safe HTTP(S) display URL without query or fragment data."""

    if not isinstance(raw_url, str):
        return "", False
    url = raw_url.strip()
    try:
        decoded = unquote(url)
    except (UnicodeDecodeError, ValueError):
        return "", False
    probe = _DISPLAY_URL_PROBE_SEPARATORS.sub(" ", decoded).strip()
    injection_flagged = looks_like_prompt_injection(probe)
    if injection_flagged:
        return "", True
    if (
        not url
        or url != raw_url
        or len(url) > _MAX_DISPLAY_URL_CHARS
        or any(character.isspace() for character in url)
        or _CONTROL_CHARACTERS.search(url)
        or _CONTROL_CHARACTERS.search(decoded)
    ):
        return "", False

    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "", False
    if (
        scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
        or "\\" in parsed.netloc
    ):
        return "", False

    try:
        normalized_host = hostname.rstrip(".").lower().encode("idna").decode("ascii")
    except UnicodeError:
        return "", False
    if (
        not normalized_host
        or normalized_host in _BLOCKED_DISPLAY_HOSTS
        or normalized_host.endswith(".localhost")
        or normalized_host.endswith(".metadata.google.internal")
    ):
        return "", False
    try:
        literal = ipaddress.ip_address(normalized_host)
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        return "", False

    display_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    netloc = f"{display_host}:{port}" if port is not None else display_host
    return urlunsplit((scheme, netloc, parsed.path, "", "")), False
