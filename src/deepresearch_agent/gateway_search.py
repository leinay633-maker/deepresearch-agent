from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from deepresearch_agent.guardrails import safe_follow_up_query
from deepresearch_agent.llm_gateway import (
    ANTHROPIC_VERSION,
    KIMI_MIN_THINKING_BUDGET_TOKENS,
    LLM_GATEWAY_API_KEY_ENV,
    _requires_thinking,
    response_model_matches,
)
from deepresearch_agent.schemas import Source


WEB_SEARCH_BETA = "web-search-2025-03-05"
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
DEFAULT_GATEWAY_WEB_SEARCH_MODEL = "claude-4.6-opus"
DEFAULT_GATEWAY_WEB_SEARCH_MAX_TOKENS = 500
DEFAULT_EMPTY_RESULT_RETRIES = 2
MAX_GATEWAY_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class GatewayWebSearchUsage:
    """One successfully parsed Gateway web-search response's token usage."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


GatewayWebSearchUsageRecorder = Callable[[GatewayWebSearchUsage], None]
_USAGE_RECORDER: ContextVar[GatewayWebSearchUsageRecorder | None] = ContextVar(
    "gateway_web_search_usage_recorder",
    default=None,
)


@contextmanager
def capture_gateway_web_search_usage(
    recorder: GatewayWebSearchUsageRecorder,
) -> Iterator[None]:
    """Bind a task-local recorder without coupling search results to accounting."""

    token = _USAGE_RECORDER.set(recorder)
    try:
        yield
    finally:
        _USAGE_RECORDER.reset(token)


class GatewayWebSearchError(RuntimeError):
    def __init__(self, message: str, *, request_attempts: int = 1) -> None:
        super().__init__(message)
        self.request_attempts = max(int(request_attempts), 1)


class GatewayWebSearchNoResultsError(GatewayWebSearchError):
    """Raised after the adapter has already exhausted its empty-result retries."""


class GatewayWebSearchAdapter:
    """Discover candidate URLs with the Gateway's server-side web-search tool.

    Results remain unverified snippets until SearchService safely crawls each URL.
    The adapter uses a cancellable async transport and never follows redirects.
    """

    name = "gateway-web"

    def __init__(
        self,
        *,
        base_url: str,
        model: str = DEFAULT_GATEWAY_WEB_SEARCH_MODEL,
        max_chars: int = 4000,
        max_tokens: int = DEFAULT_GATEWAY_WEB_SEARCH_MAX_TOKENS,
        empty_result_retries: int = DEFAULT_EMPTY_RESULT_RETRIES,
        timeout_seconds: float = 120.0,
        require_response_model_match: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        thinking_budget_tokens: int = KIMI_MIN_THINKING_BUDGET_TOKENS,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if empty_result_retries < 0:
            raise ValueError("empty_result_retries must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = _validated_messages_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.model = model.strip() or DEFAULT_GATEWAY_WEB_SEARCH_MODEL
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.empty_result_retries = empty_result_retries
        self.timeout_seconds = timeout_seconds
        self.require_response_model_match = require_response_model_match
        self.transport = transport
        self.thinking_budget_tokens = max(
            thinking_budget_tokens, KIMI_MIN_THINKING_BUDGET_TOKENS
        )

    async def search(self, query: str, max_results: int, timeout: float) -> list[Source]:
        api_key = os.environ.get(LLM_GATEWAY_API_KEY_ENV)
        if not api_key:
            raise GatewayWebSearchError(
                f"{LLM_GATEWAY_API_KEY_ENV} environment variable is required"
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        original_query = query.strip()
        if not original_query:
            raise ValueError("query must be non-empty")
        if max_results <= 0:
            return []

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        current_query = original_query
        async with httpx.AsyncClient(
            follow_redirects=False,
            transport=self.transport,
        ) as client:
            for attempt in range(1, self.empty_result_retries + 2):
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise GatewayWebSearchError(
                        "Gateway web search timed out",
                        request_attempts=max(attempt - 1, 1),
                    )
                try:
                    payload = await self._request(
                        client,
                        current_query,
                        remaining,
                        api_key,
                    )
                except GatewayWebSearchError as exc:
                    exc.request_attempts = attempt
                    raise
                sources = _response_sources(
                    payload,
                    original_query=original_query,
                    search_query=current_query,
                    provider=self.name,
                    max_results=max_results,
                    max_chars=self.max_chars,
                    attempt=attempt,
                    requested_model=self.model,
                )
                if sources:
                    return sources
                if attempt <= self.empty_result_retries:
                    current_query = _retry_query(
                        payload,
                        original_query=original_query,
                    )

        attempts = self.empty_result_retries + 1
        raise GatewayWebSearchNoResultsError(
            f"Gateway web search returned no results after {attempts} attempts",
            request_attempts=attempts,
        )

    async def _request(
        self,
        client: httpx.AsyncClient,
        query: str,
        timeout: float,
        api_key: str,
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "tools": [
                {
                    "type": WEB_SEARCH_TOOL_TYPE,
                    "name": "web_search",
                    "max_uses": 1,
                }
            ],
            "messages": [{"role": "user", "content": query}],
        }
        # Some gateway models (e.g. Kimi) reject requests without an explicit
        # enabled thinking block. Mirror LLMGatewayClient so web search works
        # for the same model set as structured generation.
        if _requires_thinking(self.model):
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
            body["max_tokens"] = self.max_tokens + self.thinking_budget_tokens
        headers = {
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": WEB_SEARCH_BETA,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            async with asyncio.timeout(timeout):
                async with client.stream(
                    "POST",
                    self.endpoint,
                    json=body,
                    headers=headers,
                    timeout=httpx.Timeout(timeout),
                ) as response:
                    if response.is_redirect:
                        raise GatewayWebSearchError(
                            f"Gateway web search refused HTTP {response.status_code} redirect"
                        )
                    if not 200 <= response.status_code < 300:
                        # Never read the error body: upstream diagnostics can echo
                        # credentials or other request metadata.
                        raise GatewayWebSearchError(
                            f"Gateway web search HTTP {response.status_code}"
                        )
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > MAX_GATEWAY_RESPONSE_BYTES:
                            raise GatewayWebSearchError(
                                "Gateway web search response exceeded size limit"
                            )
        except asyncio.TimeoutError as exc:
            raise GatewayWebSearchError("Gateway web search timed out") from exc
        except httpx.TimeoutException as exc:
            raise GatewayWebSearchError("Gateway web search timed out") from exc
        except GatewayWebSearchError:
            raise
        except httpx.HTTPError as exc:
            raise GatewayWebSearchError("Gateway web search request failed") from exc
        except Exception as exc:  # noqa: BLE001 - keep transport errors secret-safe.
            raise GatewayWebSearchError("Gateway web search request failed") from exc

        try:
            payload = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GatewayWebSearchError("Gateway web search returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise GatewayWebSearchError("Gateway web search response is not a JSON object")
        raw_response_model = payload.get("model")
        if self.require_response_model_match and (
            not isinstance(raw_response_model, str)
            or not response_model_matches(self.model, raw_response_model)
        ):
            raise GatewayWebSearchError(
                "Gateway web search response model did not match the requested model"
            )
        _record_response_usage(payload, requested_model=self.model)
        return payload


def _record_response_usage(
    payload: dict[str, Any],
    *,
    requested_model: str,
) -> None:
    recorder = _USAGE_RECORDER.get()
    if recorder is None:
        return
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, dict):
        return
    token_fields = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    # An absent/empty usage object is not evidence of a zero-token request.
    if not any(field in raw_usage for field in token_fields):
        return
    response_model = payload.get("model")
    model = (
        response_model.strip()
        if isinstance(response_model, str) and response_model.strip()
        else requested_model
    )
    recorder(
        GatewayWebSearchUsage(
            model=model,
            input_tokens=_non_negative_int(raw_usage.get("input_tokens")),
            output_tokens=_non_negative_int(raw_usage.get("output_tokens")),
            cache_creation_input_tokens=_non_negative_int(
                raw_usage.get("cache_creation_input_tokens")
            ),
            cache_read_input_tokens=_non_negative_int(
                raw_usage.get("cache_read_input_tokens")
            ),
        )
    )


def _non_negative_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _validated_messages_url(base_url: str) -> str:
    endpoint = _messages_url(base_url.strip().rstrip("/"))
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid LLM Gateway URL") from exc
    if not parsed.netloc or not hostname:
        raise ValueError("LLM Gateway URL must include a hostname")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("LLM Gateway URL userinfo is not allowed")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM Gateway URL must not include query or fragment")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("LLM Gateway URL port must be between 1 and 65535")
    scheme = parsed.scheme.lower()
    if scheme == "https":
        return endpoint
    if scheme == "http" and _is_loopback_host(hostname):
        return endpoint
    raise ValueError(
        "LLM Gateway Bearer requests require HTTPS; HTTP is allowed only for loopback tests"
    )


def _messages_url(base_url: str) -> str:
    lowered = base_url.lower()
    if lowered.endswith("/v1/messages"):
        return base_url
    if lowered.endswith("/v1"):
        return f"{base_url}/messages"
    return f"{base_url}/v1/messages"


def _is_loopback_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _response_sources(
    payload: dict[str, Any],
    *,
    original_query: str,
    search_query: str,
    provider: str,
    max_results: int,
    max_chars: int,
    attempt: int,
    requested_model: str,
) -> list[Source]:
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return []
    response_model = payload.get("model")
    if not isinstance(response_model, str) or not response_model.strip():
        response_model = requested_model

    sources: list[Source] = []
    seen_urls: set[str] = set()
    for result in _web_search_results(blocks):
        url = result.get("url")
        if not isinstance(url, str):
            continue
        url = url.strip()
        if not _is_http_url(url) or url in seen_urls:
            continue
        seen_urls.add(url)
        title = result.get("title")
        if not isinstance(title, str) or not title.strip():
            title = url
        page_age = result.get("page_age")
        cited_text = result.get("cited_text") or result.get("snippet")
        content = _source_snippet(
            title=title.strip(),
            page_age=page_age,
            cited_text=cited_text,
            max_chars=max_chars,
        )
        rank = len(sources)
        sources.append(
            Source(
                title=title.strip(),
                url=url,
                content=content,
                provider=provider,
                query=original_query,
                score=float(max(max_results - rank, 1)),
                metadata={
                    "search_api": "llm_gateway_web_search",
                    "search_query": search_query,
                    "gateway_model": response_model,
                    "gateway_attempt": attempt,
                    "provider_request_attempts": attempt,
                    "page_age": page_age,
                    "candidate_only": True,
                    "requires_crawl": True,
                    "verification_status": "unverified_snippet",
                    "snippet_only": True,
                    "extract_status": "snippet",
                    "content_type": "text/plain",
                },
            )
        )
        if len(sources) >= max_results:
            break
    return sources


def _web_search_results(blocks: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for value in _walk_json(blocks):
        block_type = value.get("type")
        if block_type == "web_search_result":
            results.append(value)
        elif block_type == "web_search_result_location" and value.get("url"):
            # Some gateway versions expose only text citations instead of the
            # enclosing web_search_tool_result block.
            results.append(value)
    return results


def _retry_query(payload: dict[str, Any], *, original_query: str) -> str:
    blocks = payload.get("content")
    candidate: str | None = None
    fallback: str | None = None
    if isinstance(blocks, list):
        for value in _walk_json(blocks):
            name = str(value.get("name") or "").strip().lower()
            tool_input = value.get("input")
            if name not in {"remote_web_search", "web_search"} or not isinstance(
                tool_input, dict
            ):
                continue
            query = tool_input.get("query")
            if not isinstance(query, str) or not query.strip():
                continue
            if name == "remote_web_search":
                candidate = query
                break
            fallback = fallback or query
    return safe_follow_up_query(
        candidate or fallback,
        original_question=original_query,
    )


def _source_snippet(
    *,
    title: str,
    page_age: Any,
    cited_text: Any,
    max_chars: int,
) -> str:
    parts = [title]
    if isinstance(page_age, str) and page_age.strip():
        parts.append(f"Search result age: {page_age.strip()}")
    if isinstance(cited_text, str) and cited_text.strip():
        parts.append(cited_text.strip())
    return _normalize_text(". ".join(parts), max_chars=max_chars) or title


def _normalize_text(text: str, *, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0 and len(normalized) > max_chars:
        return normalized[:max_chars].rstrip()
    return normalized


def _is_http_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)


def _walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)
