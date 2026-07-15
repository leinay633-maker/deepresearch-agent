from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from ipaddress import ip_address
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


LLM_GATEWAY_API_KEY_ENV = "LLM_GATEWAY_API_KEY"
LLM_GATEWAY_DEFAULT_BASE_URL = "https://llmapi.bilibili.co"
ANTHROPIC_VERSION = "2023-06-01"
KIMI_MIN_THINKING_BUDGET_TOKENS = 1024
MAX_GATEWAY_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class GatewayMessageResult:
    content: str
    model: str
    usage: dict[str, int]
    stop_reason: str | None = None


class LLMGatewayModelMismatchError(RuntimeError):
    """Strict-routing failure that retains the auditable Gateway model identity."""

    def __init__(
        self,
        *,
        requested_model: str,
        actual_model: str | None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.requested_model = requested_model
        self.actual_model = normalize_gateway_model_identifier(
            actual_model,
            requested_model=requested_model,
        )
        self.usage = dict(usage or {})
        super().__init__("LLM Gateway response model did not match the requested model")


class LLMGatewayNoTextContentError(RuntimeError):
    """A response completed without a usable text block.

    Only aggregate-safe metadata is retained. In particular, the raw response and
    any thinking block body are intentionally absent from the exception.
    """

    def __init__(
        self,
        *,
        requested_model: str,
        actual_model: str | None,
        stop_reason: str | None,
        content_block_types: tuple[str, ...],
        usage: dict[str, int],
        response_bytes: int,
        raw_response_sha256: str,
    ) -> None:
        self.requested_model = requested_model
        self.actual_model = normalize_gateway_model_identifier(
            actual_model,
            requested_model=requested_model,
        )
        self.stop_reason = _safe_stop_reason(stop_reason)
        self.content_block_types = content_block_types
        self.usage = dict(usage)
        self.response_bytes = max(int(response_bytes), 0)
        self.raw_response_sha256 = raw_response_sha256
        super().__init__("LLM Gateway returned no text content")


class _RejectRedirectHandler(HTTPRedirectHandler):
    """Do not permit a Bearer-bearing request to follow any redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


def no_redirect_urlopen(request: Request, timeout: float):
    return build_opener(_RejectRedirectHandler()).open(request, timeout=timeout)


class LLMGatewayClient:
    """Minimal Anthropic Messages client for the internal LLM Gateway."""

    def __init__(
        self,
        *,
        base_url: str = LLM_GATEWAY_DEFAULT_BASE_URL,
        timeout_seconds: float = 120.0,
        thinking_budget_tokens: int = 1024,
        require_response_model_match: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if thinking_budget_tokens < KIMI_MIN_THINKING_BUDGET_TOKENS:
            raise ValueError(
                "thinking_budget_tokens must be at least "
                f"{KIMI_MIN_THINKING_BUDGET_TOKENS}"
            )
        self.base_url = validate_gateway_base_url(base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.thinking_budget_tokens = thinking_budget_tokens
        self.require_response_model_match = require_response_model_match

    def create_message(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> GatewayMessageResult:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        request_timeout_seconds = (
            self.timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if request_timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        api_key = os.environ.get(LLM_GATEWAY_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"{LLM_GATEWAY_API_KEY_ENV} environment variable is required"
            )

        system, conversation = _anthropic_messages(messages)
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": conversation,
        }
        if system:
            body["system"] = system
        if _requires_thinking(model):
            # Kimi requires thinking to be enabled. Its thinking budget is in
            # addition to the caller's requested text-output budget.
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
            body["max_tokens"] = max_tokens + self.thinking_budget_tokens

        request = Request(
            _messages_url(self.base_url),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with no_redirect_urlopen(
                request,
                timeout=request_timeout_seconds,
            ) as response:
                raw_response = _read_response_bounded(response)
                payload = json.loads(raw_response.decode("utf-8"))
        except HTTPError as exc:
            # Do not include the response body: upstream proxies sometimes
            # echo request metadata and credentials in diagnostic payloads.
            raise RuntimeError(f"LLM Gateway HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM Gateway request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM Gateway returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise ValueError("LLM Gateway response is not a JSON object")
        raw_response_model = payload.get("model")
        if self.require_response_model_match and (
            not isinstance(raw_response_model, str)
            or not response_model_matches(model, raw_response_model)
        ):
            mismatch_error = LLMGatewayModelMismatchError(
                requested_model=model,
                actual_model=raw_response_model,
                usage=_normalize_usage(payload.get("usage")),
            )
            # As with no-text failures, do not retain a response body (which may
            # include thinking) in the raised exception's create_message frame.
            del payload
            del raw_response
            raise mismatch_error
        content = _extract_text_content(payload)
        if not content:
            no_text_error = LLMGatewayNoTextContentError(
                requested_model=model,
                actual_model=_actual_response_model(payload, requested_model=model),
                stop_reason=_safe_stop_reason(payload.get("stop_reason")),
                content_block_types=_content_block_types(payload),
                usage=_normalize_usage(payload.get("usage")),
                response_bytes=len(raw_response),
                raw_response_sha256=hashlib.sha256(raw_response).hexdigest(),
            )
            # Do not retain the response (including thinking bodies) in the
            # exception traceback's create_message frame.
            del payload
            del raw_response
            raise no_text_error
        return GatewayMessageResult(
            content=content,
            model=_response_model(payload, requested_model=model),
            usage=_normalize_usage(payload.get("usage")),
            stop_reason=_safe_stop_reason(payload.get("stop_reason")),
        )


def _read_response_bounded(response: Any) -> bytes:
    headers = getattr(response, "headers", None)
    content_length = getattr(headers, "get", lambda _name: None)("Content-Length")
    try:
        if content_length is not None and int(content_length) > MAX_GATEWAY_RESPONSE_BYTES:
            raise RuntimeError("LLM Gateway response exceeded size limit")
    except (TypeError, ValueError):
        pass
    chunks: list[bytes] = []
    total = 0
    while True:
        read_size = min(64 * 1024, MAX_GATEWAY_RESPONSE_BYTES + 1 - total)
        try:
            chunk = response.read(read_size)
        except TypeError:
            # A few test doubles and compatibility transports only implement
            # read() without a size argument.  Read them once, then enforce
            # the same hard limit before accepting their payload.
            if chunks:
                raise RuntimeError("LLM Gateway response does not support bounded reads")
            chunk = response.read()
            if not isinstance(chunk, bytes):
                raise RuntimeError("LLM Gateway response returned non-byte content")
            if len(chunk) > MAX_GATEWAY_RESPONSE_BYTES:
                raise RuntimeError("LLM Gateway response exceeded size limit")
            return chunk
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise RuntimeError("LLM Gateway response returned non-byte content")
        total += len(chunk)
        if total > MAX_GATEWAY_RESPONSE_BYTES:
            raise RuntimeError("LLM Gateway response exceeded size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _anthropic_messages(
    messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    conversation: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM Gateway messages require non-empty string content")
        if role == "system":
            system_parts.append(content)
        elif role in {"user", "assistant"}:
            conversation.append({"role": role, "content": content})
        else:
            raise ValueError(f"unsupported LLM Gateway message role: {role or '<empty>'}")
    if not conversation:
        raise ValueError("LLM Gateway request requires at least one user or assistant message")
    return "\n\n".join(system_parts), conversation


def _messages_url(base_url: str) -> str:
    lowered = base_url.lower()
    if lowered.endswith("/v1/messages"):
        return base_url
    if lowered.endswith("/v1"):
        return f"{base_url}/messages"
    return f"{base_url}/v1/messages"


def validate_gateway_base_url(base_url: str) -> str:
    normalized = base_url.strip()
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid LLM Gateway base URL") from exc
    if (
        not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid LLM Gateway base URL")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("invalid LLM Gateway base URL port")
    if parsed.scheme == "https":
        return normalized
    if parsed.scheme == "http" and _is_loopback_host(hostname):
        return normalized
    raise ValueError("LLM Gateway base URL must use HTTPS unless it is a loopback address")


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _requires_thinking(model: str) -> bool:
    return "kimi" in model.strip().lower()


def _extract_text_content(payload: dict[str, Any]) -> str:
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return ""
    text_parts = [
        str(block.get("text"))
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block.get("text", "").strip()
    ]
    return "\n".join(text_parts).strip()


def _content_block_types(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return bounded, ordered type names without retaining block bodies."""

    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return ()
    known_types = {
        "redacted_thinking",
        "server_tool_use",
        "text",
        "thinking",
        "tool_result",
        "tool_use",
        "web_search_tool_result",
    }
    normalized: list[str] = []
    for block in blocks[:32]:
        if not isinstance(block, dict):
            normalized.append("invalid")
            continue
        block_type = block.get("type")
        if not isinstance(block_type, str) or not block_type.strip():
            normalized.append("missing")
            continue
        candidate = block_type.strip().lower()
        normalized.append(candidate if candidate in known_types else "unknown")
    if len(blocks) > 32:
        normalized.append("truncated")
    return tuple(normalized)


def _normalize_usage(raw_usage: Any) -> dict[str, int]:
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    return {
        "input_tokens": _non_negative_int(usage.get("input_tokens")),
        "output_tokens": _non_negative_int(usage.get("output_tokens")),
        "cache_creation_input_tokens": _non_negative_int(
            usage.get("cache_creation_input_tokens")
        ),
        "cache_read_input_tokens": _non_negative_int(
            usage.get("cache_read_input_tokens")
        ),
    }


def _non_negative_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _response_model(payload: dict[str, Any], *, requested_model: str) -> str:
    return (
        normalize_gateway_model_identifier(
            payload.get("model"),
            requested_model=requested_model,
        )
        or requested_model
    )


def _actual_response_model(
    payload: dict[str, Any],
    *,
    requested_model: str,
) -> str | None:
    return normalize_gateway_model_identifier(
        payload.get("model"),
        requested_model=requested_model,
    )


def response_model_matches(requested_model: str, response_model: str) -> bool:
    """Allow exact models and the observed valid YYYYMM/YYYYMMDD aliases."""

    requested = requested_model.strip().lower()
    response = response_model.strip().lower()
    if not requested:
        return False
    if response == requested:
        return True
    prefix = f"{requested}-"
    if not response.startswith(prefix):
        return False
    suffix = response[len(prefix) :]
    date_format = {6: "%Y%m", 8: "%Y%m%d"}.get(len(suffix))
    if date_format is None or not suffix.isdigit():
        return False
    try:
        datetime.strptime(suffix, date_format)
    except ValueError:
        return False
    return True


def normalize_gateway_model_identifier(
    value: Any,
    *,
    requested_model: str | None = None,
) -> str | None:
    """Keep only bounded protocol-like model identifiers in audit metadata."""

    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if len(candidate) > 200 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]*",
        candidate,
    ):
        return "unknown"
    if (
        requested_model
        and candidate.lower().startswith(f"{requested_model.strip().lower()}-")
        and not response_model_matches(requested_model, candidate)
    ):
        return "unknown"
    return candidate


def _safe_stop_reason(value: Any) -> str | None:
    """Normalize only aggregate protocol enums; never retain provider prose."""

    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip().lower().replace("-", "_")
    allowed = {
        "end_turn",
        "max_tokens",
        "pause_turn",
        "refusal",
        "stop_sequence",
        "tool_use",
    }
    return candidate if candidate in allowed else "unknown"
