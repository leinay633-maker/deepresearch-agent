from __future__ import annotations

import json
import os
from dataclasses import dataclass
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
    ) -> GatewayMessageResult:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
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
            with no_redirect_urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(_read_response_bounded(response).decode("utf-8"))
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
            raise RuntimeError("LLM Gateway response model did not match the requested model")
        return GatewayMessageResult(
            content=_extract_text_content(payload),
            model=_response_model(payload, requested_model=model),
            usage=_normalize_usage(payload.get("usage")),
            stop_reason=_optional_string(payload.get("stop_reason")),
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
        raise ValueError("LLM Gateway response missing content blocks")
    text_parts = [
        str(block.get("text"))
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block.get("text", "").strip()
    ]
    content = "\n".join(text_parts).strip()
    if not content:
        raise ValueError("LLM Gateway returned no text content")
    return content


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
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return requested_model


def response_model_matches(requested_model: str, response_model: str) -> bool:
    """Allow a dated Gateway alias, but never a different model family."""

    requested = requested_model.strip().lower()
    response = response_model.strip().lower()
    return bool(requested) and (
        response == requested or response.startswith(f"{requested}-")
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
