from __future__ import annotations

import asyncio
import hashlib
import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError

import pytest

import deepresearch_agent.llm as llm_module
import deepresearch_agent.llm_gateway as gateway_module
from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.cost import CostTracker
from deepresearch_agent.llm import LLMGatewayLLMProvider, _subquestions_from_payload
from deepresearch_agent.llm_gateway import (
    MAX_GATEWAY_RESPONSE_BYTES,
    GatewayMessageResult,
    LLMGatewayClient,
    LLMGatewayModelMismatchError,
    LLMGatewayNoTextContentError,
    response_model_matches,
)
from deepresearch_agent.orchestrator import DeepResearchOrchestrator
from deepresearch_agent.schemas import (
    EvidenceItem,
    Finding,
    ResearchBrief,
    ResearchRequest,
    Source,
    SubQuestion,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _SizedFakeHTTPResponse:
    headers: dict[str, str] = {}

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.payload) - self.offset
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def _gateway_no_text_error(
    *,
    stop_reason: str | None = "end_turn",
    content_block_types: tuple[str, ...] = ("text",),
    usage: dict[str, int] | None = None,
    actual_model: str | None = "claude-opus-4-8-20260701",
    response_sha: str = "a" * 64,
) -> LLMGatewayNoTextContentError:
    return LLMGatewayNoTextContentError(
        requested_model="claude-opus-4-8",
        actual_model=actual_model,
        stop_reason=stop_reason,
        content_block_types=content_block_types,
        usage={"input_tokens": 19, "output_tokens": 0} if usage is None else usage,
        response_bytes=211,
        raw_response_sha256=response_sha,
    )


@contextmanager
def _local_http_server(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gateway_client_uses_anthropic_messages_and_ignores_thinking(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeHTTPResponse(
            {
                "model": "glm-5.2",
                "content": [
                    {"type": "thinking", "thinking": "private reasoning"},
                    {"type": "text", "text": '{"ok":true}'},
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 5,
                },
                "stop_reason": "end_turn",
            }
        )

    monkeypatch.setattr(gateway_module, "no_redirect_urlopen", fake_urlopen)
    client = LLMGatewayClient(base_url="https://gateway.local", timeout_seconds=9.0)

    result = client.create_message(
        model="glm-5.2",
        messages=[
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Say ok."},
        ],
        max_tokens=400,
    )

    assert captured["url"] == "https://gateway.local/v1/messages"
    assert captured["timeout"] == 9.0
    assert captured["headers"]["authorization"] == "Bearer test-gateway-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"] == {
        "model": "glm-5.2",
        "max_tokens": 400,
        "messages": [{"role": "user", "content": "Say ok."}],
        "system": "Return JSON.",
    }
    assert result.content == '{"ok":true}'
    assert result.model == "glm-5.2"
    assert result.usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 5,
    }


def test_gateway_client_applies_required_kimi_thinking_parameters(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    captured: dict = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured.update(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse(
            {
                "model": "kimi-k2.7-code-highspeed",
                "content": [{"type": "text", "text": "{}"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    monkeypatch.setattr(gateway_module, "no_redirect_urlopen", fake_urlopen)
    client = LLMGatewayClient(thinking_budget_tokens=1024)

    client.create_message(
        model="kimi-k2.7-code-highspeed",
        messages=[{"role": "user", "content": "Return JSON."}],
        max_tokens=500,
    )

    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert captured["max_tokens"] == 1524
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "top_k" not in captured


def test_gateway_client_requires_runtime_environment_key(monkeypatch) -> None:
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="LLM_GATEWAY_API_KEY"):
        LLMGatewayClient().create_message(
            model="claude-4.6-opus",
            messages=[{"role": "user", "content": "Return JSON."}],
            max_tokens=100,
        )


def test_gateway_client_rejects_kimi_budget_below_gateway_minimum() -> None:
    with pytest.raises(ValueError, match="at least 1024"):
        LLMGatewayClient(thinking_budget_tokens=1023)


def test_gateway_client_rejects_model_routing_drift_when_strict(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")

    def fake_urlopen(request, timeout):
        del request, timeout
        return _FakeHTTPResponse(
            {
                "model": "glm-5.2",
                "content": [{"type": "text", "text": "{}"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    monkeypatch.setattr(gateway_module, "no_redirect_urlopen", fake_urlopen)

    with pytest.raises(
        LLMGatewayModelMismatchError, match="response model did not match"
    ) as captured:
        LLMGatewayClient(
            base_url="https://gateway.local",
            require_response_model_match=True,
        ).create_message(
            model="claude-4.6-opus",
            messages=[{"role": "user", "content": "Return JSON."}],
            max_tokens=100,
        )

    assert isinstance(captured.value, RuntimeError)
    assert captured.value.requested_model == "claude-4.6-opus"
    assert captured.value.actual_model == "glm-5.2"
    assert captured.value.usage == {
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert not hasattr(captured.value, "raw_response")


def test_gateway_client_retains_safe_metadata_for_thinking_only_response(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    thinking_body = "private reasoning that must never enter artifacts"
    payload = {
        "model": "claude-opus-4-8-20260701",
        "content": [{"type": "thinking", "thinking": thinking_body}],
        "usage": {
            "input_tokens": 101,
            "output_tokens": 10000,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 11,
        },
        "stop_reason": "max_tokens",
    }
    raw_response = json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        gateway_module,
        "no_redirect_urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    with pytest.raises(LLMGatewayNoTextContentError) as captured:
        LLMGatewayClient(base_url="https://gateway.local").create_message(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Return JSON."}],
            max_tokens=10_000,
        )

    error = captured.value
    assert error.requested_model == "claude-opus-4-8"
    assert error.actual_model == "claude-opus-4-8-20260701"
    assert error.stop_reason == "max_tokens"
    assert error.content_block_types == ("thinking",)
    assert error.usage == {
        "input_tokens": 101,
        "output_tokens": 10000,
        "cache_creation_input_tokens": 7,
        "cache_read_input_tokens": 11,
    }
    assert error.response_bytes == len(raw_response)
    assert error.raw_response_sha256 == hashlib.sha256(raw_response).hexdigest()
    assert thinking_body not in str(error)
    assert thinking_body not in repr(error)
    assert not hasattr(error, "raw_response")


def test_gateway_client_retains_safe_metadata_for_empty_content(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    payload = {
        "model": "claude-opus-4-8",
        "content": [],
        "usage": {"input_tokens": 13, "output_tokens": 0},
        "stop_reason": "end_turn",
    }
    monkeypatch.setattr(
        gateway_module,
        "no_redirect_urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    with pytest.raises(LLMGatewayNoTextContentError) as captured:
        LLMGatewayClient(base_url="https://gateway.local").create_message(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Return JSON."}],
            max_tokens=100,
        )

    assert captured.value.content_block_types == ()
    assert captured.value.actual_model == "claude-opus-4-8"
    assert captured.value.stop_reason == "end_turn"
    assert captured.value.usage["input_tokens"] == 13


def test_gateway_client_normalizes_v10_blank_text_shape_for_retry_policy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    payload = {
        "model": "claude-opus-4-8-20260701",
        "content": [{"type": "text", "text": " \n\t"}],
        # v10 omitted output_tokens; the Gateway client normalizes it to zero.
        "usage": {"input_tokens": 19_422},
        "stop_reason": "end_turn",
    }
    monkeypatch.setattr(
        gateway_module,
        "no_redirect_urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    with pytest.raises(LLMGatewayNoTextContentError) as captured:
        LLMGatewayClient(
            base_url="https://gateway.local",
            require_response_model_match=True,
        ).create_message(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Return report JSON."}],
            max_tokens=10_000,
        )

    assert captured.value.actual_model == "claude-opus-4-8-20260701"
    assert captured.value.stop_reason == "end_turn"
    assert captured.value.content_block_types == ("text",)
    assert captured.value.usage["input_tokens"] == 19_422
    assert captured.value.usage["output_tokens"] == 0


def test_gateway_unknown_stop_reason_never_reaches_exception_or_ledger(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    private_stop_reason = "private reasoning that must never enter artifacts"
    payload = {
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": "   "}],
        "usage": {"input_tokens": 19, "output_tokens": 0},
        "stop_reason": private_stop_reason,
    }
    monkeypatch.setattr(
        gateway_module,
        "no_redirect_urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    with pytest.raises(LLMGatewayNoTextContentError) as gateway_error:
        LLMGatewayClient(
            base_url="https://gateway.local",
            require_response_model_match=True,
        ).create_message(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Return report JSON."}],
            max_tokens=10_000,
        )

    assert gateway_error.value.stop_reason == "unknown"
    assert private_stop_reason not in str(gateway_error.value)
    assert private_stop_reason not in repr(gateway_error.value)

    client = _SequenceGatewayClient(
        [gateway_error.value],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=2,
        client=client,
    )
    with pytest.raises(RuntimeError) as provider_error:
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return report JSON."}],
                max_tokens=10_000,
            )
        )

    assert len(client.calls) == 1
    assert provider_error.value.attempt_ledger[0]["stop_reason"] == "unknown"
    assert private_stop_reason not in repr(provider_error.value)
    assert private_stop_reason not in json.dumps(provider_error.value.attempt_ledger)


def test_gateway_mismatch_bounds_untrusted_actual_model_metadata(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    private_model = "claude-opus-4-8-private-reasoning-must-not-leak"
    payload = {
        "model": private_model,
        "content": [{"type": "text", "text": "   "}],
        "usage": {"input_tokens": 2, "output_tokens": 1},
        "stop_reason": "end_turn",
    }
    monkeypatch.setattr(
        gateway_module,
        "no_redirect_urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    with pytest.raises(LLMGatewayModelMismatchError) as captured:
        LLMGatewayClient(
            base_url="https://gateway.local",
            require_response_model_match=True,
        ).create_message(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Return report JSON."}],
            max_tokens=100,
        )

    assert captured.value.actual_model == "unknown"
    assert private_model not in str(captured.value)
    assert private_model not in repr(captured.value)

    client = _SequenceGatewayClient([captured.value])
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        client=client,
    )
    with pytest.raises(LLMGatewayModelMismatchError) as provider_error:
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return report JSON."}],
                max_tokens=100,
            )
        )

    assert provider_error.value.attempt_ledger[0]["actual_model"] == "unknown"
    assert private_model not in json.dumps(provider_error.value.attempt_ledger)


def test_gateway_model_match_allows_only_exact_or_valid_date_alias() -> None:
    requested = "claude-opus-4-8"

    assert response_model_matches(requested, requested) is True
    assert response_model_matches(requested, "claude-opus-4-8-20260701") is True
    assert response_model_matches(
        "kimi-k2.7-code-highspeed",
        "kimi-k2.7-code-highspeed-202607",
    ) is True
    assert response_model_matches(
        requested,
        "claude-opus-4-8-private-reasoning",
    ) is False
    assert response_model_matches(requested, "claude-opus-4-8-20261340") is False
    assert response_model_matches(
        "kimi-k2.7-code-highspeed",
        "kimi-k2.7-code-highspeed-202613",
    ) is False
    assert response_model_matches(requested, "claude-opus-4-8-20260701-extra") is False


def test_gateway_client_preserves_duplicate_blank_text_block_types(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    payload = {
        "model": "claude-opus-4-8",
        "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": " \n"},
        ],
        "usage": {"input_tokens": 21, "output_tokens": 0},
        "stop_reason": "end_turn",
    }
    monkeypatch.setattr(
        gateway_module,
        "no_redirect_urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    with pytest.raises(LLMGatewayNoTextContentError) as captured:
        LLMGatewayClient(
            base_url="https://gateway.local",
            require_response_model_match=True,
        ).create_message(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Return report JSON."}],
            max_tokens=10_000,
        )

    assert captured.value.content_block_types == ("text", "text")
    assert not hasattr(captured.value, "raw_response")


def test_gateway_client_marks_content_block_type_metadata_truncation(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    payload = {
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": ""} for _index in range(33)],
        "usage": {"input_tokens": 21, "output_tokens": 0},
        "stop_reason": "end_turn",
    }
    monkeypatch.setattr(
        gateway_module,
        "no_redirect_urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    with pytest.raises(LLMGatewayNoTextContentError) as captured:
        LLMGatewayClient(
            base_url="https://gateway.local",
            require_response_model_match=True,
        ).create_message(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Return report JSON."}],
            max_tokens=10_000,
        )

    assert captured.value.content_block_types == ("text",) * 32 + ("truncated",)


def test_gateway_strict_model_check_precedes_no_text_classification(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    payload = {
        "model": "glm-5.2",
        "content": [{"type": "thinking", "thinking": "must not surface"}],
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "stop_reason": "max_tokens",
    }
    monkeypatch.setattr(
        gateway_module,
        "no_redirect_urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    with pytest.raises(LLMGatewayModelMismatchError) as captured:
        LLMGatewayClient(
            base_url="https://gateway.local",
            require_response_model_match=True,
        ).create_message(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Return JSON."}],
            max_tokens=100,
        )

    assert captured.value.actual_model == "glm-5.2"


def test_gateway_client_rejects_plain_http_for_non_loopback_hosts() -> None:
    with pytest.raises(ValueError, match="must use HTTPS"):
        LLMGatewayClient(base_url="http://llmapi.example.com")


def test_gateway_client_allows_plain_http_for_loopback_development() -> None:
    client = LLMGatewayClient(base_url="http://127.0.0.1:8402")

    assert client.base_url == "http://127.0.0.1:8402"


def test_gateway_client_rejects_url_userinfo_query_and_fragment() -> None:
    for base_url in (
        "https://user:pass@gateway.local",
        "https://gateway.local?token=value",
        "https://gateway.local#fragment",
    ):
        with pytest.raises(ValueError, match="invalid LLM Gateway"):
            LLMGatewayClient(base_url=base_url)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://gateway.local", "https://gateway.local/v1/messages"),
        ("https://gateway.local/v1", "https://gateway.local/v1/messages"),
        ("https://gateway.local/v1/messages", "https://gateway.local/v1/messages"),
    ],
)
def test_gateway_client_accepts_host_version_or_endpoint_base_url(
    monkeypatch,
    base_url,
    expected,
) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["url"] = request.full_url
        return _FakeHTTPResponse(
            {
                "model": "claude-4.6-opus",
                "content": [{"type": "text", "text": "{}"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    monkeypatch.setattr(gateway_module, "no_redirect_urlopen", fake_urlopen)

    LLMGatewayClient(base_url=base_url).create_message(
        model="claude-4.6-opus",
        messages=[{"role": "user", "content": "Return JSON."}],
        max_tokens=100,
    )

    assert captured["url"] == expected


def test_gateway_http_error_does_not_echo_credentials(monkeypatch) -> None:
    secret = "gateway-secret-that-must-not-leak"
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", secret)

    def fake_urlopen(request, timeout):
        del request, timeout
        raise HTTPError(
            "https://gateway.local/v1/messages",
            401,
            "unauthorized",
            {},
            io.BytesIO(f"echoed credential: {secret}".encode()),
        )

    monkeypatch.setattr(gateway_module, "no_redirect_urlopen", fake_urlopen)

    with pytest.raises(RuntimeError) as error:
        LLMGatewayClient(base_url="https://gateway.local").create_message(
            model="claude-4.6-opus",
            messages=[{"role": "user", "content": "Return JSON."}],
            max_tokens=100,
        )

    assert secret not in str(error.value)
    assert str(error.value) == "LLM Gateway HTTP 401"


def test_gateway_client_rejects_redirect_before_target_receives_authorization(
    monkeypatch,
) -> None:
    secret = "gateway-secret-that-must-not-reach-redirect-target"
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", secret)
    source_authorizations: list[str | None] = []
    target_authorizations: list[str | None] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - HTTP handler callback name.
            target_authorizations.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    with _local_http_server(TargetHandler) as target_base_url:

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - HTTP handler callback name.
                source_authorizations.append(self.headers.get("Authorization"))
                content_length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(content_length)
                self.send_response(302)
                self.send_header("Location", f"{target_base_url}/redirect-target")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        with _local_http_server(RedirectHandler) as gateway_base_url:
            client = LLMGatewayClient(base_url=gateway_base_url)
            with pytest.raises(RuntimeError, match="LLM Gateway HTTP 302"):
                client.create_message(
                    model="claude-4.6-opus",
                    messages=[{"role": "user", "content": "Return JSON."}],
                    max_tokens=100,
                )

    assert source_authorizations == [f"Bearer {secret}"]
    assert target_authorizations == []


def test_gateway_client_rejects_response_larger_than_hard_limit(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-gateway-key")

    def fake_urlopen(request, timeout):
        del request, timeout
        return _SizedFakeHTTPResponse(b"x" * (MAX_GATEWAY_RESPONSE_BYTES + 1))

    monkeypatch.setattr(gateway_module, "no_redirect_urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="response exceeded size limit"):
        LLMGatewayClient(base_url="https://gateway.local").create_message(
            model="claude-4.6-opus",
            messages=[{"role": "user", "content": "Return JSON."}],
            max_tokens=100,
        )


def test_gateway_provider_repairs_invalid_structured_output_and_records_usage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [
            GatewayMessageResult(
                content="not valid JSON",
                model="claude-4.6-opus",
                usage={"input_tokens": 4, "output_tokens": 2},
            ),
            GatewayMessageResult(
                content=(
                    '{"normalized_query":"How does gateway routing work?",'
                    '"scope":"Verify the real provider path.",'
                    '"constraints":["Use runtime credentials."],'
                    '"assumptions":[]}'
                ),
                model="claude-4.6-opus",
                usage={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 4,
                },
            ),
        ]
    )
    provider = LLMGatewayLLMProvider(
        model="claude-4.6-opus",
        base_url="https://gateway.local",
        max_retries=1,
        client=client,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)

    brief = asyncio.run(
        provider.create_brief(
            ResearchRequest(query="How does gateway routing work?"),
            cost,
        )
    )

    assert brief.scope == "Verify the real provider path."
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-2]["role"] == "assistant"
    assert "failed JSON schema validation" in client.calls[1]["messages"][-1]["content"]
    assert len(cost.records) == 2
    assert cost.records[0].provider == "llm-gateway"
    assert cost.records[0].estimated_cost_usd == 0.0
    assert cost.records[0].input_tokens == 4
    assert cost.records[0].output_tokens == 2
    assert cost.records[1].cache_creation_input_tokens == 3
    assert cost.records[1].cache_read_input_tokens == 4
    assert cost.summary().total_input_tokens == 21
    assert cost.summary().total_output_tokens == 7
    assert cost.summary().total_tokens == 28


def test_gateway_provider_records_all_usage_when_retries_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [
            GatewayMessageResult(
                content="not JSON one",
                model="claude-4.6-opus",
                usage={"input_tokens": 4, "output_tokens": 2},
            ),
            GatewayMessageResult(
                content="not JSON two",
                model="claude-4.6-opus",
                usage={"input_tokens": 6, "output_tokens": 3},
            ),
        ]
    )
    provider = LLMGatewayLLMProvider(
        model="claude-4.6-opus",
        base_url="https://gateway.local",
        max_retries=1,
        client=client,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)

    with pytest.raises(RuntimeError, match="JSON validation failed"):
        asyncio.run(
            provider.create_brief(
                ResearchRequest(query="How does gateway routing work?"),
                cost,
            )
        )

    assert [(item.input_tokens, item.output_tokens) for item in cost.records] == [
        (4, 2),
        (6, 3),
    ]
    assert cost.summary().total_tokens == 15


def test_deep_synthesis_transport_timeout_is_single_fail_closed_attempt() -> None:
    client = _SequenceGatewayClient(
        [TimeoutError("The read operation timed out")]
    )
    provider = LLMGatewayLLMProvider(
        model="claude-4.6-opus",
        base_url="https://gateway.local",
        timeout_seconds=240.0,
        synthesis_timeout_seconds=360.0,
        max_retries=2,
        client=client,
    )

    with pytest.raises(RuntimeError, match="read operation timed out") as captured:
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return the full report JSON."}],
                max_tokens=10_000,
            )
        )

    assert len(client.calls) == 1
    assert client.calls[0]["timeout_seconds"] == 360.0
    assert client.calls[0]["max_tokens"] == 10_000
    assert captured.value.failure_class == "transport_timeout"
    assert captured.value.attempt_ledger == [
        {
            "attempt": 1,
            "request_kind": "initial",
            "failure_class": "transport_timeout",
            "duration_ms": captured.value.attempt_ledger[0]["duration_ms"],
            "timeout_seconds": 360.0,
            "max_tokens": 10_000,
            "requested_model": "claude-4.6-opus",
            "actual_model": None,
            "usage": None,
            "error": "The read operation timed out",
        }
    ]


def test_deep_synthesis_retries_exact_empty_text_with_original_request_and_cost(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    original_messages = [
        {"role": "system", "content": "Return strict report JSON."},
        {"role": "user", "content": "Use the complete evidence context."},
    ]
    client = _SequenceGatewayClient(
        [
            _gateway_no_text_error(usage={"input_tokens": 19, "output_tokens": 0}),
            GatewayMessageResult(
                content='{"answer":"Supported fact [S1]","claims":["Supported fact [S1]"]}',
                model="claude-opus-4-8-20260701",
                usage={"input_tokens": 23, "output_tokens": 7},
            ),
        ],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        synthesis_timeout_seconds=600.0,
        max_retries=2,
        client=client,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)

    with provider._capture_attempt_usage(cost, "synthesis"):
        result = asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=original_messages,
                max_tokens=10_000,
            )
        )

    assert result.parsed["claims"] == ["Supported fact [S1]"]
    assert result.final_request_kind == "retry"
    assert [item["request_kind"] for item in result.attempt_ledger] == ["initial"]
    assert len(client.calls) == 2
    assert client.calls[0] == client.calls[1]
    assert client.calls[1]["messages"] == original_messages
    assert client.calls[1]["max_tokens"] == 10_000
    assert client.calls[1]["timeout_seconds"] == 600.0
    assert [(item.input_tokens, item.output_tokens) for item in cost.records] == [
        (19, 0),
        (23, 7),
    ]


def test_brief_generation_retries_exact_empty_text_with_original_request(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [
            _gateway_no_text_error(),
            GatewayMessageResult(
                content=(
                    '{"normalized_query":"How does gateway routing work?",'
                    '"scope":"Verify the provider path.",'
                    '"constraints":["Use runtime credentials."],'
                    '"assumptions":[]}'
                ),
                model="claude-opus-4-8-20260701",
                usage={"input_tokens": 23, "output_tokens": 7},
            ),
        ],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        timeout_seconds=240.0,
        synthesis_timeout_seconds=600.0,
        max_retries=2,
        client=client,
    )

    brief = asyncio.run(
        provider.create_brief(
            ResearchRequest(query="How does gateway routing work?"),
            CostTracker(provider=provider.name, model=provider.model),
        )
    )

    assert brief.scope == "Verify the provider path."
    assert len(client.calls) == 2
    assert client.calls[0] == client.calls[1]
    assert client.calls[1]["timeout_seconds"] == 240.0
    assert client.calls[1]["max_tokens"] == 1200


def test_exact_empty_text_respects_zero_retry_budget() -> None:
    client = _SequenceGatewayClient(
        [_gateway_no_text_error()],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=0,
        client=client,
    )

    with pytest.raises(RuntimeError, match="returned no text content") as captured:
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return report JSON."}],
                max_tokens=10_000,
            )
        )

    assert len(client.calls) == 1
    assert [item["request_kind"] for item in captured.value.attempt_ledger] == [
        "initial"
    ]


def test_one_retry_budget_does_not_add_repair_after_empty_retry(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [
            _gateway_no_text_error(),
            GatewayMessageResult(
                content='{"answer":"Uncited retry fact","claims":[]}',
                model="claude-opus-4-8",
                usage={"input_tokens": 23, "output_tokens": 4},
            ),
        ],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=1,
        client=client,
    )

    def require_claim(payload: dict) -> None:
        if not payload.get("claims"):
            raise ValueError("factual text lacks a cited claim")

    with pytest.raises(RuntimeError, match="factual text lacks a cited claim") as captured:
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return report JSON."}],
                max_tokens=10_000,
                validator=require_claim,
            )
        )

    assert len(client.calls) == 2
    assert [item["request_kind"] for item in captured.value.attempt_ledger] == [
        "initial",
        "retry",
    ]


def test_timeout_after_empty_text_retry_does_not_open_repair(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [_gateway_no_text_error(), TimeoutError("The read operation timed out")],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=2,
        client=client,
    )

    with pytest.raises(RuntimeError, match="read operation timed out") as captured:
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return report JSON."}],
                max_tokens=10_000,
            )
        )

    assert len(client.calls) == 2
    assert [item["request_kind"] for item in captured.value.attempt_ledger] == [
        "initial",
        "retry",
    ]
    assert [item["failure_class"] for item in captured.value.attempt_ledger] == [
        "no_text_content",
        "transport_timeout",
    ]


def test_no_text_during_repair_does_not_open_empty_text_retry(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [
            GatewayMessageResult(
                content='{"answer":"Uncited initial fact","claims":[]}',
                model="claude-opus-4-8",
                usage={"input_tokens": 21, "output_tokens": 4},
            ),
            _gateway_no_text_error(),
        ],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=2,
        client=client,
    )

    def require_claim(payload: dict) -> None:
        if not payload.get("claims"):
            raise ValueError("factual text lacks a cited claim")

    with pytest.raises(RuntimeError, match="returned no text content") as captured:
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return report JSON."}],
                max_tokens=10_000,
                validator=require_claim,
            )
        )

    assert len(client.calls) == 2
    assert [item["request_kind"] for item in captured.value.attempt_ledger] == [
        "initial",
        "repair",
    ]


def test_successful_empty_text_retry_is_auditable_in_synthesis_context(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [
            _gateway_no_text_error(),
            GatewayMessageResult(
                content=(
                    '{"answer":"Alpha has documented property A [S1].",'
                    '"claims":["Alpha has documented property A [S1]."]}'
                ),
                model="claude-opus-4-8",
                usage={"input_tokens": 23, "output_tokens": 7},
            ),
        ],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=2,
        client=client,
    )
    brief = ResearchBrief(
        original_query="What property does Alpha have?",
        normalized_query="What property does Alpha have?",
        scope="Verify Alpha's property.",
        constraints=[],
        assumptions=[],
    )
    subquestion = SubQuestion(
        id="Q1",
        question="What property does Alpha have?",
        rationale="Verify the property.",
    )
    source = Source(
        id="S1",
        title="Alpha source",
        url="https://example.com/alpha",
        content="Alpha has documented property A.",
        provider="fixture",
        query="Alpha property",
    )
    finding = Finding(
        subquestion_id="Q1",
        subquestion=subquestion.question,
        summary="Alpha has documented property A.",
        source_ids=["S1"],
        sources=[source],
    )

    answer, _claims = asyncio.run(
        provider.synthesize(
            brief,
            [subquestion],
            [finding],
            [source],
            CostTracker(provider=provider.name, model=provider.model),
        )
    )

    assert answer == "Alpha has documented property A [S1]."
    assert len(client.calls) == 2
    assert client.calls[0] == client.calls[1]
    context = provider.last_synthesis_context
    assert context["final_request_kind"] == "retry"
    assert [item["request_kind"] for item in context["attempt_ledger"]] == [
        "initial"
    ]
    assert context["attempt_ledger"][0]["failure_class"] == "no_text_content"
    assert "Alpha has documented property A" not in str(context["attempt_ledger"])


def test_deep_synthesis_two_empty_text_attempts_fail_closed_with_safe_ledger(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [
            _gateway_no_text_error(
                usage={"input_tokens": 19, "output_tokens": 0},
                response_sha="a" * 64,
            ),
            _gateway_no_text_error(
                usage={"input_tokens": 21, "output_tokens": 0},
                response_sha="b" * 64,
            ),
        ],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        synthesis_timeout_seconds=600.0,
        max_retries=2,
        client=client,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)

    with pytest.raises(RuntimeError, match="returned no text content") as captured:
        with provider._capture_attempt_usage(cost, "synthesis"):
            asyncio.run(
                provider._chat_json_result(
                    stage="synthesis",
                    messages=[{"role": "user", "content": "Return report JSON."}],
                    max_tokens=10_000,
                )
            )

    assert len(client.calls) == 2
    assert [item["request_kind"] for item in captured.value.attempt_ledger] == [
        "initial",
        "retry",
    ]
    assert [(item.input_tokens, item.output_tokens) for item in cost.records] == [
        (19, 0),
        (21, 0),
    ]
    for ledger_item in captured.value.attempt_ledger:
        assert "response" not in ledger_item
        assert "raw_response" not in ledger_item
        assert "thinking" not in ledger_item
        assert "content" not in ledger_item
    assert "private reasoning" not in json.dumps(captured.value.attempt_ledger)


def test_empty_text_retry_can_be_followed_by_only_one_content_repair(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    invalid_draft = "draft body must never be replayed"
    original_messages = [
        {"role": "system", "content": "Return strict report JSON."},
        {"role": "user", "content": "Use all supplied evidence."},
    ]
    client = _SequenceGatewayClient(
        [
            _gateway_no_text_error(),
            GatewayMessageResult(
                content=json.dumps({"answer": invalid_draft, "claims": []}),
                model="claude-opus-4-8",
                usage={"input_tokens": 23, "output_tokens": 4},
            ),
            GatewayMessageResult(
                content=json.dumps({"answer": "still invalid", "claims": []}),
                model="claude-opus-4-8",
                usage={"input_tokens": 25, "output_tokens": 4},
            ),
        ],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        synthesis_timeout_seconds=600.0,
        max_retries=2,
        client=client,
    )

    def require_claim(payload: dict) -> None:
        if not payload.get("claims"):
            raise ValueError("factual text lacks a cited claim")

    with pytest.raises(RuntimeError, match="factual text lacks a cited claim") as captured:
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=original_messages,
                max_tokens=10_000,
                validator=require_claim,
            )
        )

    assert len(client.calls) == 3
    assert [item["request_kind"] for item in captured.value.attempt_ledger] == [
        "initial",
        "retry",
        "repair",
    ]
    assert client.calls[0] == client.calls[1]
    assert client.calls[2]["max_tokens"] == 10_000
    assert client.calls[2]["timeout_seconds"] == 600.0
    assert len(client.calls[2]["messages"]) == len(original_messages)
    assert invalid_draft not in str(client.calls[2]["messages"])
    assert "previous complete response was received" in client.calls[2]["messages"][-1][
        "content"
    ].lower()


def test_synthesis_failure_context_preserves_three_bounded_attempts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [
            _gateway_no_text_error(),
            GatewayMessageResult(
                content='{"answer":"Uncited retry fact","claims":[]}',
                model="claude-opus-4-8",
                usage={"input_tokens": 23, "output_tokens": 4},
            ),
            GatewayMessageResult(
                content='{"answer":"Uncited repair fact","claims":[]}',
                model="claude-opus-4-8",
                usage={"input_tokens": 25, "output_tokens": 4},
            ),
        ],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=2,
        client=client,
    )
    brief = ResearchBrief(
        original_query="Compare Alpha evidence.",
        normalized_query="Compare Alpha evidence.",
        scope="Compare verified Alpha evidence.",
        constraints=[],
        assumptions=[],
        report_depth="deep",
        expected_format="markdown",
    )
    subquestion = SubQuestion(
        id="Q1",
        question="What verified Alpha evidence exists?",
        rationale="Collect evidence.",
    )
    source = Source(
        id="S1",
        title="Alpha source",
        url="https://example.com/alpha",
        content="Alpha has documented property A.",
        provider="fixture",
        query="Alpha evidence",
    )
    finding = Finding(
        subquestion_id="Q1",
        subquestion=subquestion.question,
        summary="Alpha has documented property A.",
        source_ids=["S1"],
        sources=[source],
    )

    asyncio.run(
        provider.synthesize(
            brief,
            [subquestion],
            [finding],
            [source],
            CostTracker(provider=provider.name, model=provider.model),
        )
    )

    assert [
        item["request_kind"]
        for item in provider.last_synthesis_context["attempt_ledger"]
    ] == ["initial", "retry", "repair"]
    assert "Uncited retry fact" not in str(
        provider.last_synthesis_context["attempt_ledger"]
    )
    assert "Uncited repair fact" not in str(
        provider.last_synthesis_context["attempt_ledger"]
    )


@pytest.mark.parametrize(
    "no_text",
    [
        _gateway_no_text_error(
            stop_reason="end_turn",
            content_block_types=("thinking",),
        ),
        _gateway_no_text_error(stop_reason="max_tokens"),
        _gateway_no_text_error(content_block_types=()),
        _gateway_no_text_error(content_block_types=("text", "text")),
        _gateway_no_text_error(content_block_types=("text", "thinking")),
        _gateway_no_text_error(usage={"input_tokens": 19}),
        _gateway_no_text_error(usage={"input_tokens": 0, "output_tokens": 0}),
        _gateway_no_text_error(usage={"input_tokens": 19, "output_tokens": 1}),
    ],
    ids=[
        "thinking-only",
        "max-tokens",
        "empty-content-array",
        "duplicate-empty-text-blocks",
        "mixed-text-thinking-blocks",
        "missing-output-usage",
        "missing-normalized-usage",
        "nonzero-output",
    ],
)
def test_non_exact_no_text_shapes_are_not_retried(no_text) -> None:
    client = _SequenceGatewayClient(
        [no_text],
        require_response_model_match=True,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=2,
        client=client,
    )

    with pytest.raises(RuntimeError, match="returned no text content"):
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return report JSON."}],
                max_tokens=10_000,
            )
        )

    assert len(client.calls) == 1


def test_exact_empty_text_is_not_retried_without_strict_model_check() -> None:
    client = _SequenceGatewayClient(
        [_gateway_no_text_error()],
        require_response_model_match=False,
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=2,
        client=client,
    )

    with pytest.raises(RuntimeError, match="returned no text content"):
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return report JSON."}],
                max_tokens=10_000,
            )
        )

    assert len(client.calls) == 1


def test_deep_synthesis_allows_one_complete_targeted_repair(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = _SequenceGatewayClient(
        [
            GatewayMessageResult(
                content='{"answer":"Uncited fact","claims":[]}',
                model="claude-4.6-opus",
                usage={"input_tokens": 20, "output_tokens": 5},
            ),
            GatewayMessageResult(
                content=(
                    '{"answer":"# Findings\\n\\nSupported fact [S1]",'
                    '"claims":["Supported fact [S1]"]}'
                ),
                model="claude-4.6-opus",
                usage={"input_tokens": 22, "output_tokens": 8},
            ),
        ]
    )
    provider = LLMGatewayLLMProvider(
        model="claude-4.6-opus",
        base_url="https://gateway.local",
        synthesis_timeout_seconds=360.0,
        max_retries=2,
        client=client,
    )

    def require_claim(payload: dict) -> None:
        if not payload.get("claims"):
            raise ValueError("factual text lacks a cited claim")

    result = asyncio.run(
        provider._chat_json_result(
            stage="synthesis",
            messages=[
                {"role": "system", "content": "Return strict report JSON."},
                {
                    "role": "user",
                    "content": "Use the complete evidence context and preserve all tables.",
                },
            ],
            max_tokens=10_000,
            validator=require_claim,
        )
    )

    assert result.parsed["claims"] == ["Supported fact [S1]"]
    assert len(client.calls) == 2
    assert all(call["timeout_seconds"] == 360.0 for call in client.calls)
    repair_messages = client.calls[1]["messages"]
    assert len(repair_messages) == 2
    assert "Uncited fact" not in str(repair_messages)
    assert "Preserve the requested multi-section report" in repair_messages[-1]["content"]
    assert "do not collapse it into a short summary" in repair_messages[-1]["content"]


def test_synthesis_model_mismatch_is_not_retried_and_preserves_actual_model() -> None:
    mismatch = LLMGatewayModelMismatchError(
        requested_model="claude-4.6-opus",
        actual_model="glm-5.2",
        usage={"input_tokens": 17, "output_tokens": 2},
    )
    client = _SequenceGatewayClient([mismatch])
    provider = LLMGatewayLLMProvider(
        model="claude-4.6-opus",
        base_url="https://gateway.local",
        synthesis_timeout_seconds=360.0,
        max_retries=2,
        client=client,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)

    with pytest.raises(LLMGatewayModelMismatchError) as captured:
        with provider._capture_attempt_usage(cost, "synthesis"):
            asyncio.run(
                provider._chat_json_result(
                    stage="synthesis",
                    messages=[{"role": "user", "content": "Return report JSON."}],
                    max_tokens=10_000,
                )
            )

    assert len(client.calls) == 1
    assert [(item.input_tokens, item.output_tokens) for item in cost.records] == [
        (17, 2)
    ]
    assert captured.value.requested_model == "claude-4.6-opus"
    assert captured.value.actual_model == "glm-5.2"
    assert captured.value.attempt_ledger[0]["failure_class"] == "model_mismatch"
    assert captured.value.attempt_ledger[0]["actual_model"] == "glm-5.2"
    assert captured.value.attempt_ledger[0]["usage"] == {
        "input_tokens": 17,
        "output_tokens": 2,
    }


def test_synthesis_no_text_attempt_retains_only_safe_gateway_metadata() -> None:
    no_text = LLMGatewayNoTextContentError(
        requested_model="claude-opus-4-8",
        actual_model="claude-opus-4-8-20260701",
        stop_reason="max_tokens",
        content_block_types=("thinking",),
        usage={"input_tokens": 15881, "output_tokens": 10000},
        response_bytes=654321,
        raw_response_sha256="a" * 64,
    )
    client = _SequenceGatewayClient([no_text])
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        synthesis_timeout_seconds=360.0,
        max_retries=2,
        client=client,
    )
    cost = CostTracker(provider=provider.name, model=provider.model)

    with pytest.raises(RuntimeError, match="returned no text content") as captured:
        with provider._capture_attempt_usage(cost, "synthesis"):
            asyncio.run(
                provider._chat_json_result(
                    stage="synthesis",
                    messages=[{"role": "user", "content": "Return report JSON."}],
                    max_tokens=10_000,
                )
            )

    assert len(client.calls) == 1
    assert cost.summary().total_tokens == 25_881
    assert captured.value.failure_class == "no_text_content"
    assert captured.value.actual_model == "claude-opus-4-8-20260701"
    ledger = captured.value.attempt_ledger[0]
    assert ledger["failure_class"] == "no_text_content"
    assert ledger["requested_model"] == "claude-opus-4-8"
    assert ledger["actual_model"] == "claude-opus-4-8-20260701"
    assert ledger["usage"] == {"input_tokens": 15881, "output_tokens": 10000}
    assert ledger["stop_reason"] == "max_tokens"
    assert ledger["content_block_types"] == ["thinking"]
    assert ledger["response_bytes"] == 654321
    assert ledger["raw_response_sha256"] == "a" * 64
    assert "thinking" not in ledger["error"]


def test_deterministic_gateway_4xx_is_not_retried() -> None:
    client = _SequenceGatewayClient([RuntimeError("LLM Gateway HTTP 403")])
    provider = LLMGatewayLLMProvider(
        model="claude-4.6-opus",
        base_url="https://gateway.local",
        max_retries=2,
        client=client,
    )

    with pytest.raises(RuntimeError, match="HTTP 403") as captured:
        asyncio.run(
            provider._chat_json_result(
                stage="synthesis",
                messages=[{"role": "user", "content": "Return report JSON."}],
                max_tokens=10_000,
            )
        )

    assert len(client.calls) == 1
    assert captured.value.failure_class == "http_4xx"


def test_orchestrator_attaches_failed_gateway_usage_to_exception(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    provider = LLMGatewayLLMProvider(
        model="claude-4.6-opus",
        base_url="https://gateway.local",
        max_retries=1,
        client=_SequenceGatewayClient(
            [
                GatewayMessageResult(
                    content="not JSON one",
                    model="claude-4.6-opus",
                    usage={"input_tokens": 4, "output_tokens": 2},
                ),
                GatewayMessageResult(
                    content="not JSON two",
                    model="claude-4.6-opus",
                    usage={"input_tokens": 6, "output_tokens": 3},
                ),
            ]
        ),
    )
    orchestrator = DeepResearchOrchestrator(
        settings=Settings(local_retrieval_mode="none"),
        llm_provider=provider,
    )

    with pytest.raises(RuntimeError) as error:
        asyncio.run(orchestrator.run(ResearchRequest(query="How does routing work?")))

    failed_cost = error.value.deepresearch_cost
    assert failed_cost.total_tokens == 15
    assert error.value.deepresearch_run_id
    assert error.value.deepresearch_trace_events[-1].status == "error"


def test_gateway_provider_normalizes_structured_claim_objects() -> None:
    client = _SequenceGatewayClient(
        [
            GatewayMessageResult(
                content=json.dumps(
                    {
                        "answer": "阶段路由可以按任务选模型 [S1]。",
                        "claims": [
                            {
                                "statement": "阶段路由可以按任务选模型",
                                "citation_ids": ["S1"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                model="claude-opus-4-8",
                usage={"input_tokens": 10, "output_tokens": 5},
            )
        ]
    )
    provider = LLMGatewayLLMProvider(
        model="claude-opus-4-8",
        base_url="https://gateway.local",
        max_retries=0,
        client=client,
    )
    brief = ResearchBrief(
        original_query="阶段路由有什么价值？",
        normalized_query="阶段路由有什么价值？",
        scope="解释阶段路由。",
        constraints=[],
        assumptions=[],
    )
    subquestion = SubQuestion(id="Q1", question="阶段路由有什么价值？", rationale="验证。")
    source = Source(
        id="S1",
        title="Source",
        url="https://example.com",
        content="阶段路由可以按任务选模型。",
        provider="mock",
        query="阶段路由",
    )
    finding = Finding(
        subquestion_id="Q1",
        subquestion=subquestion.question,
        summary="阶段路由可以按任务选模型。",
        source_ids=["S1"],
        sources=[source],
    )

    answer, claims = asyncio.run(
        provider.synthesize(
            brief,
            [subquestion],
            [finding],
            [source],
            CostTracker(provider=provider.name, model=provider.model),
        )
    )

    assert answer == "阶段路由可以按任务选模型 [S1]"
    assert claims == ["阶段路由可以按任务选模型 [S1]"]


def test_planner_preserves_distinctive_named_entities_in_each_subquestion() -> None:
    original = (
        "In June 1637, Thomas Ballard of Wandsworth accused Richard Kestian of "
        "calling him a liar at which man's house in Putney?"
    )
    payload = {
        "subquestions": [
            {
                "id": "Q1",
                "question": "What do records say about Thomas Ballard and Richard Kestian?",
                "search_query": "Thomas Ballard Richard Kestian Putney 1637",
                "rationale": "Direct record search.",
            },
            {
                "id": "Q2",
                # No original entities at all — completely off-topic
                "question": "What are general local governance practices?",
                "search_query": "local governance practices england",
                "rationale": "Too generic — no original entity.",
            },
        ]
    }

    with pytest.raises(ValueError, match="dropped too many distinctive"):
        _subquestions_from_payload(
            payload,
            max_researchers=2,
            original_query=original,
        )

    # With at least one entity (Putney or 1637) it should pass
    payload["subquestions"][1]["search_query"] = "Putney householders 1637"
    result = _subquestions_from_payload(
        payload,
        max_researchers=2,
        original_query=original,
    )
    assert len(result) == 2


def test_gateway_settings_and_orchestrator_routing(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("LLM_GATEWAY_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("LLM_SYNTHESIS_TIMEOUT_SECONDS", "390")
    monkeypatch.setenv("LLM_GATEWAY_THINKING_BUDGET_TOKENS", "2048")
    settings = load_settings()
    orchestrator = DeepResearchOrchestrator(
        settings=Settings(
            llm_provider="llm-gateway",
            llm_gateway_model=settings.llm_gateway_model,
            llm_gateway_base_url=settings.llm_gateway_base_url,
            llm_gateway_timeout_seconds=settings.llm_gateway_timeout_seconds,
            llm_synthesis_timeout_seconds=(
                settings.llm_synthesis_timeout_seconds
            ),
            llm_gateway_thinking_budget_tokens=(
                settings.llm_gateway_thinking_budget_tokens
            ),
        )
    )

    provider = orchestrator._build_llm_provider(
        ResearchRequest(query="How is the gateway selected?")
    )

    assert isinstance(provider, LLMGatewayLLMProvider)
    assert provider.name == "llm-gateway"
    assert provider.model == "claude-opus-4-8"
    assert provider.base_url == "https://gateway.example"
    assert provider.timeout_seconds == 45.0
    assert provider.synthesis_timeout_seconds == 390.0
    assert provider.client.thinking_budget_tokens == 2048


@pytest.mark.parametrize(
    ("raw_action", "evidence", "expected_action"),
    [
        (
            "skip",
            [
                EvidenceItem(
                    source_id="S1",
                    source_title="Source",
                    source_url="https://example.com",
                    quote="Verified evidence.",
                    query="gateway",
                )
            ],
            "stop",
        ),
        ("answer", [], "need_follow_up"),
        (
            "synthesize",
            [
                EvidenceItem(
                    source_id="S1",
                    source_title="Source",
                    source_url="https://example.com",
                    quote="Verified evidence.",
                    query="gateway",
                )
            ],
            "stop",
        ),
        (
            "sufficient",
            [
                EvidenceItem(
                    source_id="S1",
                    source_title="Source",
                    source_url="https://example.com",
                    quote="Verified evidence.",
                    query="gateway",
                )
            ],
            "stop",
        ),
        (
            "search",
            [
                EvidenceItem(
                    source_id="S1",
                    source_title="Source",
                    source_url="https://example.com",
                    quote="Verified evidence.",
                    query="gateway",
                )
            ],
            "need_follow_up",
        ),
    ],
)
def test_gateway_provider_normalizes_observed_decision_aliases(
    raw_action,
    evidence,
    expected_action,
) -> None:
    client = _SequenceGatewayClient(
        [
            GatewayMessageResult(
                content=json.dumps(
                    {"action": raw_action, "reason": "No more search is needed."}
                ),
                model="claude-4.6-opus",
                usage={"input_tokens": 2, "output_tokens": 1},
            )
        ]
    )
    provider = LLMGatewayLLMProvider(
        model="claude-4.6-opus",
        base_url="https://gateway.local",
        max_retries=0,
        client=client,
    )

    decision = asyncio.run(
        provider.decide_research(
            SubQuestion(id="Q1", question="What evidence exists?", rationale="Verify."),
            evidence=evidence,
            min_evidence_items=1,
            round_index=1,
            cost=CostTracker(provider=provider.name, model=provider.model),
        )
    )

    assert decision.action == expected_action
    if not evidence:
        assert decision.follow_up_query


def test_unknown_llm_provider_fails_closed() -> None:
    orchestrator = DeepResearchOrchestrator(settings=Settings())

    with pytest.raises(ValueError, match="unknown LLM provider"):
        orchestrator._build_llm_provider(
            ResearchRequest(query="Do not silently use a mock.", llm_provider="typo")
        )


class _SequenceGatewayClient:
    def __init__(
        self,
        responses: list[GatewayMessageResult | Exception],
        *,
        require_response_model_match: bool = False,
    ) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.require_response_model_match = require_response_model_match

    def create_message(self, *, model, messages, max_tokens, timeout_seconds=None):
        self.calls.append(
            {
                "model": model,
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# ---------------------------------------------------------------------------
# Tests for multi-model compatibility fixes
# ---------------------------------------------------------------------------


def test_string_array_tolerates_semicolon_separated_string():
    """Opus sometimes outputs constraints as a single string instead of array."""
    from deepresearch_agent.llm import _string_array

    # Normal array case
    result = _string_array({"items": ["a", "b"]}, "items")
    assert result == ["a", "b"]

    # Single string with semicolons — should split
    result = _string_array(
        {"items": "Use official sources; Limit to 3 claims; No speculation"}, "items"
    )
    assert result == ["Use official sources", "Limit to 3 claims", "No speculation"]

    # Single string without semicolons — kept as one-element list
    result = _string_array({"items": "Single constraint without semicolons"}, "items")
    assert result == ["Single constraint without semicolons"]


def test_string_array_rejects_non_string_non_list():
    from deepresearch_agent.llm import _string_array

    with pytest.raises(ValueError, match="must be an array of strings"):
        _string_array({"items": 42}, "items")


def test_brief_from_payload_tolerates_empty_scope():
    """Opus 4.8 occasionally outputs scope as empty string."""
    from deepresearch_agent.llm import _brief_from_payload

    payload = {
        "normalized_query": "What is X?",
        "scope": "",
        "constraints": ["c1"],
        "assumptions": ["a1"],
    }
    brief = _brief_from_payload(payload, "What is X?")
    assert brief.scope  # Should get default scope, not empty
    assert "evidence" in brief.scope.lower()

    # Missing scope entirely
    payload2 = {
        "normalized_query": "What is X?",
        "constraints": ["c1"],
        "assumptions": ["a1"],
    }
    brief2 = _brief_from_payload(payload2, "What is X?")
    assert brief2.scope


def test_planner_entity_anchors_includes_acronyms_and_numbers():
    """Entity detection should catch TPLF, 1922, etc."""
    from deepresearch_agent.llm import _planner_entity_anchors

    # Original: only matched [A-Z][a-z]{2,}
    anchors = _planner_entity_anchors(
        "What month was the TPLF removed from the terrorist list in 1922?"
    )
    assert "tplf" in anchors
    assert "1922" in anchors

    # Mixed case proper nouns still work
    anchors2 = _planner_entity_anchors("Thomas Ballard accused Richard Kestian in 1637")
    assert "thomas" in anchors2
    assert "ballard" in anchors2
    assert "1637" in anchors2


def test_planner_entity_check_requires_at_least_one_overlap():
    """Relaxed check: subquestions need ≥1 entity overlap, not ≥2."""
    payload = {
        "subquestions": [
            {
                "id": "Q1",
                "question": "What is Kiyoshi Oka's background?",
                "search_query": "Kiyoshi Oka education 1922",
                "rationale": "Direct search.",
            },
            {
                "id": "Q2",
                # Has "1922" from the original — should pass
                "question": "What universities existed in 1922?",
                "search_query": "Imperial University 1922 departments",
                "rationale": "Related context.",
            },
        ]
    }
    # Should not raise — Q2 has "1922" which overlaps
    result = _subquestions_from_payload(
        payload,
        max_researchers=2,
        original_query="Kiyoshi Oka entered the Imperial University of Kyoto in 1922 to study what?",
    )
    assert len(result) == 2
