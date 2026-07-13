from __future__ import annotations

import asyncio
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

    with pytest.raises(RuntimeError, match="response model did not match"):
        LLMGatewayClient(
            base_url="https://gateway.local",
            require_response_model_match=True,
        ).create_message(
            model="claude-4.6-opus",
            messages=[{"role": "user", "content": "Return JSON."}],
            max_tokens=100,
        )


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
                "question": "Which householders were active in Putney?",
                "search_query": "Putney householders 1637",
                "rationale": "Too broad.",
            },
        ]
    }

    with pytest.raises(ValueError, match="dropped too many distinctive"):
        _subquestions_from_payload(
            payload,
            max_researchers=2,
            original_query=original,
        )

    payload["subquestions"][1]["search_query"] = (
        "Thomas Ballard Richard Kestian Putney house 1637"
    )
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
    monkeypatch.setenv("LLM_GATEWAY_THINKING_BUDGET_TOKENS", "2048")
    settings = load_settings()
    orchestrator = DeepResearchOrchestrator(
        settings=Settings(
            llm_provider="llm-gateway",
            llm_gateway_model=settings.llm_gateway_model,
            llm_gateway_base_url=settings.llm_gateway_base_url,
            llm_gateway_timeout_seconds=settings.llm_gateway_timeout_seconds,
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
    def __init__(self, responses: list[GatewayMessageResult]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create_message(self, *, model, messages, max_tokens):
        self.calls.append(
            {
                "model": model,
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
            }
        )
        return self.responses.pop(0)
