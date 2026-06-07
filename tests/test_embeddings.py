from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from deepresearch_agent.config import Settings
from deepresearch_agent.embeddings import (
    DashScopeEmbeddingProvider,
    LocalEmbeddingProvider,
    build_embedding_provider,
)


def test_build_embedding_provider_defaults_to_local() -> None:
    provider = build_embedding_provider(Settings())

    assert isinstance(provider, LocalEmbeddingProvider)
    assert provider.model == "BAAI/bge-small-zh-v1.5"


def test_dashscope_embedding_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    provider = DashScopeEmbeddingProvider(endpoint="http://127.0.0.1:9")

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        asyncio.run(provider.embed_text("agent retrieval"))


def test_dashscope_embedding_parses_compatible_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            assert self.headers["Authorization"] == "Bearer test-key"
            assert payload["model"] == "text-embedding-v4"
            body = json.dumps(
                {
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                        {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                    ]
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    provider = DashScopeEmbeddingProvider(
        endpoint=f"http://127.0.0.1:{server.server_port}",
        timeout=2,
    )
    try:
        vectors = asyncio.run(provider.embed_texts(["a", "b"]))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
