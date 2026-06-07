from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from deepresearch_agent.config import Settings
from deepresearch_agent.embeddings import EmbeddingProvider
from deepresearch_agent.rag import LocalRagRetriever
from deepresearch_agent.rerankers import (
    DashScopeRerankProvider,
    RerankProvider,
    RerankResult,
)
from deepresearch_agent.schemas import Source


class StaticEmbeddingProvider(EmbeddingProvider):
    name = "static"
    model = "static-test"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class StaticRerankProvider(RerankProvider):
    name = "static"
    model = "static-rerank"

    async def rerank(self, query: str, sources: list[Source]) -> list[RerankResult]:
        del query
        scores = [
            RerankResult(index=index, score=10.0 if "second" in source.content else 1.0)
            for index, source in enumerate(sources)
        ]
        return sorted(scores, key=lambda item: item.score, reverse=True)


def test_local_retriever_applies_optional_rerank(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus,
        [
            {
                "id": "first",
                "title": "First",
                "url": "file://first",
                "content": "first candidate",
            },
            {
                "id": "second",
                "title": "Second",
                "url": "file://second",
                "content": "second candidate",
            },
        ],
    )
    retriever = LocalRagRetriever(
        corpus_path=corpus,
        settings=Settings(
            local_retrieval_mode="hybrid",
            rerank_enabled=True,
            local_rerank_candidate_k=2,
        ),
        embedding_provider=StaticEmbeddingProvider(),
        rerank_provider=StaticRerankProvider(),
    )

    results = asyncio.run(retriever.retrieve("candidate", max_results=1))

    assert results[0].metadata["local_doc_id"] == "second"
    assert results[0].metadata["rerank_enabled"] is True
    assert results[0].metadata["rerank_provider"] == "static"
    assert results[0].score == 10.0


def test_dashscope_rerank_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    provider = DashScopeRerankProvider(endpoint="http://127.0.0.1:9")
    source = Source(
        title="Doc",
        url="file://doc",
        content="content",
        provider="local_rag",
        query="q",
    )

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        asyncio.run(provider.rerank("q", [source]))


def test_dashscope_rerank_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            assert self.headers["Authorization"] == "Bearer test-key"
            assert payload["model"] == "gte-rerank-v2"
            body = json.dumps(
                {
                    "output": {
                        "results": [
                            {"index": 1, "relevance_score": 0.9},
                            {"index": 0, "relevance_score": 0.2},
                        ]
                    }
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
    provider = DashScopeRerankProvider(
        endpoint=f"http://127.0.0.1:{server.server_port}",
        timeout=2,
    )
    sources = [
        Source(title="A", url="file://a", content="a", provider="local", query="q"),
        Source(title="B", url="file://b", content="b", provider="local", query="q"),
    ]
    try:
        results = asyncio.run(provider.rerank("q", sources))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert results == [RerankResult(index=1, score=0.9), RerankResult(index=0, score=0.2)]


def _write_corpus(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
