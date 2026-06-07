from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import urllib.error
import urllib.request

from deepresearch_agent.config import Settings
from deepresearch_agent.embeddings import EmbeddingProvider
from deepresearch_agent.rag import LocalRagRetriever


class StaticEmbeddingProvider(EmbeddingProvider):
    name = "static"
    model = "static-test"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        vectors = []
        for text in texts:
            if "neural meaning" in text or "latent idea" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


def test_keyword_retrieval_mode_keeps_keyword_ranking(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus,
        [
            {
                "id": "keyword",
                "title": "Keyword evidence",
                "url": "file://keyword",
                "content": "citation checker overlap support",
            },
            {
                "id": "other",
                "title": "Other note",
                "url": "file://other",
                "content": "latency budget retry fallback",
            },
        ],
    )
    retriever = LocalRagRetriever(
        corpus_path=corpus,
        settings=Settings(local_retrieval_mode="keyword"),
    )

    results = asyncio.run(retriever.retrieve("citation support", max_results=1))

    assert results[0].metadata["retrieval_mode"] == "keyword"
    assert results[0].metadata["local_doc_id"] == "keyword"


def test_hybrid_retrieval_fuses_keyword_and_vector_results(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus,
        [
            {
                "id": "keyword",
                "title": "Keyword note",
                "url": "file://keyword",
                "content": "literal overlap term",
            },
            {
                "id": "vector",
                "title": "Vector note",
                "url": "file://vector",
                "content": "neural meaning",
            },
        ],
    )
    retriever = LocalRagRetriever(
        corpus_path=corpus,
        settings=Settings(
            local_retrieval_mode="hybrid",
            local_keyword_top_k=2,
            local_vector_top_k=2,
            local_keyword_weight=1.0,
            local_vector_weight=4.0,
        ),
        embedding_provider=StaticEmbeddingProvider(),
    )

    results = asyncio.run(retriever.retrieve("latent idea", max_results=1))

    assert results[0].metadata["retrieval_mode"] == "hybrid"
    assert results[0].metadata["fusion"] == "rrf"
    assert results[0].metadata["local_doc_id"] == "vector"
    assert results[0].metadata["vector_rank"] == 1


def test_persistent_vector_index_reuses_existing_collection(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    index_path = tmp_path / "vector_index"
    _write_corpus(
        corpus,
        [
            {
                "id": "keyword",
                "title": "Keyword note",
                "url": "file://keyword",
                "content": "literal overlap term",
            },
            {
                "id": "vector",
                "title": "Vector note",
                "url": "file://vector",
                "content": "neural meaning",
            },
        ],
    )
    settings = Settings(
        local_retrieval_mode="hybrid",
        local_keyword_top_k=2,
        local_vector_top_k=2,
        local_vector_weight=4.0,
        local_vector_index_persist=True,
        local_vector_index_path=str(index_path),
    )

    first_provider = StaticEmbeddingProvider()
    first = LocalRagRetriever(
        corpus_path=corpus,
        settings=settings,
        embedding_provider=first_provider,
    )
    first_results = asyncio.run(first.retrieve("latent idea", max_results=1))

    second_provider = StaticEmbeddingProvider()
    second = LocalRagRetriever(
        corpus_path=corpus,
        settings=settings,
        embedding_provider=second_provider,
    )
    second_results = asyncio.run(second.retrieve("latent idea", max_results=1))

    assert first_results[0].metadata["local_doc_id"] == "vector"
    assert second_results[0].metadata["local_doc_id"] == "vector"
    assert first_provider.batch_sizes == [2, 1]
    assert second_provider.batch_sizes == [1]
    assert second._vector_index is not None
    assert second._vector_index.reused_existing is True


def test_qdrant_vector_index_builds_and_searches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus,
        [
            {
                "id": "keyword",
                "title": "Keyword note",
                "url": "file://keyword",
                "content": "literal overlap term",
            },
            {
                "id": "vector",
                "title": "Vector note",
                "url": "file://vector",
                "content": "neural meaning",
            },
        ],
    )
    calls: list[tuple[str, str, dict | None, dict[str, str]]] = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        headers = dict(request.header_items())
        calls.append((request.get_method(), request.full_url, body, headers))
        if request.get_method() == "GET":
            raise urllib.error.HTTPError(
                request.full_url,
                404,
                "missing",
                hdrs=None,
                fp=io.BytesIO(b"{}"),
            )
        if request.get_method() == "POST" and request.full_url.endswith("/points/search"):
            return _JsonResponse(
                {"result": [{"payload": {"chunk_id": "vector:0"}, "score": 0.97}]}
            )
        return _JsonResponse({"result": {"points_count": 0}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("TEST_QDRANT_API_KEY", "fake-qdrant-key")
    retriever = LocalRagRetriever(
        corpus_path=corpus,
        settings=Settings(
            local_retrieval_mode="hybrid",
            local_vector_index_provider="qdrant",
            qdrant_base_url="http://qdrant.local",
            qdrant_collection="unit_test_collection",
            qdrant_api_key_env="TEST_QDRANT_API_KEY",
            local_keyword_top_k=2,
            local_vector_top_k=2,
        ),
        embedding_provider=StaticEmbeddingProvider(),
    )

    results = asyncio.run(retriever.retrieve("latent idea", max_results=1))

    assert results[0].metadata["local_doc_id"] == "vector"
    assert results[0].metadata["vector_index_provider"] == "qdrant"
    methods = [call[0] for call in calls]
    assert methods == ["GET", "PUT", "PUT", "POST"]
    upsert_body = calls[2][2]
    assert upsert_body is not None
    assert upsert_body["points"][1]["payload"]["chunk_id"] == "vector:0"
    assert any(key.lower() == "api-key" for key in calls[0][3])


def test_qdrant_vector_index_reuses_existing_collection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus,
        [
            {
                "id": "keyword",
                "title": "Keyword note",
                "url": "file://keyword",
                "content": "literal overlap term",
            },
            {
                "id": "vector",
                "title": "Vector note",
                "url": "file://vector",
                "content": "neural meaning",
            },
        ],
    )
    calls: list[str] = []

    def fake_urlopen(request, timeout):
        calls.append(request.get_method())
        if request.get_method() == "GET":
            return _JsonResponse({"result": {"points_count": 2}})
        if request.get_method() == "POST" and request.full_url.endswith("/points/search"):
            return _JsonResponse(
                {"result": [{"payload": {"chunk_id": "vector:0"}, "score": 0.95}]}
            )
        raise AssertionError(f"unexpected Qdrant request: {request.get_method()}")

    provider = StaticEmbeddingProvider()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    retriever = LocalRagRetriever(
        corpus_path=corpus,
        settings=Settings(
            local_retrieval_mode="hybrid",
            local_vector_index_provider="qdrant",
            qdrant_base_url="http://qdrant.local",
            qdrant_collection="unit_test_collection",
            local_keyword_top_k=2,
            local_vector_top_k=2,
        ),
        embedding_provider=provider,
    )

    results = asyncio.run(retriever.retrieve("latent idea", max_results=1))

    assert results[0].metadata["local_doc_id"] == "vector"
    assert provider.batch_sizes == [1]
    assert calls == ["GET", "POST"]
    assert retriever._vector_index is not None
    assert retriever._vector_index.reused_existing is True


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _write_corpus(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
