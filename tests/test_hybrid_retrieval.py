from __future__ import annotations

import asyncio
import json
from pathlib import Path

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


def _write_corpus(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
