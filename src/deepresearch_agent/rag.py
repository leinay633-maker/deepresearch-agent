from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.embeddings import EmbeddingProvider, build_embedding_provider
from deepresearch_agent.rerankers import RerankProvider, build_rerank_provider
from deepresearch_agent.schemas import Source


@dataclass(frozen=True)
class LocalChunk:
    id: str
    document_id: str
    title: str
    url: str
    content: str
    chunk_index: int


@dataclass(frozen=True)
class RankedChunk:
    chunk: LocalChunk
    score: float
    rank: int
    method: str


class LocalRagRetriever:
    name = "local_rag"

    def __init__(
        self,
        corpus_path: Path | None = None,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        rerank_provider: RerankProvider | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self.settings = settings or load_settings()
        self.corpus_path = corpus_path or root / "data" / "local_corpus.jsonl"
        self.documents = self._load()
        self.chunks = self._chunk_documents(self.documents)
        self.embedding_provider = embedding_provider
        self.rerank_provider = rerank_provider
        self._vector_index: ChromaVectorIndex | None = None

    async def retrieve(self, query: str, max_results: int = 2) -> list[Source]:
        mode = self.settings.local_retrieval_mode.strip().lower()
        keyword_top_k = max(max_results, self.settings.local_keyword_top_k)
        keyword_results = self._keyword_retrieve(query, top_k=keyword_top_k)
        if mode == "keyword":
            return self._sources_from_ranked(query, keyword_results[:max_results], mode="keyword")
        if mode != "hybrid":
            raise ValueError(f"unknown local retrieval mode: {mode}")

        vector_top_k = max(max_results, self.settings.local_vector_top_k)
        vector_results = await self._vector_retrieve(query, top_k=vector_top_k)
        fused = self._rrf_fuse(keyword_results, vector_results)
        candidate_count = (
            max(max_results, self.settings.local_rerank_candidate_k)
            if self.settings.rerank_enabled
            else max_results
        )
        candidates = self._sources_from_fused(query, fused[:candidate_count])
        if self.settings.rerank_enabled:
            return await self._rerank_sources(query, candidates, max_results=max_results)
        return candidates[:max_results]

    def _keyword_retrieve(self, query: str, top_k: int) -> list[RankedChunk]:
        query_tokens = _tokens(query)
        ranked: list[tuple[float, LocalChunk]] = []
        for chunk in self.chunks:
            text = f"{chunk.title} {chunk.content}"
            overlap = len(query_tokens & _tokens(text))
            score = overlap + 0.05
            ranked.append((score, chunk))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RankedChunk(chunk=chunk, score=score, rank=index + 1, method="keyword")
            for index, (score, chunk) in enumerate(ranked[:top_k])
        ]

    async def _vector_retrieve(self, query: str, top_k: int) -> list[RankedChunk]:
        if not self.chunks:
            return []
        index = await self._ensure_vector_index()
        return await index.search(query, top_k=top_k)

    async def _ensure_vector_index(self) -> "ChromaVectorIndex":
        if self._vector_index is None:
            provider = self.embedding_provider or build_embedding_provider(self.settings)
            self._vector_index = ChromaVectorIndex(
                chunks=self.chunks,
                embedding_provider=provider,
            )
            await self._vector_index.build()
        return self._vector_index

    def _rrf_fuse(
        self, keyword_results: list[RankedChunk], vector_results: list[RankedChunk]
    ) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        rrf_k = max(1, self.settings.local_hybrid_rrf_k)
        for result, weight in [
            *[(item, self.settings.local_keyword_weight) for item in keyword_results],
            *[(item, self.settings.local_vector_weight) for item in vector_results],
        ]:
            state = by_id.setdefault(
                result.chunk.id,
                {
                    "chunk": result.chunk,
                    "fusion_score": 0.0,
                    "keyword_score": None,
                    "keyword_rank": None,
                    "vector_score": None,
                    "vector_rank": None,
                },
            )
            state["fusion_score"] += weight / (rrf_k + result.rank)
            state[f"{result.method}_score"] = result.score
            state[f"{result.method}_rank"] = result.rank
        fused = list(by_id.values())
        fused.sort(key=lambda item: item["fusion_score"], reverse=True)
        return fused

    def _sources_from_ranked(
        self, query: str, ranked: list[RankedChunk], mode: str
    ) -> list[Source]:
        return [
            self._source_from_chunk(
                query,
                item.chunk,
                score=item.score,
                metadata={
                    "local_doc_id": item.chunk.document_id,
                    "chunk_index": item.chunk.chunk_index,
                    "retrieval_mode": mode,
                    f"{item.method}_rank": item.rank,
                    f"{item.method}_score": item.score,
                },
            )
            for item in ranked
        ]

    def _sources_from_fused(self, query: str, fused: list[dict[str, Any]]) -> list[Source]:
        sources = []
        for item in fused:
            chunk: LocalChunk = item["chunk"]
            metadata = {
                "local_doc_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "retrieval_mode": "hybrid",
                "fusion": "rrf",
                "keyword_rank": item["keyword_rank"],
                "keyword_score": item["keyword_score"],
                "vector_rank": item["vector_rank"],
                "vector_score": item["vector_score"],
                "rrf_k": self.settings.local_hybrid_rrf_k,
                "keyword_weight": self.settings.local_keyword_weight,
                "vector_weight": self.settings.local_vector_weight,
            }
            sources.append(
                self._source_from_chunk(
                    query,
                    chunk,
                    score=float(item["fusion_score"]),
                    metadata=metadata,
                )
            )
        return sources

    async def _rerank_sources(
        self, query: str, sources: list[Source], max_results: int
    ) -> list[Source]:
        provider = self.rerank_provider or build_rerank_provider(self.settings)
        reranked = await provider.rerank(query, sources)
        output = []
        for new_rank, item in enumerate(reranked[:max_results], start=1):
            source = sources[item.index]
            metadata = {
                **source.metadata,
                "rerank_enabled": True,
                "rerank_provider": provider.name,
                "rerank_model": provider.model,
                "rerank_rank": new_rank,
                "rerank_score": item.score,
                "pre_rerank_score": source.score,
                "pre_rerank_rank": item.index + 1,
            }
            output.append(source.model_copy(update={"score": item.score, "metadata": metadata}))
        return output

    def _source_from_chunk(
        self,
        query: str,
        chunk: LocalChunk,
        score: float,
        metadata: dict[str, Any],
    ) -> Source:
        return Source(
            title=chunk.title,
            url=chunk.url,
            content=chunk.content,
            provider=self.name,
            query=query,
            score=score,
            metadata=metadata,
        )

    def _load(self) -> list[dict[str, str]]:
        if not self.corpus_path.exists():
            return []
        documents = []
        with self.corpus_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    documents.append(json.loads(line))
        return documents

    def _chunk_documents(self, documents: list[dict[str, str]]) -> list[LocalChunk]:
        chunks: list[LocalChunk] = []
        chunk_size = max(100, self.settings.local_chunk_size_chars)
        overlap = max(0, min(self.settings.local_chunk_overlap_chars, chunk_size - 1))
        for document in documents:
            content = document["content"]
            document_id = document.get("id", document["url"])
            if not content:
                continue
            start = 0
            chunk_index = 0
            while start < len(content):
                end = min(len(content), start + chunk_size)
                chunk_text = content[start:end]
                chunks.append(
                    LocalChunk(
                        id=f"{document_id}:{chunk_index}",
                        document_id=document_id,
                        title=document["title"],
                        url=document["url"],
                        content=chunk_text,
                        chunk_index=chunk_index,
                    )
                )
                if end >= len(content):
                    break
                start = max(end - overlap, start + 1)
                chunk_index += 1
        return chunks


class ChromaVectorIndex:
    def __init__(
        self,
        chunks: list[LocalChunk],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self.collection: Any | None = None
        self.chunk_by_id = {chunk.id: chunk for chunk in chunks}

    async def build(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise RuntimeError("hybrid local retrieval requires chromadb") from exc

        client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
        collection_name = f"local_rag_{uuid.uuid4().hex[:12]}"
        self.collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        embeddings = await self.embedding_provider.embed_texts(
            [f"{chunk.title}\n{chunk.content}" for chunk in self.chunks]
        )
        self.collection.add(
            ids=[chunk.id for chunk in self.chunks],
            documents=[chunk.content for chunk in self.chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "url": chunk.url,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in self.chunks
            ],
            embeddings=embeddings,
        )

    async def search(self, query: str, top_k: int) -> list[RankedChunk]:
        if self.collection is None:
            raise RuntimeError("vector index has not been built")
        query_vector = await self.embedding_provider.embed_text(query)
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, len(self.chunks)),
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ranked = []
        for index, (chunk_id, distance) in enumerate(zip(ids, distances, strict=False)):
            chunk = self.chunk_by_id[chunk_id]
            score = 1.0 / (1.0 + float(distance))
            ranked.append(
                RankedChunk(chunk=chunk, score=score, rank=index + 1, method="vector")
            )
        return ranked


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", text)}
