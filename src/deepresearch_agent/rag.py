from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from deepresearch_agent.config import Settings, load_settings
from deepresearch_agent.embeddings import EmbeddingProvider, build_embedding_provider
from deepresearch_agent.rerankers import RerankProvider, build_rerank_provider
from deepresearch_agent.schemas import Source
from deepresearch_agent.text_utils import tokenize

logger = logging.getLogger(__name__)


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


class VectorIndex(Protocol):
    reused_existing: bool

    async def build(self) -> None:
        raise NotImplementedError

    async def search(self, query: str, top_k: int) -> list[RankedChunk]:
        raise NotImplementedError


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
        self._vector_index: VectorIndex | None = None
        self.last_retrieval_degraded = False
        self.last_degrade_reason: str | None = None

    async def retrieve(self, query: str, max_results: int = 2) -> list[Source]:
        self.last_retrieval_degraded = False
        self.last_degrade_reason = None
        mode = self.settings.local_retrieval_mode.strip().lower()
        keyword_top_k = max(max_results, self.settings.local_keyword_top_k)
        keyword_results = self._keyword_retrieve(query, top_k=keyword_top_k)
        if mode == "keyword":
            return self._sources_from_ranked(query, keyword_results[:max_results], mode="keyword")
        if mode != "hybrid":
            raise ValueError(f"unknown local retrieval mode: {mode}")

        vector_top_k = max(max_results, self.settings.local_vector_top_k)
        try:
            vector_results = await self._vector_retrieve(query, top_k=vector_top_k)
        except Exception as exc:
            return self._degraded_keyword_sources(
                query,
                keyword_results,
                max_results=max_results,
                reason=f"{type(exc).__name__}: {exc}",
            )
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

    async def _ensure_vector_index(self) -> VectorIndex:
        if self._vector_index is None:
            provider = self.embedding_provider or build_embedding_provider(self.settings)
            index_provider = self.settings.local_vector_index_provider.strip().lower()
            if index_provider == "chroma":
                self._vector_index = ChromaVectorIndex(
                    chunks=self.chunks,
                    embedding_provider=provider,
                    persist_path=(
                        Path(self.settings.local_vector_index_path)
                        if self.settings.local_vector_index_persist
                        else None
                    ),
                )
            elif index_provider == "qdrant":
                self._vector_index = QdrantVectorIndex(
                    chunks=self.chunks,
                    embedding_provider=provider,
                    base_url=self.settings.qdrant_base_url,
                    collection_prefix=self.settings.qdrant_collection,
                    timeout=self.settings.request_timeout_seconds,
                    api_key=os.environ.get(self.settings.qdrant_api_key_env),
                )
            else:
                raise ValueError(f"unknown local vector index provider: {index_provider}")
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
                "vector_index_provider": self.settings.local_vector_index_provider,
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

    def _degraded_keyword_sources(
        self,
        query: str,
        ranked: list[RankedChunk],
        *,
        max_results: int,
        reason: str,
    ) -> list[Source]:
        self.last_retrieval_degraded = True
        self.last_degrade_reason = reason
        logger.warning(
            "local hybrid retrieval degraded to keyword-only: %s",
            reason,
        )
        sources = self._sources_from_ranked(
            query,
            ranked[:max_results],
            mode="keyword",
        )
        output = []
        for source in sources:
            metadata = {
                **source.metadata,
                "retrieval_degraded": True,
                "degrade_reason": reason,
            }
            output.append(source.model_copy(update={"metadata": metadata}))
        return output

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
        persist_path: Path | None = None,
    ) -> None:
        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self.persist_path = persist_path
        self.collection: Any | None = None
        self.chunk_by_id = {chunk.id: chunk for chunk in chunks}
        self.collection_name = self._collection_name()
        self.reused_existing = False

    async def build(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise RuntimeError("hybrid local retrieval requires chromadb") from exc

        settings = ChromaSettings(anonymized_telemetry=False)
        if self.persist_path is not None:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_path), settings=settings)
            self.collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            if self.collection.count() == len(self.chunks):
                self.reused_existing = True
                return
            client.delete_collection(self.collection_name)
            self.collection = client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            client = chromadb.Client(settings)
            self.collection = client.create_collection(
                name=f"local_rag_{uuid.uuid4().hex[:12]}",
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

    def _collection_name(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.embedding_provider.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.embedding_provider.model.encode("utf-8"))
        for chunk in self.chunks:
            digest.update(b"\0")
            digest.update(chunk.id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.title.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.url.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.content.encode("utf-8"))
        return f"local_rag_{digest.hexdigest()[:16]}"


class QdrantVectorIndex:
    def __init__(
        self,
        chunks: list[LocalChunk],
        embedding_provider: EmbeddingProvider,
        base_url: str,
        collection_prefix: str,
        timeout: float,
        api_key: str | None = None,
    ) -> None:
        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.chunk_by_id = {chunk.id: chunk for chunk in chunks}
        self.collection_name = self._collection_name(collection_prefix)
        self.reused_existing = False

    async def build(self) -> None:
        info = await asyncio.to_thread(self._collection_info)
        points_count = _qdrant_points_count(info)
        if points_count == len(self.chunks):
            self.reused_existing = True
            return
        if info is not None:
            await asyncio.to_thread(self._delete_collection)

        embeddings = await self.embedding_provider.embed_texts(
            [f"{chunk.title}\n{chunk.content}" for chunk in self.chunks]
        )
        if not embeddings:
            return
        await asyncio.to_thread(self._create_collection, len(embeddings[0]))
        await asyncio.to_thread(self._upsert_points, embeddings)

    async def search(self, query: str, top_k: int) -> list[RankedChunk]:
        query_vector = await self.embedding_provider.embed_text(query)
        payload = {
            "vector": query_vector,
            "limit": min(top_k, len(self.chunks)),
            "with_payload": True,
        }
        result = await asyncio.to_thread(
            self._request_json,
            "POST",
            f"/collections/{self.collection_name}/points/search",
            payload,
        )
        ranked: list[RankedChunk] = []
        for index, item in enumerate(result.get("result", []), start=1):
            payload = item.get("payload") or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id not in self.chunk_by_id:
                continue
            ranked.append(
                RankedChunk(
                    chunk=self.chunk_by_id[chunk_id],
                    score=float(item.get("score", 0.0)),
                    rank=index,
                    method="vector",
                )
            )
        return ranked

    def _collection_info(self) -> dict[str, Any] | None:
        return self._request_json(
            "GET",
            f"/collections/{self.collection_name}",
            allow_404=True,
        )

    def _create_collection(self, vector_size: int) -> None:
        self._request_json(
            "PUT",
            f"/collections/{self.collection_name}",
            {
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine",
                }
            },
        )

    def _delete_collection(self) -> None:
        self._request_json("DELETE", f"/collections/{self.collection_name}")

    def _upsert_points(self, embeddings: list[list[float]]) -> None:
        points = []
        for chunk, vector in zip(self.chunks, embeddings, strict=False):
            points.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)),
                    "vector": vector,
                    "payload": {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "url": chunk.url,
                        "chunk_index": chunk.chunk_index,
                    },
                }
            )
        self._request_json(
            "PUT",
            f"/collections/{self.collection_name}/points?wait=true",
            {"points": points},
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        allow_404: bool = False,
    ) -> dict[str, Any] | None:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if allow_404 and exc.code == 404:
                return None
            raise RuntimeError(
                f"Qdrant request failed: {method} {path} -> {exc.code} {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Qdrant request failed: {method} {path} -> {exc.reason}"
            ) from exc
        return json.loads(raw) if raw else {}

    def _collection_name(self, prefix: str) -> str:
        safe_prefix = (
            re.sub(r"[^a-zA-Z0-9_-]+", "_", prefix.strip())
            or "deepresearch_local_rag"
        )
        digest = hashlib.sha256()
        digest.update(self.embedding_provider.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.embedding_provider.model.encode("utf-8"))
        for chunk in self.chunks:
            digest.update(b"\0")
            digest.update(chunk.id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.title.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.url.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.content.encode("utf-8"))
        return f"{safe_prefix}_{digest.hexdigest()[:16]}"


def _qdrant_points_count(info: dict[str, Any] | None) -> int | None:
    if info is None:
        return None
    result = info.get("result") or {}
    count = result.get("points_count", result.get("vectors_count"))
    return int(count) if count is not None else None


def _tokens(text: str) -> set[str]:
    return tokenize(text)
