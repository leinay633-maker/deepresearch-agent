from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from deepresearch_agent.config import Settings


class EmbeddingProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]


@dataclass
class LocalEmbeddingProvider(EmbeddingProvider):
    model: str = "BAAI/bge-small-zh-v1.5"
    batch_size: int = 16
    normalize_embeddings: bool = True

    name: str = "local"

    def __post_init__(self) -> None:
        self._model: Any | None = None

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        encoded = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in encoded]

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "LocalEmbeddingProvider requires sentence-transformers. "
                    "Install the project with its runtime dependencies first."
                ) from exc
            self._model = SentenceTransformer(self.model)
        return self._model


@dataclass
class DashScopeEmbeddingProvider(EmbeddingProvider):
    model: str = "text-embedding-v4"
    dimensions: int = 1024
    timeout: float = 30.0
    batch_size: int = 16
    endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

    name: str = "dashscope"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY environment variable is required")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(await asyncio.to_thread(self._embed_batch_sync, batch, api_key))
        return vectors

    def _embed_batch_sync(self, texts: list[str], api_key: str) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions > 0:
            payload["dimensions"] = self.dimensions
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope embedding request failed: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DashScope embedding request failed: {exc.reason}") from exc
        parsed = json.loads(raw)
        items = sorted(parsed.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in items]
        if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
            raise RuntimeError("DashScope embedding response did not contain one vector per text")
        return vectors


def build_embedding_provider(
    settings: Settings, provider: str | None = None
) -> EmbeddingProvider:
    selected = (provider or settings.embedding_provider).strip().lower()
    if selected == "dashscope":
        return DashScopeEmbeddingProvider(
            model=settings.dashscope_embedding_model,
            dimensions=settings.dashscope_embedding_dimensions,
            timeout=settings.request_timeout_seconds,
            batch_size=settings.embedding_batch_size,
        )
    if selected == "local":
        return LocalEmbeddingProvider(
            model=settings.local_embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    raise ValueError(f"unknown embedding provider: {selected}")
