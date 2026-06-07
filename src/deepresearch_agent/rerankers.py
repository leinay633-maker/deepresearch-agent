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
from deepresearch_agent.schemas import Source


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class RerankProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def rerank(self, query: str, sources: list[Source]) -> list[RerankResult]:
        raise NotImplementedError


@dataclass
class LocalRerankProvider(RerankProvider):
    model: str = "BAAI/bge-reranker-base"
    batch_size: int = 16

    name: str = "local"

    def __post_init__(self) -> None:
        self._model: Any | None = None

    async def rerank(self, query: str, sources: list[Source]) -> list[RerankResult]:
        if not sources:
            return []
        return await asyncio.to_thread(self._rerank_sync, query, sources)

    def _rerank_sync(self, query: str, sources: list[Source]) -> list[RerankResult]:
        model = self._load_model()
        pairs = [(query, f"{source.title}\n{source.content}") for source in sources]
        scores = model.predict(pairs, batch_size=self.batch_size)
        results = [
            RerankResult(index=index, score=float(score))
            for index, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "LocalRerankProvider requires sentence-transformers. "
                    "Install the project with its runtime dependencies first."
                ) from exc
            self._model = CrossEncoder(self.model)
        return self._model


@dataclass
class DashScopeRerankProvider(RerankProvider):
    model: str = "gte-rerank-v2"
    timeout: float = 30.0
    endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    name: str = "dashscope"

    async def rerank(self, query: str, sources: list[Source]) -> list[RerankResult]:
        if not sources:
            return []
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY environment variable is required")
        return await asyncio.to_thread(self._rerank_sync, query, sources, api_key)

    def _rerank_sync(
        self, query: str, sources: list[Source], api_key: str
    ) -> list[RerankResult]:
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": [f"{source.title}\n{source.content}" for source in sources],
            },
            "parameters": {
                "top_n": len(sources),
                "return_documents": False,
            },
        }
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
            raise RuntimeError(f"DashScope rerank request failed: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DashScope rerank request failed: {exc.reason}") from exc
        parsed = json.loads(raw)
        items = parsed.get("output", {}).get("results", [])
        results = [
            RerankResult(
                index=int(item["index"]),
                score=float(item["relevance_score"]),
            )
            for item in items
            if "index" in item and "relevance_score" in item
        ]
        if len(results) != len(sources):
            raise RuntimeError("DashScope rerank response did not score every source")
        results.sort(key=lambda item: item.score, reverse=True)
        return results


def build_rerank_provider(settings: Settings, provider: str | None = None) -> RerankProvider:
    selected = (provider or settings.rerank_provider).strip().lower()
    if selected == "dashscope":
        return DashScopeRerankProvider(
            model=settings.dashscope_rerank_model,
            timeout=settings.request_timeout_seconds,
        )
    if selected == "local":
        return LocalRerankProvider(
            model=settings.local_rerank_model,
            batch_size=settings.embedding_batch_size,
        )
    raise ValueError(f"unknown rerank provider: {selected}")
