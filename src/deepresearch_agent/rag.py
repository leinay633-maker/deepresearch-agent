from __future__ import annotations

import json
import re
from pathlib import Path

from deepresearch_agent.schemas import Source


class LocalRagRetriever:
    name = "local_rag"

    def __init__(self, corpus_path: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.corpus_path = corpus_path or root / "data" / "local_corpus.jsonl"
        self.documents = self._load()

    async def retrieve(self, query: str, max_results: int = 2) -> list[Source]:
        query_tokens = _tokens(query)
        ranked = []
        for document in self.documents:
            text = f"{document['title']} {document['content']}"
            overlap = len(query_tokens & _tokens(text))
            ranked.append((overlap + 0.05, document))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for score, document in ranked[:max_results]:
            results.append(
                Source(
                    title=document["title"],
                    url=document["url"],
                    content=document["content"],
                    provider=self.name,
                    query=query,
                    score=score,
                    metadata={"local_doc_id": document.get("id", "")},
                )
            )
        return results

    def _load(self) -> list[dict[str, str]]:
        if not self.corpus_path.exists():
            return []
        documents = []
        with self.corpus_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    documents.append(json.loads(line))
        return documents


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", text)}
