from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import pickle
import ssl
import time
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import certifi

from deepresearch_agent.config import load_settings
from deepresearch_agent.embeddings import EmbeddingProvider, build_embedding_provider
from deepresearch_agent.rag import LocalRagRetriever
from deepresearch_agent.rerankers import RerankProvider, RerankResult, build_rerank_provider
from deepresearch_agent.schemas import Source


# Public BEIR benchmark dataset. SciFact is not project-owned data; it is used
# here only to evaluate retrieval quality independently from the end-to-end LLM
# benchmark.
BEIR_SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
BEIR_SCIFACT_QRELS_SPLIT = "test"
DEFAULT_SCIFACT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RETRIEVAL_EVAL_MODES = ("keyword", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class ScifactDocument:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class ScifactQuery:
    query_id: str
    text: str


@dataclass(frozen=True)
class ScifactDataset:
    name: str
    source_url: str
    qrels_split: str
    root: Path
    corpus: dict[str, ScifactDocument]
    queries: dict[str, ScifactQuery]
    qrels: dict[str, dict[str, int]]


@dataclass(frozen=True)
class RetrievalEvalConfig:
    modes: list[str]
    top_k: int
    max_queries: int | None
    embedding_provider: str
    embedding_model: str
    rerank_provider: str
    rerank_candidate_k: int
    include_rankings: bool = False


def load_scifact_dataset(
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> ScifactDataset:
    root = _ensure_scifact_dataset(cache_dir=cache_dir, force_download=force_download)
    corpus = _load_corpus(root / "corpus.jsonl")
    queries = _load_queries(root / "queries.jsonl")
    qrels = _load_qrels(root / "qrels" / f"{BEIR_SCIFACT_QRELS_SPLIT}.tsv")
    return ScifactDataset(
        name="BEIR/scifact",
        source_url=BEIR_SCIFACT_URL,
        qrels_split=BEIR_SCIFACT_QRELS_SPLIT,
        root=root,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )


def validate_scifact_dataset(dataset: ScifactDataset) -> dict[str, int | str]:
    qrel_query_ids = set(dataset.qrels)
    relevant_doc_ids = {
        doc_id for doc_scores in dataset.qrels.values() for doc_id in doc_scores
    }
    missing_query_ids = qrel_query_ids - set(dataset.queries)
    missing_doc_ids = relevant_doc_ids - set(dataset.corpus)
    if missing_query_ids or missing_doc_ids:
        raise RuntimeError(
            "BEIR/scifact qrels do not align with queries/corpus: "
            f"missing_query_ids={len(missing_query_ids)}, "
            f"missing_doc_ids={len(missing_doc_ids)}"
        )
    return {
        "dataset": dataset.name,
        "source_url": dataset.source_url,
        "qrels_split": dataset.qrels_split,
        "corpus_doc_count": len(dataset.corpus),
        "query_count": len(dataset.queries),
        "qrels_query_count": len(dataset.qrels),
        "qrels_relevant_pair_count": sum(len(items) for items in dataset.qrels.values()),
    }


def compute_retrieval_metrics(
    qrels: dict[str, dict[str, int]],
    rankings: dict[str, list[dict[str, Any]]],
    k: int,
) -> dict[str, float | int]:
    query_ids = [query_id for query_id in qrels if query_id in rankings]
    if not query_ids:
        raise ValueError("no overlapping query ids between qrels and rankings")

    recall_values: list[float] = []
    ndcg_values: list[float] = []
    reciprocal_ranks: list[float] = []
    for query_id in query_ids:
        relevant_scores = qrels[query_id]
        ranked_doc_ids = _dedupe_ranked_doc_ids(rankings[query_id])[:k]
        recall_values.append(_recall_at_k(relevant_scores, ranked_doc_ids))
        ndcg_values.append(_ndcg_at_k(relevant_scores, ranked_doc_ids, k))
        reciprocal_ranks.append(_reciprocal_rank(relevant_scores, ranked_doc_ids))

    return {
        "query_count": len(query_ids),
        f"recall@{k}": _mean(recall_values),
        f"ndcg@{k}": _mean(ndcg_values),
        "mrr": _mean(reciprocal_ranks),
    }


async def run_scifact_retrieval(
    dataset: ScifactDataset,
    config: RetrievalEvalConfig,
) -> dict[str, Any]:
    query_ids = list(dataset.qrels)
    if config.max_queries is not None:
        query_ids = query_ids[: config.max_queries]
    corpus_path = _write_local_corpus(dataset)
    embedding_provider = _build_cached_embedding_provider(dataset, config)
    runs = {}
    for mode in config.modes:
        rankings: dict[str, list[dict[str, Any]]] = {}
        started_at = time.perf_counter()
        if mode == "hybrid_rerank":
            sources_by_query_id = await _retrieve_hybrid_candidates(
                dataset=dataset,
                query_ids=query_ids,
                corpus_path=corpus_path,
                config=config,
                embedding_provider=embedding_provider,
            )
            reranked = await _rerank_eval_candidates(dataset, sources_by_query_id, config)
            rankings = {
                query_id: [_source_to_ranking_item(source) for source in sources]
                for query_id, sources in reranked.items()
            }
        else:
            retriever = _build_retriever(mode, corpus_path, config, embedding_provider)
            for query_id in query_ids:
                query = dataset.queries[query_id].text
                sources = await retriever.retrieve(query, max_results=config.top_k)
                rankings[query_id] = [_source_to_ranking_item(source) for source in sources]
        elapsed_seconds = time.perf_counter() - started_at
        run_record: dict[str, Any] = {
            "mode": mode,
            "top_k": config.top_k,
            "query_count": len(query_ids),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "average_query_latency_seconds": round(
                elapsed_seconds / len(query_ids), 6
            )
            if query_ids
            else 0.0,
            "metrics": compute_retrieval_metrics(
                _subset_qrels(dataset.qrels, query_ids),
                rankings,
                k=config.top_k,
            ),
        }
        if config.include_rankings:
            run_record["rankings"] = rankings
        runs[mode] = run_record
    return {
        "dataset": {
            "name": dataset.name,
            "source_url": dataset.source_url,
            "qrels_split": dataset.qrels_split,
            "corpus_doc_count": len(dataset.corpus),
            "query_count": len(dataset.queries),
            "evaluated_query_count": len(query_ids),
            "qrels_query_count": len(dataset.qrels),
            "qrels_relevant_pair_count": sum(
                len(items) for items in dataset.qrels.values()
            ),
        },
        "config": {
            "modes": config.modes,
            "top_k": config.top_k,
            "max_queries": config.max_queries,
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "rerank_provider": config.rerank_provider,
            "rerank_candidate_k": config.rerank_candidate_k,
            "include_rankings": config.include_rankings,
        },
        "runs": runs,
    }


async def _retrieve_hybrid_candidates(
    dataset: ScifactDataset,
    query_ids: list[str],
    corpus_path: Path,
    config: RetrievalEvalConfig,
    embedding_provider: EmbeddingProvider,
) -> dict[str, list[Source]]:
    retriever = _build_retriever("hybrid", corpus_path, config, embedding_provider)
    candidate_count = max(config.top_k, config.rerank_candidate_k)
    sources_by_query_id: dict[str, list[Source]] = {}
    for query_id in query_ids:
        query = dataset.queries[query_id].text
        sources_by_query_id[query_id] = await retriever.retrieve(
            query,
            max_results=candidate_count,
        )
    return sources_by_query_id


async def _rerank_eval_candidates(
    dataset: ScifactDataset,
    sources_by_query_id: dict[str, list[Source]],
    config: RetrievalEvalConfig,
) -> dict[str, list[Source]]:
    provider = _build_eval_rerank_provider(config)
    if provider.name == "local" and hasattr(provider, "_load_model"):
        return await asyncio.to_thread(
            _rerank_local_candidates_sync,
            provider,
            dataset,
            sources_by_query_id,
            config.top_k,
        )

    output: dict[str, list[Source]] = {}
    for query_id, sources in sources_by_query_id.items():
        query = dataset.queries[query_id].text
        reranked = await provider.rerank(query, sources)
        output[query_id] = _apply_rerank_results(
            sources=sources,
            reranked=reranked,
            provider=provider,
            max_results=config.top_k,
        )
    return output


def _build_eval_rerank_provider(config: RetrievalEvalConfig) -> RerankProvider:
    settings = replace(
        load_settings(),
        rerank_provider=config.rerank_provider,
    )
    return build_rerank_provider(settings)


def _rerank_local_candidates_sync(
    provider: RerankProvider,
    dataset: ScifactDataset,
    sources_by_query_id: dict[str, list[Source]],
    max_results: int,
) -> dict[str, list[Source]]:
    model = provider._load_model()  # type: ignore[attr-defined]
    pairs: list[tuple[str, str]] = []
    index: list[tuple[str, int, Source]] = []
    for query_id, sources in sources_by_query_id.items():
        query = dataset.queries[query_id].text
        for source_index, source in enumerate(sources):
            pairs.append((query, f"{source.title}\n{source.content}"))
            index.append((query_id, source_index, source))

    batch_size = int(getattr(provider, "batch_size", 16))
    scores = model.predict(pairs, batch_size=batch_size)
    grouped: dict[str, list[RerankResult]] = {
        query_id: [] for query_id in sources_by_query_id
    }
    for (query_id, source_index, _source), score in zip(index, scores):
        grouped[query_id].append(RerankResult(index=source_index, score=float(score)))
    for query_results in grouped.values():
        query_results.sort(key=lambda item: item.score, reverse=True)

    return {
        query_id: _apply_rerank_results(
            sources=sources_by_query_id[query_id],
            reranked=reranked,
            provider=provider,
            max_results=max_results,
        )
        for query_id, reranked in grouped.items()
    }


def _apply_rerank_results(
    sources: list[Source],
    reranked: list[RerankResult],
    provider: RerankProvider,
    max_results: int,
) -> list[Source]:
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


def _subset_qrels(
    qrels: dict[str, dict[str, int]], query_ids: list[str]
) -> dict[str, dict[str, int]]:
    return {query_id: qrels[query_id] for query_id in query_ids if query_id in qrels}


def _dedupe_ranked_doc_ids(ranking: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    doc_ids: list[str] = []
    for item in ranking:
        doc_id = str(item.get("doc_id", ""))
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return doc_ids


def _recall_at_k(relevant_scores: dict[str, int], ranked_doc_ids: list[str]) -> float:
    if not relevant_scores:
        return 0.0
    hits = {doc_id for doc_id in ranked_doc_ids if doc_id in relevant_scores}
    return len(hits) / len(relevant_scores)


def _ndcg_at_k(
    relevant_scores: dict[str, int], ranked_doc_ids: list[str], k: int
) -> float:
    ideal_relevance = sorted(relevant_scores.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_relevance)
    if ideal_dcg == 0:
        return 0.0
    observed_relevance = [relevant_scores.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    return _dcg(observed_relevance) / ideal_dcg


def _dcg(relevance_values: list[int]) -> float:
    return sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(relevance_values, start=1)
        if relevance > 0
    )


def _reciprocal_rank(
    relevant_scores: dict[str, int], ranked_doc_ids: list[str]
) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_scores:
            return 1.0 / rank
    return 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_retriever(
    mode: str,
    corpus_path: Path,
    config: RetrievalEvalConfig,
    embedding_provider: EmbeddingProvider,
) -> LocalRagRetriever:
    if mode not in RETRIEVAL_EVAL_MODES:
        raise ValueError(f"unknown retrieval eval mode: {mode}")
    settings = load_settings()
    retrieval_mode = "keyword" if mode == "keyword" else "hybrid"
    rerank_enabled = mode == "hybrid_rerank"
    settings = replace(
        settings,
        embedding_provider=config.embedding_provider,
        local_embedding_model=config.embedding_model,
        local_retrieval_mode=retrieval_mode,
        local_keyword_top_k=max(config.top_k, config.rerank_candidate_k),
        local_vector_top_k=max(config.top_k, config.rerank_candidate_k),
        rerank_enabled=rerank_enabled,
        rerank_provider=config.rerank_provider,
        local_rerank_candidate_k=max(config.top_k, config.rerank_candidate_k),
        # SciFact abstracts should be evaluated at document granularity.
        local_chunk_size_chars=100_000,
        local_chunk_overlap_chars=0,
    )
    rerank_provider = build_rerank_provider(settings) if rerank_enabled else None
    return LocalRagRetriever(
        corpus_path=corpus_path,
        settings=settings,
        embedding_provider=embedding_provider,
        rerank_provider=rerank_provider,
    )


def _build_cached_embedding_provider(
    dataset: ScifactDataset, config: RetrievalEvalConfig
) -> EmbeddingProvider:
    settings = replace(
        load_settings(),
        embedding_provider=config.embedding_provider,
        local_embedding_model=config.embedding_model,
    )
    base_provider = build_embedding_provider(settings, config.embedding_provider)
    cache_path = dataset.root / f"embedding_cache_{_safe_name(base_provider.model)}.pkl"
    return CachedCorpusEmbeddingProvider(base_provider=base_provider, cache_path=cache_path)


@dataclass
class CachedCorpusEmbeddingProvider(EmbeddingProvider):
    base_provider: EmbeddingProvider
    cache_path: Path

    @property
    def name(self) -> str:
        return self.base_provider.name

    @property
    def model(self) -> str:
        return self.base_provider.model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= 1:
            return await self.base_provider.embed_texts(texts)
        digest = _texts_digest(texts)
        cached = self._load_cache(digest)
        if cached is not None:
            return cached
        vectors = await self.base_provider.embed_texts(texts)
        self._save_cache(digest, vectors)
        return vectors

    def _load_cache(self, digest: str) -> list[list[float]] | None:
        if not self.cache_path.exists():
            return None
        with self.cache_path.open("rb") as file:
            payload = pickle.load(file)
        if (
            payload.get("model") == self.model
            and payload.get("digest") == digest
            and isinstance(payload.get("vectors"), list)
        ):
            return payload["vectors"]
        return None

    def _save_cache(self, digest: str, vectors: list[list[float]]) -> None:
        with self.cache_path.open("wb") as file:
            pickle.dump(
                {"model": self.model, "digest": digest, "vectors": vectors},
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def _texts_digest(texts: list[str]) -> str:
    hasher = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _write_local_corpus(dataset: ScifactDataset) -> Path:
    path = dataset.root / "deepresearch_local_corpus.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for document in dataset.corpus.values():
            row = {
                "id": document.doc_id,
                "title": document.title,
                "url": f"beir://scifact/{document.doc_id}",
                "content": document.text,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _source_to_ranking_item(source: Source) -> dict[str, Any]:
    doc_id = str(source.metadata.get("local_doc_id", ""))
    return {
        "doc_id": doc_id,
        "score": source.score,
        "title": source.title,
        "metadata": source.metadata,
    }


def _ensure_scifact_dataset(
    cache_dir: Path | None,
    force_download: bool,
) -> Path:
    root = Path(__file__).resolve().parents[2]
    base_dir = cache_dir or root / "data" / "beir"
    scifact_dir = base_dir / "scifact"
    required_files = [
        scifact_dir / "corpus.jsonl",
        scifact_dir / "queries.jsonl",
        scifact_dir / "qrels" / f"{BEIR_SCIFACT_QRELS_SPLIT}.tsv",
    ]
    if not force_download and all(path.exists() for path in required_files):
        return scifact_dir

    base_dir.mkdir(parents=True, exist_ok=True)
    zip_path = base_dir / "scifact.zip"
    _download(BEIR_SCIFACT_URL, zip_path)
    with ZipFile(zip_path) as archive:
        archive.extractall(base_dir)
    return scifact_dir


def _download(url: str, destination: Path) -> None:
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": "deepresearch-agent/0.1"})
    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    last_error: Exception | None = None
    for _ in range(3):
        try:
            if temp_destination.exists():
                temp_destination.unlink()
            with urllib.request.urlopen(request, timeout=300, context=context) as response:
                expected_length = response.headers.get("Content-Length")
                bytes_written = 0
                with temp_destination.open("wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        file.write(chunk)
            if expected_length is not None and bytes_written != int(expected_length):
                raise RuntimeError(
                    f"incomplete download: expected {expected_length} bytes, got {bytes_written}"
                )
            _assert_valid_zip(temp_destination)
            temp_destination.replace(destination)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"failed to download valid BEIR/scifact zip: {last_error}") from last_error


def _assert_valid_zip(path: Path) -> None:
    try:
        with ZipFile(path) as archive:
            bad_member = archive.testzip()
    except BadZipFile as exc:
        raise RuntimeError(f"downloaded file is not a valid zip: {path}") from exc
    if bad_member is not None:
        raise RuntimeError(f"downloaded zip has a corrupt member: {bad_member}")


def _load_corpus(path: Path) -> dict[str, ScifactDocument]:
    corpus: dict[str, ScifactDocument] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = str(item["_id"])
            corpus[doc_id] = ScifactDocument(
                doc_id=doc_id,
                title=item.get("title", ""),
                text=item.get("text", ""),
            )
    return corpus


def _load_queries(path: Path) -> dict[str, ScifactQuery]:
    queries: dict[str, ScifactQuery] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            query_id = str(item["_id"])
            queries[query_id] = ScifactQuery(query_id=query_id, text=item["text"])
    return queries


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as file:
        header = file.readline().strip().split("\t")
        try:
            query_index = header.index("query-id")
            doc_index = header.index("corpus-id")
            score_index = header.index("score")
        except ValueError as exc:
            raise RuntimeError(f"unexpected qrels header in {path}: {header}") from exc
        for line in file:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            query_id = parts[query_index]
            doc_id = parts[doc_index]
            score = int(parts[score_index])
            if score > 0:
                qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and evaluate local retrieval on BEIR/scifact."
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--run", action="store_true", help="Run retrieval and output rankings.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=RETRIEVAL_EVAL_MODES,
        default=list(RETRIEVAL_EVAL_MODES),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--embedding-provider", choices=["local", "dashscope"], default="local")
    parser.add_argument("--embedding-model", default=DEFAULT_SCIFACT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-provider", choices=["local", "dashscope"], default="local")
    parser.add_argument("--rerank-candidate-k", type=int, default=20)
    parser.add_argument(
        "--include-rankings",
        action="store_true",
        help="Include per-query top-k rankings in the output JSON.",
    )
    parser.add_argument(
        "--output",
        default="results/retrieval_eval_scifact.json",
        help="Where to write retrieval metrics when --run is set.",
    )
    args = parser.parse_args()
    dataset = load_scifact_dataset(
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        force_download=args.force_download,
    )
    validation = validate_scifact_dataset(dataset)
    if not args.run:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    result = asyncio.run(
        run_scifact_retrieval(
            dataset,
            RetrievalEvalConfig(
                modes=args.modes,
                top_k=args.top_k,
                max_queries=args.max_queries,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
                rerank_provider=args.rerank_provider,
                rerank_candidate_k=args.rerank_candidate_k,
                include_rankings=args.include_rankings,
            ),
        )
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"validation": validation, "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
