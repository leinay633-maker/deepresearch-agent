from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pickle
import ssl
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import certifi

from deepresearch_agent.config import load_settings
from deepresearch_agent.embeddings import EmbeddingProvider, build_embedding_provider
from deepresearch_agent.rag import LocalRagRetriever
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
        retriever = _build_retriever(mode, corpus_path, config, embedding_provider)
        rankings: dict[str, list[dict[str, Any]]] = {}
        for query_id in query_ids:
            query = dataset.queries[query_id].text
            sources = await retriever.retrieve(query, max_results=config.top_k)
            rankings[query_id] = [_source_to_ranking_item(source) for source in sources]
        runs[mode] = {
            "mode": mode,
            "top_k": config.top_k,
            "query_count": len(query_ids),
            "rankings": rankings,
        }
    return {
        "dataset": {
            "name": dataset.name,
            "source_url": dataset.source_url,
            "qrels_split": dataset.qrels_split,
            "corpus_doc_count": len(dataset.corpus),
            "query_count": len(dataset.queries),
            "evaluated_query_count": len(query_ids),
            "qrels_query_count": len(dataset.qrels),
        },
        "config": {
            "modes": config.modes,
            "top_k": config.top_k,
            "max_queries": config.max_queries,
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "rerank_provider": config.rerank_provider,
            "rerank_candidate_k": config.rerank_candidate_k,
        },
        "runs": runs,
    }


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
    return LocalRagRetriever(
        corpus_path=corpus_path,
        settings=settings,
        embedding_provider=embedding_provider,
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
        "--output",
        default="results/retrieval_eval_scifact_rankings.json",
        help="Where to write retrieval rankings when --run is set.",
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
            ),
        )
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"validation": validation, "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
