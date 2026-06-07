from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import certifi


# Public BEIR benchmark dataset. SciFact is not project-owned data; it is used
# here only to evaluate retrieval quality independently from the end-to-end LLM
# benchmark.
BEIR_SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
BEIR_SCIFACT_QRELS_SPLIT = "test"


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
    args = parser.parse_args()
    dataset = load_scifact_dataset(
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        force_download=args.force_download,
    )
    print(json.dumps(validate_scifact_dataset(dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
