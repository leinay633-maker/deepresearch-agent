from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INCLUDE_EXTENSIONS = {".md", ".markdown", ".txt"}
DEFAULT_EXCLUDE_DIRS = {".git", ".obsidian", ".claude", "node_modules", "__pycache__"}


@dataclass
class IngestSummary:
    input_dir: str
    output_path: str
    document_count: int
    skipped_count: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "input_dir": self.input_dir,
            "output_path": self.output_path,
            "document_count": self.document_count,
            "skipped_count": self.skipped_count,
        }


def ingest_directory(
    input_dir: Path,
    output_path: Path,
    *,
    include_extensions: set[str] | None = None,
    exclude_dirs: set[str] | None = None,
    max_chars_per_document: int | None = None,
) -> IngestSummary:
    root = input_dir.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    extensions = _normalize_extensions(include_extensions or DEFAULT_INCLUDE_EXTENSIONS)
    excluded = exclude_dirs or DEFAULT_EXCLUDE_DIRS
    rows: list[dict[str, Any]] = []
    skipped_count = 0

    for path in _iter_document_paths(root, extensions, excluded):
        raw = path.read_text(encoding="utf-8", errors="replace")
        content = _clean_content(raw)
        if max_chars_per_document is not None:
            content = content[:max_chars_per_document].strip()
        if not content:
            skipped_count += 1
            continue
        relative_path = path.relative_to(root).as_posix()
        rows.append(
            {
                "id": _document_id(relative_path),
                "title": _title_from_content(content, path),
                "url": path.resolve().as_uri(),
                "content": content,
                "metadata": {
                    "source_path": relative_path,
                    "ingest_format": path.suffix.lower().lstrip("."),
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    output_path.write_text(payload, encoding="utf-8")

    return IngestSummary(
        input_dir=str(root),
        output_path=str(output_path.resolve()),
        document_count=len(rows),
        skipped_count=skipped_count,
    )


def _iter_document_paths(
    root: Path,
    include_extensions: set[str],
    exclude_dirs: set[str],
) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts[:-1]
        if any(part in exclude_dirs for part in relative_parts):
            continue
        if path.suffix.lower() in include_extensions:
            paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix().lower())


def _clean_content(raw: str) -> str:
    text = raw.replace("\x00", "")
    if text.startswith("---"):
        match = re.match(r"^---\s*\n.*?\n---\s*\n", text, flags=re.DOTALL)
        if match:
            text = text[match.end() :]
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _title_from_content(content: str, path: Path) -> str:
    for line in content.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return path.stem.replace("_", " ").replace("-", " ").strip() or path.name


def _document_id(relative_path: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", relative_path.rsplit(".", 1)[0]).strip("-").lower()
    return f"local-{slug}" if slug else "local-document"


def _normalize_extensions(values: set[str]) -> set[str]:
    return {(value if value.startswith(".") else f".{value}").lower() for value in values}


def _parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local JSONL corpus from Markdown/TXT files.")
    parser.add_argument("input_dir", help="Directory containing private knowledge-base files.")
    parser.add_argument(
        "--output",
        default="data/local_corpus.jsonl",
        help="Output JSONL path consumed by LocalRagRetriever.",
    )
    parser.add_argument(
        "--include-extensions",
        default="md,markdown,txt",
        help="Comma-separated file extensions to ingest.",
    )
    parser.add_argument(
        "--exclude-dirs",
        default=".git,.obsidian,.claude,node_modules,__pycache__",
        help="Comma-separated directory names to skip.",
    )
    parser.add_argument(
        "--max-chars-per-document",
        type=int,
        default=None,
        help="Optional truncation limit per document.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    summary = ingest_directory(
        Path(args.input_dir),
        Path(args.output),
        include_extensions=_parse_csv_set(args.include_extensions),
        exclude_dirs=_parse_csv_set(args.exclude_dirs),
        max_chars_per_document=args.max_chars_per_document,
    )
    if args.json:
        print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(
            "Ingested "
            f"{summary.document_count} documents into {summary.output_path} "
            f"(skipped={summary.skipped_count})."
        )


if __name__ == "__main__":
    main()
