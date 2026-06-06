from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

from deepresearch_agent.schemas import Source


class SourceDeduplicator:
    def dedup(self, sources: list[Source]) -> list[Source]:
        by_key: dict[str, Source] = {}
        for source in sources:
            key = self.key(source)
            existing = by_key.get(key)
            if existing is None or _rank(source) > _rank(existing):
                by_key[key] = source
        return sorted(by_key.values(), key=_rank, reverse=True)

    def key(self, source: Source) -> str:
        if source.url:
            parts = urlsplit(source.url)
            path = re.sub(r"/+$", "", parts.path.lower())
            return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))
        return re.sub(r"\W+", " ", source.title.lower()).strip()


def _rank(source: Source) -> float:
    return source.quality_score * 10 + source.score
