from __future__ import annotations

from collections import Counter
from urllib.parse import urlparse

from deepresearch_agent.schemas import Source


def source_diversity_metrics(sources: list[Source]) -> dict[str, object]:
    provider_counts = Counter(source.provider or "unknown" for source in sources)
    domain_counts = Counter(_source_domain(source.url) for source in sources)
    return {
        "source_provider_count": len(provider_counts),
        "source_domain_count": len(domain_counts),
        "source_provider_counts": dict(sorted(provider_counts.items())),
        "source_domain_counts": dict(sorted(domain_counts.items())),
    }


def _source_domain(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc:
        return parsed.netloc.lower()
    if parsed.scheme:
        return parsed.scheme.lower()
    return "unknown"
