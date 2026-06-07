from __future__ import annotations

import argparse
import asyncio

from deepresearch_agent.config import load_settings
from deepresearch_agent.embeddings import (
    DashScopeEmbeddingProvider,
    build_embedding_provider,
)


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings()
    provider = build_embedding_provider(settings, args.provider)
    if isinstance(provider, DashScopeEmbeddingProvider) and args.endpoint:
        provider.endpoint = args.endpoint
    vector = await provider.embed_text(args.text)
    print(f"provider={provider.name}")
    print(f"model={provider.model}")
    print(f"dimension={len(vector)}")
    print(f"sample={vector[:5]}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an embedding provider.")
    parser.add_argument("--provider", choices=["local", "dashscope"], default=None)
    parser.add_argument(
        "--text",
        default="混合检索需要同时保留关键词召回和向量召回，再做融合排序。",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional DashScope-compatible embeddings endpoint for transport tests.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
