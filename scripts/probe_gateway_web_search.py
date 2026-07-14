#!/usr/bin/env python3
"""Write a secret-free capability matrix for Gateway server-side web search."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from deepresearch_agent.gateway_search import GatewayWebSearchAdapter


DEFAULT_MODELS = (
    "claude-4.6-opus",
    "claude-opus-4-8",
    "kimi-k2.7-code-highspeed",
    "glm-5.2",
)


async def probe_models(
    models: list[str],
    *,
    base_url: str,
    timeout_seconds: float,
) -> dict[str, object]:
    probes = []
    for model in models:
        adapter = GatewayWebSearchAdapter(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            require_response_model_match=True,
        )
        probes.append(asdict(await adapter.probe_capability(timeout=timeout_seconds)))
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url.rstrip("/"),
        "probe_retains_response_text": False,
        "probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Gateway web-search tool support without retaining result bodies."
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--base-url", default="https://llmapi.bilibili.co")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = asyncio.run(
        probe_models(
            args.models or list(DEFAULT_MODELS),
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
