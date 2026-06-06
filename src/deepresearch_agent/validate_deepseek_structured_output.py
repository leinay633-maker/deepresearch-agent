from __future__ import annotations

import argparse
import asyncio
import json
import os

from deepresearch_agent.cost import CostTracker
from deepresearch_agent.llm import DEEPSEEK_DEFAULT_MODEL, DeepSeekLLMProvider
from deepresearch_agent.schemas import ResearchBrief


async def _run(args: argparse.Namespace) -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is required")

    brief = ResearchBrief(
        original_query=args.query,
        normalized_query=args.query.strip().rstrip("?") + "?",
        scope="Validate that DeepSeek can produce structured planner output for this project.",
        constraints=[
            "Return only schema-valid JSON.",
            "Produce concrete searchable subquestions.",
        ],
        assumptions=["This script validates planner structured output only."],
    )
    provider = DeepSeekLLMProvider(model=args.model, max_retries=args.max_retries)
    cost = CostTracker(provider=provider.name, model=provider.model)
    subquestions = await provider.plan(brief, max_researchers=args.max_researchers, cost=cost)
    print(
        json.dumps(
            {
                "model": provider.model,
                "validated_schema": "list[SubQuestion]",
                "subquestions": [item.model_dump() for item in subquestions],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate DeepSeek JSON-mode planner output against SubQuestion schema."
    )
    parser.add_argument(
        "--query",
        default="How should citation checking reduce hallucination in deep research agents?",
    )
    parser.add_argument("--model", default=DEEPSEEK_DEFAULT_MODEL)
    parser.add_argument("--max-researchers", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
