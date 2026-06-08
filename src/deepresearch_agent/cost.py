from __future__ import annotations

import math

from deepresearch_agent.schemas import CostRecord, CostSummary

DEEPSEEK_PRICING_SOURCE_URL = "https://api-docs.deepseek.com/quick_start/pricing"
DEEPSEEK_PRICING_CHECKED_ON = "2026-06-07"
DEEPSEEK_V4_FLASH_PRICING_USD_PER_1M = {
    # Official DeepSeek Models & Pricing page, checked on 2026-06-07.
    "input_cache_hit": 0.0028,
    "input_cache_miss": 0.14,
    "output": 0.28,
}
DEEPSEEK_MODEL_PRICING_USD_PER_1M = {
    "deepseek-v4-flash": DEEPSEEK_V4_FLASH_PRICING_USD_PER_1M,
}


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def deepseek_usage_cost_usd(model: str, usage: dict) -> tuple[int, int, float]:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if prompt_tokens <= 0 or completion_tokens <= 0:
        raise ValueError(f"DeepSeek usage missing token counts: {usage}")
    pricing = DEEPSEEK_MODEL_PRICING_USD_PER_1M.get(model.lower())
    if pricing is None:
        raise ValueError(f"DeepSeek pricing is not configured for model: {model}")

    cache_hit_tokens = int(usage.get("prompt_cache_hit_tokens") or 0)
    cache_miss_tokens = usage.get("prompt_cache_miss_tokens")
    if cache_miss_tokens is None:
        cache_miss_tokens = max(prompt_tokens - cache_hit_tokens, 0)
    cache_miss_tokens = int(cache_miss_tokens)

    input_cost = (
        cache_hit_tokens * pricing["input_cache_hit"]
        + cache_miss_tokens * pricing["input_cache_miss"]
    ) / 1_000_000
    output_cost = completion_tokens * pricing["output"] / 1_000_000
    return prompt_tokens, completion_tokens, input_cost + output_cost


class CostTracker:
    def __init__(
        self,
        provider: str,
        model: str,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.records: list[CostRecord] = []

    def add(
        self,
        stage: str,
        input_text: str,
        output_text: str,
        model: str | None = None,
    ) -> CostRecord:
        input_tokens = estimate_tokens(input_text)
        output_tokens = estimate_tokens(output_text)
        cost = (
            input_tokens * self.input_cost_per_1m / 1_000_000
            + output_tokens * self.output_cost_per_1m / 1_000_000
        )
        record = CostRecord(
            stage=stage,
            provider=self.provider,
            model=model or self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 8),
        )
        self.records.append(record)
        return record

    def add_usage(
        self,
        stage: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        model: str | None = None,
        provider: str | None = None,
    ) -> CostRecord:
        record = CostRecord(
            stage=stage,
            provider=provider or self.provider,
            model=model or self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(estimated_cost_usd, 8),
        )
        self.records.append(record)
        return record

    def summary(self) -> CostSummary:
        input_tokens = sum(record.input_tokens for record in self.records)
        output_tokens = sum(record.output_tokens for record in self.records)
        cost = sum(record.estimated_cost_usd for record in self.records)
        return CostSummary(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            total_estimated_cost_usd=round(cost, 8),
            records=list(self.records),
        )
