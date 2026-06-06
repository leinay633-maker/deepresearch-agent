from __future__ import annotations

import math

from deepresearch_agent.schemas import CostRecord, CostSummary


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


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

    def add(self, stage: str, input_text: str, output_text: str) -> CostRecord:
        input_tokens = estimate_tokens(input_text)
        output_tokens = estimate_tokens(output_text)
        cost = (
            input_tokens * self.input_cost_per_1m / 1_000_000
            + output_tokens * self.output_cost_per_1m / 1_000_000
        )
        record = CostRecord(
            stage=stage,
            provider=self.provider,
            model=self.model,
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
    ) -> CostRecord:
        record = CostRecord(
            stage=stage,
            provider=self.provider,
            model=self.model,
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
