from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "DeepResearch Agent"
    llm_provider: str = "mock"
    search_provider: str = "mock"
    request_timeout_seconds: float = 4.0
    max_retries: int = 2
    circuit_breaker_failure_threshold: int = 2
    circuit_breaker_cooldown_seconds: float = 30.0
    max_researchers: int = 3
    mock_model_name: str = "mock-structured-tool-model"
    mock_input_cost_per_1m_tokens: float = 0.0
    mock_output_cost_per_1m_tokens: float = 0.0
    trace_dir: str = "logs"
    deepseek_model: str = "deepseek-chat"


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower() or "mock",
        search_provider=os.getenv("SEARCH_PROVIDER", "mock").strip().lower() or "mock",
        request_timeout_seconds=_float_env("REQUEST_TIMEOUT_SECONDS", 4.0),
        max_retries=_int_env("MAX_RETRIES", 2),
        circuit_breaker_failure_threshold=_int_env("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 2),
        circuit_breaker_cooldown_seconds=_float_env("CIRCUIT_BREAKER_COOLDOWN_SECONDS", 30.0),
        max_researchers=_int_env("MAX_RESEARCHERS", 3),
        trace_dir=os.getenv("TRACE_DIR", "logs"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat",
    )
