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
    run_store_path: str = "data/runs.sqlite"
    deepseek_model: str = "deepseek-v4-flash"
    embedding_provider: str = "local"
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    dashscope_embedding_model: str = "text-embedding-v4"
    dashscope_embedding_dimensions: int = 1024
    embedding_batch_size: int = 16
    local_retrieval_mode: str = "hybrid"
    local_keyword_top_k: int = 4
    local_vector_top_k: int = 4
    local_hybrid_rrf_k: int = 60
    local_keyword_weight: float = 1.0
    local_vector_weight: float = 1.0
    local_chunk_size_chars: int = 600
    local_chunk_overlap_chars: int = 80
    rerank_enabled: bool = False
    rerank_provider: str = "local"
    local_rerank_model: str = "BAAI/bge-reranker-base"
    dashscope_rerank_model: str = "gte-rerank-v2"
    local_rerank_candidate_k: int = 6
    searxng_base_url: str = ""
    web_crawler_provider: str = "none"
    jina_reader_base_url: str = "https://r.jina.ai/"
    jina_search_base_url: str = "https://s.jina.ai/"
    crawler_max_chars: int = 4000
    mcp_transport: str = "stdio"
    mcp_command: str = ""
    mcp_args: str = ""
    mcp_http_url: str = ""
    mcp_search_tool: str = ""
    mcp_query_argument: str = "query"
    run_lease_seconds: int = 120


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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
        run_store_path=os.getenv("RUN_STORE_PATH", "data/runs.sqlite"),
        deepseek_model=(
            os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
            or "deepseek-v4-flash"
        ),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
        or "local",
        local_embedding_model=(
            os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip()
            or "BAAI/bge-small-zh-v1.5"
        ),
        dashscope_embedding_model=(
            os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4").strip()
            or "text-embedding-v4"
        ),
        dashscope_embedding_dimensions=_int_env("DASHSCOPE_EMBEDDING_DIMENSIONS", 1024),
        embedding_batch_size=_int_env("EMBEDDING_BATCH_SIZE", 16),
        local_retrieval_mode=os.getenv("LOCAL_RETRIEVAL_MODE", "hybrid").strip().lower()
        or "hybrid",
        local_keyword_top_k=_int_env("LOCAL_KEYWORD_TOP_K", 4),
        local_vector_top_k=_int_env("LOCAL_VECTOR_TOP_K", 4),
        local_hybrid_rrf_k=_int_env("LOCAL_HYBRID_RRF_K", 60),
        local_keyword_weight=_float_env("LOCAL_KEYWORD_WEIGHT", 1.0),
        local_vector_weight=_float_env("LOCAL_VECTOR_WEIGHT", 1.0),
        local_chunk_size_chars=_int_env("LOCAL_CHUNK_SIZE_CHARS", 600),
        local_chunk_overlap_chars=_int_env("LOCAL_CHUNK_OVERLAP_CHARS", 80),
        rerank_enabled=_bool_env("RERANK_ENABLED", False),
        rerank_provider=os.getenv("RERANK_PROVIDER", "local").strip().lower() or "local",
        local_rerank_model=(
            os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-base").strip()
            or "BAAI/bge-reranker-base"
        ),
        dashscope_rerank_model=(
            os.getenv("DASHSCOPE_RERANK_MODEL", "gte-rerank-v2").strip()
            or "gte-rerank-v2"
        ),
        local_rerank_candidate_k=_int_env("LOCAL_RERANK_CANDIDATE_K", 6),
        searxng_base_url=os.getenv("SEARXNG_BASE_URL", "").strip(),
        web_crawler_provider=os.getenv("WEB_CRAWLER_PROVIDER", "none").strip().lower()
        or "none",
        jina_reader_base_url=os.getenv("JINA_READER_BASE_URL", "https://r.jina.ai/").strip()
        or "https://r.jina.ai/",
        jina_search_base_url=os.getenv("JINA_SEARCH_BASE_URL", "https://s.jina.ai/").strip()
        or "https://s.jina.ai/",
        crawler_max_chars=_int_env("CRAWLER_MAX_CHARS", 4000),
        mcp_transport=os.getenv("MCP_TRANSPORT", "stdio").strip().lower() or "stdio",
        mcp_command=os.getenv("MCP_COMMAND", "").strip(),
        mcp_args=os.getenv("MCP_ARGS", "").strip(),
        mcp_http_url=os.getenv("MCP_HTTP_URL", "").strip(),
        mcp_search_tool=os.getenv("MCP_SEARCH_TOOL", "").strip(),
        mcp_query_argument=os.getenv("MCP_QUERY_ARGUMENT", "query").strip() or "query",
        run_lease_seconds=_int_env("RUN_LEASE_SECONDS", 120),
    )
