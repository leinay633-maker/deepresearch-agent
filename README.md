# DeepResearch Agent

一个故意收窄的 DeepResearch Agent：从用户问题出发，生成 research brief，拆分子问题，并发检索，去重和来源质量过滤后，合成带引用的结构化报告，并对引用做 faithfulness 检查。

默认运行不需要 API key。默认 LLM 和 search 都走 `mock`，用于稳定演示、测试和 mock plumbing benchmark；local retrieval 默认使用本地 BGE embedding 做 keyword + vector hybrid，不需要 API key。也可以显式切到 DeepSeek 真实 LLM provider、Wikipedia 真实无 key search adapter，以及 DashScope embedding/rerank provider。所有 key 只从环境变量读取：DeepSeek 用 `DEEPSEEK_API_KEY`，DashScope 用 `DASHSCOPE_API_KEY`。

## Quickstart

```powershell
py -3.11 -m pip install -e ".[dev]"
py -3.11 -m deepresearch_agent.cli "How does citation checking reduce hallucination in agentic RAG?"
```

启动 API：

```powershell
py -3.11 -m deepresearch_agent.api --host 127.0.0.1 --port 8000
```

请求普通 JSON：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/research `
  -ContentType "application/json" `
  -Body '{"query":"How does a supervisor-researcher deep research agent differ from normal RAG?"}'
```

请求 SSE 流式输出：

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/research/stream -H "Content-Type: application/json" -d "{\"query\":\"How should agent tools fail safely?\"}"
```

运行 benchmark：

```powershell
py -3.11 -m deepresearch_agent.benchmark --search-provider mock --seed 20260606
```

运行 DeepSeek + Wikipedia 真实 provider benchmark：

```powershell
$env:DEEPSEEK_API_KEY="<your-key>"
$env:REQUEST_TIMEOUT_SECONDS="8"
py -3.11 -m deepresearch_agent.benchmark --llm-provider deepseek --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3
```

输出会写入 `logs/benchmark-*.jsonl` 和 `results/benchmark_summary.json`。

运行 retrieval 对比 benchmark：

```powershell
$env:REQUEST_TIMEOUT_SECONDS="8"
py -3.11 -m deepresearch_agent.benchmark --llm-provider deepseek --llm-model deepseek-v4-flash --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3 --local-retrieval-mode keyword
py -3.11 -m deepresearch_agent.benchmark --llm-provider deepseek --llm-model deepseek-v4-flash --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3 --local-retrieval-mode hybrid --embedding-provider local
```

## Providers

LLM providers：

- `mock`：默认 LLM provider，离线可复现，用于测试和 mock plumbing benchmark。
- `deepseek`：真实 OpenAI-compatible LLM provider，默认模型 `deepseek-v4-flash`，使用 DeepSeek JSON mode 生成 brief、plan 和 cited synthesis；需要 `DEEPSEEK_API_KEY`，可用 `DEEPSEEK_MODEL` 覆盖默认模型。

Search providers：

- `mock`：默认 search provider，离线可复现。
- `wikipedia`：真实网络检索 adapter，调用 Wikipedia Search API；失败、超时或熔断时自动使用 mock fallback，并在 trace/metrics 里暴露 fallback。

Local retrieval：

- `keyword`：旧基线，只按本地语料 token overlap 排序。
- `hybrid`：默认本地检索模式，关键词召回 + BGE 向量召回 + Chroma index + RRF 融合，仍输出统一 `Source`。
- embedding provider 默认 `local`，模型是 `BAAI/bge-small-zh-v1.5`；可显式切到 `dashscope`，key 从 `DASHSCOPE_API_KEY` 读。
- rerank provider 默认 `local`，模型是 `BAAI/bge-reranker-base`，但 `--rerank-enabled` 才会启用；也可以显式切到 DashScope rerank。

## Project Layout

```text
src/deepresearch_agent/
  api.py              FastAPI + SSE endpoints
  orchestrator.py     End-to-end research spine
  llm.py              Mock and DeepSeek structured-output model providers
  search.py           Search adapters, retry, timeout, circuit breaker, fallback
  rag.py              Local keyword/vector hybrid RAG retriever
  embeddings.py       Local and DashScope embedding providers
  rerankers.py        Optional local and DashScope rerank providers
  verifier.py         Source quality filtering
  citation.py         Citation faithfulness check
  cost.py             Token/cost attribution
  tracing.py          Structured JSONL trace events
  benchmark.py        Reproducible benchmark harness
```

See `KNOWLEDGE_BASE.md` and `INTERVIEW_QA.md` for implementation notes and interview drill material.
