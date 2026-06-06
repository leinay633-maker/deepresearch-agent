# DeepResearch Agent

一个故意收窄的 DeepResearch Agent：从用户问题出发，生成 research brief，拆分子问题，并发检索，去重和来源质量过滤后，合成带引用的结构化报告，并对引用做 faithfulness 检查。

默认运行不需要 API key。`mock` provider 用于稳定演示和 benchmark；`wikipedia` provider 是真实无 key 网络检索 adapter，网络失败时会降级到 mock。

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

输出会写入 `logs/benchmark-*.jsonl` 和 `results/benchmark_summary.json`。

## Providers

- `mock`：默认 provider，离线可复现，用于测试和 benchmark。
- `wikipedia`：真实网络检索 adapter，调用 Wikipedia Search API；失败、超时或熔断时自动使用 mock fallback。

## Project Layout

```text
src/deepresearch_agent/
  api.py              FastAPI + SSE endpoints
  orchestrator.py     End-to-end research spine
  llm.py              Mock structured-output/tool-capable model provider
  search.py           Search adapters, retry, timeout, circuit breaker, fallback
  rag.py              Local keyword RAG retriever
  verifier.py         Source quality filtering
  citation.py         Citation faithfulness check
  cost.py             Token/cost attribution
  tracing.py          Structured JSONL trace events
  benchmark.py        Reproducible benchmark harness
```

See `KNOWLEDGE_BASE.md` and `INTERVIEW_QA.md` for implementation notes and interview drill material.
