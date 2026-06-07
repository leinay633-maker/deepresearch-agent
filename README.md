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

启用一次 bounded reflection loop：

```powershell
py -3.11 -m deepresearch_agent.cli "How should citation grounding work in a research agent?" --search-provider mock --llm-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --reflection-enabled --max-reflection-rounds 1 --reflection-min-sources 4
```

默认不开启 reflection；开启后系统会先压缩已有 findings，再按启发式判断是否追加 1 个 follow-up subquestion，相关 `compression.roundN` 和 `reflection.roundN` 会进入 trace。

创建可审核的长任务 run，并在 planner 后 approve 继续：

```powershell
$body = '{"query":"How should agent run control reduce wasted LLM cost?","search_provider":"mock","llm_provider":"mock","max_researchers":1,"max_results_per_researcher":1,"require_approval":true}'
$run = Invoke-RestMethod -Method Post http://127.0.0.1:8000/runs -ContentType "application/json" -Body $body
$run.run_id
Invoke-RestMethod http://127.0.0.1:8000/runs/$($run.run_id)
Invoke-RestMethod http://127.0.0.1:8000/runs/$($run.run_id)/steps
Invoke-RestMethod -Method Post http://127.0.0.1:8000/runs/$($run.run_id)/approve
Invoke-RestMethod http://127.0.0.1:8000/runs/$($run.run_id)/trace
```

订阅持久化 run 事件，断线后可用 `Last-Event-ID` 续传：

```powershell
curl.exe -N http://127.0.0.1:8000/runs/<run_id>/events
curl.exe -N http://127.0.0.1:8000/runs/<run_id>/events -H "Last-Event-ID: 2"
```

Run control 默认 SQLite 文件是 `data/runs.sqlite`，可用 `RUN_STORE_PATH` 覆盖。默认路径已被 `.gitignore` 忽略。

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

运行公开 Deep Research 端到端评测 artifact：

```powershell
py -3.11 -m deepresearch_agent.deep_research_eval --dataset livedrbench-preview --limit 1 --llm-provider mock --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1
```

运行 DeepSeek + Wikipedia 的公开评测口径：

```powershell
$env:DEEPSEEK_API_KEY="<your-key>"
py -3.11 -m deepresearch_agent.deep_research_eval --dataset livedrbench-preview --limit 1 --llm-provider deepseek --search-provider wikipedia --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --request-timeout-seconds 8
```

输出会写入 `logs/deep-research-eval-*.jsonl`、`results/deep_research_eval_summary.json` 和 `results/livedrbench_predictions.json`。当前脚本先产可复查 artifact 和 LiveDRBench-style predictions；官方 judge/answer-quality scoring 尚未接入。

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
- `searxng`：真实 web search adapter，读取 `SEARXNG_BASE_URL` 或 CLI `--searxng-base-url` 指向自建/可访问 SearxNG 实例；可配 `WEB_CRAWLER_PROVIDER=jina` 把搜索结果 URL 交给 Jina Reader 抽正文。
- `jina`：Jina Search adapter，调用 `https://s.jina.ai/`；如配置 `JINA_API_KEY` 会带 Bearer token，未配置时仍会尝试公开 endpoint，失败会走 mock fallback。
- `mcp`：MCP tool search adapter，支持 `MCP_TRANSPORT=stdio/http`，用 `MCP_COMMAND`/`MCP_ARGS` 或 `MCP_HTTP_URL` 连接 server，并调用 `MCP_SEARCH_TOOL`；tool result 会转换成统一 `Source`。

Crawler：

- `none`：默认，不抽网页正文，只使用 search snippet。
- `jina` / `jina_reader`：调用 `https://r.jina.ai/<url>` 抽取 URL 的 LLM-friendly 文本；可用 `JINA_READER_BASE_URL`、`JINA_API_KEY`、`CRAWLER_MAX_CHARS` 配置。

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
  run_control.py      Run state machine, approval, cancel, retry
  run_store.py        SQLite run/step/event checkpoint store
  run_models.py       Pydantic models for run control API
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
  deep_research_eval.py Public end-to-end Deep Research eval artifact runner
```

See `KNOWLEDGE_BASE.md` and `INTERVIEW_QA.md` for implementation notes and interview drill material.
