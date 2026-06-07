# DeepResearch Agent

一个故意收窄的 DeepResearch Agent：从用户问题出发，生成 research brief，拆分子问题，并发检索，去重和来源质量过滤后，合成带引用的结构化报告，并对引用做 faithfulness 检查。

默认运行不需要 API key。默认 LLM 和 search 都走 `mock`，用于稳定演示、测试和 mock plumbing benchmark；local retrieval 默认使用本地 BGE embedding 做 keyword + vector hybrid，不需要 API key。也可以显式切到 DeepSeek 真实 LLM provider、OpenAI-compatible LLM adapter、Wikipedia 真实无 key search adapter、Brave/Tavily/Jina 搜索 adapter、DashScope embedding/rerank provider，以及可选 Qdrant HTTP vector index。所有 key 只从环境变量读取：DeepSeek 用 `DEEPSEEK_API_KEY`，OpenAI-compatible adapter 默认读 `OPENAI_COMPATIBLE_API_KEY`（可配置为非必需，方便接本地 Ollama 兼容端点），DashScope 用 `DASHSCOPE_API_KEY`，Brave 用 `BRAVE_SEARCH_API_KEY`，Tavily 用 `TAVILY_API_KEY`，Jina 用 `JINA_API_KEY`，Qdrant 可选读 `QDRANT_API_KEY`。

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

按阶段覆盖模型名：

```powershell
py -3.11 -m deepresearch_agent.cli "How should model routing work in a research agent?" --llm-provider mock --llm-model mock-default --brief-model mock-brief --planner-model mock-planner --synthesis-model mock-synthesis --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --json
```

真实 DeepSeek 路径也支持同一组参数和环境变量：`LLM_BRIEF_MODEL`、`LLM_PLANNER_MODEL`、`LLM_SYNTHESIS_MODEL`。未设置时全部回落到 `--llm-model` / `DEEPSEEK_MODEL`。

导出报告文件：

```powershell
py -3.11 -m deepresearch_agent.cli "How should report export work?" --llm-provider mock --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --export-dir reports --export-formats markdown,html,json,pdf,docx,pptx,wav
```

当前导出支持 Markdown、HTML、JSON、PDF、DOCX、PPTX、WAV；PDF/DOCX/PPTX 是文本版报告导出，保留 answer、sources、citation assessments 和 evidence quotes。WAV 通过本机 Windows SAPI 把报告摘要转成语音，不是完整 podcast 制作系统。

从本地 Markdown/TXT/PDF/DOCX 文件夹生成私有知识库 JSONL：

```powershell
py -3.11 -m deepresearch_agent.ingest_corpus .\my_notes --output data\local_corpus.jsonl --json
```

安装 editable package 后，也可以用 console script：`deepresearch-ingest-corpus .\my_notes --output data\local_corpus.jsonl --json`。生成的 JSONL 可被 `LocalRagRetriever` 的 keyword / hybrid 两种模式直接读取；默认会跳过 `.git`、`.obsidian`、`.claude`、`node_modules` 和 `__pycache__`。PDF 通过 `pypdf` 抽取文本，DOCX 通过 `python-docx` 抽取段落和表格文本；扫描件 OCR 不在当前范围内。

可选导出 OpenTelemetry OTLP HTTP trace：

```powershell
$env:TRACE_EXPORTER="otlp_http"
$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4318"
$env:OTEL_SERVICE_NAME="deepresearch-agent"
py -3.11 -m deepresearch_agent.cli "How should trace export work?" --llm-provider mock --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1
```

默认 `TRACE_EXPORTER=jsonl`，只写本地 `logs/research-<run_id>.jsonl`。`otlp_http` 会把每个 trace event 额外 POST 到 `<endpoint>/v1/traces`；export 失败只写本地 `trace_exporter` error event，不中断 research run。

可选启用 citation judge：

```powershell
py -3.11 -m deepresearch_agent.cli "How should citation judges work?" --llm-provider mock --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --citation-judge-provider heuristic
```

默认 `CITATION_JUDGE_PROVIDER=none`，保持原来的 lexical citation checker 口径。`heuristic` 是本地无 key judge；`deepseek` 会调用 DeepSeek JSON mode，key 仍只从 `DEEPSEEK_API_KEY` 读取，并把 judge usage 计入 `citation_judge` 成本阶段。

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

打开本地审核页面：

```text
http://127.0.0.1:8000/ui
```

这个页面复用上面的 run control API，支持创建 mock run、查看最近 runs、编辑 planner subquestions、approve/reject/cancel，并展示 SSE events、report、sources 和 citation evidence。它是本地审核面，不包含登录权限或多人协作。

创建后先排队，再由单次 worker 消费下一条 queued run：

```powershell
$deferredBody = '{"query":"How should deferred agent runs work?","search_provider":"mock","llm_provider":"mock","max_researchers":1,"max_results_per_researcher":1,"require_approval":false,"defer_execution":true}'
$queued = Invoke-RestMethod -Method Post http://127.0.0.1:8000/runs -ContentType "application/json" -Body $deferredBody
Invoke-RestMethod -Method Post http://127.0.0.1:8000/runs/worker/next
```

`defer_execution=true` 只影响 run control 创建路径：系统会把本次请求快照存到 SQLite 的 `request_json`，worker 执行时从该快照恢复 provider、模型、并发数和检索参数。`/runs/worker/next` 是 SQLite 单机 worker-once 入口，当前不是 Redis/Celery worker pool。

也可以启动本地轮询 worker：

```powershell
py -3.11 -m deepresearch_agent.run_worker --max-runs 1 --idle-exit --json
```

安装 editable package 后，也可以用 console script：`deepresearch-worker --max-runs 1 --idle-exit --json`。不传 `--max-runs` 时会持续轮询，直到手动中断。

手动验证 worker lease / heartbeat：

```powershell
$leaseBody = '{"worker_id":"demo-worker","lease_seconds":120}'
Invoke-RestMethod -Method Post http://127.0.0.1:8000/runs/$($run.run_id)/lease -ContentType "application/json" -Body $leaseBody
Invoke-RestMethod -Method Post http://127.0.0.1:8000/runs/$($run.run_id)/heartbeat -ContentType "application/json" -Body $leaseBody
Invoke-RestMethod http://127.0.0.1:8000/runs/stale
Invoke-RestMethod -Method Post http://127.0.0.1:8000/runs/recover-stale -ContentType "application/json" -Body '{"reason":"manual stale lease recovery"}'
```

Lease 目前用于 SQLite 单机 worker ownership 和 stale run recovery，不是完整分布式任务队列。

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

输出会写入 `logs/deep-research-eval-*.jsonl`、`results/deep_research_eval_summary.json` 和 `results/livedrbench_predictions.json`。当前脚本先产可复查 artifact 和 LiveDRBench-style predictions；可选 `--judge-provider heuristic` 会按 case 里的 `ground_truths` / `answer` / `expected_answer` 做本地字符串命中评分。官方 judge/answer-quality scoring 尚未接入。

运行 retrieval 对比 benchmark：

```powershell
$env:REQUEST_TIMEOUT_SECONDS="8"
py -3.11 -m deepresearch_agent.benchmark --llm-provider deepseek --llm-model deepseek-v4-flash --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3 --local-retrieval-mode keyword
py -3.11 -m deepresearch_agent.benchmark --llm-provider deepseek --llm-model deepseek-v4-flash --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3 --local-retrieval-mode hybrid --embedding-provider local
```

## Providers

LLM providers：

- `mock`：默认 LLM provider，离线可复现，用于测试和 mock plumbing benchmark。
- `deepseek`：真实 OpenAI-compatible LLM provider，默认模型 `deepseek-v4-flash`，使用 DeepSeek JSON mode 生成 brief、plan 和 cited synthesis；需要 `DEEPSEEK_API_KEY`，可用 `DEEPSEEK_MODEL` 覆盖默认模型，也可用 `LLM_BRIEF_MODEL` / `LLM_PLANNER_MODEL` / `LLM_SYNTHESIS_MODEL` 做阶段模型覆盖。
- `openai-compatible`：通用 OpenAI-compatible Chat Completions adapter，默认 base URL `http://localhost:11434/v1`、默认模型 `llama3.1`，适合接本地 Ollama 或 OpenRouter 等兼容端点。配置项：`OPENAI_COMPATIBLE_BASE_URL`、`OPENAI_COMPATIBLE_MODEL`、`OPENAI_COMPATIBLE_API_KEY_ENV`、`OPENAI_COMPATIBLE_API_KEY_REQUIRED`、`OPENAI_COMPATIBLE_INPUT_COST_PER_1M_TOKENS`、`OPENAI_COMPATIBLE_OUTPUT_COST_PER_1M_TOKENS`。当前只做了 stub 单测，未做 live benchmark。

Search providers：

- `mock`：默认 search provider，离线可复现。
- `wikipedia`：真实网络检索 adapter，调用 Wikipedia Search API；失败、超时或熔断时自动使用 mock fallback，并在 trace/metrics 里暴露 fallback。
- `searxng`：真实 web search adapter，读取 `SEARXNG_BASE_URL` 或 CLI `--searxng-base-url` 指向自建/可访问 SearxNG 实例；可配 `WEB_CRAWLER_PROVIDER=jina` 把搜索结果 URL 交给 Jina Reader 抽正文。
- `jina`：Jina Search adapter，调用 `https://s.jina.ai/`；如配置 `JINA_API_KEY` 会带 Bearer token，未配置时仍会尝试公开 endpoint，失败会走 mock fallback。
- `brave`：Brave Search API adapter，调用 `https://api.search.brave.com/res/v1/web/search`，key 从 `BRAVE_SEARCH_API_KEY` 读。
- `tavily`：Tavily Search API adapter，调用 `https://api.tavily.com/search`，key 从 `TAVILY_API_KEY` 读；`TAVILY_SEARCH_DEPTH` 默认 `basic`。
- `mcp`：MCP tool search adapter，支持 `MCP_TRANSPORT=stdio/http`，用 `MCP_COMMAND`/`MCP_ARGS` 或 `MCP_HTTP_URL` 连接 server，并调用 `MCP_SEARCH_TOOL`；tool result 会转换成统一 `Source`。

Crawler：

- `none`：默认，不抽网页正文，只使用 search snippet。
- `jina` / `jina_reader`：调用 `https://r.jina.ai/<url>` 抽取 URL 的 LLM-friendly 文本；可用 `JINA_READER_BASE_URL`、`JINA_API_KEY`、`CRAWLER_MAX_CHARS` 配置。
- `html`：本地无 key HTML 正文抽取，直接抓 URL，用标准库 HTMLParser 去掉 script/style/noscript/svg 后抽取可读文本；适合无 Jina key 的 smoke，不是完整浏览器渲染 crawler。

Local retrieval：

- `keyword`：旧基线，只按本地语料 token overlap 排序。
- `hybrid`：默认本地检索模式，关键词召回 + BGE 向量召回 + Chroma index + RRF 融合，仍输出统一 `Source`。
- embedding provider 默认 `local`，模型是 `BAAI/bge-small-zh-v1.5`；可显式切到 `dashscope`，key 从 `DASHSCOPE_API_KEY` 读。
- rerank provider 默认 `local`，模型是 `BAAI/bge-reranker-base`，但 `--rerank-enabled` 才会启用；也可以显式切到 DashScope rerank。
- 持久化向量索引默认关闭；设置 `LOCAL_VECTOR_INDEX_PERSIST=true` 后会把 Chroma index 写到 `LOCAL_VECTOR_INDEX_PATH`（默认 `data/vector_index`，已 gitignore），按 corpus + embedding provider + model 指纹复用 collection。
- vector index provider 默认 `chroma`；可设置 `LOCAL_VECTOR_INDEX_PROVIDER=qdrant` 或 CLI `--local-vector-index-provider qdrant`，通过 `QDRANT_BASE_URL`、`QDRANT_COLLECTION`、`QDRANT_API_KEY_ENV` 访问 Qdrant HTTP API。当前只做 stub 单测，未做真实 Qdrant live benchmark。

Citation judge：

- `none`：默认，只使用 lexical citation checker，保持 benchmark 旧口径。
- `heuristic`：本地无 key judge，基于 evidence quote overlap 给出 supported / partial / unsupported / unverifiable。
- `deepseek`：可选 LLM citation judge，默认模型 `deepseek-v4-flash`，可用 `CITATION_JUDGE_MODEL` 或 CLI `--citation-judge-model` 覆盖；需要 `DEEPSEEK_API_KEY`。

## Project Layout

```text
src/deepresearch_agent/
  api.py              FastAPI + SSE endpoints
  orchestrator.py     End-to-end research spine
  run_control.py      Run state machine, approval, cancel, retry
  run_store.py        SQLite run/step/event checkpoint store
  run_models.py       Pydantic models for run control API
  run_worker.py       Local polling worker for queued runs
  llm.py              Mock and DeepSeek structured-output model providers
  search.py           Search adapters, retry, timeout, circuit breaker, fallback
  rag.py              Local keyword/vector hybrid RAG retriever, Chroma/Qdrant index adapters
  ingest_corpus.py    Markdown/TXT/PDF/DOCX to local JSONL corpus ingestion
  embeddings.py       Local and DashScope embedding providers
  rerankers.py        Optional local and DashScope rerank providers
  verifier.py         Source quality filtering
  source_metrics.py   Source provider/domain diversity metrics
  citation.py         Citation faithfulness check
  citation_judge.py   Optional heuristic and DeepSeek citation judge providers
  report_exporter.py  Markdown/HTML/JSON/PDF/DOCX/PPTX/WAV report exports
  tts.py              Optional Windows SAPI TTS provider
  cost.py             Token/cost attribution
  tracing.py          Structured JSONL trace events and optional OTLP HTTP export
  benchmark.py        Reproducible benchmark harness
  deep_research_eval.py Public end-to-end Deep Research eval artifact runner
  eval_judge.py       Optional heuristic answer judge for public eval cases
```

See `KNOWLEDGE_BASE.md` and `INTERVIEW_QA.md` for implementation notes and interview drill material.
