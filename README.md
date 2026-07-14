# DeepResearch Agent

一个故意收窄的 DeepResearch Agent：从用户问题出发，生成 research brief，拆分子问题，并发检索，去重和来源质量过滤后，合成带引用的结构化报告，并对引用做 faithfulness 检查。

默认运行不需要 API key。默认 LLM 和 search 都走 `mock`，用于稳定演示、测试和 mock plumbing benchmark；local retrieval 默认使用本地 BGE embedding 做 keyword + vector hybrid，不需要 API key。也可以显式切到 DeepSeek 真实 LLM provider、OpenAI-compatible LLM adapter、Wikipedia 真实无 key search adapter、Brave/Tavily/Jina 搜索 adapter、DashScope embedding/rerank provider，以及可选 Qdrant HTTP vector index。所有 key 只从环境变量读取：DeepSeek 用 `DEEPSEEK_API_KEY`，OpenAI-compatible adapter 默认读 `OPENAI_COMPATIBLE_API_KEY`（可配置为非必需，方便接本地 Ollama 兼容端点），DashScope 用 `DASHSCOPE_API_KEY`，Brave 用 `BRAVE_SEARCH_API_KEY`，Tavily 用 `TAVILY_API_KEY`，Jina 用 `JINA_API_KEY`，Qdrant 可选读 `QDRANT_API_KEY`。

## Quickstart

推荐使用 uv，Python 版本固定在 3.11：

```bash
uv python install 3.11
uv sync --extra dev --python 3.11
uv run deepresearch "How does citation checking reduce hallucination in agentic RAG?"
```

跨平台离线演示（不依赖 PowerShell，也不需要 API key）：

```bash
uv run python scripts/demo.py "How should a research agent recover from tool failures?"
```

Windows 也可以使用等价的 `py -3.11 -m pip install -e ".[dev]"` 和 `py -3.11 -m deepresearch_agent.cli ...`。

启动 API：

```bash
uv run deepresearch-api --host 127.0.0.1 --port 8000
```

macOS/Linux 可直接请求：

```bash
curl -sS -X POST http://127.0.0.1:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does a supervisor-researcher deep research agent differ from normal RAG?"}'
```

Windows PowerShell 等价命令：

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

Researcher 还支持有界证据循环。默认仍只执行一轮；需要演示预算停止时可以显式设置：

```bash
uv run deepresearch "How should citation grounding work?" --llm-provider mock --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --max-rounds 3 --max-tool-calls 2 --min-evidence-items 3
```

`fallback-policy` 有 `mock`、`degraded`、`fail` 三种模式。离线演示用 `mock`；普通 `/research` 和 CLI 的 live search 默认是 `degraded`（保留弱结果并标记原因）；benchmark 的 live search 默认是 `fail`，避免把 mock 来源混入 live 结果。

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

这个页面复用上面的 run control API，支持创建 mock run、查看最近 runs、编辑 planner subquestions、approve/reject/cancel，并展示 SSE events、report、sources 和 citation evidence。运行中取消是阶段边界协作式取消：系统会保留 `cancelled` 终态并释放 lease，但不会强制打断已经发出的 LLM/search 请求。`retry` 会优先复用 planner checkpoint；如果 researcher 已成功，还会复用 researcher checkpoint，避免后续阶段失败时重复检索。它是本地审核面，不包含登录权限或多人协作。

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

Lease 目前用于 SQLite 单机 worker ownership、stale run recovery 和取消后的 lease 清理，不是完整分布式任务队列。

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

输出会写入带 `manifest_id` 的 `logs/benchmark-<timestamp>-<manifest_id>.jsonl`、
`results/benchmarks/<manifest_id>/summary.json`，并同步更新兼容路径
`results/benchmark_summary.json`。后者是“最近一次运行”指针，不是不可变 baseline。

运行公开 Deep Research 端到端评测 artifact：

```powershell
py -3.11 -m deepresearch_agent.deep_research_eval --dataset livedrbench-preview --limit 1 --llm-provider mock --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1
```

运行 DeepSeek + Wikipedia 的公开评测口径：

```powershell
$env:DEEPSEEK_API_KEY="<your-key>"
py -3.11 -m deepresearch_agent.deep_research_eval --dataset livedrbench-preview --limit 1 --llm-provider deepseek --search-provider wikipedia --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --request-timeout-seconds 8
```

输出会写入 `logs/deep-research-eval-*.jsonl`、`results/deep_research_eval_summary.json` 和 `results/livedrbench_predictions.json`。当前脚本先产可复查 artifact 和 LiveDRBench-style predictions；可选 `--judge-provider heuristic` 会按 case 里的 `ground_truths` / `answer` / `expected_answer` 做本地字符串命中评分；`deepseek` 与 `llm-gateway` judge 会返回严格 JSON。答案判分固定为 `correct / incorrect / not_attempted / unscored`，始终以全部题目为分母；citation grounding 另算，不能用引用质量替代事实正确性。裁判实际返回模型不匹配、字段缺失或三次重试后仍无效时记为 `unscored`，不会静默跳过。官方 judge 尚未接入。

### SimpleQA 公开质量评测

`evals/simpleqa_public32_v1.jsonl` 是从 OpenAI `simple_qa_test_set.csv` 按固定 seed `20260714` 抽取的独立 32 题公开主集；source hash、上游 commit、抽样算法、源行号和 case hash 记录在同名 manifest。抽样同时平衡 topic（10 类，每类 3–4 题）与 answer type（5 类，每类 6–7 题）。历史 8 题集合只作为诊断回归集，不能代表整体 SimpleQA 质量。

重新生成公开主集：

```bash
python scripts/build_simpleqa_public32.py \
  --source-file /path/to/simple_qa_test_set.csv \
  --exclude-cases /path/to/simpleqa-diagnostic.jsonl \
  --output evals/simpleqa_public32_v1.jsonl \
  --manifest evals/simpleqa_public32_v1.manifest.json
```

正式生成使用 `--single-model-run --require-clean-worktree`，brief、planner、synthesis 与 Gateway web search 必须显式使用同一模型；生成期禁止配置外部 LLM judge。Gateway 搜索摘要只是候选信息，只有安全 HTML crawler 成功得到的正文才能进入 evidence、合成上下文和引用。失败候选只在 trace 中保存不含正文、query string 或原始错误的审计 hint。

离线检查固定 artifact 中答案事实在哪一层丢失：

```bash
python scripts/analyze_eval_snapshot.py --cases evals/simpleqa_public32_v1.jsonl \
  --artifact /path/to/raw.jsonl --output /path/to/snapshot-audit.json
```

该工具分别统计 gold URL 候选、可引用正文、snippet/crawl-failed 候选，以及 650/1200 token 打包上下文，避免把搜索摘要命中误写成证据命中。Gateway server-side web-search 能力可用 `scripts/probe_gateway_web_search.py` 生成不保留响应正文的 capability artifact。两份事后重判可用 `scripts/summarize_dual_judges.py` 合并；合并器会保留自评、分歧和 `unscored`，双判一致只称为一致性证据，不冒充官方真值。

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
- `wikipedia`：真实网络检索 adapter，调用 Wikipedia Search API；失败、超时或熔断时按 `fallback-policy` 选择 mock fallback、显式 degraded 或 fail，并在 trace/metrics 里暴露原因。
- `searxng`：真实 web search adapter，读取 `SEARXNG_BASE_URL` 或 CLI `--searxng-base-url` 指向自建/可访问 SearxNG 实例；可配 `WEB_CRAWLER_PROVIDER=jina` 把搜索结果 URL 交给 Jina Reader 抽正文。
- `jina`：Jina Search adapter，调用 `https://s.jina.ai/`；如配置 `JINA_API_KEY` 会带 Bearer token，未配置时仍会尝试公开 endpoint，失败会走 mock fallback。
- `brave`：Brave Search API adapter，调用 `https://api.search.brave.com/res/v1/web/search`，key 从 `BRAVE_SEARCH_API_KEY` 读。
- `tavily`：Tavily Search API adapter，调用 `https://api.tavily.com/search`，key 从 `TAVILY_API_KEY` 读；`TAVILY_SEARCH_DEPTH` 默认 `basic`。
- `mcp`：MCP tool search adapter，支持 `MCP_TRANSPORT=stdio/http`，用 `MCP_COMMAND`/`MCP_ARGS` 或 `MCP_HTTP_URL` 连接 server，并调用 `MCP_SEARCH_TOOL`；tool result 会转换成统一 `Source`。

Search reliability knobs：

- `SEARCH_RATE_LIMIT_PER_SECOND`：默认 `0` 关闭；设置为正数后，`SearchService` 会在 primary search 调用前做进程内节流，mock fallback 不限流。
- `SEARCH_RETRY_BACKOFF_SECONDS`：默认 `0` 关闭；设置为正数后，primary search 失败重试前按 `base * 2^attempt` 等待。
- `MAX_RETRIES`、`REQUEST_TIMEOUT_SECONDS`、`CIRCUIT_BREAKER_FAILURE_THRESHOLD`、`CIRCUIT_BREAKER_COOLDOWN_SECONDS` 控制 retry、timeout、circuit breaker 和 fallback 行为。

Crawler：

- `none`：默认，不抽网页正文，只使用 search snippet。
- `jina` / `jina_reader`：调用 `https://r.jina.ai/<url>` 抽取 URL 的 LLM-friendly 文本；可用 `JINA_READER_BASE_URL`、`JINA_API_KEY`、`CRAWLER_MAX_CHARS` 配置。
- `html`：本地无 key HTML 正文抽取，直接抓 URL，用标准库 HTMLParser 去掉 script/style/noscript/svg 后抽取可读文本；适合无 Jina key 的 smoke，不是完整浏览器渲染 crawler。

Local retrieval：

- `keyword`：旧基线，只按本地语料 token overlap 排序。
- `hybrid`：默认本地检索模式，关键词召回 + BGE 向量召回 + Chroma index + RRF 融合，仍输出统一 `Source`。如果 Chroma、embedding 或 vector search 不可用，会降级为 keyword-only，并在 source metadata 里记录 `retrieval_degraded=True` 和 `degrade_reason`。
- embedding provider 默认 `local`，模型是 `BAAI/bge-small-zh-v1.5`；可显式切到 `dashscope`，key 从 `DASHSCOPE_API_KEY` 读。
- rerank provider 默认 `local`，模型是 `BAAI/bge-reranker-base`，但 `--rerank-enabled` 才会启用；也可以显式切到 DashScope rerank。
- 持久化向量索引默认关闭；设置 `LOCAL_VECTOR_INDEX_PERSIST=true` 后会把 Chroma index 写到 `LOCAL_VECTOR_INDEX_PATH`（默认 `data/vector_index`，已 gitignore），按 corpus + embedding provider + model 指纹复用 collection。
- vector index provider 默认 `chroma`；可设置 `LOCAL_VECTOR_INDEX_PROVIDER=qdrant` 或 CLI `--local-vector-index-provider qdrant`，通过 `QDRANT_BASE_URL`、`QDRANT_COLLECTION`、`QDRANT_API_KEY_ENV` 访问 Qdrant HTTP API。当前只做 stub 单测，未做真实 Qdrant live benchmark。

Citation judge：

- `none`：默认，只使用 lexical citation checker，保持 benchmark 旧口径。
- `heuristic`：本地无 key judge，基于 evidence quote overlap 给出 supported / partial / unsupported / unverifiable。
- `deepseek`：可选 LLM citation judge，默认模型 `deepseek-v4-flash`，可用 `CITATION_JUDGE_MODEL` 或 CLI `--citation-judge-model` 覆盖；需要 `DEEPSEEK_API_KEY`。

Answer judge：

- `none`：默认，public eval 只产 artifacts，不做 answer-quality scoring。
- `heuristic`：本地无 key answer judge，只检查 ground-truth 字符串是否出现在答案里。
- `deepseek`：可选 LLM answer judge，使用 DeepSeek JSON mode，默认沿用生效的 DeepSeek 模型，也可用 CLI `--judge-model` 覆盖；需要 `DEEPSEEK_API_KEY`。这不是官方 LiveDRBench/Deep Research Bench judge。

## Interview-quality evaluation and replay

`data/benchmark_cases.jsonl` 现在包含 24 条离线回归题，覆盖英文/中文、单事实、对比、多跳、引用冲突、工具失败和 JSON 输出。运行：

```bash
LLM_PROVIDER=mock SEARCH_PROVIDER=mock LOCAL_RETRIEVAL_MODE=keyword \
  uv run deepresearch-benchmark --benchmark-name interview-baseline --max-researchers 1 --max-results 1
```

benchmark 会区分 `execution_success`、`task_format_valid`、`answer_quality`、`citation_grounding`、`citation_coverage`、`unsupported_claim_rate`、`source_quality`、`tool_failure_recovery`、延迟、token 和成本。没有显式 answer judge 时，`answer_quality` 保持 `null`；旧的 `success` 字段仅作为兼容性的执行成功别名，不代表答案正确。

每次运行的 manifest 会记录 commit SHA、数据集 hash、prompt bundle hash、provider/model、配置快照、确定性说明和 replay 标识；若工作区不干净，还会把 staged/unstaged diff 与未跟踪文件内容汇总成只存摘要的 `git_worktree_hash`，避免同一 commit 下的本地改动覆盖同一 manifest。可以用上一轮 JSONL case artifact 离线重放：

```bash
uv run deepresearch-benchmark --cases data/benchmark_cases.jsonl \
  --replay-dir logs/previous-benchmark.jsonl
```

这个 `--replay-dir` 入口是严格的 case-result snapshot replay：它复用报告、来源和 claims，并重新计算当前版本可重算的评测指标；它不重新调用 LLM/search。

这里有两种故意分开的 replay：

- `--replay-dir` 是 benchmark snapshot replay。它读取一次运行保存的 case-result artifact，跳过 LLM/search，再用当前版本的格式和 citation evaluator 重算可重算指标；旧指标保留在 `recorded_*` 字段中，不能把它当成重新调用 provider。
- provider cassette 是 `deepresearch_agent.replay` 中的 `CassetteLLMProvider` / `CassetteSearchAdapter`。它按严格的 kind、operation、request 顺序重新执行 orchestrator，未消费完或请求漂移都会失败；当前提供 fixture 和 `deepresearch-cassette` 检查命令，还没有自动录制真实 HTTP 流量的 recorder。

当前最终离线 24-case baseline（manifest `949aab1f58a8e33c`）只用于冻结可靠性口径：执行成功率 `1.0`、结构化格式有效率 `0.875`、`answer_quality=null`（未配置 judge）、citation grounding `0.5417`、citation precision `0.5417`、citation coverage `1.0`、unsupported claim rate `0.4583`、claim extraction 有效率 `1.0`、source quality `0.9`、工具失败恢复适用 3 条且平均 `1.0`、fallback 总数 `3`。这是 deterministic mock/keyword plumbing 结果，不是真实答案质量提升；manifest 记录了代码提交 `07a624b` 且 `git_dirty=false`，后续版本必须使用同一题集和同一指标字段比较。

产物位置：`results/benchmarks/949aab1f58a8e33c/summary.json` 和对应的
`logs/benchmark-20260712T185935Z-949aab1f58a8e33c.jsonl`。审查前的 `e515b3e6d8bca06f` 与审查后、最终指纹修复前的 `cfc88658446ed557` 仍按 manifest 分目录保留，便于追溯口径变化。

## 五分钟面试演示路径

1. 用上面的 `scripts/demo.py` 展示正常问题、plan、sources、citation 和 cost。
2. 运行 `uv run python scripts/demo.py --scenario failure`，强制 search timeout，展示 retry 后的 `researcher.Q1=fallback`、`degraded_count=1` 和 trace；再跑 24-case benchmark 查看 3 条固定 failure-injection case。
3. 启动 API，创建 `require_approval=true` 的 `/runs`，查看 planner step，编辑或 approve，再查看 `/runs/<run_id>/trace` 和 SSE events。
4. 在报告 JSON 的 `citation_check.assessments[].evidence_quotes` 中展示 claim、source ID、quote、URL、抓取时间和 `extract_status`，同时查看 `cost.records`。
5. 保存一次 summary 后，用同一命令的 `--replay-dir` 做 snapshot replay；如需演示真正重新跑主链，使用 provider cassette fixture，并明确它和 snapshot replay 的区别。没有第二个经过同一题集/口径生成的版本时，不宣称质量提升，只比较可靠性与可观测性。

## 能力成熟度矩阵

| 能力 | 当前状态 | 口径说明 |
|---|---|---|
| Mock/keyword 主链、24-case 回归 | replay-tested | CI、pytest 和 deterministic baseline 可离线重放 |
| bounded researcher action contract、中文 citation | replay-tested | 有预算边界、evidence quote 和固定回归；不是原生 function calling |
| provider cassette orchestrator replay | replay-tested | fixture 会重新执行 orchestrator，并严格检查请求顺序/消费完毕 |
| snapshot benchmark replay | replay-tested | 复用已保存报告并按当前 evaluator 重算格式/citation 指标 |
| DeepSeek + Wikipedia | live-tested | 历史小样本曾真实跑通；当前环境无 key，不把它当本轮新结果 |
| Jina Reader / HTML crawler | live-tested / stub-tested | 有少量 URL smoke；没有生产级网页质量评测 |
| Brave、Tavily、SearxNG、MCP | stub-tested | 请求、解析或 fake client 单测；没有稳定 live benchmark |
| OpenAI-compatible、DashScope、Qdrant | stub-tested | adapter/HTTP contract 已测；真实 endpoint 未验证 |
| SQLite run control、lease、SSE、checkpoint | replay-tested | 本地集成测试覆盖；不是分布式队列 |
| citation judge、answer judge | stub-tested | heuristic/DeepSeek 请求和 schema 有测试；没有 live judge 结论 |
| PDF/DOCX/PPTX/WAV 导出、`/ui` | optional/demo-only | 面试展示能力，非本阶段主线或生产级交付 |

矩阵中的 `live-tested` 只表示至少有一次真实 provider/URL 路径运行记录，不表示 SLA、生产质量或安全性；`replay-tested`、`stub-tested` 和 `optional/demo-only` 都不应写成 live 成果。

## Limitations / Future work

当前完整回归：`.venv/bin/python -m pytest -q` 通过，`308 passed, 1 warning`。warning 来自 FastAPI/Starlette 的 TestClient deprecation，未影响功能；ruff、compileall 和 `git diff --check` 同样通过。默认 hybrid 需要首次下载本地 embedding 模型；CI 和离线演示显式使用 keyword，避免把模型下载耗时混入功能回归。

实测口径需要严格区分：mock benchmark 只证明离线路径、trace、citation ID 和记录链路能跑，不能当真实性能、真实成本或真实答案质量成果；DeepSeek v4-flash + Wikipedia benchmark 是真实 provider 小样本，延迟包含网络/API 时间，citation retention 仍是 lexical checker 口径，不是语义级事实评分。

当前主要限制：

- 真实 LLM 路径主要实测 DeepSeek v4-flash；OpenAI-compatible adapter 只有 stub/路由测试，OpenAI/Anthropic 原生 provider 尚未实现。
- Wikipedia adapter 能真实跑，但不是生产级搜索；Brave/Tavily 需要 API key，当前只做请求/解析单测；SearxNG 需要自建 endpoint。
- Local hybrid retrieval 已实现并有 BEIR/scifact 检索评测，但历史 5-case 端到端小样本里旧 success_rate 低于 keyword baseline，不能包装成质量稳定提升，也不能与当前 24-case 指标横比。
- DashScope embedding/rerank provider 已有 stub 测试，本机未配置 `DASHSCOPE_API_KEY`，所以没有真实百炼延迟、费用或质量数字。
- Qdrant HTTP vector index 只有 stub HTTP 单测，没有真实 Qdrant live benchmark、索引生命周期、权限过滤或增量 reindex。
- Citation checker 默认仍是 lexical grounding；heuristic/DeepSeek judge 是可选扩展，DeepSeek judge 尚未做 live benchmark，也不是官方 LiveDRBench/Deep Research Bench judge。
- Researcher 的 bounded loop 使用结构化 action contract；当前 DeepSeek/Mock provider 都没有声明原生 function calling，不能把它描述成已经接入模型 tool-call loop。
- snapshot replay 复用完整 case artifact；provider cassette fixture 能重新执行主链，但当前没有自动录制真实 LLM/search HTTP 流量的 recorder，live 输出仍可能受网络和模型版本影响。
- Run control 当前是 SQLite 单机版本；没有 Redis/Postgres/Celery worker pool、跨进程强制取消、权限系统或多人协作。
- Report export 已支持 Markdown/HTML/JSON/PDF/DOCX/PPTX/WAV，但都是文本版交付；WAV 是 Windows SAPI 摘要，不是完整 podcast/TTS 制作系统。

## Project Layout

```text
src/deepresearch_agent/
  api.py              FastAPI + SSE endpoints
  orchestrator.py     End-to-end research spine
  execution.py        Shared clarify/planner/researcher/synthesis/verifier stage runner
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
  text_utils.py       Multilingual tokenization and sentence splitting
  report_metrics.py   Execution/quality metric separation
  replay.py           Strict JSONL cassette and case-artifact replay helpers
  citation_judge.py   Optional heuristic and DeepSeek citation judge providers
  report_exporter.py  Markdown/HTML/JSON/PDF/DOCX/PPTX/WAV report exports
  tts.py              Optional Windows SAPI TTS provider
  cost.py             Token/cost attribution
  tracing.py          Structured JSONL trace events and optional OTLP HTTP export
  benchmark.py        Reproducible benchmark harness
  deep_research_eval.py Public end-to-end Deep Research eval artifact runner
  eval_judge.py       Optional heuristic and DeepSeek answer judges for public eval cases
```

See `KNOWLEDGE_BASE.md` and `INTERVIEW_QA.md` for implementation notes and interview drill material.
