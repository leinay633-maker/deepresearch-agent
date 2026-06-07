# DeepResearch Agent Review Package

这是给外部 AI / 网页版 AI 审阅本项目进展用的交接说明。请优先看 `KNOWLEDGE_BASE.md`，再看代码和测试。

## 当前完成状态

- 已从空仓库实现一个可运行的 DeepResearch Agent MVP。
- 已实现主链路：用户问题 -> clarify/normalize -> research brief -> planner 拆子问题 -> 3 个 researcher 并发检索 -> source dedup -> verifier -> synthesizer 带引用合成 -> citation check -> structured report。
- 已实现 FastAPI JSON 接口 `/research` 和 SSE 接口 `/research/stream`。
- 已实现 mock LLM/search provider、DeepSeek 真实 LLM provider、Wikipedia 真实无 key search adapter。
- 已实现本地 hybrid retrieval：关键词召回 + BGE 向量召回 + Chroma index + RRF 融合，并保留 keyword baseline；rerank provider 可选开启。
- 已实现工具失败处理：retry、timeout、circuit breaker、fallback。
- 已实现结构化 trace、阶段级 token/cost 归因、benchmark harness；DeepSeek 路径记录 provider 返回的真实 usage，并按当前实现里的价格常量估算成本。
- 已生成 `KNOWLEDGE_BASE.md` 和 `INTERVIEW_QA.md`。

## 优先审阅文件

1. `KNOWLEDGE_BASE.md`：项目复盘、真实问题、实测数据、和参考项目差异。
2. `INTERVIEW_QA.md`：面试检索和 drill 用问答。
3. `src/deepresearch_agent/orchestrator.py`：端到端编排主线。
4. `src/deepresearch_agent/search.py`：search adapter、retry、timeout、circuit breaker、fallback。
5. `src/deepresearch_agent/citation.py`：citation faithfulness 第一层检查。
6. `src/deepresearch_agent/verifier.py`：source quality filtering。
7. `src/deepresearch_agent/benchmark.py`：可复现 benchmark。
8. `tests/`：当前自动化测试。

## 实测结果

测试命令：

```powershell
py -3.11 -m pytest -q
```

最近一次结果：`20 passed, 2 warnings`。warning 是 FastAPI TestClient / Starlette 关于 httpx 的 deprecation 提示，以及 Chroma/OpenTelemetry 的 deprecation 提示，未影响功能。

benchmark 命令：

```powershell
$env:REQUEST_TIMEOUT_SECONDS='8'
py -3.11 -m deepresearch_agent.benchmark --llm-provider deepseek --llm-model deepseek-v4-flash --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3 --local-retrieval-mode keyword
py -3.11 -m deepresearch_agent.benchmark --llm-provider deepseek --llm-model deepseek-v4-flash --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3 --local-retrieval-mode hybrid --embedding-provider local
```

当前 `results/benchmark_summary.json` 是最后一次 local hybrid run；retrieval 对比汇总在 `results/retrieval_benchmark_comparison.json`。keyword baseline 原始记录在 `logs/benchmark-20260607T080835Z.jsonl`，local hybrid 原始记录在 `logs/benchmark-20260607T081104Z.jsonl`。

核心实测指标解释：下面这些是真实 provider local benchmark 记录，适合证明本机这次配置下 LLM provider、search adapter、usage/cost 记录和 citation checker 已经端到端跑通；它仍不是线上 SLA 或广义质量分数。延迟包含 DeepSeek 和 Wikipedia 网络/API 时间；citation_retention_rate 是 lexical checker 结果，不是语义级事实评估。

- keyword baseline: success_rate 1.0, citation_retention_rate_avg 0.8867, p50 24595.506ms, total_tokens 17740, estimated_cost_usd_total 0.00368858, fallback_count_total 1
- local hybrid: success_rate 0.6, citation_retention_rate_avg 0.8929, p50 30747.284ms, total_tokens 18842, estimated_cost_usd_total 0.00399994, fallback_count_total 2

对比用 mock plumbing 原始记录仍保留在 `logs/benchmark-20260606T152954Z.jsonl`。mock 数字只证明离线路径和记录链路能跑，不能当真实性能、真实成本或真实答案质量成果。

## 运行方式

安装：

```powershell
py -3.11 -m pip install --timeout 180 -e ".[dev]"
```

CLI：

```powershell
py -3.11 -m deepresearch_agent.cli "How does citation checking reduce hallucination in agentic RAG?"
```

API：

```powershell
py -3.11 -m deepresearch_agent.api --host 127.0.0.1 --port 8000
```

health check：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## 明确没做或未实测

- 只接了 DeepSeek 一个真实 LLM provider；OpenAI/Anthropic 等其他 provider 未接。
- DeepSeek 已实测真实 token usage 和 cost，但没有做长时间多轮稳定性/限流压测。
- 当前默认 DeepSeek 模型已切到显式 `deepseek-v4-flash`；legacy alias 只为兼容旧配置保留在代码价格表里。
- Citation checker 当前是 lexical overlap，不是语义级事实校验。
- Wikipedia adapter 已实测能跑，但本次 benchmark 真实出现 2 次 fallback；它不是生产级搜索 provider。
- Local hybrid retrieval 已实现，但 5 case 小样本里 success_rate 低于 keyword baseline；不能包装成质量稳定提升。
- DashScope embedding/rerank provider 已实现 stub 测试，但因本机未配置 `DASHSCOPE_API_KEY`，真实百炼版本未实测。
- mock benchmark 的 latency/success/citation/cost 不应作为面试成果开场，只能说它证明 pipeline plumbing 和记录链路能跑。
- 未接 Redis/PostgreSQL、OpenTelemetry/LangSmith、LangGraph checkpoint。
- 未做 PDF/Docx 多格式导出。

## Git 提交

当前主要提交：

- `de058f2 chore: initialize deep research project`
- `d52a83e feat: implement observable deep research spine`
- `236608a docs: clarify mock benchmark limitations`
- `2eee6fe feat: add DeepSeek structured-output validation`
- `c6eb9a6 feat: wire DeepSeek provider into end-to-end synthesis`
- `7d371a9 feat: record real DeepSeek usage and cost`
- `cdf48fa feat: run DeepSeek Wikipedia benchmark without mock fallback`
- `5ec1be6 feat: add embedding provider abstraction`
- `6259a99 feat: add hybrid local retrieval fusion`
- `92e19e4 feat: add optional retrieval reranking`

本交接文件用于帮助审阅者快速定位项目状态，不替代 `KNOWLEDGE_BASE.md`。
