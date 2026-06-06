# DeepResearch Agent Review Package

这是给外部 AI / 网页版 AI 审阅本项目进展用的交接说明。请优先看 `KNOWLEDGE_BASE.md`，再看代码和测试。

## 当前完成状态

- 已从空仓库实现一个可运行的 DeepResearch Agent MVP。
- 已实现主链路：用户问题 -> clarify/normalize -> research brief -> planner 拆子问题 -> 3 个 researcher 并发检索 -> source dedup -> verifier -> synthesizer 带引用合成 -> citation check -> structured report。
- 已实现 FastAPI JSON 接口 `/research` 和 SSE 接口 `/research/stream`。
- 已实现 mock LLM/search provider、DeepSeek 真实 LLM provider、Wikipedia 真实无 key search adapter。
- 已实现工具失败处理：retry、timeout、circuit breaker、fallback。
- 已实现结构化 trace、阶段级 token/cost 归因、benchmark harness；DeepSeek 路径记录 provider 返回的真实 usage，并按当前实现里的价格常量估算成本。
- 已生成 `KNOWLEDGE_BASE.md` 和 `INTERVIEW_QA.md`。

## 优先审阅文件

1. `KNOWLEDGE_BASE.md`：项目复盘、真实问题、实测数据、和参考项目差异。
2. `INTERVIEW_QA.md`：面试检索和 drill 用问答，共 41 题。
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

最近一次结果：`12 passed, 1 warning`。warning 是 FastAPI TestClient / Starlette 关于 httpx 的 deprecation 提示，未影响功能。

benchmark 命令：

```powershell
$env:REQUEST_TIMEOUT_SECONDS='8'
py -3.11 -m deepresearch_agent.benchmark --llm-provider deepseek --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3
```

最近一次 summary 在 `results/benchmark_summary.json`，原始记录在 `logs/benchmark-20260606T171739Z.jsonl`。本次是 DeepSeek `deepseek-v4-flash` + Wikipedia search，`fallback_count_total=2`，有 2 次 researcher 检索降级，已按真实结果记录。

核心实测指标解释：下面这些是真实 provider local benchmark 记录，适合证明本机这次配置下 LLM provider、search adapter、usage/cost 记录和 citation checker 已经端到端跑通；它仍不是线上 SLA 或广义质量分数。延迟包含 DeepSeek 和 Wikipedia 网络/API 时间；citation_retention_rate 是 lexical checker 结果，不是语义级事实评估。

- case_count: 5
- success_count: 4
- success_rate: 0.8
- latency p50 / p90 / max: 30235.414ms / 48390.038ms / 54020.348ms
- total_tokens: 19843
- citation_retention_rate_avg: 0.8778
- estimated_cost_usd_total: 0.00425096（按当前实现中的 `deepseek-v4-flash` 价格常量估算，价格核对日期：2026-06-07）
- fallback_count_total: 2

对比用 mock plumbing 原始记录仍保留在 `logs/benchmark-20260606T152954Z.jsonl`。mock 数字只证明离线路径和记录链路能跑，不能当真实性能、真实成本或真实答案质量成果。

## 运行方式

安装：

```powershell
py -3.11 -m pip install --timeout 120 -e ".[dev]"
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

本交接文件用于帮助审阅者快速定位项目状态，不替代 `KNOWLEDGE_BASE.md`。
