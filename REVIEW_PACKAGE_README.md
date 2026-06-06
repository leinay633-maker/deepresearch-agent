# DeepResearch Agent Review Package

这是给外部 AI / 网页版 AI 审阅本项目进展用的交接说明。请优先看 `KNOWLEDGE_BASE.md`，再看代码和测试。

## 当前完成状态

- 已从空仓库实现一个可运行的 DeepResearch Agent MVP。
- 已实现主链路：用户问题 -> clarify/normalize -> research brief -> planner 拆子问题 -> 3 个 researcher 并发检索 -> source dedup -> verifier -> synthesizer 带引用合成 -> citation check -> structured report。
- 已实现 FastAPI JSON 接口 `/research` 和 SSE 接口 `/research/stream`。
- 已实现 mock search provider 和 Wikipedia 真实无 key search adapter。
- 已实现工具失败处理：retry、timeout、circuit breaker、fallback。
- 已实现结构化 trace、阶段级 token/cost 估算、benchmark harness。
- 已生成 `KNOWLEDGE_BASE.md` 和 `INTERVIEW_QA.md`。

## 优先审阅文件

1. `KNOWLEDGE_BASE.md`：项目复盘、真实问题、实测数据、和参考项目差异。
2. `INTERVIEW_QA.md`：面试检索和 drill 用问答，共 40 题。
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

最近一次结果：`9 passed, 1 warning`。warning 是 FastAPI TestClient / Starlette 关于 httpx 的 deprecation 提示，未影响功能。

benchmark 命令：

```powershell
py -3.11 -m deepresearch_agent.benchmark --search-provider mock --seed 20260606
```

最近一次 summary 在 `results/benchmark_summary.json`，原始记录在 `logs/benchmark-20260606T145935Z.jsonl`。

核心实测指标：

- case_count: 5
- success_count: 5
- success_rate: 1.0
- latency p50: 8.681ms
- latency p90: 9.249ms
- total_tokens: 22281
- citation_retention_rate_avg: 1.0
- estimated_cost_usd_total: 0.0

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

- 未接真实 LLM provider；当前 planner/synthesizer 是 deterministic mock。
- 未实测真实 LLM token usage 和真实 LLM cost。
- Citation checker 当前是 lexical overlap，不是语义级事实校验。
- Wikipedia adapter 已实测能跑，但不是生产级搜索 provider。
- 未接 Redis/PostgreSQL、OpenTelemetry/LangSmith、LangGraph checkpoint。
- 未做 PDF/Docx 多格式导出。

## Git 提交

当前主要提交：

- `de058f2 chore: initialize deep research project`
- `d52a83e feat: implement observable deep research spine`

本交接文件用于帮助审阅者快速定位项目状态，不替代 `KNOWLEDGE_BASE.md`。
