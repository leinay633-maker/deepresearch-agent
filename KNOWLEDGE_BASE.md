# 0 项目一句话介绍

本项目是我从空仓库开始手写的一个收窄版 DeepResearch Agent，目标不是复刻大而全的 open_deep_research，而是把「问题澄清、research brief、并发 researcher、来源去重、带引用合成、citation check、trace 和 benchmark」这条主链路做干净。它解决的是普通 RAG 一次性检索后直接回答时，难以解释检索路径、引用是否支撑论断、工具失败如何降级的问题。当前版本默认使用 mock LLM 和 mock search，保证无 API key 也能一条命令跑通；同时已经接入 DeepSeek 真实 LLM provider 和 Wikipedia 真实检索 adapter，用显式参数切换并记录真实 usage/cost。这个项目体现的 Agent 后端能力主要是多阶段编排、并发工具调用、失败兜底、可观测性、成本归因和可复现评测。

# 1 岗位匹配

我做这个项目时刻意对齐 Agent 后端 / LLM 应用岗，而不是做一个只会调用 LLM 的 demo。JD 里常见的 LangGraph、RAG、MCP、并发、可观测性、评测这些关键词，在本项目里对应到清晰的工程模块：`orchestrator.py` 做轻量编排，`rag.py` 做本地 keyword RAG，`search.py` 做工具 adapter、重试、超时、熔断和降级，`tracing.py` 和 `cost.py` 做观测和成本归因，`benchmark.py` 做可复现评测。

我在第一阶段没有强行让默认路径依赖真实 LLM provider，因为没有 API key 时会阻塞陌生人 clone 运行。最终选择是默认保留 `MockLLMProvider` 做可复现测试和 mock plumbing benchmark；当环境变量 `DEEPSEEK_API_KEY` 存在时，可以显式启用 `DeepSeekLLMProvider` 跑真实 structured output、synthesis、token usage 和 cost。OpenAI/Anthropic 等其他 provider 仍作为 v2 扩展。

# 2 总体架构

API 层：`src/deepresearch_agent/api.py`。输入是 `ResearchRequest`，输出是 `StructuredReport`。提供 `/research` JSON 接口、`/research/stream` SSE 接口和 `/health`。我参考 FastAPI + LangGraph 模板时只吸收了「服务层薄封装、每次请求创建编排器、接口返回结构化对象」这个思路，没有引入 JWT、数据库、Langfuse 或 Prometheus。

Agent 编排层：`src/deepresearch_agent/orchestrator.py`。输入是用户 query 和配置，输出是完整报告。它按 clarify/normalize、planner、并发 researcher、source dedup、synthesizer、citation check 的顺序执行。这里我没有直接用 LangGraph，是因为当前目标是可讲清楚的收窄项目，轻量 orchestrator 更便于展示每个阶段的输入输出和失败边界。

工具 Adapter 层：`src/deepresearch_agent/search.py`、`src/deepresearch_agent/rag.py`。搜索层有 `MockSearchAdapter` 和 `WikipediaSearchAdapter`，外加 `SearchService` 负责 retry、timeout、circuit breaker 和 fallback。本地 RAG 用 `data/local_corpus.jsonl`，用于展示普通 RAG 与 agentic RAG 的区别。

检索质量层：`src/deepresearch_agent/dedup.py`、`src/deepresearch_agent/verifier.py`。Dedup 按规范化 URL 合并重复来源，Verifier 按标题、正文长度、稳定 URL、已知 adapter、低质量模式打分过滤。

评测层：`src/deepresearch_agent/benchmark.py`、`data/benchmark_cases.jsonl`、`tests/`。benchmark 固定 seed 和配置快照，记录 latency、tokens、cost、source count、citation retention、success。

可观测层：`src/deepresearch_agent/tracing.py`、`src/deepresearch_agent/cost.py`。Trace 每阶段写 JSONL，Cost 按 brief_generation、planning、synthesis 归因 token 和成本；mock 路径仍是字符数近似，DeepSeek 路径使用 provider 返回的真实 usage。

# 3 核心流程

完整链路是：用户问题进入 `ResearchRequest` 后，配置化 LLM provider 先做 normalize 和 research brief；`plan` 生成子问题；`orchestrator` 用 `asyncio.gather` 并发启动 2 到 3 个 researcher；每个 researcher 同时拿 search 和 local RAG 的来源，做 dedup 和 verifier；全局再做一次 source dedup 并分配 `S1`、`S2` 这样的引用 ID；`synthesize` 生成带引用的报告；`CitationChecker` 对每条 claim 的 citation ID 和 source text 做词重叠校验；最后返回结构化报告，同时写 trace log 和 cost summary。默认 provider 是 mock；显式传 `--llm-provider deepseek` 时 brief、plan 和 synthesis 都由 DeepSeek JSON mode 生成。

# 4 关键设计决策

## 决策 1：多 agent vs 单 agent

背景：DeepResearch 类任务通常不是一次检索就能回答，尤其是架构、风险、评测这类问题需要多视角。
可选方案：单 agent 顺序检索；supervisor-researcher 并发；完全通用 LangGraph 多 agent。
最终选择：轻量 supervisor-researcher，并发执行 3 个 researcher。
理由：它保留 open_deep_research 的核心骨架，但目录和状态都由我自己重写，面试时能解释每个阶段。
代价：没有 LangGraph Studio 的图可视化和 checkpoint。
面试怎么答：我会说我不是为了炫多 agent，而是把 research task 拆成可以并发、可观测、可失败隔离的子任务。

## 决策 2：RAG 怎么搭

背景：普通 RAG 容易变成一次 retrieve + answer，看不出 Agent 工程深度。
可选方案：只用 web search；只用 local RAG；web search + local RAG 混合。
最终选择：web search adapter 和 local keyword RAG 混合，每个 researcher 都会合并两类来源。
理由：可以展示本地知识和外部检索的统一 Source 抽象，也方便无 key 时稳定运行。
代价：当前 local RAG 只是 keyword overlap，未实测 embedding 召回效果。
面试怎么答：我会说当前重点不是向量数据库，而是把检索结果纳入可验证、可追踪的 Agent pipeline。

## 决策 3：工具失败怎么兜

背景：真实搜索 API 会超时、限流或返回空结果，DeepResearch 不能因为一个工具失败就整体失败。
可选方案：直接抛错；只 retry；retry + timeout + circuit breaker + fallback。
最终选择：`SearchService` 里做 bounded retry、timeout、circuit breaker，失败后降级到 mock search。
理由：这个组合能把外部不稳定性限制在 researcher 层。
代价：fallback 结果不等于真实搜索结果，报告必须标明 provider 和 fallback_count。
面试怎么答：我会强调「可用性优先，真实性不造假」，fallback 是保证流程不断，不是伪装成真实外部检索。

## 决策 4：citation 怎么校验

背景：带引用不等于引用真实支撑论断，报告可能引用 A 但说 B。
可选方案：只检查 citation ID 存在；用 LLM judge；先做轻量 lexical faithfulness。
最终选择：`CitationChecker` 提取 `[S1]` 这类 citation ID，用 claim/source 词重叠做第一层检查。
理由：无 API key、可复现、测试稳定，能先拦住明显 unsupported claim。
代价：它不能理解复杂语义蕴含，未来需要 LLM judge 或 NLI 模型。
面试怎么答：我会说这不是终局评测，而是便宜、确定、可 CI 化的第一道闸。

## 决策 5：并发怎么控限流

背景：researcher 并发能降低延迟，但并发过高会放大 API 限流和成本。
可选方案：无限 gather；固定串行；`asyncio.Semaphore` 控制并发。
最终选择：`asyncio.Semaphore(max_researchers)`，默认最多 3。
理由：足够展示并发，同时保持输出和 trace 易读。
代价：没有 per-provider QPS bucket，未实测高并发流量。
面试怎么答：我会说 MVP 先控制任务级并发，v2 再加 provider 级 token bucket。

## 决策 6：为什么先用自定义 orchestrator 而不是 LangGraph

背景：参考项目大量使用 LangGraph，但我这个项目的目标是求职展示，不是做平台级通用 agent harness。
可选方案：直接上 LangGraph；自定义轻量 orchestrator；先自定义后抽成 LangGraph graph。
最终选择：先自定义轻量 orchestrator。
理由：每个阶段都能直观看到输入输出、trace 和错误边界；也避免为了 checkpoint、Studio UI 引入大量复杂度。
代价：没有 LangGraph 原生可视化和 durable execution。
面试怎么答：我会说我理解 LangGraph 的价值，但 MVP 的核心风险在 citation、fallback、benchmark，而不是图框架本身。

## 决策 7：为什么第一版真实 LLM provider 选择 DeepSeek

背景：mock 路径能证明 pipeline plumbing，但无法回答“真实 LLM structured output、真实 token usage、真实成本记录是否能跑通”。同时默认路径不能依赖 API key，否则陌生人 clone 后会被阻塞。
可选方案：继续 mock-only；接 OpenAI/Anthropic；接 DeepSeek OpenAI-compatible API；一次性做多 provider。
最终选择：先接一个显式启用的 DeepSeek provider，默认仍是 mock；API key 只读环境变量 `DEEPSEEK_API_KEY`，模型名允许用 `DEEPSEEK_MODEL` 覆盖。
理由：DeepSeek API 兼容 OpenAI Chat Completions，适合用标准 `/chat/completions` 接入；官方 JSON Output 支持 `response_format={"type":"json_object"}`，满足 brief/plan/synthesis 的结构化输出验证；官方 Tool Calls 能力存在，但本项目当前工具调用由 Python orchestrator 管控，没有让模型直接发 tool call；本机有可用 key，可以在不提交密钥的前提下跑出真实 usage/cost。
核对过的官方文档：JSON Output `https://api-docs.deepseek.com/guides/json_mode/`，Tool Calls `https://api-docs.deepseek.com/guides/function_calling/`，当前模型与价格 `https://api-docs.deepseek.com/quick_start/pricing`。
代价：当前只代表 DeepSeek 一个 provider，不能泛化到所有模型；默认模型已从 legacy alias 迁移到显式 `deepseek-v4-flash`，legacy alias 仅为旧配置兼容保留在价格表中。当前 `estimated_cost_usd` 是根据 provider usage 和代码里的 `deepseek-v4-flash` 价格常量估算，不等同于长期稳定账单或产品级成本承诺。
面试怎么答：我会说我没有把 mock 数字包装成真实成果，而是先用 DeepSeek 把 structured output、usage 解析、成本归因和真实搜索 benchmark 打通；迁移 v4-flash 后又重跑了 schema validation 和 5 case benchmark。但我也会主动说明它只是单 provider 小样本，下一步是 provider 抽象扩展和更强评测。

# 5 实现细节

Planner：`src/deepresearch_agent/llm.py`。输入是 `ResearchBrief`，输出是 `SubQuestion` 列表。默认 deterministic mock planner 会生成 background、evidence、tradeoffs 三类问题，用于离线可复现；DeepSeek planner 会用 JSON mode 生成符合同一 Pydantic schema 的子问题。局限是 planner 还不会根据 researcher 中间结果动态追加子问题。

DeepSeek Planner 验证：`src/deepresearch_agent/llm.py` 里新增了 `DeepSeekLLMProvider.plan`，第一步先独立验证结构化输出；验证脚本是 `src/deepresearch_agent/validate_deepseek_structured_output.py`，它从环境变量读取 `DEEPSEEK_API_KEY`，用 JSON mode 请求默认 `deepseek-v4-flash`，并用现有 `SubQuestion` Pydantic schema 解析输出。后续步骤再把同一个 provider 扩展到 `create_brief` 和 `synthesize`，避免一次性接太多导致错误边界不清。

DeepSeek Synthesizer 接入：步骤 2 以后，`DeepSeekLLMProvider.create_brief`、`plan`、`synthesize` 都走 DeepSeek JSON mode。CLI 和 API 可以通过 `llm_provider="deepseek"` 或 CLI 参数 `--llm-provider deepseek` 显式启用；默认仍是 mock，保证离线测试不受 API key 影响。当前 synthesis 要求模型输出 `{"answer": "...", "claims": [...]}`，并要求每条 factual claim 使用输入 sources 中已有的 `[Sx]` citation ID。

Researcher：`src/deepresearch_agent/orchestrator.py` 的 `_research_one`。输入是子问题，输出是 `Finding`。它调用 `SearchService` 和 `LocalRagRetriever`，再 dedup、verify、summary。局限是 summary 仍是模板化，不是自然语言 LLM 压缩。

Verifier：`src/deepresearch_agent/verifier.py`。输入是 source 列表，输出是过滤后的 source。关键设计是可解释 quality reasons。局限是规则打分，不能真正判断来源权威性。

Synthesizer：`src/deepresearch_agent/llm.py` 的 `synthesize`。输入是 brief、plan、findings、sources，输出 answer 和 claims。默认 mock 会生成可测报告，DeepSeek provider 会用 JSON mode 生成 markdown answer 和结构化 claims。局限是 DeepSeek 输出目前只靠 prompt 约束和后置 citation checker，没有做二次 LLM judge 或强制 source quote。

Citation Checker：`src/deepresearch_agent/citation.py`。输入是 claims 和 sources，输出 `CitationCheckReport`。关键设计是每条 claim 都落到 citation ID 和 overlap score。局限是只能做 lexical support。

Cost Tracker：`src/deepresearch_agent/cost.py`。mock provider 仍使用字符数近似估算；DeepSeek provider 已接入 API 返回的真实 `prompt_tokens` / `completion_tokens`，并通过 `CostTracker.add_usage()` 记录到同一套 `CostSummary`。当前 `deepseek-v4-flash` 成本计算按 DeepSeek 官方 Models & Pricing 页，核对日期 `2026-06-07`：input cache hit `$0.0028/1M tokens`，input cache miss `$0.14/1M tokens`，output `$0.28/1M tokens`。legacy alias 只作为 v4-flash 兼容入口使用同一价格表；未配置价格的模型会直接报错，避免 silently 用错单价。如果响应没有 token usage，DeepSeek 路径会直接失败，不会退回字符估算伪装成真实 usage。

Trace Logger：`src/deepresearch_agent/tracing.py`。每个 run 写 `logs/research-<run_id>.jsonl`，记录 stage、status、duration_ms、payload。runtime trace 默认不提交 Git，benchmark 原始记录提交。

# 6 遇到的问题与修复

## 问题 1：仓库初始化后 git commit 失败

现象：第一次 commit 报 `Author identity unknown`。
原因：本机没有全局 Git user.name/user.email。
排查：`git commit` 直接暴露错误。
修复：只在本仓库设置 `git config user.name "Codex Engineer"` 和 `git config user.email "codex@example.local"`，不改全局。
复盘：自动化项目初始化时，Git author 也是环境依赖。
面试可能追问：为什么不改 global？回答：这是用户机器，我只改项目局部配置，避免影响其他仓库。

## 问题 2：默认 Python 3.14 安装 Pydantic 失败

现象：`python -m pip install -e ".[dev]"` 解析失败，提示 `pydantic-core` 没有匹配发行包。
原因：本机默认 Python 是 3.14，而当时依赖 wheel 对 3.14 不完整。
排查：`py -0p` 发现本机还有 Python 3.11。
修复：`pyproject.toml` 改成 `>=3.11,<3.14`，README 改成 `py -3.11`。
复盘：Agent 后端项目最好明确 Python minor version，尤其依赖有 native wheel 时。
面试可能追问：为什么不改成纯 dataclass？回答：FastAPI 服务层需要 Pydantic，既然本机有 3.11，用受支持解释器比砍掉技术栈更合理。

## 问题 3：pip 下载 pydantic-core 超时

现象：3.11 安装时依赖解析通过，但下载 `pydantic_core` wheel 发生 read timeout。
原因：网络下载超时。
排查：pip traceback 指向 files.pythonhosted.org read timeout。
修复：用 `py -3.11 -m pip install --timeout 120 -e ".[dev]"` 重试成功。
复盘：安装文档里应该保留可重试命令或建议使用镜像。
面试可能追问：这算不算项目 bug？回答：不是业务 bug，但属于可复现环境风险，需要在 README 和知识库里说明。

## 问题 4：Wikipedia adapter 初版相关性排序不合理

现象：真实查询 `What is Model Context Protocol?` 时，报告里出现 GPS/Data center 作为最强来源。
原因：我一开始把 Wikipedia `size` 当 score，导致页面越大越靠前，而不是越相关越靠前。
排查：查看 live run 的 source title 和 metrics，发现 `fallback_count=0` 但结果质量不对。
修复：把 score 改成 query/source token overlap + 原始搜索排序位置。
复盘：真实 adapter 不只要能连通，还要避免把 provider 的元字段误当相关性。
面试可能追问：修完后是否完全解决？回答：只解决了明显 size 污染，Wikipedia 仍不是专业 web search provider，source quality 还需要更强 reranker。

## 问题 5：接入 DeepSeek synthesis 时 plan 阶段 cost 变量被误删

现象：步骤 2 第一次运行 `py -3.11 -m deepresearch_agent.cli "How should citation checking reduce hallucination in deep research agents?" --llm-provider deepseek --search-provider mock --max-researchers 2 --max-results 3 --json` 失败，报 `UnboundLocalError: cannot access local variable 'cost' where it is not associated with a value`。
原因：步骤 1 只验证 planner schema 时，`DeepSeekLLMProvider.plan()` 里写了 `del cost`；步骤 2 给 DeepSeek plan 增加了 `cost.add()` 后，没有删除这行。
排查：traceback 直接指向 `llm.py` 的 `cost.add("planning", ...)`。
修复：删除 `del cost`，保留 plan 阶段的成本记录调用。注意此阶段仍是字符估算，真实 usage 接入在步骤 3。
复盘：分步骤推进是有价值的，步骤 2 立刻暴露了步骤 1 临时代码和后续真实接入的冲突。
面试可能追问：为什么把这个写进知识库？回答：这是接真实 provider 过程中真实发生的 bug，不编造也不隐藏，能说明我是按验证链路推进的。

## 问题 6：第一次 DeepSeek + Wikipedia benchmark 出现 mock fallback

现象：步骤 4 第一次运行 DeepSeek LLM + Wikipedia search benchmark 时，summary 显示 `fallback_count_total=6`，说明部分 researcher 的 Wikipedia 检索降级到了 mock，不能算“LLM 和检索都不是 mock”的真实 benchmark。
原因：DeepSeek planner 生成的子问题是长自然语言问题，直接传给 Wikipedia Search API 时，有些查询返回 no results，有些查询触发 live search timeout；`SearchService` 按设计降级到 mock。
排查：查看 `logs/research-*.jsonl`，失败原因包括 `wikipedia returned no results` 和 timeout 空错误字符串。
修复：在 `WikipediaSearchAdapter` 内增加 `_wikipedia_query_candidates()`，把长问题压缩成关键词查询，并在无结果时逐步尝试候选查询；同时真实 benchmark 运行时把 `REQUEST_TIMEOUT_SECONDS` 设置为 `8`。当时重跑曾把 fallback 降到 0，但最新 v4-flash benchmark 又出现 `fallback_count_total=2`，说明这只是降低 fallback，不是彻底消灭 fallback。
复盘：真实检索 adapter 不只是“能联网”，还要把 agent planner 产出的长问题转换成搜索引擎能吃的 query。这个 bug 也证明 fallback 指标必须进入 benchmark，否则会误以为检索全是真实的；最新结果里 fallback 再次出现，所以面试时不能说“已经彻底解决”。
面试可能追问：为什么不用更强搜索？回答：本次按约束先用无 key 的 Wikipedia，后续换 Tavily/Brave 是新增 adapter，不需要改 orchestrator。

## 问题 7：DeepSeek 默认模型和价格表需要迁移到 v4-flash

现象：官方模型与价格页已经把主模型列成 `deepseek-v4-flash` / `deepseek-v4-pro`，并说明历史模型名只是兼容别名；继续默认 legacy alias 会让代码和最新文档脱节。
原因：步骤 1 到步骤 4 接入 DeepSeek 时使用的是历史兼容模型名和旧价格常量，后来核对官方文档发现需要迁移。
排查：重新查 DeepSeek 官方 Models & Pricing、Create Chat Completion、JSON Output 和 Tool Calls 文档，确认 `deepseek-v4-flash` 支持 JSON Output 和 Tool Calls；价格核对日期是 `2026-06-07`，当前价格是 cache hit `$0.0028/1M`、cache miss `$0.14/1M`、output `$0.28/1M`。
修复：把默认模型改成 `deepseek-v4-flash`，验证脚本默认值同步迁移；成本计算改成按模型查价格表，legacy alias 仅为兼容旧配置保留；新增单测覆盖 cache hit/cache miss/output 成本计算。随后用 `deepseek-v4-flash` 重跑 planner schema validation 和真实 benchmark。
复盘：模型 provider 不是一次接完就结束，模型名、价格和功能支持都会变；代码里必须有显式定价表和失败策略，不能把过期单价悄悄沿用。
面试可能追问：为什么不直接删掉 legacy alias？回答：删掉会让旧运行记录和用户自定义旧模型名立即失效；我保留兼容入口，但当前默认和新 benchmark 都走显式 `deepseek-v4-flash`。

# 7 实测数据

本节所有 mock benchmark 数字只用于证明 pipeline plumbing 能端到端跑通，不能当作真实性能、真实成本或真实答案质量成果。尤其不能在面试里说“我的 DeepResearch p50 是个位数毫秒”这类话，因为这个延迟测的是本机 Python 跑 deterministic mock 的速度，换机器、换进程热身状态、换依赖版本都会变。

实测环境：Windows PowerShell，`py -3.11`，mock search provider，seed `20260606`，5 条 benchmark case，max_researchers=3，max_results=4。

安装验证：`py -3.11 -m pip install --timeout 120 -e ".[dev]"` 成功。
测试验证：`py -3.11 -m pytest -q`，最新结果 `12 passed, 1 warning in 0.88s`。warning 来自 FastAPI TestClient / Starlette 对 httpx 的 deprecation 提示，未影响功能。
CLI example：`py -3.11 -m deepresearch_agent.cli "How does citation checking reduce hallucination in agentic RAG?"` 成功，raw_search_result_count `12`，deduped_source_count `8`，total_tokens `4417`。这次运行记录的 latency 是 `10.63ms`，但它只是 mock plumbing run 的本机样本，不作为性能指标引用。citation_retention_rate `1.0` 只说明 mock synthesis 生成的 citation ID 能被当前 checker 找到，不代表真实 LLM 场景下的引用可靠性。estimated_cost_usd `0.0` 是因为 mock provider 单价配置为 0，不代表真实成本。
真实 adapter probe：`py -3.11 -m deepresearch_agent.cli "What is Model Context Protocol?" --search-provider wikipedia --json` 成功，修复后 sample 输出显示 `fallback_count=0`，latency 约 `1506.501ms`。注意：Wikipedia 是真实无 key adapter，但不是高质量通用搜索，结果质量仍有限。

DeepSeek 结构化输出验证：`py -3.11 -m deepresearch_agent.validate_deepseek_structured_output --query "How should citation checking reduce hallucination in deep research agents?" --max-researchers 3` 成功。迁移后默认 `deepseek-v4-flash` 返回了 3 条合法 `SubQuestion`，Pydantic schema 解析通过。真实输出主题分别覆盖 citation checking 机制、citation accuracy 与 hallucination rate 的关系、以及 citation checking 的失败模式。

DeepSeek 端到端单条验证（LLM 真，search 仍是 mock）：`py -3.11 -m deepresearch_agent.cli "How should citation checking reduce hallucination in deep research agents?" --llm-provider deepseek --search-provider mock --max-researchers 2 --max-results 3 --json` 成功。模型生成的 brief 不再是模板回填，scope 是“Methods and effectiveness of citation verification in mitigating factual inaccuracies in AI-driven research agents”；planner 拆出 automated citation verification 和 human-in-the-loop 对比两个子问题；synthesis 生成了 markdown 报告和 6 条 claims。步骤 2 首次成功运行记录：latency `18987.535ms`，raw_search_result_count `6`，deduped_source_count `7`，citation_retention_rate `0.8333`，supported_claims `5/6`。其中 1 条 human-in-the-loop claim 被当前 lexical citation checker 标为 unsupported。这是接真实 usage 前的历史记录，所以当时的 cost 不能当真实成本；步骤 3 已补上 provider usage 解析。

DeepSeek usage/cost 单条验证（LLM 真，search 仍是 mock）：接入真实 usage 后重跑同一条命令成功。运行记录：latency `18524.693ms`，raw_search_result_count `6`，deduped_source_count `7`，citation_retention_rate `1.0`，supported_claims `6/6`。真实 usage：input_tokens `1842`，output_tokens `1310`，total_tokens `3152`，estimated_cost_usd `0.00193834`。分阶段成本：brief_generation `118 + 140 tokens / $0.00018586`，planning `277 + 226 tokens / $0.00032339`，synthesis `1447 + 944 tokens / $0.00142909`。注意：search 仍是 mock，所以这还不是“LLM + search 全真实”的 benchmark；步骤 4 再切 Wikipedia。

mock benchmark 原始记录：`logs/benchmark-20260606T152954Z.jsonl`。
当前 benchmark summary：`results/benchmark_summary.json`，已被真实 DeepSeek + Wikipedia benchmark 覆盖。

benchmark 汇总：管线 plumbing 指标，mock，非真实性能。具体 latency/token/cost 数字保留在 `results/benchmark_summary.json` 和 raw log 里，面试时不把这些数字当成果开场。

| 指标 | mock plumbing run 记录 | 面试口径 |
|---|---|---|
| case_count | 5 条本地 smoke case | 只说明 benchmark harness 能批量跑 case |
| success_count / success_rate | 当前 mock run 全部通过 | 不代表真实任务成功率 |
| latency p50 / p90 / max | JSON 里有记录 | 只说明系统会记录延迟，不报具体 ms 作为性能成果 |
| total_tokens / avg_tokens | JSON 里有记录 | 字符数近似估算，不是真实 tokenizer usage |
| estimated_cost_usd_total | mock 配置下为 0 | 不代表真实成本 |
| citation_retention_rate_avg | 当前 mock run 为 1.0 | mock 自生成自引用，只说明 checker 链路没断 |
| fallback_count_total | 当前 mock run 为 0 | mock provider 本身不触发外部失败 |

旧 legacy alias benchmark 原始记录仍保留在 `logs/benchmark-20260606T160617Z.jsonl`，只作为历史对照，不再展开旧数字作为当前主口径。

真实 DeepSeek v4-flash + Wikipedia benchmark：`$env:REQUEST_TIMEOUT_SECONDS='8'; py -3.11 -m deepresearch_agent.benchmark --llm-provider deepseek --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3`。本次 LLM 是 `deepseek-v4-flash`，检索 primary 是 Wikipedia，最终 `fallback_count_total=2`，说明有 2 次 researcher 检索降级；这是实际运行结果，不做调参美化。原始记录：`logs/benchmark-20260606T171739Z.jsonl`，summary：`results/benchmark_summary.json`。

| 指标 | 真实 benchmark 记录 | 怎么解释 |
|---|---:|---|
| case_count | 5 | 仍是小型本地 benchmark，不是公开权威评测 |
| success_count / success_rate | 4 / 0.8 | 真实 citation checker 下有 1 条 case 未达当前 success 条件 |
| latency p50 | 30235.414ms | 包含 DeepSeek + Wikipedia live 网络时间，不是 SLA |
| latency p90 | 48390.038ms | 同上 |
| latency max | 54020.348ms | 同上 |
| total_tokens | 19843 | 来自 DeepSeek usage 字段 |
| avg_tokens | 3968.6 | 来自 DeepSeek usage 字段 |
| estimated_cost_usd_total | 0.00425096 | 按当前实现中的 `deepseek-v4-flash` 价格常量估算，价格核对日期：2026-06-07 |
| citation_retention_rate_avg | 0.8778 | lexical citation checker 结果，不是语义级事实评估 |
| fallback_count_total | 2 | 有 2 次 researcher 检索降级，如实记录 |

逐 case 结果：case-001 失败，retention `0.5556`，cost `$0.00080836`，fallback `0`；case-002 成功，retention `1.0`，cost `$0.00076818`，fallback `0`；case-003 成功，retention `0.8333`，cost `$0.00092302`，fallback `1`；case-004 成功，retention `1.0`，cost `$0.00099176`，fallback `1`；case-005 成功，retention `1.0`，cost `$0.00075964`，fallback `0`。这组数据比 mock plumbing 更有意义，因为 LLM token/cost 是 provider usage；但它仍受 Wikipedia 搜索质量、fallback、LLM 输出波动和 lexical citation checker 限制。

未实测：真实搜索 API 高并发限流、语义级 citation faithfulness、Redis/PostgreSQL 缓存、OpenTelemetry/LangSmith tracing、真实用户流量。

# 8 评测设计

answer completeness：当前未做 LLM judge，只用 case success 间接衡量，未实测完整性。
citation faithfulness：当前实测指标是 claim/source lexical overlap。mock plumbing run 平均 retention 是 `1.0`，只能说明 mock 引用链路没断；DeepSeek v4-flash + Wikipedia benchmark 平均 retention 是 `0.8778`，但仍有 1 条 case 未达当前 success 条件。
source diversity：当前记录 deduped_source_count，但没有按 domain/provider 多样性打分。
hallucination rate：当前用 unsupported citation count 作为 proxy，不能覆盖无引用幻觉。
latency：benchmark 记录每 case latency_ms，并计算 P50/P90/max；mock latency 只能作为 plumbing 回归信号，DeepSeek + Wikipedia latency 包含真实网络/API 时间，也不能当线上 SLA。
cost：mock provider 成本为 0，token 用字符估算；DeepSeek provider 已接真实 usage，并按当前实现里的 v4-flash 价格常量估算成本，价格核对日期 `2026-06-07`。其他真实 LLM provider 未实测。
工具失败恢复：有 unit test 覆盖 primary failure fallback 和 circuit breaker open；第一次 DeepSeek + Wikipedia benchmark 出现过 fallback，修复 Wikipedia 长查询压缩后 fallback 曾降到 0，但最新 v4-flash benchmark 真实出现 `fallback_count_total=2`，已在第 7 节如实记录。
multi-hop 成功率：当前没有真实 multi-hop 标注集，未实测。

评测集构造方式：我先放了 5 条围绕本项目核心能力的问题，覆盖 supervisor-researcher、citation faithfulness、tool failure、cost tracking、benchmark reproducibility。它不是公开标准 benchmark，目标是本地可复现 smoke benchmark；现在同时保留 mock plumbing 记录和 DeepSeek + Wikipedia 真实 provider 小样本记录。

# 9 与参考项目的差异

## open_deep_research

参考了什么：我读了 README 和 CLAUDE.md，参考了它的 deep research 三段式、multi-agent / parallel researcher、模型需要 structured output + tool calling、评测和配置化思想。
没照搬什么：没有复制它的 LangGraph graph、prompt、state、配置文件或 eval 代码。
我做了哪些改造：改成自定义轻量 orchestrator，把 citation checker、source verifier、fallback、trace/cost、benchmark 都做成直接可读的小模块。
为什么更适合求职展示：代码量小，面试时能从 API 到 citation check 一路讲清楚，不会被大框架细节淹没。

## deep_research_from_scratch

参考了什么：参考了它按 notebook 拆 scope、research agent、MCP、supervisor、full agent 的学习路径。
没照搬什么：没有复制 notebook 代码，也没有依赖 Tavily/OpenAI key。
我做了哪些改造：把学习型 building blocks 改成可安装 Python package、CLI、FastAPI、pytest 和 benchmark。
为什么更适合求职展示：它像课程，本项目像一个可运行的后端工程。

## DeerFlow v1

参考了什么：按要求只看了 `main-1.x` 分支 README，参考它 Coordinator/Planner/Researcher/Reporter 的角色划分，以及 web UI/API/工具集分层思路。
没照搬什么：没有复制它的 web UI、crawler、TTS、presentation、checkpoint、配置系统。
我做了哪些改造：砍掉内容生产和平台能力，只保留 deep research 主干和后端可观测部分。
为什么更适合求职展示：范围更窄，重点在 Agent 后端可解释性和评测，而不是完整产品形态。

## gpt-researcher

参考了什么：参考了 planner/execution/publisher、source tracking、并发 researcher 和报告生成的宏观思路。
没照搬什么：没有复制它的 package、retriever、crawler、frontend 或 report exporter。
我做了哪些改造：把 source tracking 后面补上 citation faithfulness check，并把 fallback、trace、cost、benchmark 做成一等模块。
为什么更适合求职展示：我可以明确讲「我借鉴了 report/source 思路，但我更强调引用是否支撑论断和工具失败恢复」。

## FastAPI + LangGraph production template

参考了什么：参考了服务层结构化 API、LLM service fallback、observability、rate limiting 这些后端关注点。
没照搬什么：没有复制认证、数据库、memory、Langfuse、Prometheus、Alembic。
我做了哪些改造：只保留薄 API 和 SSE，把生产模板里的重组件作为 v2 optional。
为什么更适合求职展示：MVP 先展示 Agent pipeline，本项目不是 SaaS 后台模板。

# 10 局限与优化空间

真实 LLM provider 覆盖还窄：当前只接了 DeepSeek v4-flash，一个 provider 不能代表所有模型/价格/限流行为。可行方案是继续实现 OpenAI/Anthropic 等 OpenAI-compatible 或原生 provider，并统一 structured output、usage 解析、重试、模型定价表和测试替身。工程代价是 API key、价格、限流、错误码差异和 CI mock。面试怎么讲：我会说我已经把真实 provider 接入路径跑通，但不会把单 provider 小样本夸成通用生产能力。

Citation checker 语义能力弱：当前问题是 lexical overlap 只能拦明显错引。可行方案是加 LLM judge、NLI 模型或 sentence embedding entailment。工程代价是成本、延迟和 judge 可靠性评估。面试怎么讲：我会说现在是 CI 友好的第一道闸，不是最终事实评审。

Wikipedia 不是专业 search provider：当前问题是真实 adapter 能跑但相关性和覆盖有限。可行方案是接 Tavily/Brave/SerpAPI 或自建 SearxNG。工程代价是 key、限流、费用和 provider schema 差异。面试怎么讲：我会强调 adapter 已经抽象好，替换 provider 不影响 orchestrator。

Local RAG 只是 keyword overlap：当前问题是召回能力弱。可行方案是加 embedding、向量库和 reranker。工程代价是 embedding cost、index lifecycle、缓存和评测集。面试怎么讲：我会说当前 RAG 是为了展示接口和流程，不把数据库作为 MVP 阻塞项。

没有 durable execution：当前问题是服务重启会丢 run state。可行方案是 LangGraph checkpoint、PostgreSQL 或 Redis。工程代价是部署复杂度和 schema 维护。面试怎么讲：我会说单机 MVP 先保证可测，生产化再加持久化。

没有 OpenTelemetry/LangSmith：当前问题是 trace 只写本地 JSONL。可行方案是接 OTel exporter 或 LangSmith。工程代价是外部账号、采样、隐私和成本。面试怎么讲：我会说本地 JSONL 先保证无外部依赖，后续可以从同一 trace event 结构导出。
