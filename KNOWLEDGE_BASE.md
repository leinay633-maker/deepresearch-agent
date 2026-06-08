# 0 项目一句话介绍

本项目是我从空仓库开始手写的一个收窄版 DeepResearch Agent，目标不是复刻大而全的 open_deep_research，而是把「问题澄清、research brief、并发 researcher、来源去重、带引用合成、citation check、trace 和 benchmark」这条主链路做干净，并补上可管理的 run control plane。它解决的是普通 RAG 一次性检索后直接回答时，难以解释检索路径、引用是否支撑论断、工具失败如何降级的问题。当前版本默认使用 mock LLM 和 mock search，保证无 API key 也能一条命令跑通；同时已经接入 DeepSeek 真实 LLM provider、OpenAI-compatible 通用 LLM adapter、阶段级模型覆盖、Wikipedia 真实检索 adapter、本地关键词 + 向量 + RRF 融合的 hybrid local retrieval、可选 Qdrant HTTP vector index provider、SQLite run_id / checkpoint / HITL / SSE replay 控制平面，以及 LiveDRBench 这类公开 Deep Research 任务的端到端 artifact 评测 runner。这个项目体现的 Agent 后端能力主要是多阶段编排、并发工具调用、失败兜底、混合检索、可观测性、成本归因、可复现评测和长任务状态管理。

# 1 岗位匹配

我做这个项目时刻意对齐 Agent 后端 / LLM 应用岗，而不是做一个只会调用 LLM 的 demo。JD 里常见的 LangGraph、RAG、MCP、并发、可观测性、评测这些关键词，在本项目里对应到清晰的工程模块：`orchestrator.py` 做轻量编排，`rag.py` 做本地 keyword/vector hybrid RAG，`embeddings.py` 和 `rerankers.py` 做可切换 provider，`search.py` 做工具 adapter、重试、超时、熔断和降级，`tracing.py` 和 `cost.py` 做观测和成本归因，`benchmark.py` 做可复现评测。

我在第一阶段没有强行让默认路径依赖真实 LLM provider，因为没有 API key 时会阻塞陌生人 clone 运行。最终选择是默认保留 `MockLLMProvider` 做可复现测试和 mock plumbing benchmark；当环境变量 `DEEPSEEK_API_KEY` 存在时，可以显式启用 `DeepSeekLLMProvider` 跑真实 structured output、synthesis、token usage 和 cost；如果有本地或托管的 OpenAI-compatible endpoint，也可以显式启用 `OpenAICompatibleLLMProvider`。OpenAI/Anthropic 原生 SDK 仍作为 v2 扩展。

# 2 总体架构

API 层：`src/deepresearch_agent/api.py`。输入是 `ResearchRequest` 或 `CreateRunRequest`，输出是 `StructuredReport`、`AgentRun` 或 run trace。保留 `/research` JSON 接口、`/research/stream` SSE 接口和 `/health`；新增 `/runs`、`/runs/{run_id}/approve`、`/edit`、`/reject`、`/cancel`、`/retry`、`/steps`、`/events`、`/trace`，用于 worker ownership 的 `/runs/{run_id}/lease`、`/heartbeat`、`/runs/stale`、`/runs/recover-stale`，以及用于消费下一条 queued run 的 `/runs/worker/next`。我参考 FastAPI + LangGraph 模板时只吸收了「服务层薄封装、每次请求创建编排器、接口返回结构化对象」这个思路，没有引入 JWT、Postgres、Redis、Langfuse 或 Prometheus。

Agent 编排层：`src/deepresearch_agent/orchestrator.py`。输入是用户 query 和配置，输出是完整报告。它按 clarify/normalize、planner、并发 researcher、source dedup、synthesizer、citation check 的顺序执行。这里我没有直接用 LangGraph，是因为当前目标是可讲清楚的收窄项目，轻量 orchestrator 更便于展示每个阶段的输入输出和失败边界。

工具 Adapter 层：`src/deepresearch_agent/search.py`、`src/deepresearch_agent/rag.py`、`src/deepresearch_agent/ingest_corpus.py`、`src/deepresearch_agent/embeddings.py`、`src/deepresearch_agent/rerankers.py`。搜索层有 `MockSearchAdapter`、`WikipediaSearchAdapter`、`SearxngSearchAdapter`、`JinaSearchAdapter` 和 `JinaReaderCrawler`，外加 `SearchService` 负责 retry、timeout、circuit breaker 和 fallback。本地 RAG 用 `data/local_corpus.jsonl`，默认走关键词 + BGE 向量 + Chroma + RRF 融合；`ingest_corpus.py` 可以把 Markdown/TXT/PDF/DOCX 私有文档生成同一 JSONL 格式；也可以显式切回 keyword baseline，开启持久化 Chroma index，切到 Qdrant HTTP vector index，或者开启本地 / DashScope rerank。

检索质量层：`src/deepresearch_agent/dedup.py`、`src/deepresearch_agent/verifier.py`。Dedup 按规范化 URL 合并重复来源，Verifier 按标题、正文长度、稳定 URL、已知 adapter、低质量模式打分过滤。

评测层：`src/deepresearch_agent/benchmark.py`、`src/deepresearch_agent/retrieval_eval.py`、`src/deepresearch_agent/deep_research_eval.py`、`src/deepresearch_agent/source_metrics.py`、`data/benchmark_cases.jsonl`、`tests/`。端到端 benchmark 固定 seed 和配置快照，记录 latency、tokens、cost、source count、source provider/domain diversity、citation retention、success；独立检索评测只加载 BEIR/scifact 的 corpus/query/qrels，计算 Recall@10、nDCG@10 和 MRR，不调用 LLM、Wikipedia 或 orchestrator 主链路；公开 Deep Research 评测 runner 会加载 LiveDRBench 等公开任务，跑完整 orchestrator 并输出 answer、sources、trace、cost、citation check 和 predictions artifact。

可观测层：`src/deepresearch_agent/tracing.py`、`src/deepresearch_agent/cost.py`。Trace 每阶段写 JSONL，Cost 按 brief_generation、planning、synthesis 归因 token、成本和实际模型名；mock 路径仍是字符数近似，DeepSeek 路径使用 provider 返回的真实 usage。

Run Control Plane：`src/deepresearch_agent/run_models.py`、`src/deepresearch_agent/run_store.py`、`src/deepresearch_agent/run_control.py`、`src/deepresearch_agent/run_worker.py`。输入是 run create/approval/cancel/retry 请求，输出是持久化 run 状态、step trace、event stream 和最终 report checkpoint。它用 SQLite 保存 `agent_runs`、`agent_steps`、`agent_events`，默认文件是 `data/runs.sqlite`，可以用 `RUN_STORE_PATH` 覆盖；`agent_runs.request_json` 会保存创建时的完整请求快照，让 deferred worker 后续按本次配置恢复执行。`run_worker.py` 提供本地轮询 worker CLI，复用同一个 `RunController.process_next_queued()`。测试用临时 SQLite 文件。

# 3 核心流程

完整链路是：用户问题进入 `ResearchRequest` 后，配置化 LLM provider 先做 normalize 和 research brief；`plan` 生成子问题；`orchestrator` 用 `asyncio.gather` 并发启动 2 到 3 个 researcher；每个 researcher 同时拿 search 和 local RAG 的来源。local RAG 内部可以是 keyword baseline，也可以是 keyword + vector + RRF hybrid，并可选 rerank，但输出仍是统一 `Source`。之后做 dedup 和 verifier；全局再做一次 source dedup 并分配 `S1`、`S2` 这样的引用 ID；`synthesize` 生成带引用的报告；`CitationChecker` 对每条 claim 的 citation ID 和 source text 做词重叠校验；最后返回结构化报告，同时写 trace log 和 cost summary。默认 provider 是 mock；显式传 `--llm-provider deepseek` 时 brief、plan 和 synthesis 都由 DeepSeek JSON mode 生成。

# 4 关键设计决策

## 决策 1：多 agent vs 单 agent

背景：DeepResearch 类任务通常不是一次检索就能回答，尤其是架构、风险、评测这类问题需要多视角。
可选方案：单 agent 顺序检索；supervisor-researcher 并发；完全通用 LangGraph 多 agent。
最终选择：轻量 supervisor-researcher，并发执行 3 个 researcher。
理由：它保留 open_deep_research 的核心骨架，但目录和状态都由我自己重写，面试时能解释每个阶段。
代价：没有 LangGraph Studio 的图可视化和 checkpoint。
面试怎么答：我会说我不是为了炫多 agent，而是把 research task 拆成可以并发、可观测、可失败隔离的子任务。

## 决策 2：RAG 怎么搭

背景：普通 RAG 容易变成一次 retrieve + answer，看不出 Agent 工程深度；早期本地 RAG 只有 keyword overlap，能跑但语义召回弱。
可选方案：只用 web search；只用 local RAG；web search + local RAG；local RAG 内部升级为 keyword/vector hybrid。
最终选择：web search adapter 和 hybrid local RAG 并存。local RAG 保留 keyword baseline，同时新增 BGE embedding、Chroma vector index、RRF 融合、可选持久化 Chroma index、可选 Qdrant HTTP vector index 和可选 rerank；每个 researcher 仍合并 web search 与 local RAG 来源。
理由：keyword 对精确术语稳定，vector 对语义相近问题更友好，RRF 不要求两路分数同尺度；统一 `Source` 抽象让下游 dedup、verifier、synthesizer 不需要改。
代价：本地 embedding / Chroma / Qdrant / rerank 会增加依赖、外部服务或延迟；最新真实 benchmark 里 local hybrid 的 citation retention 略高于 keyword baseline，但 success_rate 更低，说明混合检索不是自动变好，需要更大语料和 rerank/权重调优。
面试怎么答：我会说我没有用向量替换关键词，而是保留两路召回再融合；实测结果不全是好看的，反而暴露了小语料场景下 hybrid 可能引入不稳定来源。

## 决策 3：工具失败怎么兜

背景：真实搜索 API 会超时、限流或返回空结果，DeepResearch 不能因为一个工具失败就整体失败。
可选方案：直接抛错；只 retry；retry + timeout + circuit breaker + fallback；再加 retry backoff 和进程内 rate limiter。
最终选择：`SearchService` 里做 bounded retry、可选指数 backoff、timeout、circuit breaker、可选 primary search rate limiter，失败后降级到 mock search。
理由：这个组合能把外部不稳定性限制在 researcher 层；`SEARCH_RETRY_BACKOFF_SECONDS` 默认 0，只有显式配置时才在失败重试之间等待；rate limiter 默认关闭，只有配置 `SEARCH_RATE_LIMIT_PER_SECOND` 时才对 primary search 生效。
代价：fallback 结果不等于真实搜索结果，报告必须标明 provider 和 fallback_count；进程内 rate limiter 不是跨 worker 的全局配额控制。
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
可选方案：无限 gather；固定串行；`asyncio.Semaphore` 控制并发；再给 search provider 做请求节流。
最终选择：`asyncio.Semaphore(max_researchers)` 控制 researcher 并发，默认最多 3；`SearchRateLimiter` 可通过 `SEARCH_RATE_LIMIT_PER_SECOND` 对 primary search 做本地进程内节流。
理由：足够展示并发，同时保持输出和 trace 易读；真实搜索 provider 有 key/额度时，可以先用本地节流降低 burst 风险。
代价：不是分布式限流，也没有 provider 级 quota 感知；多个 worker 进程仍需要 Redis/Postgres/网关级全局限流。
面试怎么答：我会说 MVP 先控制任务级并发，再补 SearchService 级本地节流；真正生产要按 provider 和 worker 池做全局 rate limit。

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
代价：当前只代表 DeepSeek 一个 provider，不能泛化到所有模型；默认模型已从 legacy alias 迁移到显式 `deepseek-v4-flash`，价格表只保留核对过的 v4-flash。未配置价格的模型名会 fail fast，不再借用 v4-flash 价格。当前 `estimated_cost_usd` 是根据 provider usage 和代码里的 `deepseek-v4-flash` 价格常量估算，不等同于长期稳定账单或产品级成本承诺。
面试怎么答：我会说我没有把 mock 数字包装成真实成果，而是先用 DeepSeek 把 structured output、usage 解析、成本归因和真实搜索 benchmark 打通；迁移 v4-flash 后又重跑了 schema validation 和 5 case benchmark。但我也会主动说明它只是单 provider 小样本，下一步是 provider 抽象扩展和更强评测。

## 决策 8：为什么做 hybrid retrieval，而不是直接换成向量检索

背景：keyword-only local RAG 对本项目里的固定英文术语很稳，但对同义表达和中文问题不友好；直接改成 pure vector 又会丢掉精确术语匹配优势。
可选方案：继续 keyword-only；直接 pure vector；keyword + vector 加权相加；keyword + vector 用 RRF 融合；再加 rerank。
最终选择：默认 local hybrid：关键词召回 + BGE local embedding 向量召回 + Chroma index + RRF 融合；rerank 做成可选 provider，默认关闭。
理由：RRF 只依赖排序名次，不要求 keyword score、cosine score 和 rerank score 同尺度；本地 BGE 默认无 API key，保证 clone 后仍能跑；DashScope embedding/rerank 作为显式 provider，只从 `DASHSCOPE_API_KEY` 读 key。
代价：依赖变重，首次加载 embedding/rerank 模型延迟明显；本次 5 case benchmark 里 hybrid 的 retention 从 `0.8867` 到 `0.8929` 略升，但 success_rate 从 `1.0` 降到 `0.6`，p50 从 `24595.506ms` 升到 `30747.284ms`。这说明检索结构更完整，不等于短期指标一定更好。
面试怎么答：我会说这次我做的是工程能力升级，不是调参刷分。混合检索给后续扩展语料、中文 query 和 rerank 留了接口，但当前小语料 benchmark 反而说明需要更细的评测和参数选择。

## 决策 9：为什么新增 BEIR/scifact 独立检索评测

背景：DeepSeek + Wikipedia 的 5 case benchmark 会同时受到 planner、search adapter、synthesis、citation checker 和网络波动影响，不能单独回答“local retriever 本身有没有比 keyword-only 更好”。
可选方案：继续只看端到端 benchmark；手写小语料相关性标注；接入完整 BEIR 框架；只加载 BEIR/scifact 数据并自己算指标。
最终选择：新增 `src/deepresearch_agent/retrieval_eval.py`，下载公开 BEIR/scifact 数据集，只用 corpus/query/qrels 评测本地 retriever 的 keyword、hybrid、hybrid+rerank 三种模式，并自己计算 Recall@10、nDCG@10、MRR。
理由：SciFact 是公开信息检索 benchmark，不是我项目自造数据；它有 qrels，可以直接评价召回和排序质量；自己写 loader 和 metric 能避免引入完整 BEIR 框架的重量，也能把评测逻辑讲清楚。
代价：SciFact 是英文科学摘要任务，所以评测时 embedding 模型切到 `BAAI/bge-small-en-v1.5`，不等于中文求职知识库场景；本地 reranker 在 CPU 上明显变慢，hybrid+rerank 的平均 query latency 到 `3.4431s`。
面试怎么答：我会说端到端 benchmark 看系统整体，BEIR/scifact 看检索模块本身；这次独立评测证明 hybrid 的 Recall@10 从 `0.6000` 到 `0.8239`。因为本次 `rerank_candidate_k=10` 且 `top_k=10`，rerank 只重排同一批 top10 候选，所以 Recall@10 与 hybrid 保持一致；它主要提升排序质量，把 nDCG@10 从 `0.6597` 提到 `0.7307`，但延迟代价也很明显。

## 决策 10：为什么做轻量 Run Control Plane 而不是迁移 LangGraph

背景：DeepResearch 不是一个瞬时 CRUD 请求，planner、多个 researcher、synthesis、citation check 都可能耗时、失败或方向跑偏；原来的 `/research` 更像一次性脚本调用，服务重启后只能看本地 JSONL trace，不能管理 run 生命周期。
可选方案：直接迁移 LangGraph durable execution；引入 Redis/Postgres/队列；在现有 FastAPI + async pipeline 外包一层轻量 run control；继续只保留一次性接口。
最终选择：不迁移 LangGraph，不重构主链路；新增自己的 `run_id + SQLite checkpoint + step trace + approval gate + SSE replay` 控制平面。
理由：本项目已有清晰的 orchestrator 主链路，这次目标是补生产化能力而不是换框架。SQLite 足够让本地 demo 和测试在服务重启后读回 run、steps、events；planner 后 HITL 能在 researcher 和 synthesis 之前阻止错误方向继续消耗 token/search 成本；SSE replay 能让客户端断线后按 `Last-Event-ID` 补发历史事件。
代价：它不是分布式调度系统；现在只有 SQLite 单机 lease/heartbeat、worker-once 消费入口和阶段边界协作式取消，没有常驻 worker pool、强制抢占正在进行的 LLM/search 调用；阶段级恢复当前以 planner checkpoint 后从 researcher 重跑为主，没有精确恢复到某个 researcher 子任务内部。
面试怎么答：我会说我借鉴的是 LangGraph 的 durable execution、checkpoint 和 human-in-the-loop 思想，但没有为了框架迁移牺牲项目可读性；我实现的是后端控制平面最小闭环：状态机、SQLite checkpoint、approval/resume/cancel/retry、SSE replay。

## 决策 11：为什么先补公开 Deep Research artifact 评测，而不是先做 judge 打分

背景：之前只有 5 条本地端到端 smoke benchmark 和 BEIR/scifact 检索模块评测。它们能证明 plumbing 和 retriever，但不能回答“面对公开 Deep Research 任务，完整报告链路表现怎样”，这是和 open_deep_research 这类项目的可信度差距。
可选方案：直接接官方 judge/LLM 评分；先手写更多本地 case；先复用 LiveDRBench/Deep Research Bench 任务格式输出 artifacts；继续只看 BEIR/scifact。
最终选择：新增 `src/deepresearch_agent/deep_research_eval.py`，优先做公开任务加载、完整 orchestrator 运行、raw JSONL、summary JSON 和 LiveDRBench-style predictions 输出；之后补两个可选 answer judge：`--judge-provider heuristic` 只按 case 里的 ground truth 字符串做本地命中率评分，`--judge-provider deepseek` 调 DeepSeek JSON mode 返回 `score/verdict/reason/matched/missing`。官方 judge 仍标记为 `not_run`，不编官方分数。
理由：没有 judge key 或官方 scoring 环境时，最重要的是先把每题的 query、配置快照、answer、claims、sources、trace、token、cost、citation check 和失败原因保存下来，保证结果可复查。heuristic answer judge 给无 key 的可复现弱信号；DeepSeek answer judge 复用本项目已有 v4-flash provider 和价格常量，能把非官方 LLM 评分、judge token 和估算成本记录进 artifact，但不会冒充官方分数。
代价：当前 `success_rate` 仍沿用本项目 citation retention 阈值，不等于官方 Deep Research Bench 分数；heuristic answer judge 只是 normalized substring matching，不能理解语义等价、格式错误或部分正确；DeepSeek answer judge 是非官方 LLM judge，受 prompt、模型版本和 ground truth 质量影响，当前只做了 stub/smoke，没有 live benchmark；LiveDRBench 任务常要求精确 JSON/论文标题，当前 synthesizer 还不是专门为该格式训练或约束的，所以真实组可以跑但质量很差。
面试怎么答：我会说我先补的是“公开任务可跑、artifact 可审计、本地弱评分可复现、可选 LLM judge 可审计”的评测底座，而不是假装已经有官方 leaderboard 分数。最新 1 条 LiveDRBench preview 真实口径就是失败样本：DeepSeek v4-flash + Wikipedia 跑通但 `success_rate=0.0`、citation retention `0.5`，这反而暴露了搜索覆盖和 citation 语义评测短板。

## 决策 12：为什么把 web search 和 crawler 分开接

背景：Wikipedia adapter 太窄，公开 Deep Research 任务需要真正的 web search 和网页正文抽取；如果 search adapter 只返回 title/snippet，synthesizer 很容易在证据不足时失败。
可选方案：继续强化 Wikipedia；直接接 Tavily/Brave 这种一体化 API；先接 SearxNG 搜索和 Jina Reader crawler；把 crawler 混在每个 search provider 里。
最终选择：在 `search.py` 里新增 provider registry，保留 `mock/wikipedia`，再加 `SearxngSearchAdapter`、`JinaSearchAdapter`、`BraveSearchAdapter`、`TavilySearchAdapter` 和独立 crawler provider。crawler 目前有 `JinaReaderCrawler` 和本地无 key 的 `HtmlTextCrawler`；SearxNG/Brave/Tavily/Jina 负责搜索候选 URL 或 snippet，crawler 负责把 URL 转成正文；`SearchService` 的 retry、timeout、circuit breaker、fallback 仍复用。
理由：search 和 crawler 的失败模式不同，分开后可以替换任一层：有 Jina 时走 LLM-friendly Reader，没有 key 时也可以用标准库 HTMLParser 做基础正文抽取，而 orchestrator 仍只看统一 `Source`。默认仍是 mock，不破坏无 key 路径；SearxNG 用 `SEARXNG_BASE_URL`，Jina/Brave/Tavily key 都只从环境变量读。
代价：当前没有自建 SearxNG 实例，所以 SearxNG 只做了 stub 单测；Jina Reader live crawl `https://example.com` 成功，但 Jina Search live smoke 在当前网络下返回 `401/403`；Brave/Tavily 因没有 key 也只做了 stub 单测；本地 HTML crawler 不执行 JavaScript、不处理登录、反爬、正文主内容识别或 robots。它是 provider 结构升级，还不是“真实搜索质量已经解决”。
面试怎么答：我会说这一步补的是工具层边界，不是刷 benchmark。真实网页搜索要分成 query→URL 和 URL→content 两层，否则后面 citation grounding 没法定位证据片段。

## 决策 13：为什么先做 claim-level evidence quote，而不是直接上 LLM judge

背景：当前 citation checker 只有 claim/source lexical overlap，能算 retention，但不能告诉我“具体哪段 source 支撑了这个 claim”。如果直接接 LLM judge，成本、延迟和 judge 自身可靠性都会让问题变复杂。
可选方案：继续只算 overlap；直接上 DeepSeek/OpenAI judge；先做 evidence span/quote grounding；用本地 NLI 模型做 entailment。
最终选择：在 `CitationAssessment` 里新增 `support_level` 和 `evidence_quotes`。checker 仍从 `[Sx]` 找 source，但会在每个 cited source 内按句子找最接近 claim 的证据 quote，输出 `supported / partial / unsupported / unverifiable` 四种 level。
理由：这一步不需要额外 API key，能保持 CI 可跑，也为后续 LLM judge/NLI 提供结构化输入：claim、citation_id、source_title、evidence_quote、overlap_score。旧的 `supported` 和 `retention_rate` 语义保留，已有 benchmark 不会被强行换口径。
代价：它仍然是 lexical grounding，不等于语义蕴含；中文和长表格/列表的句子切分仍粗糙；`partial` 只是提示证据不足，不是严格事实分类。
面试怎么答：我会说我没有一步跳到昂贵 judge，而是先把 citation 从“一个分数”升级成“claim -> cited source -> evidence quote”的可审计结构。后续接 LLM judge 时，judge 看的是具体 quote，不是整篇网页。

## 决策 14：为什么 reflection loop 默认关闭且有轮次上限

背景：原主链路是一次 plan、一次并发 research、一次 synthesis；这比真实 deep research agent 少了“发现证据不足后再研究”的循环。直接做无限循环会带来成本、延迟、状态恢复和测试不稳定问题。
可选方案：保持一次性 plan；让 LLM 每轮自由决定是否继续；做固定 N 轮；做启发式 bounded reflection；迁移 LangGraph 循环节点。
最终选择：在 `ResearchRequest` 上新增 `reflection_enabled`、`max_reflection_rounds`、`reflection_min_sources`。默认关闭；显式开启后，每轮先写 `compression.roundN`，压缩已有 findings，再按 source coverage/fallback 启发式判断是否追加一个 `R<N>` follow-up question，最多跑配置的轮数。
理由：这一步先补 control flow 边界，不引入新的 LLM judge 或 planner prompt。默认关闭能保护现有 benchmark 口径；轮次上限和阈值能避免成本失控；trace 里保留 compression/reflection payload，后续可以替换成 LLM reflection policy。
代价：当前 reflection 是启发式，不是真正语义判断“信息是否足够”；追加问题模板也比较通用，可能召回重复信息。run control 已复用同一 helper，但还没有阶段级 checkpoint 到单个 reflection round。
面试怎么答：我会说这是从 pipeline 到 agent loop 的第一步：先把循环、压缩、追加问题和 trace 边界做出来，再把启发式 policy 换成 LLM reflection 或 evaluator。

## 决策 15：为什么 MCP 先作为 search/tool adapter 接入

背景：open_deep_research 和 DeerFlow 都把 MCP 当外部工具扩展点；本项目之前只有 Python adapter，工具生态和它们相比差距明显。但直接把所有 researcher tool call 都迁成 MCP 会影响主链路稳定性。
可选方案：暂不做 MCP；引入完整 MCP SDK 并重构工具层；先写一个最小 JSON-RPC stdio/http client；只做 MCP search adapter，把 tool result 转成 `Source`。
最终选择：新增 `src/deepresearch_agent/mcp_tools.py`，支持 stdio/http JSON-RPC 的 `tools/list` 和 `tools/call`，并在 `search.py` 里加 `mcp` search provider。通过 `MCP_SEARCH_TOOL` 调一个明确的搜索工具，结果统一转换成 `Source`，下游 dedup/verifier/synthesizer 不需要改。
理由：这保持了本项目的边界：researcher 仍消费 `Source`，MCP 只是新工具来源；默认不启用，不影响无 key / 无 server 路径。配置全部来自环境变量：`MCP_TRANSPORT`、`MCP_COMMAND`、`MCP_ARGS`、`MCP_HTTP_URL`、`MCP_SEARCH_TOOL`、`MCP_QUERY_ARGUMENT`。
代价：当前没有接真实 MCP server 做 live run，只用 fake client 和 adapter 单测验证协议边界；stdio client 是最小实现，不包含长连接复用、server capability 细分、资源订阅和复杂认证。
面试怎么答：我会说我没有把主链路绑死在某个 MCP server 上，而是先把 MCP 的工具结果纳入统一 Source 模型。后续接 filesystem、browser、company search MCP server 时，不需要改 synthesizer 和 citation checker。

## 决策 16：为什么先做 FastAPI 内置审核页，而不是单独前端工程

背景：Run control plane 已经有 approval/edit/reject/cancel/SSE replay API，但没有 DeerFlow 那种可视化 plan 修改、报告编辑和 citation 查看页面。直接引入 React/Vite 会新增构建链、依赖和部署复杂度。
可选方案：继续只用 API；新建 React 前端；用 FastAPI 返回一个内置 HTML/JS 页面；接入现成 admin UI。
最终选择：新增 `src/deepresearch_agent/ui.py`，由 `GET /ui` 返回一个无构建链的审核页面；新增 `GET /runs` 列出最近 run。页面能创建 run、查看 run list、编辑 plan JSON、approve/reject/cancel、订阅 SSE events，并展示 report、sources、citation assessments/evidence quotes。
理由：本项目当前更需要证明 HITL 后端闭环可用，而不是做复杂产品前端。内置页面一条 `deepresearch-api` 就能打开，适合本地 demo 和面试展示；API 仍保持独立，后续替换成 React 不影响 run control。
代价：它不是完整产品 UI，没有登录权限、多人协作、富文本报告编辑、可视化 diff、持久草稿和前端测试截图；本机没有 Playwright 包，所以这次只做了 TestClient + HTTP probe，没有浏览器截图验证。
面试怎么答：我会说这一步把 HITL 从“只有 API”变成“能看、能改、能 approve 的最小审核面”，但不会把它包装成 DeerFlow 级别前端。

## 决策 17：为什么先做 SQLite worker lease，而不是直接引入队列系统

背景：现在 run control 已能持久化状态和 checkpoint，但同步 API worker 之间没有 ownership 语义。生产系统里如果两个 worker 同时 resume 同一个 run，会造成重复检索、重复扣费和状态覆盖。
可选方案：直接上 Redis/RQ/Celery；用 Postgres advisory lock；在 SQLite `agent_runs` 上加 lease/heartbeat 字段；暂时不管并发 worker。
最终选择：在 `agent_runs` 增加 `leased_by`、`heartbeat_at`、`lease_expires_at`，用原子 SQL `UPDATE ... WHERE lease is null/expired/same worker` 获取 lease；新增 heartbeat、stale list 和 recover stale API。内部 planner/researcher/synthesizer/verifier 执行路径也会自动 acquire/heartbeat/release。
理由：这个项目默认仍要无外部依赖跑通，SQLite lease 能先把 worker ownership、过期检测、stale recovery 这些生产概念打出来。以后迁移 Postgres/Redis 时，业务层只需要替换 `RunStore` 的 lease 实现。
代价：这不是完整任务队列，没有常驻 worker pool、分布式调度、公平排队、幂等 stage replay、长任务强制抢占和强事务隔离；运行中取消只在 planner/researcher/synthesizer/verifier 阶段边界生效。SQLite 锁竞争下也不适合高并发。
面试怎么答：我会说我先补的是“同一个 run 只能被一个 worker 拿走”的控制平面语义，而不是假装做了生产队列。这个取舍让项目保持可跑，同时能解释下一步怎么演进到 Postgres/Redis/worker pool。

## 决策 18：为什么先做可选持久化 Chroma index，而不是直接上 Qdrant/Milvus

背景：local hybrid retrieval 每次新建 retriever 都要为 `data/local_corpus.jsonl` 重新 embedding 并创建 Chroma collection，审核页 smoke 也暴露出 hybrid 冷启动会拖慢 run。
可选方案：保持临时内存 index；默认启用本地持久化 Chroma；接 Qdrant/Milvus；先做 embedding pickle cache。
最终选择：新增 `LOCAL_VECTOR_INDEX_PERSIST` 和 `LOCAL_VECTOR_INDEX_PATH`，默认关闭；显式开启后使用 Chroma PersistentClient，并用 corpus chunk、embedding provider、embedding model 的 fingerprint 生成稳定 collection 名。collection count 与当前 chunk 数一致时直接复用，不重新 embed corpus。
理由：这一步只改检索层，不影响 orchestrator 的 `Source` contract，也不需要额外服务、端口或账号。它先解决“本地私有知识库索引生命周期”这个能力缺口，同时保持无 API key 可跑通。
代价：Chroma 本地持久化仍是单机文件索引，不是生产级向量数据库；当前复用判断主要靠 fingerprint collection name 和 count，没有后台 reindex 任务、版本迁移、TTL、压缩、并发写控制或跨机器共享。
面试怎么答：我会说我先把私有知识库从“每次临时建索引”推进到“可复用的本地持久化索引”，但不会把它说成 Qdrant/Milvus 级别的生产向量库。

## 决策 19：为什么先做阶段模型覆盖，而不是一次性接多家模型 provider

背景：open_deep_research / DeerFlow 这类系统通常会按 planner、research、synthesis、compression 等角色选择不同模型。本项目之前只有一个 `llm_model`，所有 LLM stage 共用同一个模型名，无法展示“便宜模型规划、强模型合成”这类策略。
可选方案：直接接 OpenAI/Anthropic/OpenRouter/Ollama；只保留单模型；先做阶段模型路由，再扩展 provider；把模型选择硬编码在 provider 内。
最终选择：保留默认 `deepseek-v4-flash` 和 mock provider，新增 `brief_model`、`planner_model`、`synthesis_model` 请求字段，以及 `LLM_BRIEF_MODEL`、`LLM_PLANNER_MODEL`、`LLM_SYNTHESIS_MODEL` 环境变量。DeepSeek 请求体会按 stage 使用对应模型，`CostTracker` 每条 record 也记录真实 stage model。
理由：这一步不引入新的 API key 和 provider 复杂度，但先把多模型策略最关键的控制面打出来。后续接 OpenAI/Anthropic 时，可以复用同一套 stage model 字段和成本归因。
代价：当前仍只有 mock 和 DeepSeek 两类 LLM provider；没有按 source quality 自动选模型，没有动态降级策略，也没有证明不同 stage model 会提升质量或降低成本。
面试怎么答：我会说我先做的是“模型路由接口和可观测归因”，不是宣称已经有完整 model zoo。它让面试官看到我知道多 Agent 系统里模型选择应该是按角色管理的，而不是一个全局模型名打到底。

## 决策 20：为什么内容导出分阶段做，而不是一开始做全套办公文档

背景：之前报告只能从 API/CLI 读结构化 JSON 或 markdown answer，不方便把一次 run 交给别人审阅，也没有稳定 artifact 文件。
可选方案：只保留 API JSON；先导出 Markdown/HTML/JSON；直接生成 PDF/DOCX/PPT；接第三方文档服务。
最终选择：第一步新增 `src/deepresearch_agent/report_exporter.py`，支持 Markdown、静态 HTML 和完整 JSON；之后继续扩展到文本版 PDF、DOCX、PPTX 和可选 WAV，分别用 `reportlab`、`python-docx`、`python-pptx` 与 `src/deepresearch_agent/tts.py` 的 Windows SAPI provider 生成文件。CLI 仍通过 `--export-dir` 和 `--export-formats` 控制格式。
理由：Markdown/HTML/JSON 保留完整可审计结构，PDF/DOCX/PPTX 解决“发给别人直接打开或展示”的交付形态，WAV 解决“把报告变成可听摘要”的最小闭环。这些格式仍从同一个 `StructuredReport` 生成，保留 answer、sources、citation assessments 和 evidence quotes，不改变主链路。
代价：新增 `reportlab`、`python-docx`、`python-pptx` 依赖；WAV 依赖本机 Windows SAPI；PDF/DOCX/PPTX 是文本版报告，不是复杂版式系统，没有封面模板、目录、图表、图片布局、批注、修订模式或完整 podcast/TTS 工作流。
面试怎么答：我会说我先交付可审计 artifact，再补常见文档格式和一个本地语音导出口；但我不会把文本版 PDF/DOCX/PPTX 或 Windows SAPI WAV 包装成完整办公自动化 / 播客生产平台。

## 决策 21：为什么先做 OTLP HTTP trace export，而不是直接接 LangSmith 或完整 OpenTelemetry SDK

背景：之前 trace 只写本地 JSONL，适合无依赖 demo 和排查，但和生产系统常见的 collector / APM / tracing 平台之间没有出口。
可选方案：继续只写 JSONL；直接引入 OpenTelemetry SDK；直接接 LangSmith；先做一个可选 OTLP HTTP exporter；把 trace 写入数据库。
最终选择：在 `src/deepresearch_agent/tracing.py` 里新增 `TraceExporter` 抽象和 `OtlpHttpTraceExporter`，配置 `TRACE_EXPORTER=otlp_http` 与 `OTEL_EXPORTER_OTLP_ENDPOINT` 时，把每条 `TraceEvent` 额外 POST 到 `<endpoint>/v1/traces`；默认仍是本地 JSONL。
理由：这一步不新增依赖、不需要外部账号，也不影响无 API key 路径；同时给现有 trace event 一个标准化外送边界。export 失败不会打断 research run，而是写一条本地 `trace_exporter` error event，便于排查 collector 不可用。
代价：这不是完整 OpenTelemetry instrumentation，没有上下文传播、采样、batch processor、metrics/logs pipeline、LangSmith run tree 或线上 collector 验证；当前只验证了本地 HTTP test server 收到 OTLP 风格 JSON。
面试怎么答：我会说我先把 trace 从“只能本地看”推进到“可以接 collector 的出口”，但仍保留 JSONL 作为默认和兜底，不把它包装成完整可观测平台。

## 决策 22：为什么 citation judge 默认关闭，但做成可切换 provider

背景：原来的 `CitationChecker` 已能找 citation ID、evidence quote 和 lexical support_level，但 lexical overlap 不能判断语义蕴含、反义或复杂跨句支撑。直接把 LLM judge 默认打开会改变历史 benchmark 口径，并引入额外成本、延迟和 judge 自身不稳定。
可选方案：继续只做 lexical；默认启用 LLM judge；新增本地 heuristic judge；做 provider 抽象，默认 none，显式启用 heuristic/deepseek。
最终选择：新增 `src/deepresearch_agent/citation_judge.py`，提供 `HeuristicCitationJudgeProvider` 和 `DeepSeekCitationJudgeProvider`；`CitationChecker.check()` 接收可选 judge provider 和 cost tracker。默认 `CITATION_JUDGE_PROVIDER=none`，CLI/benchmark/public eval 可以显式传 `--citation-judge-provider heuristic|deepseek`。
理由：默认不改变既有结果；无 key 环境可以用 heuristic smoke 验证 judge plumbing；有 `DEEPSEEK_API_KEY` 时可以用 DeepSeek JSON mode 做 claim/evidence 判断，并把 usage 计入 `citation_judge` 成本阶段。
代价：heuristic judge 本质仍是 overlap；DeepSeek judge 目前只用 stub 测了解析和 usage 成本，没有做真实 live benchmark，也没有 NLI 模型、人工标注集或 judge 可靠性评估。
面试怎么答：我会说 citation faithfulness 现在是两层结构：默认 lexical 负责便宜可复现，optional judge 负责语义评审接口；我不会声称 LLM judge 已经实测提升质量。

## 决策 23：为什么补 Brave/Tavily provider，但默认仍不依赖它们

背景：Wikipedia 对公开 Deep Research 任务覆盖太弱，SearxNG 需要自建实例，Jina Search 当前环境 live 返回过 401/403。open_deep_research / DeerFlow 这类项目通常会支持多个真实 web search provider。
可选方案：继续只用 Wikipedia；只接 SearxNG/Jina；新增 Brave/Tavily 这种常见商业搜索 API；直接接 crawler 平台。
最终选择：在 `src/deepresearch_agent/search.py` 新增 `BraveSearchAdapter` 和 `TavilySearchAdapter`。Brave 按官方 Web Search API 用 `GET /res/v1/web/search` 和 `X-Subscription-Token`；Tavily 按官方 Search endpoint 用 Bearer auth 和 JSON body 的 `query/search_depth/max_results`。key 分别只从 `BRAVE_SEARCH_API_KEY` 和 `TAVILY_API_KEY` 读取。
理由：这一步只扩展 search adapter 层，仍输出统一 `Source`，不改 orchestrator；默认 provider 仍是 mock，保证无 key 可跑。benchmark/public eval 的 `--search-provider` choices 同步支持 `brave/tavily`，后续有 key 时可以直接重跑同一套评测。
代价：当前只用 stub 单测验证请求格式和响应解析，没有真实 Brave/Tavily API key，也没有 live benchmark、限流、配额成本或相关性评测。
面试怎么答：我会说我把搜索层从“Wikipedia 兜底”扩展成可插拔真实搜索 provider，但不会把未实测的 Brave/Tavily 包装成质量提升。

## 决策 24：为什么先做 deferred run + worker-once，而不是直接上 Redis/Celery

背景：上一阶段只有 lease/heartbeat，能避免两个 worker 同时执行同一个 run，但 `POST /runs` 默认仍在 API 请求内同步跑 planner；这和生产队列还有距离。
可选方案：直接接 Redis/RQ/Celery；引入 Postgres job table；保留同步 API；在 SQLite run store 上先补 queued run 和单次 worker 消费。
最终选择：给 `CreateRunRequest` 增加 `defer_execution`，默认 `false`，不改变原 `/runs` 行为；创建 run 时把完整请求写入 `agent_runs.request_json`；`/runs/worker/next` 用原子 SQL claim 最早的 queued run，拿到 lease 后从 `request_json` 恢复 provider、模型、并发和检索参数，再执行 planner/researcher/synthesis/verifier。
理由：这一步把“API 只入队、worker 后台执行”的控制面打通，但仍保持无外部依赖和可测试。`request_json` 是关键：否则 worker 只能看到 query，会退回默认 settings，容易重现以前 benchmark 配置快照和实际运行不一致的问题。
代价：它只是 worker-once，不是常驻 worker pool；没有 Redis 可见队列、并发 worker 调度、任务优先级、幂等 stage replay、强制抢占或 backpressure。运行中取消是阶段边界检查，不会打断已经发出的 LLM/search 请求。SQLite 锁竞争下仍不适合高并发生产。
面试怎么答：我会说这一步不是宣称完成 Celery/RQ，而是把生产 run control 的下一块拼上：入队、请求快照、worker claim、lease ownership、状态和事件可追踪。后续迁移 Redis/Postgres 时，可以保留同样的 API 语义。

## 决策 25：为什么加本地 polling worker CLI，但仍不引入 Celery

背景：`/runs/worker/next` 已能处理一条 queued run，但真实使用时需要一个进程持续轮询队列；如果每次都手动 POST，展示和回归都不方便。
可选方案：继续只保留 worker-once API；直接接 Celery/RQ；新增一个本地 polling worker CLI；把 worker loop 放进 FastAPI lifespan。
最终选择：新增 `src/deepresearch_agent/run_worker.py` 和 console script `deepresearch-worker`。它循环调用 `RunController.process_next_queued()`，支持 `--max-runs`、`--idle-exit`、`--poll-interval-seconds` 和 `--json`，默认不改变 API 服务进程。
理由：这个做法让本地 demo 可以像生产 worker 一样“排队后后台消费”，同时继续保持无 Redis、无 broker、无外部部署账号。worker loop 复用 controller，不复制 planner/researcher/synthesis 逻辑，也不会绕过 lease。
代价：它仍是单机轮询 worker，不提供任务优先级、并发 worker pool、backpressure、队列可视化、死信队列、broker ack 或跨机器调度。FastAPI 进程也没有自动托管 worker，部署时需要单独启动。
面试怎么答：我会说我做的是从 worker-once 到本地 worker loop 的最小生产化增量，目的是让 run control 的队列消费闭环能实际运行；真正上生产仍会换成 Redis/Postgres + worker pool，但 API 和 store contract 已经先稳定下来。

## 决策 26：为什么先做离线文档 ingest，而不是直接上 RAGFlow/Qdrant/Milvus

背景：本地 RAG 已经有 keyword/vector/hybrid 和可选持久化 Chroma，但语料入口仍是手写的 `data/local_corpus.jsonl`。这不适合把一批私有笔记或项目文档快速接进来。
可选方案：继续手写 JSONL；直接接 RAGFlow；直接上 Qdrant/Milvus document pipeline；先做 Markdown/TXT 到 JSONL 的离线 ingest，再补 PDF/DOCX 文本抽取。
最终选择：新增 `src/deepresearch_agent/ingest_corpus.py` 和 console script `deepresearch-ingest-corpus`。它递归读取 `.md/.markdown/.txt/.pdf/.docx`，默认跳过 `.git/.obsidian/.claude/node_modules/__pycache__`，Markdown 会清理 YAML frontmatter，PDF 用 `pypdf` 抽取每页文本，DOCX 用 `python-docx` 抽取段落和表格文本，最后输出现有 `LocalRagRetriever` 可直接消费的 JSONL。
理由：这一步补的是私有知识库的入口和格式契约，不引入服务端向量库、后台任务或账号。生成的 JSONL 仍能走 keyword baseline、hybrid vector、持久化 Chroma、Qdrant provider 和 rerank，所以下游能力不需要改。
代价：当前只做文本抽取，不做扫描件 OCR、网页递归爬取、增量 manifest、embedding cache、文档删除同步、权限隔离或版本化 reindex。大规模知识库仍需要 Qdrant/Milvus/RAGFlow 这类生产系统。
面试怎么答：我会说我先把“私有文档如何进入本地 RAG”打通，而不是直接引入重基础设施。PDF/DOCX 支持让入口更接近真实知识库，但我不会把它说成完整文档管理平台。

## 决策 27：为什么先做 OpenAI-compatible adapter，而不是一次性接多家原生 SDK

背景：本项目已有 DeepSeek provider 和阶段模型路由，但多模型策略仍偏单一。OpenRouter、Ollama、LM Studio 以及很多自建网关都暴露 OpenAI-compatible Chat Completions 接口；直接接每家原生 SDK 会引入大量 key、错误码和依赖差异。
可选方案：继续只用 DeepSeek；直接接 OpenAI/Anthropic/OpenRouter/Ollama 全家桶；抽一个 OpenAI-compatible adapter；先只做文档计划不写代码。
最终选择：新增 `OpenAICompatibleLLMProvider`，复用 DeepResearch 的 JSON mode prompt、stage model 路由和 usage/cost 记录逻辑；通过 `OPENAI_COMPATIBLE_BASE_URL`、`OPENAI_COMPATIBLE_MODEL`、`OPENAI_COMPATIBLE_API_KEY_ENV`、`OPENAI_COMPATIBLE_API_KEY_REQUIRED`、`OPENAI_COMPATIBLE_INPUT_COST_PER_1M_TOKENS`、`OPENAI_COMPATIBLE_OUTPUT_COST_PER_1M_TOKENS` 配置。默认 provider 仍是 mock。
理由：这一步先把“兼容 Chat Completions 的模型网关”接入同一个 LLM provider contract，不新增默认 key 依赖，也不影响 DeepSeek 真实 benchmark。对本地 Ollama 这类无 key endpoint，`OPENAI_COMPATIBLE_API_KEY_REQUIRED=false` 可以直接尝试；对 OpenRouter 等托管网关，可以把 key 环境变量名显式配置出来。
代价：当前只用 stub provider 测了请求模型路由和 usage 成本记录，没有 live 调 OpenRouter/Ollama/OpenAI-compatible endpoint；不同网关对 `response_format={"type":"json_object"}` 的支持不完全一致，真实质量和错误处理仍需分别验证。
面试怎么答：我会说我没有把“多 provider”做成一堆硬编码分支，而是先抽了 OpenAI-compatible 这条公共接口；这能覆盖很多本地/托管模型，但我不会声称已经完成所有 provider 的真实评测。

## 决策 28：为什么补 Qdrant HTTP vector index，但默认仍用 Chroma

背景：本地私有知识库已经有 Markdown/TXT ingest 和 Chroma 持久化 index，但 Qdrant/Milvus 这类外部向量库仍是和 DeerFlow / RAGFlow 类项目对齐时会被问到的生产化缺口。
可选方案：直接把默认 index 切成 Qdrant；接 Qdrant/Milvus SDK；只写计划不实现；做一个可选 Qdrant HTTP adapter，默认仍保留 Chroma。
最终选择：在 `rag.py` 里新增 `QdrantVectorIndex`，通过 `LOCAL_VECTOR_INDEX_PROVIDER=qdrant` 或 CLI `--local-vector-index-provider qdrant` 显式启用；配置项是 `QDRANT_BASE_URL`、`QDRANT_COLLECTION`、`QDRANT_API_KEY_ENV`，key 只从环境变量读取。默认仍是 `chroma`。
理由：HTTP adapter 不新增 SDK 依赖，也不破坏无外部服务可跑通的默认路径；同时把 vector index provider 边界抽出来，后续接 Milvus 或 Qdrant Cloud 只需要替换索引层，不碰 orchestrator、dedup、verifier 或 synthesizer。
代价：当前只用 stub HTTP 单测验证 collection create/upsert/search 和 reuse 逻辑，没有启动真实 Qdrant 服务，也没有做大规模向量库延迟、并发写、删除同步、payload filter 或权限隔离评测。
面试怎么答：我会说我已经把 Chroma 单机索引升级成可替换的 vector index provider，并补了 Qdrant HTTP 入口；但默认不切过去，因为项目要保证 clone 后无服务依赖也能跑，真实 Qdrant 质量和运维还需要单独 benchmark。

## 决策 29：为什么运行中取消做成阶段边界检查，而不是强制杀任务

背景：run control 已经有 `/cancel`，但如果用户在 researcher 或 synthesizer 正在执行时取消，原来的通用异常处理可能把后续 `run cancelled` 当成失败，导致终态从 `cancelled` 被覆盖成 `failed`。生产长任务系统需要区分“用户主动取消”和“系统失败”。
可选方案：直接取消 asyncio task；给每个 provider 做 abort signal；在 run control 阶段边界做协作式取消；继续只支持 waiting_approval 前取消。
最终选择：新增 `RunCancelledError`，`planner/researcher/synthesizer/verifier` 阶段边界调用 `_raise_if_cancelled()`；`create_run/retry/process_next_queued/_continue_from_plan` 单独捕获取消异常，返回并保留 `cancelled` 终态，不再写 failed step/event。`_acquire_execution_lease()` 和 `_heartbeat_execution_lease()` 遇到已取消 run 也转成同一类取消异常。
理由：这个改法不重构 orchestrator，不改变 provider 接口，也不会破坏无 key mock 路径；它先保证 run state machine 的语义正确：用户取消不是系统失败，后续 approve/retry/fail 不能覆盖取消终态。
代价：这是协作式取消，只在阶段边界生效。已经发出去的 LLM 请求、search 请求或本地 embedding/rerank 计算不会被强制中断；真正的抢占式取消需要 provider 级 timeout/abort、worker task registry，甚至进程级隔离。
面试怎么答：我会说当前版本先把控制平面的状态一致性修好：取消后不会被误报为失败，lease 会释放，事件也保持可追踪。它不是 Kubernetes/Celery 那种任务抢占，生产化下一步才是给每类 provider 加 abort signal 和更细粒度 checkpoint。

## 决策 30：为什么 retry 先复用 researcher checkpoint，而不是做每个 researcher 子任务幂等

背景：之前 failed run 的 retry 只复用 planner checkpoint，从 researcher 阶段重新跑。这样如果真实失败点在 synthesizer 或 verifier，系统会重复 search/retrieval，浪费时间和外部 API 调用。
可选方案：继续只复用 planner；把每个 researcher 子任务拆成独立 checkpoint；保存 researcher 阶段整体输出，在后续阶段失败时先复用；直接引入 LangGraph durable node replay。
最终选择：在 researcher 成功 step 的 `output_json.checkpoint` 中保存 `findings/all_sources/sources/raw_search_result_count/fallback_count`；`retry_count > 0` 且存在成功 researcher checkpoint 时，`_execute_research_flow()` 直接复用该 checkpoint，写 `researcher.checkpoint_reused` event，再进入 synthesizer/verifier。
理由：这是最小的幂等恢复增量：不改 orchestrator、不改 search adapter，也不需要引入队列框架，就能避免“后处理失败后重新检索”。对求职展示来说，这比空谈分布式恢复更具体。
代价：checkpoint 粒度是整个 researcher 阶段，不是单个 subquestion；SQLite 里会保存 findings 和 source 内容，结果变大；如果失败发生在 researcher 内部，仍然需要重跑 researcher。真正生产化还需要按 researcher 子任务、reflection round、synthesis、verifier 分别做幂等节点和大对象存储。
面试怎么答：我会说我先解决最常见的浪费：检索已经成功，但合成或校验失败时不应该重新打搜索。当前是阶段级 checkpoint reuse，不是完整 DAG replay。

## 决策 31：为什么 search retry backoff / rate limit 做成进程内可选配置

背景：真实 web search provider 可能有 QPS、并发或额度限制；多个 researcher 并发时，即使有 retry/circuit breaker，也可能在短时间内打出 burst 请求；失败后立即重试也会放大 429 或瞬时错误。
可选方案：不做 backoff/限流；在每个 provider adapter 里硬编码 sleep；在 `SearchService` 外层做统一进程内 retry backoff 和 limiter；直接引入 Redis/网关级全局限流。
最终选择：新增 `SearchRateLimiter`，用 async lock 和 min-interval 控制 primary search 调用间隔；新增 `SEARCH_RETRY_BACKOFF_SECONDS`，失败后按 `base * 2^attempt` 做简单指数 backoff。两者默认都关闭；fallback mock 不限流，避免外部失败后本地兜底也被拖慢。
理由：backoff/限流都应该贴近工具调用层，而不是散落在每个 provider；默认关闭能保持 mock/CI 快速和无 key 路径不变。有真实 Brave/Tavily/SearxNG/Jina key 时，可以先用本地 limiter 降低 burst 风险，用 backoff 给 429/瞬时错误恢复窗口。
代价：它只约束当前 Python 进程，不是跨进程/跨机器全局 quota；backoff 也没有解析 provider-specific `Retry-After`，不理解不同套餐或动态配额。生产化仍需要 provider-specific backoff、分布式 rate limit 和集中任务队列。
面试怎么答：我会说我先把 retry backoff 和 rate limit 作为 SearchService 的可配置保护补上，避免并发 researcher 直接打爆外部搜索，失败时也不要马上重试；但我不会把它说成生产级全局限流。

## 决策 32：为什么 hybrid local retrieval 缺向量能力时降级到 keyword

背景：默认 `LOCAL_RETRIEVAL_MODE=hybrid` 能展示关键词 + 向量 + RRF 的完整检索结构，但在没有 Chroma、没有本地 embedding 模型，或者向量索引 build/search 失败的机器上，原实现会直接抛错并打断整条 research run。
可选方案：把默认改回 keyword；要求所有环境必须安装 ML extras；在 API/spine 测试里手动覆盖成 keyword；保留默认 hybrid，但向量侧不可用时自动降级到 keyword-only。
最终选择：保留默认 hybrid，不改 CLI / HTTP API；`LocalRagRetriever.retrieve()` 在 hybrid 模式下捕获 vector build/search/embedding 异常，写 `logging.warning`，把本次 retrieval 标成 `last_retrieval_degraded=True`，并返回 keyword 结果。返回的 `Source.metadata` 会带 `retrieval_degraded=True` 和 `degrade_reason`。
理由：这符合项目“无 key / 弱环境也能跑通”的底线，同时不把降级伪装成正常 hybrid。下游 orchestrator 仍只接收统一 `Source`，不用改主链路。
代价：降级后没有向量召回和 rerank，结果质量回到 keyword baseline；metadata 只标注 local RAG source，最终 dedup 后如果同 URL 被其他 provider 覆盖，需要看 trace/source metadata 分析。
面试怎么答：我会说 hybrid 是默认能力，但不是让运行环境因为一个向量依赖失败就整体挂掉。真正工程化要可观测地降级，而不是静默吞错或把 keyword 结果说成 hybrid。

# 5 实现细节

Planner：`src/deepresearch_agent/llm.py`。输入是 `ResearchBrief`，输出是 `SubQuestion` 列表。默认 deterministic mock planner 会生成 background、evidence、tradeoffs 三类问题，用于离线可复现；DeepSeek planner 会用 JSON mode 生成符合同一 Pydantic schema 的子问题。`ResearchRequest.planner_model` 或 `LLM_PLANNER_MODEL` 可以覆盖 planning stage 的模型名。局限是 planner 还不会根据 researcher 中间结果做 LLM 语义级动态追加子问题。

DeepSeek Planner 验证：`src/deepresearch_agent/llm.py` 里新增了 `DeepSeekLLMProvider.plan`，第一步先独立验证结构化输出；验证脚本是 `src/deepresearch_agent/validate_deepseek_structured_output.py`，它从环境变量读取 `DEEPSEEK_API_KEY`，用 JSON mode 请求默认 `deepseek-v4-flash`，并用现有 `SubQuestion` Pydantic schema 解析输出。后续步骤再把同一个 provider 扩展到 `create_brief` 和 `synthesize`，避免一次性接太多导致错误边界不清。

DeepSeek Synthesizer 接入：步骤 2 以后，`DeepSeekLLMProvider.create_brief`、`plan`、`synthesize` 都走 DeepSeek JSON mode。CLI 和 API 可以通过 `llm_provider="deepseek"` 或 CLI 参数 `--llm-provider deepseek` 显式启用；默认仍是 mock，保证离线测试不受 API key 影响。当前 synthesis 要求模型输出 `{"answer": "...", "claims": [...]}`，并要求每条 factual claim 使用输入 sources 中已有的 `[Sx]` citation ID。`brief_model`、`planner_model`、`synthesis_model` 会分别覆盖 DeepSeek 请求体中的 `model` 字段；不填时回落到 `llm_model` / `DEEPSEEK_MODEL`。

OpenAI-compatible LLM Provider：`src/deepresearch_agent/llm.py` 的 `OpenAICompatibleLLMProvider` 复用 DeepSeek provider 的 brief/plan/synthesis JSON prompt 和 schema validation，但 `_post_chat_completions()` 改为读取通用 `base_url`、model 和可选 API key。`orchestrator.py` 里 `llm_provider="openai-compatible"` 或 `openai_compatible` 会构造它；`cli.py`、`benchmark.py`、`deep_research_eval.py` 的 provider choices 也已同步。成本记录不使用 DeepSeek 官方价格，而是用 `OPENAI_COMPATIBLE_INPUT_COST_PER_1M_TOKENS` / `OPENAI_COMPATIBLE_OUTPUT_COST_PER_1M_TOKENS` 这两个显式配置，默认 0。局限是尚未 live 测试任何真实 OpenAI-compatible endpoint。

Researcher：`src/deepresearch_agent/orchestrator.py` 的 `_research_one`。输入是子问题，输出是 `Finding`。它调用 `SearchService` 和 `LocalRagRetriever`，再 dedup、verify、summary。`LocalRagRetriever` 内部已经从 keyword overlap 升级为可配置的 keyword / hybrid retrieval，但 orchestrator 仍只接收 `Source` 列表。局限是 summary 仍是模板化，不是自然语言 LLM 压缩。

Reflection / Compression Loop：`src/deepresearch_agent/orchestrator.py` 的 `_run_reflection_rounds`、`_compress_findings`、`_reflect_on_evidence`。输入是初始 plan 和 researcher results，输出是可能扩展后的 plan/results。开启 `reflection_enabled` 后，每轮先把 findings 压成短文本写入 `compression.roundN` trace，再根据 fallback_count 和每个 finding 的唯一 source 数是否低于 `reflection_min_sources` 来决定是否追加 `R<N>` 子问题。`run_control.py` 的 researcher 阶段也调用同一个 helper，所以 `/research` 和 `/runs` 语义一致。局限是当前 policy 是启发式，不是 LLM reflection。

Web Search / Crawler Provider：`src/deepresearch_agent/search.py`。`build_search_adapter()` 现在按 provider name 构造 `mock`、`wikipedia`、`searxng`、`jina`、`brave`、`tavily` 或 `mcp`，`build_crawler()` 按配置构造 `JinaReaderCrawler` 或 `HtmlTextCrawler`。`SearxngSearchAdapter` 调 `SEARXNG_BASE_URL/search?format=json`，解析 title/url/snippet，再可选调用 crawler 抽正文；`JinaReaderCrawler` 用 `https://r.jina.ai/<url>` 抽 LLM-friendly text；`HtmlTextCrawler` 直接抓 URL，用标准库 `HTMLParser` 去掉 script/style/noscript/svg 后抽正文；`JinaSearchAdapter` 用 `https://s.jina.ai/<query>`。`BraveSearchAdapter` 读 `BRAVE_SEARCH_API_KEY`，调用 Brave Web Search API；`TavilySearchAdapter` 读 `TAVILY_API_KEY`，调用 Tavily Search API，`TAVILY_SEARCH_DEPTH` 默认 `basic`。这些 key 都不进入 Settings 快照。

MCP Tool Adapter：`src/deepresearch_agent/mcp_tools.py`。`McpToolSearchAdapter` 用 `McpClient.call_tool()` 调配置好的 search tool，把 MCP result 里的 `sources` 或 `content[type=text]` 解析成统一 `Source`。`HttpMcpClient` 用 JSON-RPC HTTP POST，`StdioMcpClient` 用 MCP 的 `Content-Length` framing 和子进程 stdin/stdout。当前实现只覆盖 `tools/list`、`tools/call` 和 search-like result 转换，不做资源订阅或长连接池。

Embedding Provider：`src/deepresearch_agent/embeddings.py`。输入文本列表，输出向量列表。默认 `LocalEmbeddingProvider` 使用 `sentence-transformers` 加载 `BAAI/bge-small-zh-v1.5`，无 API key；`DashScopeEmbeddingProvider` 调百炼 OpenAI-compatible embeddings endpoint，key 只从 `DASHSCOPE_API_KEY` 读。验证脚本是 `src/deepresearch_agent/validate_embeddings.py`，本机 local BGE 实测维度 `512`；DashScope 因未配置 key，只做了 stub endpoint 解析测试。

Hybrid Local Retriever：`src/deepresearch_agent/rag.py`。输入 query 和 top-k，输出统一 `Source`。keyword 路按 token overlap 排序；vector 路先把 `data/local_corpus.jsonl` 分块，用 embedding 建 vector index，再做相似度检索；融合用 RRF，metadata 记录 keyword_rank、vector_rank、vector_index_provider、fusion score 和权重。默认 vector index provider 是 Chroma；设置 `LOCAL_VECTOR_INDEX_PERSIST=true` 后，Chroma 会写入 `LOCAL_VECTOR_INDEX_PATH`，collection 名由 corpus chunk、embedding provider 和 model 指纹决定，count 匹配时复用已有 collection；设置 `LOCAL_VECTOR_INDEX_PROVIDER=qdrant` 后会通过 Qdrant HTTP API 创建/复用 collection、upsert point 并 search。现在 hybrid 模式下如果向量索引或 embedding 路径抛错，会降级到 keyword-only，并在 `LocalRagRetriever.last_retrieval_degraded`、`last_degrade_reason` 和返回 `Source.metadata` 里显式标注原因。局限是 Qdrant 目前只有 stub HTTP 单测，没有真实服务 benchmark，也没有索引管理后台、payload filter 或权限隔离。

Document Corpus Ingestor：`src/deepresearch_agent/ingest_corpus.py`。输入是本地文件夹，输出是 `LocalRagRetriever` 可直接消费的 JSONL，每行包含 `id/title/url/content/metadata`。它支持 `.md/.markdown/.txt/.pdf/.docx`，默认排除 `.git/.obsidian/.claude/node_modules/__pycache__`；Markdown 会清理 YAML frontmatter，用 H1 或文件名生成 title；PDF 通过 `pypdf` 抽取每页文本；DOCX 通过 `python-docx` 抽取段落和表格文本；metadata 记录 `source_path` 和 `ingest_format`。局限是它不做扫描件 OCR、增量 manifest、去重、权限过滤或自动触发 vector index rebuild。

Rerank Provider：`src/deepresearch_agent/rerankers.py`。输入 query 和候选 source，输出重排分数。默认 provider 是本地 `BAAI/bge-reranker-base`，但 `rerank_enabled` 默认关闭；DashScope rerank provider 调 `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`，key 仍只读 `DASHSCOPE_API_KEY`。本地 rerank 单条 smoke 跑通过，但首次模型下载/加载使 latency 约 `279692.721ms`，所以没有把它放进默认 benchmark。

Retrieval Eval Harness：`src/deepresearch_agent/retrieval_eval.py`。输入是 BEIR/scifact 的 corpus、queries、qrels，输出 `results/retrieval_eval_scifact.json`。这个脚本不调用 LLM、不调用 Wikipedia、不跑 orchestrator，只把 SciFact 文档写成本项目 local corpus 格式，然后复用 `LocalRagRetriever` 跑 keyword / hybrid / hybrid+rerank。评测默认不保存每个 query 的 ranking 明细，避免结果文件过大；需要排查时可以显式加 `--include-rankings`。

Public Deep Research Eval Harness：`src/deepresearch_agent/deep_research_eval.py`。输入可以是本地 JSONL/JSON/CSV，也可以直接从 Hugging Face datasets-server 拉 `microsoft/LiveDRBench` 的 `preview` 或 `v1-full` split。它会跑完整 `DeepResearchOrchestrator`，写 `logs/deep-research-eval-*.jsonl`、summary JSON 和 LiveDRBench-style `preds` 文件。`judge_provider=none` 时只产 artifact 和本项目已有 success/citation/cost 指标；`judge_provider=heuristic` 时会调用 `src/deepresearch_agent/eval_judge.py` 的本地字符串命中评分；`judge_provider=deepseek` 时会用 `DEEPSEEK_API_KEY` 调 DeepSeek JSON mode，写入 `answer_judgment`，并在 summary 的 `answer_judge` 里记录 judge model、token 和估算成本。局限是 heuristic 和 DeepSeek 都不是官方 answer-quality 分数。

Verifier：`src/deepresearch_agent/verifier.py`。输入是 source 列表，输出是过滤后的 source。关键设计是可解释 quality reasons。局限是规则打分，不能真正判断来源权威性。

Source Metrics：`src/deepresearch_agent/source_metrics.py`。输入是最终 dedup 后的 `Source` 列表，输出 `source_provider_count`、`source_domain_count`、`source_provider_counts` 和 `source_domain_counts`。`orchestrator.py` 和 `run_control.py` 都复用这个 helper，所以 `/research`、CLI benchmark、public eval 和 `/runs` 的 source diversity 口径一致。局限是它只看 provider/domain 分布，不判断来源相关性、权威性或证据是否独立。

Synthesizer：`src/deepresearch_agent/llm.py` 的 `synthesize`。输入是 brief、plan、findings、sources，输出 answer 和 claims。默认 mock 会生成可测报告，DeepSeek provider 会用 JSON mode 生成 markdown answer 和结构化 claims。局限是 DeepSeek 输出目前只靠 prompt 约束和后置 citation checker，没有做二次 LLM judge 或强制 source quote。

Citation Checker：`src/deepresearch_agent/citation.py`。输入是 claims 和 sources，输出 `CitationCheckReport`。每条 `CitationAssessment` 现在包含 citation IDs、`supported`、`support_level`、overlap score、最多 3 条 `evidence_quotes`，以及可选 judge metadata。checker 会从 cited source 里按句子找最接近 claim 的 quote，输出 `supported / partial / unsupported / unverifiable`。默认仍是 lexical grounding；启用 judge 后，judge verdict 会覆盖最终 support_level，并保留 `judge_provider`、`judge_model`、`judge_confidence`、`judge_reason`。

Citation Judge Provider：`src/deepresearch_agent/citation_judge.py`。输入是 claim 和 evidence quotes，输出 `CitationJudgeResult`。`HeuristicCitationJudgeProvider` 完全本地运行、无 key；`DeepSeekCitationJudgeProvider` 调 DeepSeek JSON mode，要求返回 `verdict/confidence/reason`，并用 DeepSeek usage 字段估算 `citation_judge` 成本。`orchestrator.py` 和 `run_control.py` 都通过 `build_citation_judge_provider()` 接入，所以 `/research`、CLI、`/runs` 和 eval runner 的语义一致。局限是真实 DeepSeek judge 尚未 live benchmark，也没有 judge agreement 评测。

Report Exporter：`src/deepresearch_agent/report_exporter.py`。输入是 `StructuredReport`，输出 Markdown、HTML、JSON、PDF、DOCX、PPTX、WAV 文件路径。Markdown/HTML 会展开 answer、sources、citation assessments 和 evidence quotes；JSON 保存完整 `model_dump(mode="json")`，方便后续二次处理；PDF 用 `reportlab` 生成文本版报告；DOCX 用 `python-docx` 生成 Word 文档；PPTX 用 `python-pptx` 生成标题/answer/sources/citation assessment 幻灯片；WAV 会把报告摘要文本交给 `src/deepresearch_agent/tts.py` 的 `WindowsSapiTtsProvider`。CLI 通过 `--export-dir` 和 `--export-formats` 调用它；`--json` 模式下导出路径写到 stderr，避免污染 stdout JSON。局限是这些文档格式仍是文本版交付，WAV 也只是本地语音摘要，不包含复杂模板、目录、图片、图表、批注、配乐、多角色播客或跨平台语音引擎。

Cost Tracker：`src/deepresearch_agent/cost.py`。mock provider 仍使用字符数近似估算；DeepSeek provider 已接入 API 返回的真实 `prompt_tokens` / `completion_tokens`，并通过 `CostTracker.add_usage()` 记录到同一套 `CostSummary`。每条 `CostRecord` 现在支持单独的 `model`，所以 brief_generation、planning、synthesis 可以显示不同 stage model。当前 `deepseek-v4-flash` 成本计算按 DeepSeek 官方 Models & Pricing 页，核对日期 `2026-06-07`：input cache hit `$0.0028/1M tokens`，input cache miss `$0.14/1M tokens`，output `$0.28/1M tokens`。价格表只保留核对过的模型；`deepseek-chat`、`deepseek-reasoner` 这类未配置价格的模型会直接报错，避免 silently 借用别的模型单价。如果响应没有 token usage，DeepSeek 路径会直接失败，不会退回字符估算伪装成真实 usage。

Trace Logger：`src/deepresearch_agent/tracing.py`。每个 run 写 `logs/research-<run_id>.jsonl`，记录 stage、status、duration_ms、payload。默认 `TRACE_EXPORTER=jsonl`；配置 `TRACE_EXPORTER=otlp_http` 和 `OTEL_EXPORTER_OTLP_ENDPOINT` 后，`OtlpHttpTraceExporter` 会把每条 event 额外转成 OTLP HTTP traces JSON 并 POST 到 collector endpoint。export 失败只追加本地 `trace_exporter` error event，不中断主链路。runtime trace 默认不提交 Git，benchmark 原始记录提交。

Agent Run Control Plane：`src/deepresearch_agent/run_control.py` 外包现有 DeepResearch pipeline，不替换 `/research`。`POST /runs` 会创建 `run_id`；默认行为仍是执行 brief/planner，然后在 `require_approval=true` 时进入 `waiting_approval`；如果请求带 `defer_execution=true`，它只写入 queued run 和 `request_json` 快照，不立即跑 planner。`/runs/worker/next` 会 claim 最早 queued run，从 `request_json` 恢复本次 provider、模型、并发和检索参数，再执行 planner/researcher/synthesizer/verifier；没有 queued run 时返回 `null`。`run_worker.py` 提供本地 polling worker CLI，循环复用同一个 `process_next_queued()`，支持 `--max-runs`、`--idle-exit`、`--poll-interval-seconds` 和 `--json`。`approve` 会从 planner checkpoint 继续 researcher、synthesizer、verifier；`edit` 会保存修改后的 subquestions 再继续；`reject/cancel` 会终止 run；运行中取消通过 `RunCancelledError` 在阶段边界协作式生效，保证终态保留为 `cancelled` 而不是被通用失败处理覆盖；`retry` 对 failed run 优先复用 `plan_json`，如果已有成功 researcher step，则复用 researcher checkpoint 直接进入 synthesis/verifier，否则从 researcher 阶段重跑，没有 planner checkpoint 时会优先用 `request_json` 恢复原始请求。`run_store.py` 的 SQLite schema 是三张表：`agent_runs` 保存 run 状态、请求快照、plan/result、token/cost、`leased_by`、`heartbeat_at`、`lease_expires_at`；`agent_steps` 保存阶段输入输出、latency、token_usage、cost、error、retry_count，其中 researcher 成功 step 会保存可复用的 findings/source checkpoint；`agent_events` 保存可 SSE replay 的单调递增 event。内部执行路径会在 planner/researcher/synthesizer/verifier 阶段 acquire/heartbeat/release lease；外部 worker 也可以用 `/runs/{run_id}/lease`、`/heartbeat`、`/runs/stale`、`/runs/recover-stale` 做 ownership 和 stale recovery 验证。

Run Review UI：`src/deepresearch_agent/ui.py` 和 `src/deepresearch_agent/api.py`。`GET /ui` 返回内置 HTML/JS；`GET /runs` 返回最近 run list，底层是 `RunStore.list_runs()`。页面直接调用现有 `/runs/{id}/approve`、`/edit`、`/reject`、`/cancel`、`/events`，展示 planner subquestions、event stream、answer、sources 和 citation evidence quotes。局限是它只是本地审核面，没有权限、协作、前端构建/测试体系。

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
修复：把默认模型改成 `deepseek-v4-flash`，验证脚本默认值同步迁移；成本计算改成按模型查价格表，只保留已核对价格的 `deepseek-v4-flash`；新增单测覆盖 cache hit/cache miss/output 成本计算，并覆盖未配置价格模型抛 `ValueError`。随后用 `deepseek-v4-flash` 重跑 planner schema validation 和真实 benchmark。
复盘：模型 provider 不是一次接完就结束，模型名、价格和功能支持都会变；代码里必须有显式定价表和失败策略，不能把过期单价悄悄沿用。
面试可能追问：为什么不保留 legacy alias 价格？回答：模型名兼容和计价准确是两回事。没有各自官方价格时，继续把旧 alias 映射到 v4-flash 会产生静默错误；现在选择 fail fast，让调用者显式切到已核对的 `deepseek-v4-flash`，或者等补齐真实价格后再打开对应模型。

## 问题 8：BEIR/scifact 下载不能只看 HTTP 成功

现象：第一次用 `urllib` 下载 SciFact zip 时遇到 SSL CA 校验失败；修复证书后又出现过部分下载文件看起来以 `PK` 开头、但实际是截断坏 zip 的情况。
原因：Windows Python 环境的 CA bundle 不稳定，且一次性读取大响应时网络中断会留下不完整文件。
排查：用 `Format-Hex` 看到文件头像 zip，但 `ZipFile` 解压失败；对比 `Content-Length` 后确认本地文件大小明显不足。
修复：`retrieval_eval.py` 的下载逻辑改成使用 `certifi` CA、临时 `.tmp` 文件、分块写入、最多 3 次重试、校验 `Content-Length`，并用 `ZipFile.testzip()` 验证压缩包完整后才替换正式文件。
复盘：公开 benchmark 数据下载也要做完整性校验，否则后续 metric 报错会被误判成代码 bug。
面试可能追问：为什么不直接依赖 HuggingFace datasets？回答：我这里想保持评测脚本轻量，只加载 corpus/query/qrels；数据源和校验逻辑写在代码注释和知识库里，透明但不引入重框架。

## 问题 9：hybrid+rerank 全量评测第一次超时

现象：第一次把 keyword、hybrid、hybrid+rerank 三组一起跑完整 SciFact test qrels 时，30 分钟超时且没有写出最终 JSON。
原因：评测脚本最初没有向 `LocalRagRetriever` 注入可复用的 rerank provider，导致本地 `BAAI/bge-reranker-base` 在多个 query 上重复构造；即使复用后，CPU 上 cross-encoder rerank 300 个 query 仍然明显慢。
排查：先分模式运行，keyword 全量约 `99.55s`，hybrid 全量约 `171.57s`；再跑 10 query rerank，发现输出里反复出现 reranker weights loading。
修复：评测脚本中对 hybrid+rerank 先用 hybrid retriever 取候选，再复用同一个本地 reranker 对 query-candidate pair 批量打分；最新全量重跑里 hybrid+rerank 跑出 `1032.93s`。
复盘：rerank 的质量收益要和延迟一起讲，不能只展示 nDCG/MRR 变好；本地 CPU cross-encoder 在独立评测里可以接受，在默认端到端 benchmark 里仍不应默认开启。
面试可能追问：这算不算为结果调参？回答：不是调参，top_k 和 candidate_k 仍是 10；我只是让评测脚本复用同一个 provider 并批量打分，避免重复加载模型造成的工程噪声。

## 问题 10：公开评测 raw log 文件名并发撞名

现象：第一次并发跑 `deep_research_eval` 的 mock 组和 DeepSeek/Wikipedia 真实组时，两边都生成了同一个 `logs/deep-research-eval-20260607T133055Z.jsonl` raw log 路径，summary 里的 raw_log 指向同一文件。
原因：文件名时间戳只精确到秒；两个评测进程在同一秒启动，默认 raw log 名称完全相同。
排查：我同时查看两组 summary，发现 `raw_log` 字段相同，但 provider 配置不同；这说明不是 orchestrator 结果问题，而是 artifact 命名问题。
修复：把 `deep_research_eval.py` 的默认 timestamp 改成微秒级 `YYYYMMDDTHHMMSSffffffZ`，然后清掉撞名临时文件，顺序重跑 mock 和真实两组结果。
复盘：评测 artifact 是可信度的一部分，不能只保证脚本能跑，还要保证并行/重复运行时不会覆盖证据文件。
面试可能追问：为什么不直接用 UUID？回答：UUID 也可以；我这次选微秒时间戳是为了保留人类可读的运行时间，同时把秒级撞名风险降掉。下一步如果做 worker queue，可以再加 run_id 或 provider suffix。

## 问题 11：Jina Search live smoke 触发 fallback

现象：实现 `JinaSearchAdapter` 后，我用 `py -3.11 -m deepresearch_agent.cli "What is Model Context Protocol?" --search-provider jina --llm-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --json` 做 live smoke。run 成功，但 `researcher.Q1` status 是 `fallback`，error 是 `HTTP Error 401: Unauthorized`；单独用 `urllib` 请求 `https://s.jina.ai/Model+Context+Protocol` 又出现过 `403/401`。同一时间，`https://r.jina.ai/https://example.com` Reader crawl 返回 `200` 和 clean text。
原因：当前环境下 Jina Search 公开 endpoint 没有匿名搜索成功，可能需要 key、受到访问策略限制，或和 header/user-agent 有关；Reader crawl endpoint 对普通 URL 仍可匿名访问。
排查：分别测试 `s.jina.ai` search 和 `r.jina.ai` reader；再看 orchestrator trace，确认失败被 `SearchService` 捕获并降级到 mock，而不是打断整条研究链路。
修复：代码里保留公开 endpoint 尝试，同时支持可选 `JINA_API_KEY` 环境变量，存在时发送 Bearer header；未配置或请求失败时仍走既有 fallback。
复盘：真实 web search provider 不是只写 adapter 就算完成，外部服务的认证、限流和访问策略都要进入 trace。当前我只能说 Jina Reader crawl live 成功，Jina Search live 未通过。
面试可能追问：为什么还保留 Jina Search？回答：因为它的接口边界和数据格式已经接入，且官方 README 明确有 `s.jina.ai` search 能力；但我不会说它在本机已稳定可用，下一步更稳的是接自建 SearxNG 或带 key 的 Brave/Tavily。

## 问题 12：新增 crawler CLI 参数后评测单测 AttributeError

现象：把 SearxNG/Jina crawler 配置接进 `benchmark.py` 和 `deep_research_eval.py` 后，`tests/test_deep_research_eval.py` 失败：`Namespace` 对象没有 `searxng_base_url` 属性。
原因：CLI parser 会补全新参数，但单测直接手工构造 `argparse.Namespace`，不会自动带上新增字段；runner 里直接访问 `args.searxng_base_url` 就破坏了旧调用方式。
排查：失败栈定位到 `replace(settings, searxng_base_url=args.searxng_base_url ...)`，不是 orchestrator 或 provider 问题。
修复：评测 runner 读取新增可选参数时统一用 `getattr(args, "...", None)`，没有属性时回落到 settings/env 默认值。相关测试重跑 `14 passed`。
复盘：CLI 扩展不能假设所有调用者都从 parser 进来；测试、库函数、外部脚本都可能直接传 Namespace。
面试可能追问：为什么不改测试补字段？回答：测试补字段也可以，但生产代码更应该对可选参数缺省鲁棒，尤其是 benchmark runner 这种会被脚本复用的入口。

## 问题 13：组合跑 citation/API/spine 测试时超时

现象：我第一次用 `py -3.11 -m pytest tests/test_quality_and_citations.py tests/test_spine.py tests/test_api.py -q` 验证 citation grounding 时，命令在约 `124s` 超时，没有输出断言失败。
原因：`test_spine.py` 和 `test_api.py` 会走默认 local hybrid retrieval，首次/冷启动路径会加载本地 embedding/Chroma，单个文件就可能跑到 80 秒以上；组合跑在这次命令的 timeout 预算内不够。
排查：拆开跑后，`tests/test_quality_and_citations.py` 是 `6 passed in 0.26s`，`tests/test_spine.py` 是 `1 passed, 1 warning in 86.23s`，`tests/test_api.py` 是 `2 passed, 2 warnings in 84.51s`，说明不是 citation schema 破坏了 API。
修复：没有改功能代码；后续验证用更大的 full test timeout，并记录 local hybrid 测试路径偏慢。
复盘：本地 embedding/hybrid 默认路径会影响测试耗时。之后如果 CI 要稳定，可以在 API/spine 测试里显式设置 `LOCAL_RETRIEVAL_MODE=keyword` 或做 fixture 级缓存。
面试可能追问：为什么不马上优化？回答：这次功能目标是 citation grounding，不做额外性能 refactor；但这个现象说明生产/CI 里需要索引缓存和测试模式隔离。

## 问题 14：UI approve HTTP probe 30 秒超时但 run 最终成功

现象：临时启动 `/ui` 服务后，我用 PowerShell 调 `POST /runs/{run_id}/approve` 做后端 flow 验证，30 秒命令超时；随后等待 20 秒再查 run，状态已经是 `succeeded`，result_json 正常。
原因：临时服务没有设置 `LOCAL_RETRIEVAL_MODE=keyword`，走默认 hybrid local retrieval；首次 approve 触发本地 embedding/Chroma 冷启动，researcher 阶段耗时约 `36s`，超过这次手工 probe 的 30 秒 timeout。
排查：查看 run metrics，`latency_ms=36307.56`，fallback 为 0，status succeeded；说明不是 UI 或 approve API 错，而是默认检索模式冷启动慢。
修复：没有改功能代码；文档中把这次 smoke 如实记录。后续 demo 如果要快，可以在启动服务前设置 `LOCAL_RETRIEVAL_MODE=keyword`，或者开启 `LOCAL_VECTOR_INDEX_PERSIST=true` 复用本地 Chroma index。
复盘：默认功能更完整和 demo 响应更快之间有取舍。hybrid 默认展示检索能力，但审核页演示需要明确环境变量或预热。
面试可能追问：这是不是生产不可用？回答：这说明当前还没有生产级索引生命周期和 warm worker；我不会把它说成低延迟生产 UI，但 run control 能正确等待长任务完成。

## 问题 15：worker lease 阶段没有阻塞 bug，但暴露出队列边界

现象：这次实现 SQLite worker lease、heartbeat、stale recovery 时，目标测试一次通过，没有出现阻塞性 bug。
工程风险：当前 lease 只解决单个 run 的 ownership 和过期恢复；新增 worker-once 后可以消费下一条 queued run，但仍不提供常驻 worker pool、任务分发、公平调度或阶段级幂等 replay。如果一个长 researcher 阶段内部卡住，heartbeat 只能在阶段边界刷新，不能像真实 worker 进程那样持续后台续租。
修复：没有为了掩盖这个边界去加复杂基础设施，只在代码和文档里明确：这是 SQLite 单机 control-plane primitive，下一步才是 Postgres/Redis/worker pool。
复盘：这个边界比“加一个队列表名”更重要。面试时我会承认当前还不是分布式执行，但已经有了迁移到生产队列前必须定义清楚的 lease 字段、stale 判断和 recovery 语义。
面试可能追问：为什么不直接 Celery？回答：因为这个项目的下限是无外部依赖跑通；先在 SQLite 里定义 ownership 语义，后面替换底层 store 比一开始引入队列系统更稳。

## 问题 16：持久化向量索引测试第一次失败

现象：新增 `test_persistent_vector_index_reuses_existing_collection` 后，第一次运行 `tests/test_hybrid_retrieval.py` 失败，断言期望返回 `vector` 文档，但实际第一名是 `keyword`。
原因：测试没有设置 `local_vector_weight=4.0`，在默认权重下 keyword 和 vector 的 RRF 融合不是这个测试想验证的口径。PersistentClient 本身已经能写入和读取 collection，失败点是测试排序权重不一致。
排查：对照前一个 hybrid fusion 测试，发现它显式把 vector weight 设置为 4.0 来证明语义向量召回能压过 keyword tie；新测试漏了同一设置。
修复：只修测试配置，补 `local_vector_weight=4.0`，没有改业务代码。重跑 `tests/test_hybrid_retrieval.py` 成功。
复盘：检索测试必须把融合权重写进配置，否则测试失败会混淆“索引是否复用”和“RRF 排序是否符合预期”两个问题。
面试可能追问：这说明 hybrid 不稳定吗？回答：是的，RRF 权重会影响结果，所以我把它配置化并用独立评测看指标，不会把某一次排序当成普遍规律。

## 问题 17：阶段模型测试第一次跑得过慢

现象：新增 `tests/test_stage_models.py` 后第一次单测通过，但耗时 `34.33s`，明显不该出现在只验证模型路由的测试里。
原因：mock orchestrator 测试没有显式设置 `local_retrieval_mode="keyword"`，导致默认 hybrid retrieval 路径加载本地 embedding/Chroma。测试目标是验证 cost record 的 stage model，不应该把 embedding 冷启动混进来。
排查：看测试内容只有 mock provider 和 1 个 researcher，慢点只能来自 local RAG 默认 hybrid；对照其他 API/run control 测试也都显式设置了 keyword。
修复：把测试 orchestrator 改成 `DeepResearchOrchestrator(settings=Settings(local_retrieval_mode="keyword"))`。重跑 `tests/test_stage_models.py` 变成 `2 passed in 0.35s`。
复盘：单元测试要隔离目标变量。多模型路由测试只该测 LLM/cost，不该顺带测向量检索模型加载。
面试可能追问：这是不是说明默认 hybrid 有问题？回答：不是功能错误，但说明本地 embedding 冷启动会污染无关测试和 demo，需要通过显式配置或持久化索引控制。

## 问题 18：报告导出阶段无阻塞 bug，但刻意没有做富文档格式

现象：第一阶段新增 Markdown/HTML/JSON exporter 和 CLI 参数后，单测与 CLI smoke 都通过，没有出现阻塞性 bug；这次扩展 PDF/DOCX 后，`tests/test_report_exporter.py` 也通过。
工程风险：PDF/DOCX/PPTX 现在是文本版报告导出，WAV 是 Windows SAPI 本地语音摘要，不等于完整办公文档或 podcast 生成。它们保留 answer、sources、citation assessments 和 evidence quotes，但不支持封面、目录、分页模板、图片、图表、批注、修订模式、配乐、多角色对话或跨平台 TTS。
修复：没有把文本版 PDF/DOCX/PPTX 或 WAV 包装成富文档/podcast 系统；README、知识库和 QA 都写清 WAV 只是可选本地语音导出口，文档导出只是报告 artifact 的常见文件格式。
复盘：导出层先做稳定数据边界，再扩展常见格式是合理顺序。后续如果做高保真排版，应继续复用 `StructuredReport`，而不是让导出层反向影响主链路。
面试可能追问：为什么不直接做完整 PPT 模板？回答：高保真排版是展示工程，不是 DeepResearch 主链路的核心风险；我现在先保证引用和来源可追溯，再做格式扩展。

## 问题 19：OTLP exporter 只做出口验证，不等于生产 tracing 平台

现象：新增 `OtlpHttpTraceExporter` 后，本地测试 server 能收到 `/v1/traces` POST，exporter 失败时 JSONL 里会追加 `trace_exporter` error event，没有阻塞 run。
工程风险：这只是 trace event 外送边界，不包含采样、batch、上下文传播、metrics/logs、collector 部署、鉴权策略或 LangSmith UI。
修复：默认仍保持 `TRACE_EXPORTER=jsonl`；`otlp_http` 只有显式配置 endpoint 才启用；失败吞掉并写本地 error event。README、知识库和 QA 都写清这是轻量 OTLP HTTP exporter。
复盘：可观测性要先保证“不影响业务路径”。我没有为了贴 OpenTelemetry 标签引入完整 SDK，而是先把现有 trace event 做成可外送的稳定接口。
面试可能追问：这算 OpenTelemetry 吗？回答：它是 OTLP HTTP traces 的轻量出口，不是完整 OTel SDK 接入；如果生产化，我会换成官方 SDK + batch processor + collector 配置。

## 问题 20：citation judge 容易改变历史 benchmark 口径

现象：实现 citation judge 时，如果直接默认启用，会让 `citation_retention_rate` 从 lexical overlap 口径变成 judge verdict 口径，历史 `results/benchmark_summary.json` 和 retrieval 对比就不能直接比较。
工程风险：LLM judge 会增加成本和延迟，还可能因为 judge 本身输出波动改变 success_rate；如果不写清楚，面试时会把“引用检查口径变了”误讲成“质量提升了”。
修复：默认 `CITATION_JUDGE_PROVIDER=none`，历史 benchmark 口径不变；只有显式传 `--citation-judge-provider heuristic|deepseek` 才启用。benchmark 和 public eval 的 config snapshot 会记录 citation judge provider/model。
复盘：评测系统里，指标定义比实现更重要。新增 judge 是能力边界，不是自动替换已有指标。
面试可能追问：现在的 citation_retention 和 judge 分数怎么比较？回答：不能混着比；要把 lexical baseline、heuristic judge、DeepSeek judge 分成不同口径重新跑。

## 问题 21：public eval 测试暴露 argparse Namespace 兼容性问题

现象：新增 `citation_judge_provider` 字段后，`tests/test_deep_research_eval.py` 里手工构造的 `argparse.Namespace` 没有这个属性，第一次跑相关测试失败，报 `AttributeError: 'Namespace' object has no attribute 'citation_judge_provider'`。
原因：脚本真实 CLI 会由 parser 注入新字段，但单测和其他程序化调用可能传入旧 Namespace。
排查：失败栈指向 `deep_research_eval.py` 的 `replace(settings, citation_judge_provider=args.citation_judge_provider, ...)`。
修复：`benchmark.py` 和 `deep_research_eval.py` 都改成 `getattr(args, "citation_judge_provider", None)` / `getattr(args, "citation_judge_model", None)`，缺字段时回落到 Settings。
复盘：CLI 增参不能只考虑命令行路径，测试和程序化调用也需要兼容。
面试可能追问：为什么不直接改测试？回答：测试暴露的是兼容性风险；修业务代码比让所有调用方同步加字段更稳。

## 问题 22：Brave/Tavily 只完成 adapter 级验证，没有 live key

现象：新增 Brave/Tavily provider 后，单测能验证请求 header/body 和响应解析，但当前环境没有 `BRAVE_SEARCH_API_KEY` 或 `TAVILY_API_KEY`，所以没有做 live search。
工程风险：商业搜索 API 的真实可用性会受账号、额度、限流、地区网络和结果质量影响；stub 测试只能证明代码按文档组请求，不能证明生产搜索质量。
修复：缺 key 时 adapter 明确抛 `SearchError`，交给 `SearchService` 走 mock fallback；文档和 KB 写清未 live benchmark。默认仍是 mock，不破坏无 key 运行。
复盘：搜索 provider 扩展要把“能解析 API”与“真实质量提升”分开讲。
面试可能追问：为什么不直接跑 Brave/Tavily？回答：没有 key 就不伪造 live 结果；有 key 后用同一套 benchmark/public eval 重跑并记录真实数字。

## 问题 23：运行中取消不能被误记成 failed

现象：run control 已经有 `_raise_if_cancelled()` 阶段检查，但取消异常原来会被通用 `except Exception` 捕获并走 `_fail_run()`，存在把用户主动取消的 run 覆盖成 `failed` 的风险。
原因：代码没有把“用户取消”和“系统失败”建模成不同异常类型；`create_run`、`process_next_queued`、`retry` 和 `_continue_from_plan` 都只按普通异常处理。
排查：沿着 researcher/synthesizer/verifier 阶段边界看状态流，发现只要运行中途 `cancel()` 改成 `cancelled`，下一次阶段检查抛错后就可能进入 failed 路径。
修复：新增 `RunCancelledError`，取消检查、lease acquire/heartbeat 遇到 cancelled 时都抛这个异常；外层执行入口单独捕获后调用 `_cancelled_run()`，保留 `cancelled` 终态并释放 lease。新增测试用 monkeypatch 模拟 researcher 阶段被外部取消，验证最终状态不是 failed。
复盘：长任务控制平面要把 terminal state 的语义分清楚。用户取消不是系统失败，不能混到 retry/failure 指标里。
面试可能追问：这是不是实时取消？回答：不是强制抢占，当前是阶段边界协作式取消；已发出的 LLM/search 请求不会被杀掉，下一步要做 provider 级 abort signal 或 worker task registry。

## 问题 24：后处理失败时 retry 会重复 researcher 检索

现象：旧 retry 路径只复用 planner checkpoint。即使 researcher 已经成功，只要 synthesizer 或 verifier 后续失败，retry 仍会重新跑 researcher，导致重复 search/retrieval。
原因：researcher 成功 step 只记录了 count，没有保存 findings 和 source checkpoint；`_execute_research_flow()` 也没有读取历史成功阶段输出。
排查：看 `agent_steps.output_json` 只有 `raw_search_result_count/fallback_count/deduped_source_count`，无法恢复 synthesis 所需的 `findings/sources`。
修复：researcher 成功 step 增加 `output_json.checkpoint`，保存 findings、all_sources、deduped sources 和计数；retry 时如果 `retry_count > 0` 且找到成功 researcher checkpoint，就写 `researcher.checkpoint_reused` event 并跳过 researcher 阶段。
复盘：幂等恢复不一定一开始就上完整 DAG。先保存阶段输出，能解决“检索成功但后处理失败”的重复成本问题。
面试可能追问：为什么不是每个 subquestion checkpoint？回答：当前先做阶段级复用，改动小且足够证明思路；细粒度恢复需要拆 researcher 子任务状态、reflection round 和大对象存储。

## 问题 25：search backoff / 限流只能先做本地进程内保护

现象：多 researcher 并发时，SearchService 之前只有 retry、timeout、circuit breaker 和 fallback，没有主动控制 primary search 请求发出的节奏；失败后 retry 也是立即重试。
工程风险：真实 Brave/Tavily/SearxNG/Jina provider 有 QPS/额度限制；没有节流时，一次 run 的并发 researcher 可能瞬间打出 burst，请求失败后才靠 fallback 兜底；没有 backoff 时，429/瞬时错误会被立即重打。
修复：新增 `SearchRateLimiter` 和 `SEARCH_RATE_LIMIT_PER_SECOND` 配置，默认关闭；开启后只在 primary search 调用前等待，不限制 mock fallback。新增 `SEARCH_RETRY_BACKOFF_SECONDS`，失败后按 `base * 2^attempt` 等待再重试。单测用 fake clock/sleep 验证 2 QPS 时连续三次 wait 产生两个 `0.5s` 间隔，并验证 0.25s backoff 下两次 retry 等待 `[0.25, 0.5]`。
复盘：本地 limiter/backoff 是可靠性保护，不是质量提升。它减少 burst 和立即重试风险，但不能替代 provider-specific `Retry-After`、429 策略或跨 worker 全局配额。
面试可能追问：为什么不直接做 Redis 限流？回答：当前项目默认无外部依赖；先把 SearchService 层的限流接口和测试打出来，生产化再换 Redis/网关级全局 limiter。

## 问题 26：默认 hybrid retrieval 在缺向量依赖时会打断无关测试

现象：`local_retrieval_mode` 默认是 `hybrid`，但 `ChromaVectorIndex.build()` 在 Chroma 或 embedding 路径不可用时会抛错；在没有装好 ML 依赖的机器或 CI 上，`test_api`、`test_spine`、`test_rerankers` 这类本来只想验证 API/编排/rerank 语义的测试也可能被 local vector index 冷启动或缺依赖拖垮。
原因：`LocalRagRetriever.retrieve()` 之前在 hybrid 模式下无保护地调用 `_vector_retrieve()`，没有把向量召回当成可失败的本地工具，也没有把降级状态写回 source metadata。
排查：先读 `README.md`、`AGENTS.md`、`pyproject.toml`、`rag.py`、`cost.py`、`api.py`、`orchestrator.py`、`search.py`，确认 orchestrator 主链路只依赖 `LocalRagRetriever.retrieve()` 返回统一 `Source`；因此修复可以限制在检索层，不需要改 API 或 researcher 流程。
修复：在 `retrieve()` 中只包住 vector retrieval 路径，捕获 Chroma 缺失、embedding 加载失败、索引 build/search 抛错等异常；记录 `logging.warning`，设置 `last_retrieval_degraded/last_degrade_reason`，并返回 keyword-only 结果。降级 source 的 metadata 增加 `retrieval_degraded=True` 和 `degrade_reason`。Chroma 正常时原 hybrid + RRF 路径不变。
复盘：这是可靠性修复，不是质量优化。默认 hybrid 仍保留，但工程系统不能把一个 optional vector path 失败放大成整条 research run 失败。
面试可能追问：降级会不会掩盖问题？回答：不会静默掩盖，因为 warning、retriever 状态和 source metadata 都能看到降级原因；它只是保证弱环境先跑通，真实质量评测仍要区分 keyword 和 hybrid 口径。

# 7 实测数据

本节所有 mock benchmark 数字只用于证明 pipeline plumbing 能端到端跑通，不能当作真实性能、真实成本或真实答案质量成果。尤其不能在面试里说“我的 DeepResearch p50 是个位数毫秒”这类话，因为这个延迟测的是本机 Python 跑 deterministic mock 的速度，换机器、换进程热身状态、换依赖版本都会变。

实测环境：Windows PowerShell，`py -3.11`，mock search provider，seed `20260606`，5 条 benchmark case，max_researchers=3，max_results=4。

安装验证：`py -3.11 -m pip install --timeout 180 -e ".[dev]"` 成功。为了支持本地 hybrid retrieval，新增安装了 `sentence-transformers` 和 `chromadb`；第一次安装时有一个超时遗留 pip 进程占用 `torch` 文件，结束该遗留进程后重试成功。
测试验证：`py -3.11 -m pytest -q`，最新结果 `89 passed, 2 warnings in 69.53s`。warning 来自 FastAPI TestClient / Starlette 对 httpx 的 deprecation 提示，以及 OpenTelemetry metadata 的 deprecation 提示，未影响功能。
CLI example：`py -3.11 -m deepresearch_agent.cli "How does citation checking reduce hallucination in agentic RAG?"` 成功，raw_search_result_count `12`，deduped_source_count `8`，total_tokens `4417`。这次运行记录的 latency 是 `10.63ms`，但它只是 mock plumbing run 的本机样本，不作为性能指标引用。citation_retention_rate `1.0` 只说明 mock synthesis 生成的 citation ID 能被当前 checker 找到，不代表真实 LLM 场景下的引用可靠性。estimated_cost_usd `0.0` 是因为 mock provider 单价配置为 0，不代表真实成本。
真实 adapter probe：`py -3.11 -m deepresearch_agent.cli "What is Model Context Protocol?" --search-provider wikipedia --json` 成功，修复后 sample 输出显示 `fallback_count=0`，latency 约 `1506.501ms`。注意：Wikipedia 是真实无 key adapter，但不是高质量通用搜索，结果质量仍有限。

DeepSeek 结构化输出验证：`py -3.11 -m deepresearch_agent.validate_deepseek_structured_output --query "How should citation checking reduce hallucination in deep research agents?" --max-researchers 3` 成功。迁移后默认 `deepseek-v4-flash` 返回了 3 条合法 `SubQuestion`，Pydantic schema 解析通过。真实输出主题分别覆盖 citation checking 机制、citation accuracy 与 hallucination rate 的关系、以及 citation checking 的失败模式。

DeepSeek 端到端单条验证（LLM 真，search 仍是 mock）：`py -3.11 -m deepresearch_agent.cli "How should citation checking reduce hallucination in deep research agents?" --llm-provider deepseek --search-provider mock --max-researchers 2 --max-results 3 --json` 成功。模型生成的 brief 不再是模板回填，scope 是“Methods and effectiveness of citation verification in mitigating factual inaccuracies in AI-driven research agents”；planner 拆出 automated citation verification 和 human-in-the-loop 对比两个子问题；synthesis 生成了 markdown 报告和 6 条 claims。步骤 2 首次成功运行记录：latency `18987.535ms`，raw_search_result_count `6`，deduped_source_count `7`，citation_retention_rate `0.8333`，supported_claims `5/6`。其中 1 条 human-in-the-loop claim 被当前 lexical citation checker 标为 unsupported。这是接真实 usage 前的历史记录，所以当时的 cost 不能当真实成本；步骤 3 已补上 provider usage 解析。

DeepSeek usage/cost 单条验证（LLM 真，search 仍是 mock）：接入真实 usage 后重跑同一条命令成功。运行记录：latency `18524.693ms`，raw_search_result_count `6`，deduped_source_count `7`，citation_retention_rate `1.0`，supported_claims `6/6`。真实 usage：input_tokens `1842`，output_tokens `1310`，total_tokens `3152`，estimated_cost_usd `0.00193834`。分阶段成本：brief_generation `118 + 140 tokens / $0.00018586`，planning `277 + 226 tokens / $0.00032339`，synthesis `1447 + 944 tokens / $0.00142909`。注意：search 仍是 mock，所以这还不是“LLM + search 全真实”的 benchmark；步骤 4 再切 Wikipedia。

Embedding provider 验证：`py -3.11 -m deepresearch_agent.validate_embeddings --provider local --text "混合检索需要关键词召回和向量召回一起融合。"` 成功。本地模型 `BAAI/bge-small-zh-v1.5` 返回维度 `512`。`DASHSCOPE_API_KEY` 当前未配置，所以百炼 embedding 没有做真实 API 调用；代码层用本地 HTTP stub 测过 DashScope-compatible response parsing。

Reranker smoke：`py -3.11 -m deepresearch_agent.cli "How does citation checking reduce hallucination in agentic RAG?" --search-provider mock --llm-provider mock --max-researchers 1 --max-results 2 --local-retrieval-mode hybrid --rerank-enabled --rerank-provider local` 成功。因为首次下载/加载 `BAAI/bge-reranker-base`，latency 达到 `279692.721ms`。这只证明可选 rerank provider 能接入，不作为常规性能指标。

Web search / crawler provider smoke：`py -3.11 -m pytest tests/test_web_search_providers.py tests/test_failure_handling.py -q` 成功，最新结果 `18 passed in 0.46s`，覆盖 SearxNG JSON parsing、crawler 内容替换、Jina Reader URL prefix、本地 HTML crawler 去掉 script/style 并抽正文、Jina Search JSON parsing、Brave/Tavily stub parsing、provider registry、unknown provider fail-fast、primary failure fallback、circuit breaker open、SearchRateLimiter 的 min-interval 计算、SearchService 在 primary search 前执行 limiter，以及失败 retry 的指数 backoff `[0.25, 0.5]`。真实外网 smoke：`https://r.jina.ai/https://example.com` 返回 `200` 和 clean text，说明 Jina Reader crawler 这层能 live 访问；`--search-provider jina` 的 CLI run 成功但触发 fallback，trace error 是 `HTTP Error 401: Unauthorized`，所以 Jina Search 真实检索在当前环境未通过。SearxNG 需要 `SEARXNG_BASE_URL`，当前没有自建实例，未做 live search；本地 HTML crawler 只做 stub 单测，没有 live 大规模网页抽取评测；search rate limit / backoff 目前只做本地单测，没有真实 provider 429/QPS 压测。

Brave/Tavily provider smoke：`py -3.11 -m pytest tests/test_web_search_providers.py tests/test_failure_handling.py -q` 成功，最新相关结果 `14 passed in 0.41s`。新增覆盖 Brave Search 的 `X-Subscription-Token` header、`web.results` 解析、缺 `BRAVE_SEARCH_API_KEY` fail-fast；Tavily Search 的 Bearer auth、JSON body `query/search_depth/max_results`、`results` 解析、缺 `TAVILY_API_KEY` fail-fast；provider registry 可构造 `brave` 和 `tavily`。当前没有真实 Brave/Tavily key，所以没有 live API 调用、延迟、成本或质量数字。

Claim-level evidence grounding smoke：`py -3.11 -m pytest tests/test_quality_and_citations.py -q` 成功，`6 passed in 0.26s`；新增测试覆盖 supported claim 提取 evidence quote，以及 missing source 标成 `unverifiable`。`py -3.11 -m deepresearch_agent.cli "How does citation checking reduce hallucination in agentic RAG?" --search-provider mock --llm-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --json` 成功，输出里的 citation assessment 已包含 `support_level="supported"` 和 `evidence_quotes`，quote 示例来自本地 source 句子 “Normal RAG retrieves context once for a single answer...”。这只证明 evidence quote plumbing 和 lexical grounding 生效，不代表语义事实校验完成。

Reflection loop smoke：`py -3.11 -m pytest tests/test_reflection_loop.py tests/test_run_control.py -q` 成功，`9 passed, 1 warning in 4.49s`。`tests/test_reflection_loop.py` 覆盖 orchestrator 在 `reflection_enabled=True`、`max_reflection_rounds=1`、`reflection_min_sources=4` 时追加 `R1` follow-up question，并写入 `compression.round1` / `reflection.round1` trace。`tests/test_run_control.py` 也覆盖了 `/runs` approve 后 result_json 的 plan 包含 `R1`，trace_events 包含 `reflection.round1`。CLI smoke：`py -3.11 -m deepresearch_agent.cli "How should citation grounding work in a research agent?" --search-provider mock --llm-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --reflection-enabled --max-reflection-rounds 1 --reflection-min-sources 4 --json` 成功，输出可见 `id="R1"`、`compression.round1`、`reflection.round1` 和 `should_add_question=true`。

MCP adapter smoke：`py -3.11 -m pytest tests/test_mcp_tools.py tests/test_web_search_providers.py tests/test_failure_handling.py -q` 成功，`14 passed in 0.37s`。覆盖 MCP result 的 `sources` array 转 `Source`、`content[type=text]` JSON 转 `Source`、`McpToolSearchAdapter` 调 fake client、`build_search_adapter(..., "mcp")` 构造 provider、以及缺少 `MCP_SEARCH_TOOL` 时 fail-fast。当前没有配置真实 MCP server，所以 stdio/http live call 未实测。

Run review UI smoke：`py -3.11 -m pytest tests/test_run_control.py tests/test_api.py -q` 成功，`11 passed, 2 warnings in 45.22s`，覆盖 `/ui` 返回页面、`GET /runs` 列出刚创建的 run、原 approval/edit/cancel/retry/SSE replay 仍可用。临时启动 `py -3.11 -m deepresearch_agent.api --host 127.0.0.1 --port 8010` 后，`/health` 返回 `ok`，`/ui` HTML 包含 `DeepResearch Run Review`、`planEditor`、`EventSource`，`POST /runs` 创建 run 后 `GET /runs` 能看到它。`POST /runs/{run_id}/approve` 第一次 30s probe 超时，但 20s 后查询 run 已 `succeeded`，metrics latency 约 `36307.56ms`；原因是临时服务没有设置 `LOCAL_RETRIEVAL_MODE=keyword`，默认 hybrid 冷启动加载本地模型。Node 环境没有 `playwright` 包，所以没有做截图验证。

Worker lease smoke：`py -3.11 -m pytest tests/test_run_control.py -q` 成功，`12 passed, 1 warning in 6.97s`。新增测试覆盖 `RunStore.acquire_lease()` 只能让一个 worker 获得 lease、同 worker heartbeat、release 后另一个 worker 可重新 acquire、同一 worker 即使跨过 TTL 也能在未被别人接管时续租；API 测试覆盖 `/runs/{run_id}/lease`、竞争 worker 返回 `409`、`/heartbeat` 更新 `heartbeat_at`、`/runs/stale` 能列出过期 running run、`/runs/recover-stale` 会把 stale run 标记为 `failed` 并清空 `leased_by`；migration 测试覆盖旧版 `agent_runs` 表缺少 lease 列时，`RunStore` 会自动补 `leased_by`、`heartbeat_at`、`lease_expires_at`。

Deferred worker smoke：`py -3.11 -m pytest tests/test_run_control.py -q` 成功，最新结果 `14 passed, 1 warning in 7.56s`。新增测试覆盖 `defer_execution=true` 创建后 run 保持 `queued/planner`，`agent_runs.request_json` 保存本次请求快照且 planner steps 为空；调用 `POST /runs/worker/next` 后同一 run 执行到 `succeeded`，事件里出现 `worker.claimed`，lease 最终释放；无 queued run 时 `/runs/worker/next` 返回 JSON `null`；旧版 `agent_runs` 表缺少 `request_json` 时，`RunStore` 会自动迁移补列。这里没有启动常驻 worker pool，也没有做 Redis/Celery live 验证。

Run control retry/cancellation smoke：`py -3.11 -m pytest tests/test_run_control.py -q` 成功，最新结果 `16 passed, 1 warning in 8.79s`。新增运行中取消回归用 monkeypatch 模拟 researcher 阶段被外部 `cancel()`，确认最终状态保持 `cancelled`、lease 释放、没有 failed event；新增 retry checkpoint 回归模拟第一次 synthesis 失败，确认 retry 后 `researcher` 只调用 1 次、`synthesizer` 调用 2 次、成功 researcher step 带 `output_json.checkpoint`，并写入 `researcher.checkpoint_reused` event。这只证明阶段级 researcher checkpoint 复用，不代表每个 subquestion 都可幂等恢复。

Local worker loop smoke：`py -3.11 -m pytest tests/test_run_worker.py tests/test_run_control.py -q` 最近一次成功，`17 passed, 1 warning in 8.66s`。测试覆盖 `run_worker_loop(max_runs=1)` 能消费一个 `defer_execution=true` 的 queued run 并执行到 `succeeded`，summary 记录 `processed_count=1`、`stopped_reason=max_runs`；空队列时 `idle_exit=True` 返回 `processed_count=0`、`idle_polls=1`、`stopped_reason=idle`。这里没有做多进程 worker 竞争压测，也没有 Redis/Celery broker；取消也不是强制抢占已经发出的 LLM/search 请求。

Persistent vector index / Qdrant provider / local retrieval degrade smoke：`py -3.11 -m pytest tests/test_hybrid_retrieval.py tests/test_api.py tests/test_spine.py tests/test_rerankers.py -q` 成功，最新结果 `12 passed, 2 warnings in 98.40s`。测试用静态 embedding provider 和临时 Chroma PersistentClient 验证：第一次检索会 embed 2 个 corpus chunk 和 1 个 query，第二个 `LocalRagRetriever` 指向同一 `LOCAL_VECTOR_INDEX_PATH` 时只 embed query，不重新 embed corpus；`ChromaVectorIndex.reused_existing` 为 `True`。Qdrant stub HTTP 测试覆盖 collection missing 时 create collection、upsert points、search 返回指定 chunk、`api-key` header 从环境变量读取，以及已有 collection points_count 匹配时复用 collection、不重新 embed corpus。新增 graceful degrade 回归用 monkeypatch 让 `ChromaVectorIndex.build()` 抛 `RuntimeError("synthetic vector build failure")`，断言 hybrid 不崩、返回 keyword source、metadata 带 `retrieval_degraded=True/degrade_reason`，且 `LocalRagRetriever.last_retrieval_degraded` 和 warning 日志都可观察。这里没有启动真实 Qdrant 服务，也没有做真实 BGE 冷/热启动或外部向量库延迟 benchmark。

Document corpus ingest smoke：`py -3.11 -m pytest tests/test_ingest_corpus.py tests/test_hybrid_retrieval.py -q` 成功，最新结果 `7 passed, 1 warning in 3.64s`。测试覆盖 Markdown YAML frontmatter 清理、H1 title 抽取、TXT 文件名 title、`.obsidian` 目录排除、空文档跳过、PDF 文本抽取、DOCX 段落/表格文本抽取、输出 JSONL 的 `id/title/url/content/metadata` 字段，以及生成的 corpus 能被 `LocalRagRetriever(local_retrieval_mode="keyword")` 直接检索；同组也覆盖 Chroma/Qdrant hybrid retrieval 回归。这里没有测试扫描件 OCR、增量 reindex、权限过滤或大规模 corpus。

Stage model / pricing smoke：`py -3.11 -m pytest tests/test_quality_and_citations.py tests/test_stage_models.py tests/test_citation_judge.py tests/test_deep_research_eval.py -q` 成功，最新结果 `21 passed in 0.64s`。测试覆盖 mock orchestrator 在 `brief_model`、`planner_model`、`synthesis_model` 不同时，`CostRecord.model` 分别记录 `mock-brief`、`mock-planner`、`mock-synthesis`；用不联网的 `RecordingDeepSeekProvider` 验证 DeepSeek 请求体按 stage 发送已配置价格的 `deepseek-v4-flash`；用 `RecordingOpenAICompatibleProvider` 验证 OpenAI-compatible provider 按 stage 发送 `local-brief/local-planner/local-synthesis`，并按显式配置的 input/output 单价把 usage 记为 `$0.00042`；还覆盖 orchestrator 能从 `llm_provider="openai-compatible"` 构造通用 provider。成本测试新增断言：`deepseek-chat` 这类未配置价格的模型名会抛 `ValueError`。CLI smoke：`py -3.11 -m deepresearch_agent.cli "How should model routing work in a research agent?" --llm-provider mock --llm-model mock-default --brief-model mock-brief --planner-model mock-planner --synthesis-model mock-synthesis --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --json` 后抽取 `cost.records[].model`，输出 `['mock-brief', 'mock-planner', 'mock-synthesis']`。这只验证路由和记录，不代表 OpenAI-compatible endpoint 已 live benchmark。

Report exporter smoke：`py -3.11 -m pytest tests/test_report_exporter.py -q` 成功，最新结果 `4 passed in 1.27s`。测试覆盖 Markdown/HTML/JSON/PDF/DOCX/PPTX/WAV 七种文件写出、HTML answer 内容转义、PDF 文件头为 `%PDF`、DOCX 的 `word/document.xml` 包含报告内容和 evidence quote、PPTX slide XML 包含报告标题和 evidence quote、WAV fake provider 写入 `RIFF` 头、TTS 文本把 `[S1]` 展开成可读的 `source S1`、未知格式报错。真实本机 smoke：`py -3.11 -m deepresearch_agent.cli "How should report audio export work?" --llm-provider mock --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --export-dir <temp> --export-formats wav --json` 成功，生成 WAV 文件头为 `RIFF`，文件大小 `6517700` bytes。CLI 支持 `--export-formats markdown,html,json,pdf,docx,pptx,wav`。这只验证本机 Windows SAPI WAV 摘要导出，不代表完整 podcast 或复杂办公排版已实现。

Trace exporter smoke：`py -3.11 -m pytest tests/test_tracing_exporter.py -q` 成功，`3 passed in 0.90s`。测试覆盖 `OtlpHttpTraceExporter` 向本地 HTTP server 的 `/v1/traces` POST、`build_trace_exporter()` 读取 OTLP 配置、以及 exporter 抛错时 `TraceLogger` 仍写 JSONL 并追加 `trace_exporter` error event。这里没有接真实 collector、LangSmith 或线上 APM。

Citation judge smoke：`py -3.11 -m pytest tests/test_citation_judge.py tests/test_quality_and_citations.py -q` 成功，`10 passed in 0.54s`。测试覆盖 fake judge 覆盖 verdict 并记录 `citation_judge` cost、heuristic judge 无 key 返回 `unverifiable`、DeepSeek judge stub 解析 `verdict/confidence/reason` 与 usage 成本、以及 orchestrator 配置 `citation_judge_provider="heuristic"` 后 assessment 带 `judge_provider="heuristic"`。相关集成回归：`py -3.11 -m pytest tests/test_citation_judge.py tests/test_quality_and_citations.py tests/test_spine.py tests/test_run_control.py tests/test_deep_research_eval.py -q` 成功，`27 passed, 2 warnings in 50.94s`。CLI smoke：`py -3.11 -m deepresearch_agent.cli "How should citation judges work?" --llm-provider mock --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --citation-judge-provider heuristic --json` 成功，输出包含 `judge_provider="heuristic"` 和 `success=true`。这里没有真实调用 DeepSeek citation judge，也没有重跑 benchmark。

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

真实 DeepSeek v4-flash + Wikipedia + local retrieval 对比 benchmark：两组都使用 `$env:REQUEST_TIMEOUT_SECONDS='8'`，`--llm-provider deepseek --llm-model deepseek-v4-flash --search-provider wikipedia --seed 20260607 --max-researchers 2 --max-results 3`。区别只在 `--local-retrieval-mode keyword` 和 `--local-retrieval-mode hybrid`。对比汇总写入 `results/retrieval_benchmark_comparison.json`；当前 `results/benchmark_summary.json` 被最后一次 local hybrid run 覆盖。

| 口径 | raw log | success_rate | citation_retention_rate_avg | deduped_source_count_avg | latency p50 | latency max | total_tokens | estimated_cost_usd_total | fallback_count_total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 纯关键词 local RAG baseline | `logs/benchmark-20260607T080835Z.jsonl` | 1.0 | 0.8867 | 8.2 | 24595.506ms | 39793.567ms | 17740 | 0.00368858 | 1 |
| 本地 hybrid：keyword + BGE vector + Chroma + RRF，rerank 关闭 | `logs/benchmark-20260607T081104Z.jsonl` | 0.6 | 0.8929 | 7.8 | 30747.284ms | 48727.728ms | 18842 | 0.00399994 | 2 |
| 百炼 hybrid embedding | 未实测 | 未实测 | 未实测 | 未实测 | 未实测 | 未实测 | 未实测 | 未实测 | 未实测 |

这次结果不能包装成“hybrid 一定更好”。本地 hybrid 的平均 citation retention 从 `0.8867` 小幅升到 `0.8929`，但 success_rate 从 `1.0` 降到 `0.6`，fallback 从 `1` 增到 `2`，p50 latency 从 `24595.506ms` 增到 `30747.284ms`，token 和成本也上升。我的解释是：当前 local corpus 很小，向量召回引入的额外 local source 会改变 synthesis 上下文和 citation checker 的 lexical overlap 分布；在小样本、无人工相关性标注的情况下，hybrid 是能力升级，不是质量保证。

逐 case 对比：keyword baseline 5 条全成功；local hybrid 中 case-001 retention `0.7143` 失败，case-002 retention `0.75` 失败，case-003/004/005 成功。case-003 的 fallback 从 keyword 的 `1` 变成 hybrid 的 `2`，说明同样的 planner/search 条件下仍有外部检索波动，不能把差异完全归因于 local retrieval。

百炼组没有跑：当前环境没有 `DASHSCOPE_API_KEY`。代码已经实现 DashScope embedding 和 rerank provider，并用 stub 测过 HTTP response parsing；真实百炼 embedding/rerank latency、费用和效果均未实测。

检索模块独立评测（BEIR/scifact）：这组只评测本地 retriever，不调用 DeepSeek、不调用 Wikipedia、不跑 orchestrator，也不产生 LLM token/cost。数据集是公开 BEIR/scifact，不是本项目自有数据；来源 URL 写在代码注释和结果文件里：`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`。验证结果：corpus `5183` 篇，queries `1109` 条，test qrels query `300` 条，relevant pair `339` 对。结果文件：`results/retrieval_eval_scifact.json`。

实际运行方式：先用 `py -3.11 -m deepresearch_agent.retrieval_eval` 校验数据集；然后用 `py -3.11 -m deepresearch_agent.retrieval_eval --run --modes keyword hybrid hybrid_rerank --top-k 10 --rerank-candidate-k 10 --output results\retrieval_eval_scifact.json` 全量重跑并由脚本覆盖结果 JSON。最终配置是 `top_k=10`、`rerank_candidate_k=10`、`embedding_provider=local`、`embedding_model=BAAI/bge-small-en-v1.5`、`rerank_provider=local`；本地 rerank provider 使用默认模型 `BAAI/bge-reranker-base`。这里把 embedding 模型从默认中文 BGE 切到英文 BGE，是因为 SciFact 本身是英文科学摘要检索任务；项目默认中文模型没有因此改变。

| 检索模式 | Recall@10 | nDCG@10 | MRR | 总耗时 | 平均每 query 延迟 | LLM tokens | API cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| keyword baseline | 0.6000 | 0.4823 | 0.4548 | 99.55s | 0.3318s | 0 | $0 |
| hybrid：keyword + local BGE vector + RRF | 0.8239 | 0.6597 | 0.6114 | 173.43s | 0.5781s | 0 | $0 |
| hybrid + local rerank | 0.8239 | 0.7307 | 0.7083 | 1032.93s | 3.4431s | 0 | $0 |

这组结果比 5 case 端到端 benchmark 更能说明 retriever 本身：hybrid 相对 keyword 的 Recall@10 从 `0.6000` 提到 `0.8239`，说明向量召回确实补到了更多 qrels 正例。本次 `rerank_candidate_k=10`、`top_k=10`，hybrid+rerank 只是重排 hybrid 的 top10 候选，所以 Recall@10 必须和 hybrid 一致，不能把 rerank 说成提升召回；它的价值体现在 nDCG@10 从 `0.6597` 到 `0.7307`、MRR 从 `0.6114` 到 `0.7083`，也就是相关文档排得更靠前。代价也很明显：hybrid 比 keyword 慢，rerank 在本地 CPU 上慢得更多，所以默认端到端 benchmark 仍不开 rerank。

这组评测不能外推成“线上问答质量提升”。SciFact 是英文科学摘要，query/qrels 来自公开 IR benchmark，和本项目求职知识库、小规模 local corpus、中文问题不是同一分布。面试里我会把它作为“检索模块结构升级的独立证据”，同时主动说它和 DeepSeek + Wikipedia 的端到端成功率、fallback、citation retention 是两套指标。

公开 Deep Research 端到端 artifact 评测（LiveDRBench preview）：这组是公开任务驱动完整 orchestrator，不是只测 retriever。脚本是 `src/deepresearch_agent/deep_research_eval.py`，默认从 Hugging Face datasets-server 拉 `microsoft/LiveDRBench` 的 `preview/test` 行，输出 summary、raw JSONL 和 LiveDRBench-style predictions。当前还没有接官方 judge，所以 `official_judge_score=not_run`；下面的 success 仍是本项目现有 citation retention 阈值口径，不是官方排行榜分数。`--judge-provider heuristic` 只在 case 带 ground truth 时做 normalized substring 命中评分；`--judge-provider deepseek` 是可选非官方 LLM answer judge，会记录 judge token/cost，但这次没有 live benchmark。

| 口径 | summary | raw log | case_count | success_rate | citation_retention_rate_avg | deduped_source_count_avg | latency p50 | total_tokens | estimated_cost_usd_total | fallback_count_total |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LiveDRBench preview + mock/mock plumbing | `results/deep_research_eval_livedrbench_mock_summary.json` | `logs/deep-research-eval-20260607T133222828802Z.jsonl` | 1 | 1.0 | 1.0 | 3.0 | 5.785ms | 2624 | 0.0 | 0 |
| LiveDRBench preview + DeepSeek v4-flash + Wikipedia | `results/deep_research_eval_livedrbench_deepseek_wikipedia_summary.json` | `logs/deep-research-eval-20260607T133237936766Z.jsonl` | 1 | 0.0 | 0.5 | 3.0 | 20910.052ms | 3434 | 0.00072332 | 0 |

这条公开真实 case 的 query 是让系统根据 `American Community Survey / FEMA Harvey flood depths / USDA Food Access Research Atlas / Streetlight / SafeGraph POI` 找使用全部数据集的论文，并按 JSON 返回 `paper_title`。真实组没有报错，也没有 fallback，但 success 为 0，说明当前 DeepSeek + Wikipedia + 本地 keyword RAG 没有解决这类公开精确查证任务；citation retention 只有 `0.5`，也说明 lexical citation check 已经暴露支撑不足。面试里我会把它讲成“公开评测入口已经打通，但质量短板被暴露出来”，不会把 mock 组的 `1.0` 当质量成果。

Public eval answer judge smoke：`py -3.11 -m pytest tests/test_deep_research_eval.py -q` 成功，最新结果 `6 passed in 0.55s`。测试覆盖 heuristic judge 对 ground-truth group 的命中评分、public eval raw/summary 写入 `answer_judgment`、以及 DeepSeek answer judge 的 stubbed HTTP 请求：`DEEPSEEK_API_KEY` 只从环境变量读，请求使用 `response_format={"type":"json_object"}`，模型为 `deepseek-v4-flash`，并能解析 `score=0.75 / verdict=partial / matched / missing / usage`，把 judge token 和估算成本写入 `AnswerJudgment`。CLI help 已显示 `--judge-provider {none,heuristic,deepseek}` 和 `--judge-model`；本地 CLI smoke：`py -3.11 -m deepresearch_agent.deep_research_eval --cases <temp>/cases.jsonl --benchmark-name local-cli-smoke --llm-provider mock --search-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --judge-provider heuristic --raw-log <temp>/raw.jsonl --summary-output <temp>/summary.json --predictions-output <temp>/predictions.json` 成功，summary 的 `answer_judge.provider=heuristic`、`scored_count=1`、`score_avg=1.0`、`tokens_total=0`。这是本地字符串命中和 DeepSeek stub 单测，不是官方 judge，也不是 DeepSeek live judge benchmark。

Source diversity metrics smoke：新增断言覆盖 `/research` 生成的 `StructuredReport.metrics` 和 `/runs/worker/next` 结果里包含 `source_provider_count`、`source_domain_count`、`source_provider_counts`、`source_domain_counts`；`benchmark.py` 和 `deep_research_eval.py` 也会把 provider/domain count 写入 case record 并汇总平均数。当前完整回归 `py -3.11 -m pytest -q` 成功，最新结果 `89 passed, 2 warnings in 69.53s`。这里没有重跑真实 DeepSeek/Wikipedia benchmark，指标只证明 plumbing 和 schema 已接入。

未实测：LiveDRBench/Deep Research Bench 官方 judge 分数、DeepSeek answer judge live benchmark、真实搜索 API 高并发/多进程全局限流、Brave/Tavily live search benchmark、OpenAI-compatible LLM live endpoint、DeepSeek citation judge live benchmark、Redis/PostgreSQL 缓存、多进程 worker pool / 分布式队列、真实 OpenTelemetry collector / LangSmith tracing、真实用户流量、DashScope 真实 embedding/rerank、真实 Qdrant 服务 live benchmark、扫描件 OCR、大规模私有语料增量 reindex、rerank 5 case 全量 benchmark。

# 8 评测设计

answer completeness：当前已经有可选 answer judge，会在 case 提供 ground truth 时写入 `answer_judgment` / `answer_judge` summary。`heuristic` 是本地 normalized substring 弱信号；`deepseek` 是非官方 DeepSeek JSON mode LLM judge，会记录 judge model、token 和估算成本。LiveDRBench preview 已经能驱动完整 orchestrator 产 artifact，但官方 answer-quality 分数和 DeepSeek answer judge live benchmark 仍未实测。
citation faithfulness：当前提交过的 benchmark 指标仍以 claim/source lexical overlap 为基础，但已经从单一分数升级到 `support_level`、`evidence_quotes`，并新增可选 citation judge provider。mock plumbing run 平均 retention 是 `1.0`，只能说明 mock 引用链路没断；最新 DeepSeek v4-flash + Wikipedia 对比里，keyword baseline 平均 retention 是 `0.8867`，local hybrid 是 `0.8929`，但 hybrid success_rate 更低，说明不能只看均值。DeepSeek citation judge 还没有真实 benchmark，不能把它当成已验证质量提升。
retrieval quality：端到端 benchmark 里的 citation_retention 会受 LLM 和 search 波动影响，所以我新增了 BEIR/scifact 独立检索评测，直接用 qrels 计算 Recall@10、nDCG@10、MRR。当前真实结果是 keyword `0.6000/0.4823/0.4548`，hybrid `0.8239/0.6597/0.6114`，hybrid+rerank `0.8239/0.7307/0.7083`。
source diversity：当前记录 deduped_source_count，也新增了 `source_provider_count`、`source_domain_count`、`source_provider_counts`、`source_domain_counts`，benchmark/public eval 会汇总 provider/domain 平均数；local retrieval metadata 仍记录 keyword/vector/rerank rank。但这还不是人工相关性或来源独立性评分，只是结构化多样性信号。
hallucination rate：当前用 unsupported citation count 作为 proxy，不能覆盖无引用幻觉。
latency：benchmark 记录每 case latency_ms，并计算 P50/P90/max；mock latency 只能作为 plumbing 回归信号，DeepSeek + Wikipedia latency 包含真实网络/API 时间，也不能当线上 SLA。local hybrid 比 keyword baseline p50 多 `6151.778ms`；独立检索评测里 keyword 平均每 query `0.3070s`，hybrid `0.5781s`，hybrid+rerank `3.4431s`，这些都需要如实讲。
cost：mock provider 成本为 0，token 用字符估算；DeepSeek provider 已接真实 usage，并按当前实现里的 v4-flash 价格常量估算成本，价格核对日期 `2026-06-07`。BEIR/scifact 独立检索评测不调用 LLM，LLM token 和 API cost 都是 0；本地 embedding/rerank 不产生 API 成本，但会产生本机 CPU/GPU 时间；DashScope 成本未实测。
工具失败恢复：有 unit test 覆盖 primary failure fallback、circuit breaker open、本地 SearchRateLimiter 和 retry backoff；第一次 DeepSeek + Wikipedia benchmark 出现过 fallback，修复 Wikipedia 长查询压缩后 fallback 曾降到 0，但最新 keyword/hybrid 对比里仍分别出现 `fallback_count_total=1` 和 `2`，已在第 7 节如实记录。限流/backoff 目前只是单进程 primary search 保护，没有真实 429、`Retry-After` 或跨 worker 全局 quota 实测。
multi-hop / reflection 成功率：当前没有真实 multi-hop 标注集，未实测质量提升；但 reflection loop 的控制流已用 mock/keyword smoke 验证，会在证据不足启发式触发时追加 `R<N>` follow-up question，并把 compression/reflection payload 写入 trace。下一步需要公开多跳任务或人工标注集来评估是否真的提升答案质量。

评测集构造方式：端到端 5 条 case 是围绕本项目核心能力手写的 smoke benchmark，覆盖 supervisor-researcher、citation faithfulness、tool failure、cost tracking、benchmark reproducibility。公开检索标准口径采用 BEIR/scifact test qrels，只评测 local retriever，不评测 LLM 回答质量。公开 Deep Research 端到端口径新增 LiveDRBench preview runner，会跑完整 orchestrator 并保存 answer/source/trace/cost/predictions artifacts；有 ground truth 的本地/公开 case 可以额外开 heuristic 或 DeepSeek answer judge；当前只跑了 1 条 mock 和 1 条 DeepSeek/Wikipedia 样本，官方 judge 分数和 DeepSeek answer judge live 口径尚未接入。

# 9 与参考项目的差异

## open_deep_research

参考了什么：我读了 README 和 CLAUDE.md，参考了它的 deep research 三段式、multi-agent / parallel researcher、模型需要 structured output + tool calling、评测和配置化思想；这次还参考它把公开 deep research benchmark 当可信度证据的方向。
没照搬什么：没有复制它的 LangGraph graph、prompt、state、配置文件或 eval 代码。
我做了哪些改造：改成自定义轻量 orchestrator，把 citation checker、source verifier、fallback、trace/cost、benchmark 都做成直接可读的小模块；公开评测先做成 LiveDRBench artifact runner，保留本项目自己的 summary/raw log/predictions 输出，而不是直接搬它的评测栈。
为什么更适合求职展示：代码量小，面试时能从 API 到 citation check 一路讲清楚，不会被大框架细节淹没。

## deep_research_from_scratch

参考了什么：参考了它按 notebook 拆 scope、research agent、MCP、supervisor、full agent 的学习路径。
没照搬什么：没有复制 notebook 代码，也没有依赖 Tavily/OpenAI key。
我做了哪些改造：把学习型 building blocks 改成可安装 Python package、CLI、FastAPI、pytest 和 benchmark。
为什么更适合求职展示：它像课程，本项目像一个可运行的后端工程。

## DeerFlow v1

参考了什么：按要求只看了 `main-1.x` 分支 README，参考它 Coordinator/Planner/Researcher/Reporter 的角色划分，以及 web UI/API/工具集分层思路。
没照搬什么：没有复制它的 web UI、crawler、TTS、presentation、checkpoint、配置系统。
我做了哪些改造：砍掉内容生产和平台能力，只保留 deep research 主干和后端可观测部分；这次补了 SearxNG/Jina/Brave/Tavily 这种搜索与正文抽取边界，但仍保持统一 `Source` 输出，不把 DeerFlow 的完整前端和工具生态搬进来。
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

真实 LLM provider 覆盖仍需 live 验证：当前真实主路径是 DeepSeek v4-flash，也新增了 OpenAI-compatible adapter，并支持 brief/planner/synthesis 的阶段模型覆盖；但真实端到端 benchmark 主要仍是 DeepSeek，OpenAI-compatible endpoint 没有 live 跑。可行方案是继续验证本地 Ollama/OpenRouter/OpenAI-compatible 网关，再实现 OpenAI/Anthropic 原生 provider，并统一 structured output、usage 解析、重试、模型定价表、动态降级和测试替身。工程代价是 API key、价格、限流、错误码差异和 CI mock。面试怎么讲：我会说我已经把真实 provider 接入路径、通用兼容 adapter 和阶段模型路由打通，但不会把 stub 单测夸成通用生产能力。

Citation checker 语义能力仍需实测：当前已经有可选 heuristic / DeepSeek citation judge provider，但默认 benchmark 仍是 lexical overlap，DeepSeek judge 没有 live 跑完整评测。可行方案是用公开/人工标注 claim-evidence 集测 judge agreement，再接 NLI 模型或 sentence embedding entailment。工程代价是成本、延迟、标注和 judge 可靠性评估。面试怎么讲：我会说现在是“lexical baseline + optional judge 接口”，不是最终事实评审。

搜索质量仍需 live 验证：Wikipedia 能跑但相关性和覆盖有限；SearxNG 需要自建实例；Jina Search 在当前环境返回过 401/403；Brave/Tavily 已接 adapter 但没有 key 跑 live benchmark；本地 HTML crawler 能做基础正文抽取，但不执行 JavaScript、不做正文主内容识别、反爬或 robots。可行方案是配置真实 Brave/Tavily/SerpAPI/自建 SearxNG，并做页面正文抽取与相关性评测。工程代价是 key、限流、费用、网页解析质量和 provider schema 差异。面试怎么讲：我会强调 adapter 已经抽象好，替换 provider/crawler 不影响 orchestrator，但不会把未 live 的 provider 包装成生产搜索。

Hybrid retrieval / private corpus 还没有证明质量稳定提升：当前已经实现 keyword + vector + RRF、Markdown/TXT/PDF/DOCX 到 JSONL ingest、可选持久化 Chroma index、可选 Qdrant HTTP vector index provider 和可选 rerank，但 5 case 小样本里 hybrid success_rate 反而低于 keyword baseline，Qdrant 也只做了 stub HTTP 单测。可行方案是扩大本地语料、补人工相关性标注、调 RRF 权重、做增量 manifest/embedding cache、OCR 扫描件、启动真实 Qdrant/Milvus 做索引生命周期和延迟评测，并把 rerank 纳入全量 benchmark。工程代价是索引生命周期、模型加载时间、评测集标注、外部服务运维和更多运行成本。面试怎么讲：我会说我完成了检索结构、私有文档入口和向量库 provider 边界升级，但不会把一次小样本结果或 stub Qdrant 测试包装成质量提升，也不会把基础文件 ingest 说成完整 RAGFlow。

Run control 还不是分布式调度：当前已经有 SQLite run store、request_json 请求快照、planner checkpoint、researcher 阶段 checkpoint 复用、approval/resume/cancel/retry、SSE replay、单机 worker lease/heartbeat、`defer_execution=true` + `/runs/worker/next` 的 worker-once 消费入口、本地 polling worker CLI，以及运行中取消的阶段边界状态一致性保护，但它仍是轻量实现。可行方案是引入真正的 worker queue、多进程 worker pool、PostgreSQL/Redis、provider 级 abort、阶段幂等和更细粒度 checkpoint。工程代价是并发一致性、任务抢占、schema migration、大对象存储和运维复杂度。面试怎么讲：我会说我先把长任务控制平面闭环、worker ownership、最小队列消费语义、取消终态一致性和 researcher checkpoint reuse 做出来，生产化再升级存储和调度，不把 SQLite 版本包装成高并发任务系统。

内容导出还不是完整办公文档 / podcast 生成：当前支持 Markdown、HTML、JSON、文本版 PDF、DOCX、PPTX artifact 和 Windows SAPI WAV 语音摘要，但还没有复杂版式、图表、图片、批注、模板系统、配乐、多角色播客或跨平台 TTS。可行方案是为 PDF/DOCX/PPTX 做模板、目录、分页和渲染校验，并把 TTS provider 扩展到云端或本地跨平台语音引擎。工程代价是版式、字体、分页、图片、图表、音频时长、语音质量和跨平台验证。面试怎么讲：我会说我已经把常见报告文件格式和本地语音摘要打通，但不会把它包装成完整文档/podcast 生产系统。

OpenTelemetry/LangSmith 仍是轻量出口：当前已经有可选 OTLP HTTP trace export，但只验证到本地 test server，不是完整 SDK/collector/LangSmith run tree。可行方案是引入官方 OTel SDK、batch processor、collector 配置、采样策略，或增加 LangSmith exporter。工程代价是外部账号、部署、隐私、采样和成本。面试怎么讲：我会说我先做的是稳定 trace event 和可外送边界，生产化再接完整可观测平台。
