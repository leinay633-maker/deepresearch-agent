# 0 项目一句话介绍

本项目是我从空仓库开始手写的一个收窄版 DeepResearch Agent，目标不是复刻大而全的 open_deep_research，而是把「问题澄清、research brief、并发 researcher、来源去重、带引用合成、citation check、trace 和 benchmark」这条主链路做干净，并补上可管理的 run control plane。它解决的是普通 RAG 一次性检索后直接回答时，难以解释检索路径、引用是否支撑论断、工具失败如何降级的问题。当前版本默认使用 mock LLM 和 mock search，保证无 API key 也能一条命令跑通；同时已经接入 DeepSeek 真实 LLM provider、Wikipedia 真实检索 adapter、本地关键词 + 向量 + RRF 融合的 hybrid local retrieval、SQLite run_id / checkpoint / HITL / SSE replay 控制平面，以及 LiveDRBench 这类公开 Deep Research 任务的端到端 artifact 评测 runner。这个项目体现的 Agent 后端能力主要是多阶段编排、并发工具调用、失败兜底、混合检索、可观测性、成本归因、可复现评测和长任务状态管理。

# 1 岗位匹配

我做这个项目时刻意对齐 Agent 后端 / LLM 应用岗，而不是做一个只会调用 LLM 的 demo。JD 里常见的 LangGraph、RAG、MCP、并发、可观测性、评测这些关键词，在本项目里对应到清晰的工程模块：`orchestrator.py` 做轻量编排，`rag.py` 做本地 keyword/vector hybrid RAG，`embeddings.py` 和 `rerankers.py` 做可切换 provider，`search.py` 做工具 adapter、重试、超时、熔断和降级，`tracing.py` 和 `cost.py` 做观测和成本归因，`benchmark.py` 做可复现评测。

我在第一阶段没有强行让默认路径依赖真实 LLM provider，因为没有 API key 时会阻塞陌生人 clone 运行。最终选择是默认保留 `MockLLMProvider` 做可复现测试和 mock plumbing benchmark；当环境变量 `DEEPSEEK_API_KEY` 存在时，可以显式启用 `DeepSeekLLMProvider` 跑真实 structured output、synthesis、token usage 和 cost。OpenAI/Anthropic 等其他 provider 仍作为 v2 扩展。

# 2 总体架构

API 层：`src/deepresearch_agent/api.py`。输入是 `ResearchRequest` 或 `CreateRunRequest`，输出是 `StructuredReport`、`AgentRun` 或 run trace。保留 `/research` JSON 接口、`/research/stream` SSE 接口和 `/health`；新增 `/runs`、`/runs/{run_id}/approve`、`/edit`、`/reject`、`/cancel`、`/retry`、`/steps`、`/events`、`/trace`，以及用于 worker ownership 的 `/runs/{run_id}/lease`、`/heartbeat`、`/runs/stale`、`/runs/recover-stale`。我参考 FastAPI + LangGraph 模板时只吸收了「服务层薄封装、每次请求创建编排器、接口返回结构化对象」这个思路，没有引入 JWT、Postgres、Redis、Langfuse 或 Prometheus。

Agent 编排层：`src/deepresearch_agent/orchestrator.py`。输入是用户 query 和配置，输出是完整报告。它按 clarify/normalize、planner、并发 researcher、source dedup、synthesizer、citation check 的顺序执行。这里我没有直接用 LangGraph，是因为当前目标是可讲清楚的收窄项目，轻量 orchestrator 更便于展示每个阶段的输入输出和失败边界。

工具 Adapter 层：`src/deepresearch_agent/search.py`、`src/deepresearch_agent/rag.py`、`src/deepresearch_agent/embeddings.py`、`src/deepresearch_agent/rerankers.py`。搜索层有 `MockSearchAdapter`、`WikipediaSearchAdapter`、`SearxngSearchAdapter`、`JinaSearchAdapter` 和 `JinaReaderCrawler`，外加 `SearchService` 负责 retry、timeout、circuit breaker 和 fallback。本地 RAG 用 `data/local_corpus.jsonl`，默认走关键词 + BGE 向量 + Chroma + RRF 融合；也可以显式切回 keyword baseline，开启持久化 Chroma index，或者开启本地 / DashScope rerank。

检索质量层：`src/deepresearch_agent/dedup.py`、`src/deepresearch_agent/verifier.py`。Dedup 按规范化 URL 合并重复来源，Verifier 按标题、正文长度、稳定 URL、已知 adapter、低质量模式打分过滤。

评测层：`src/deepresearch_agent/benchmark.py`、`src/deepresearch_agent/retrieval_eval.py`、`src/deepresearch_agent/deep_research_eval.py`、`data/benchmark_cases.jsonl`、`tests/`。端到端 benchmark 固定 seed 和配置快照，记录 latency、tokens、cost、source count、citation retention、success；独立检索评测只加载 BEIR/scifact 的 corpus/query/qrels，计算 Recall@10、nDCG@10 和 MRR，不调用 LLM、Wikipedia 或 orchestrator 主链路；公开 Deep Research 评测 runner 会加载 LiveDRBench 等公开任务，跑完整 orchestrator 并输出 answer、sources、trace、cost、citation check 和 predictions artifact。

可观测层：`src/deepresearch_agent/tracing.py`、`src/deepresearch_agent/cost.py`。Trace 每阶段写 JSONL，Cost 按 brief_generation、planning、synthesis 归因 token 和成本；mock 路径仍是字符数近似，DeepSeek 路径使用 provider 返回的真实 usage。

Run Control Plane：`src/deepresearch_agent/run_models.py`、`src/deepresearch_agent/run_store.py`、`src/deepresearch_agent/run_control.py`。输入是 run create/approval/cancel/retry 请求，输出是持久化 run 状态、step trace、event stream 和最终 report checkpoint。它用 SQLite 保存 `agent_runs`、`agent_steps`、`agent_events`，默认文件是 `data/runs.sqlite`，可以用 `RUN_STORE_PATH` 覆盖；测试用临时 SQLite 文件。

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
最终选择：web search adapter 和 hybrid local RAG 并存。local RAG 保留 keyword baseline，同时新增 BGE embedding、Chroma vector index、RRF 融合、可选持久化 Chroma index 和可选 rerank；每个 researcher 仍合并 web search 与 local RAG 来源。
理由：keyword 对精确术语稳定，vector 对语义相近问题更友好，RRF 不要求两路分数同尺度；统一 `Source` 抽象让下游 dedup、verifier、synthesizer 不需要改。
代价：本地 embedding / Chroma / rerank 会增加依赖和延迟；最新真实 benchmark 里 local hybrid 的 citation retention 略高于 keyword baseline，但 success_rate 更低，说明混合检索不是自动变好，需要更大语料和 rerank/权重调优。
面试怎么答：我会说我没有用向量替换关键词，而是保留两路召回再融合；实测结果不全是好看的，反而暴露了小语料场景下 hybrid 可能引入不稳定来源。

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
代价：它不是分布式调度系统；现在只有 SQLite 单机 lease/heartbeat，没有 worker queue、并发抢占和跨进程实时取消；阶段级恢复当前以 planner checkpoint 后从 researcher 重跑为主，没有精确恢复到某个 researcher 子任务内部。
面试怎么答：我会说我借鉴的是 LangGraph 的 durable execution、checkpoint 和 human-in-the-loop 思想，但没有为了框架迁移牺牲项目可读性；我实现的是后端控制平面最小闭环：状态机、SQLite checkpoint、approval/resume/cancel/retry、SSE replay。

## 决策 11：为什么先补公开 Deep Research artifact 评测，而不是先做 judge 打分

背景：之前只有 5 条本地端到端 smoke benchmark 和 BEIR/scifact 检索模块评测。它们能证明 plumbing 和 retriever，但不能回答“面对公开 Deep Research 任务，完整报告链路表现怎样”，这是和 open_deep_research 这类项目的可信度差距。
可选方案：直接接官方 judge/LLM 评分；先手写更多本地 case；先复用 LiveDRBench/Deep Research Bench 任务格式输出 artifacts；继续只看 BEIR/scifact。
最终选择：新增 `src/deepresearch_agent/deep_research_eval.py`，优先做公开任务加载、完整 orchestrator 运行、raw JSONL、summary JSON 和 LiveDRBench-style predictions 输出；judge provider 先保留为 `none`，不编官方分数。
理由：没有 judge key 或官方 scoring 环境时，最重要的是先把每题的 query、配置快照、answer、claims、sources、trace、token、cost、citation check 和失败原因保存下来，保证结果可复查。这样下一步接 judge 或人工复核时不用重构评测入口。
代价：当前 `success_rate` 仍沿用本项目 citation retention 阈值，不等于官方 Deep Research Bench 分数；LiveDRBench 任务常要求精确 JSON/论文标题，当前 synthesizer 还不是专门为该格式训练或约束的，所以真实组可以跑但质量很差。
面试怎么答：我会说我先补的是“公开任务可跑、artifact 可审计”的评测底座，而不是假装已经有官方 leaderboard 分数。最新 1 条 LiveDRBench preview 真实口径就是失败样本：DeepSeek v4-flash + Wikipedia 跑通但 `success_rate=0.0`、citation retention `0.5`，这反而暴露了搜索覆盖和 citation 语义评测短板。

## 决策 12：为什么把 web search 和 crawler 分开接

背景：Wikipedia adapter 太窄，公开 Deep Research 任务需要真正的 web search 和网页正文抽取；如果 search adapter 只返回 title/snippet，synthesizer 很容易在证据不足时失败。
可选方案：继续强化 Wikipedia；直接接 Tavily/Brave 这种一体化 API；先接 SearxNG 搜索和 Jina Reader crawler；把 crawler 混在每个 search provider 里。
最终选择：在 `search.py` 里新增 provider registry，保留 `mock/wikipedia`，再加 `SearxngSearchAdapter`、`JinaSearchAdapter` 和独立 `JinaReaderCrawler`。SearxNG 负责搜索候选 URL，crawler 负责把 URL 转成正文；`SearchService` 的 retry、timeout、circuit breaker、fallback 仍复用。
理由：search 和 crawler 的失败模式不同，分开后可以替换任一层：SearxNG 可以换 Brave/Tavily，Jina Reader 可以换 trafilatura/readability/Firecrawl，而 orchestrator 仍只看统一 `Source`。默认仍是 mock，不破坏无 key 路径；SearxNG 用 `SEARXNG_BASE_URL`，Jina 可选 `JINA_API_KEY`，所有 key 都只从环境变量读。
代价：当前没有自建 SearxNG 实例，所以 SearxNG 只做了 stub 单测；Jina Reader live crawl `https://example.com` 成功，但 Jina Search live smoke 在当前网络下返回 `401/403`，实际 query 走了 mock fallback。它是 provider 结构升级，还不是“真实搜索质量已经解决”。
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
代价：这不是完整任务队列，没有 worker pool、分布式调度、公平排队、幂等 stage replay、长任务实时取消抢占和强事务隔离；SQLite 锁竞争下也不适合高并发。
面试怎么答：我会说我先补的是“同一个 run 只能被一个 worker 拿走”的控制平面语义，而不是假装做了生产队列。这个取舍让项目保持可跑，同时能解释下一步怎么演进到 Postgres/Redis/worker queue。

## 决策 18：为什么先做可选持久化 Chroma index，而不是直接上 Qdrant/Milvus

背景：local hybrid retrieval 每次新建 retriever 都要为 `data/local_corpus.jsonl` 重新 embedding 并创建 Chroma collection，审核页 smoke 也暴露出 hybrid 冷启动会拖慢 run。
可选方案：保持临时内存 index；默认启用本地持久化 Chroma；接 Qdrant/Milvus；先做 embedding pickle cache。
最终选择：新增 `LOCAL_VECTOR_INDEX_PERSIST` 和 `LOCAL_VECTOR_INDEX_PATH`，默认关闭；显式开启后使用 Chroma PersistentClient，并用 corpus chunk、embedding provider、embedding model 的 fingerprint 生成稳定 collection 名。collection count 与当前 chunk 数一致时直接复用，不重新 embed corpus。
理由：这一步只改检索层，不影响 orchestrator 的 `Source` contract，也不需要额外服务、端口或账号。它先解决“本地私有知识库索引生命周期”这个能力缺口，同时保持无 API key 可跑通。
代价：Chroma 本地持久化仍是单机文件索引，不是生产级向量数据库；当前复用判断主要靠 fingerprint collection name 和 count，没有后台 reindex 任务、版本迁移、TTL、压缩、并发写控制或跨机器共享。
面试怎么答：我会说我先把私有知识库从“每次临时建索引”推进到“可复用的本地持久化索引”，但不会把它说成 Qdrant/Milvus 级别的生产向量库。

# 5 实现细节

Planner：`src/deepresearch_agent/llm.py`。输入是 `ResearchBrief`，输出是 `SubQuestion` 列表。默认 deterministic mock planner 会生成 background、evidence、tradeoffs 三类问题，用于离线可复现；DeepSeek planner 会用 JSON mode 生成符合同一 Pydantic schema 的子问题。局限是 planner 还不会根据 researcher 中间结果动态追加子问题。

DeepSeek Planner 验证：`src/deepresearch_agent/llm.py` 里新增了 `DeepSeekLLMProvider.plan`，第一步先独立验证结构化输出；验证脚本是 `src/deepresearch_agent/validate_deepseek_structured_output.py`，它从环境变量读取 `DEEPSEEK_API_KEY`，用 JSON mode 请求默认 `deepseek-v4-flash`，并用现有 `SubQuestion` Pydantic schema 解析输出。后续步骤再把同一个 provider 扩展到 `create_brief` 和 `synthesize`，避免一次性接太多导致错误边界不清。

DeepSeek Synthesizer 接入：步骤 2 以后，`DeepSeekLLMProvider.create_brief`、`plan`、`synthesize` 都走 DeepSeek JSON mode。CLI 和 API 可以通过 `llm_provider="deepseek"` 或 CLI 参数 `--llm-provider deepseek` 显式启用；默认仍是 mock，保证离线测试不受 API key 影响。当前 synthesis 要求模型输出 `{"answer": "...", "claims": [...]}`，并要求每条 factual claim 使用输入 sources 中已有的 `[Sx]` citation ID。

Researcher：`src/deepresearch_agent/orchestrator.py` 的 `_research_one`。输入是子问题，输出是 `Finding`。它调用 `SearchService` 和 `LocalRagRetriever`，再 dedup、verify、summary。`LocalRagRetriever` 内部已经从 keyword overlap 升级为可配置的 keyword / hybrid retrieval，但 orchestrator 仍只接收 `Source` 列表。局限是 summary 仍是模板化，不是自然语言 LLM 压缩。

Reflection / Compression Loop：`src/deepresearch_agent/orchestrator.py` 的 `_run_reflection_rounds`、`_compress_findings`、`_reflect_on_evidence`。输入是初始 plan 和 researcher results，输出是可能扩展后的 plan/results。开启 `reflection_enabled` 后，每轮先把 findings 压成短文本写入 `compression.roundN` trace，再根据 fallback_count 和每个 finding 的唯一 source 数是否低于 `reflection_min_sources` 来决定是否追加 `R<N>` 子问题。`run_control.py` 的 researcher 阶段也调用同一个 helper，所以 `/research` 和 `/runs` 语义一致。局限是当前 policy 是启发式，不是 LLM reflection。

Web Search / Crawler Provider：`src/deepresearch_agent/search.py`。`build_search_adapter()` 现在按 provider name 构造 `mock`、`wikipedia`、`searxng` 或 `jina`，`build_crawler()` 按配置构造 `JinaReaderCrawler`。`SearxngSearchAdapter` 调 `SEARXNG_BASE_URL/search?format=json`，解析 title/url/snippet，再可选调用 crawler 抽正文；`JinaReaderCrawler` 用 `https://r.jina.ai/<url>` 抽 LLM-friendly text；`JinaSearchAdapter` 用 `https://s.jina.ai/<query>`。`JINA_API_KEY` 是可选环境变量，不进入 Settings 快照。

MCP Tool Adapter：`src/deepresearch_agent/mcp_tools.py`。`McpToolSearchAdapter` 用 `McpClient.call_tool()` 调配置好的 search tool，把 MCP result 里的 `sources` 或 `content[type=text]` 解析成统一 `Source`。`HttpMcpClient` 用 JSON-RPC HTTP POST，`StdioMcpClient` 用 MCP 的 `Content-Length` framing 和子进程 stdin/stdout。当前实现只覆盖 `tools/list`、`tools/call` 和 search-like result 转换，不做资源订阅或长连接池。

Embedding Provider：`src/deepresearch_agent/embeddings.py`。输入文本列表，输出向量列表。默认 `LocalEmbeddingProvider` 使用 `sentence-transformers` 加载 `BAAI/bge-small-zh-v1.5`，无 API key；`DashScopeEmbeddingProvider` 调百炼 OpenAI-compatible embeddings endpoint，key 只从 `DASHSCOPE_API_KEY` 读。验证脚本是 `src/deepresearch_agent/validate_embeddings.py`，本机 local BGE 实测维度 `512`；DashScope 因未配置 key，只做了 stub endpoint 解析测试。

Hybrid Local Retriever：`src/deepresearch_agent/rag.py`。输入 query 和 top-k，输出统一 `Source`。keyword 路按 token overlap 排序；vector 路先把 `data/local_corpus.jsonl` 分块，用 embedding 建 Chroma collection，再按 cosine distance 检索；融合用 RRF，metadata 记录 keyword_rank、vector_rank、fusion score 和权重。默认不强制写本地索引；设置 `LOCAL_VECTOR_INDEX_PERSIST=true` 后，Chroma 会写入 `LOCAL_VECTOR_INDEX_PATH`，collection 名由 corpus chunk、embedding provider 和 model 指纹决定，count 匹配时复用已有 collection。局限是当前还没有 Qdrant/Milvus 这种独立向量数据库，也没有索引管理后台。

Rerank Provider：`src/deepresearch_agent/rerankers.py`。输入 query 和候选 source，输出重排分数。默认 provider 是本地 `BAAI/bge-reranker-base`，但 `rerank_enabled` 默认关闭；DashScope rerank provider 调 `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`，key 仍只读 `DASHSCOPE_API_KEY`。本地 rerank 单条 smoke 跑通过，但首次模型下载/加载使 latency 约 `279692.721ms`，所以没有把它放进默认 benchmark。

Retrieval Eval Harness：`src/deepresearch_agent/retrieval_eval.py`。输入是 BEIR/scifact 的 corpus、queries、qrels，输出 `results/retrieval_eval_scifact.json`。这个脚本不调用 LLM、不调用 Wikipedia、不跑 orchestrator，只把 SciFact 文档写成本项目 local corpus 格式，然后复用 `LocalRagRetriever` 跑 keyword / hybrid / hybrid+rerank。评测默认不保存每个 query 的 ranking 明细，避免结果文件过大；需要排查时可以显式加 `--include-rankings`。

Public Deep Research Eval Harness：`src/deepresearch_agent/deep_research_eval.py`。输入可以是本地 JSONL/JSON/CSV，也可以直接从 Hugging Face datasets-server 拉 `microsoft/LiveDRBench` 的 `preview` 或 `v1-full` split。它会跑完整 `DeepResearchOrchestrator`，写 `logs/deep-research-eval-*.jsonl`、summary JSON 和 LiveDRBench-style `preds` 文件。当前 `judge_provider` 只有 `none`，所以它只产 artifact 和本项目已有 success/citation/cost 指标，不产官方 answer-quality 分数。

Verifier：`src/deepresearch_agent/verifier.py`。输入是 source 列表，输出是过滤后的 source。关键设计是可解释 quality reasons。局限是规则打分，不能真正判断来源权威性。

Synthesizer：`src/deepresearch_agent/llm.py` 的 `synthesize`。输入是 brief、plan、findings、sources，输出 answer 和 claims。默认 mock 会生成可测报告，DeepSeek provider 会用 JSON mode 生成 markdown answer 和结构化 claims。局限是 DeepSeek 输出目前只靠 prompt 约束和后置 citation checker，没有做二次 LLM judge 或强制 source quote。

Citation Checker：`src/deepresearch_agent/citation.py`。输入是 claims 和 sources，输出 `CitationCheckReport`。每条 `CitationAssessment` 现在包含 citation IDs、`supported`、`support_level`、overlap score 和最多 3 条 `evidence_quotes`。checker 会从 cited source 里按句子找最接近 claim 的 quote，输出 `supported / partial / unsupported / unverifiable`。局限是证据定位仍基于 lexical overlap，还不是 NLI/LLM judge。

Cost Tracker：`src/deepresearch_agent/cost.py`。mock provider 仍使用字符数近似估算；DeepSeek provider 已接入 API 返回的真实 `prompt_tokens` / `completion_tokens`，并通过 `CostTracker.add_usage()` 记录到同一套 `CostSummary`。当前 `deepseek-v4-flash` 成本计算按 DeepSeek 官方 Models & Pricing 页，核对日期 `2026-06-07`：input cache hit `$0.0028/1M tokens`，input cache miss `$0.14/1M tokens`，output `$0.28/1M tokens`。legacy alias 只作为 v4-flash 兼容入口使用同一价格表；未配置价格的模型会直接报错，避免 silently 用错单价。如果响应没有 token usage，DeepSeek 路径会直接失败，不会退回字符估算伪装成真实 usage。

Trace Logger：`src/deepresearch_agent/tracing.py`。每个 run 写 `logs/research-<run_id>.jsonl`，记录 stage、status、duration_ms、payload。runtime trace 默认不提交 Git，benchmark 原始记录提交。

Agent Run Control Plane：`src/deepresearch_agent/run_control.py` 外包现有 DeepResearch pipeline，不替换 `/research`。`POST /runs` 会创建 `run_id`，执行 brief/planner，然后在 `require_approval=true` 时进入 `waiting_approval`；`approve` 会从 planner checkpoint 继续 researcher、synthesizer、verifier；`edit` 会保存修改后的 subquestions 再继续；`reject/cancel` 会终止 run；`retry` 对 failed run 优先复用 `plan_json` 从 researcher 阶段重跑。`run_store.py` 的 SQLite schema 是三张表：`agent_runs` 保存 run 状态、plan/result、token/cost、`leased_by`、`heartbeat_at`、`lease_expires_at`；`agent_steps` 保存阶段输入输出、latency、token_usage、cost、error、retry_count；`agent_events` 保存可 SSE replay 的单调递增 event。内部执行路径会在 planner/researcher/synthesizer/verifier 阶段 acquire/heartbeat/release lease；外部 worker 也可以用 `/runs/{run_id}/lease`、`/heartbeat`、`/runs/stale`、`/runs/recover-stale` 做 ownership 和 stale recovery 验证。

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
修复：把默认模型改成 `deepseek-v4-flash`，验证脚本默认值同步迁移；成本计算改成按模型查价格表，legacy alias 仅为兼容旧配置保留；新增单测覆盖 cache hit/cache miss/output 成本计算。随后用 `deepseek-v4-flash` 重跑 planner schema validation 和真实 benchmark。
复盘：模型 provider 不是一次接完就结束，模型名、价格和功能支持都会变；代码里必须有显式定价表和失败策略，不能把过期单价悄悄沿用。
面试可能追问：为什么不直接删掉 legacy alias？回答：删掉会让旧运行记录和用户自定义旧模型名立即失效；我保留兼容入口，但当前默认和新 benchmark 都走显式 `deepseek-v4-flash`。

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
工程风险：当前 lease 只解决单个 run 的 ownership 和过期恢复，不提供 worker queue、任务分发、公平调度或阶段级幂等 replay。如果一个长 researcher 阶段内部卡住，heartbeat 只能在阶段边界刷新，不能像真实 worker 进程那样持续后台续租。
修复：没有为了掩盖这个边界去加复杂基础设施，只在代码和文档里明确：这是 SQLite 单机 control-plane primitive，下一步才是 Postgres/Redis/worker queue。
复盘：这个边界比“加一个队列表名”更重要。面试时我会承认当前还不是分布式执行，但已经有了迁移到生产队列前必须定义清楚的 lease 字段、stale 判断和 recovery 语义。
面试可能追问：为什么不直接 Celery？回答：因为这个项目的下限是无外部依赖跑通；先在 SQLite 里定义 ownership 语义，后面替换底层 store 比一开始引入队列系统更稳。

## 问题 16：持久化向量索引测试第一次失败

现象：新增 `test_persistent_vector_index_reuses_existing_collection` 后，第一次运行 `tests/test_hybrid_retrieval.py` 失败，断言期望返回 `vector` 文档，但实际第一名是 `keyword`。
原因：测试没有设置 `local_vector_weight=4.0`，在默认权重下 keyword 和 vector 的 RRF 融合不是这个测试想验证的口径。PersistentClient 本身已经能写入和读取 collection，失败点是测试排序权重不一致。
排查：对照前一个 hybrid fusion 测试，发现它显式把 vector weight 设置为 4.0 来证明语义向量召回能压过 keyword tie；新测试漏了同一设置。
修复：只修测试配置，补 `local_vector_weight=4.0`，没有改业务代码。重跑 `tests/test_hybrid_retrieval.py` 成功。
复盘：检索测试必须把融合权重写进配置，否则测试失败会混淆“索引是否复用”和“RRF 排序是否符合预期”两个问题。
面试可能追问：这说明 hybrid 不稳定吗？回答：是的，RRF 权重会影响结果，所以我把它配置化并用独立评测看指标，不会把某一次排序当成普遍规律。

# 7 实测数据

本节所有 mock benchmark 数字只用于证明 pipeline plumbing 能端到端跑通，不能当作真实性能、真实成本或真实答案质量成果。尤其不能在面试里说“我的 DeepResearch p50 是个位数毫秒”这类话，因为这个延迟测的是本机 Python 跑 deterministic mock 的速度，换机器、换进程热身状态、换依赖版本都会变。

实测环境：Windows PowerShell，`py -3.11`，mock search provider，seed `20260606`，5 条 benchmark case，max_researchers=3，max_results=4。

安装验证：`py -3.11 -m pip install --timeout 180 -e ".[dev]"` 成功。为了支持本地 hybrid retrieval，新增安装了 `sentence-transformers` 和 `chromadb`；第一次安装时有一个超时遗留 pip 进程占用 `torch` 文件，结束该遗留进程后重试成功。
测试验证：`py -3.11 -m pytest -q`，最新结果 `52 passed, 2 warnings in 59.78s`。warning 来自 FastAPI TestClient / Starlette 对 httpx 的 deprecation 提示，以及 OpenTelemetry metadata 的 deprecation 提示，未影响功能。
CLI example：`py -3.11 -m deepresearch_agent.cli "How does citation checking reduce hallucination in agentic RAG?"` 成功，raw_search_result_count `12`，deduped_source_count `8`，total_tokens `4417`。这次运行记录的 latency 是 `10.63ms`，但它只是 mock plumbing run 的本机样本，不作为性能指标引用。citation_retention_rate `1.0` 只说明 mock synthesis 生成的 citation ID 能被当前 checker 找到，不代表真实 LLM 场景下的引用可靠性。estimated_cost_usd `0.0` 是因为 mock provider 单价配置为 0，不代表真实成本。
真实 adapter probe：`py -3.11 -m deepresearch_agent.cli "What is Model Context Protocol?" --search-provider wikipedia --json` 成功，修复后 sample 输出显示 `fallback_count=0`，latency 约 `1506.501ms`。注意：Wikipedia 是真实无 key adapter，但不是高质量通用搜索，结果质量仍有限。

DeepSeek 结构化输出验证：`py -3.11 -m deepresearch_agent.validate_deepseek_structured_output --query "How should citation checking reduce hallucination in deep research agents?" --max-researchers 3` 成功。迁移后默认 `deepseek-v4-flash` 返回了 3 条合法 `SubQuestion`，Pydantic schema 解析通过。真实输出主题分别覆盖 citation checking 机制、citation accuracy 与 hallucination rate 的关系、以及 citation checking 的失败模式。

DeepSeek 端到端单条验证（LLM 真，search 仍是 mock）：`py -3.11 -m deepresearch_agent.cli "How should citation checking reduce hallucination in deep research agents?" --llm-provider deepseek --search-provider mock --max-researchers 2 --max-results 3 --json` 成功。模型生成的 brief 不再是模板回填，scope 是“Methods and effectiveness of citation verification in mitigating factual inaccuracies in AI-driven research agents”；planner 拆出 automated citation verification 和 human-in-the-loop 对比两个子问题；synthesis 生成了 markdown 报告和 6 条 claims。步骤 2 首次成功运行记录：latency `18987.535ms`，raw_search_result_count `6`，deduped_source_count `7`，citation_retention_rate `0.8333`，supported_claims `5/6`。其中 1 条 human-in-the-loop claim 被当前 lexical citation checker 标为 unsupported。这是接真实 usage 前的历史记录，所以当时的 cost 不能当真实成本；步骤 3 已补上 provider usage 解析。

DeepSeek usage/cost 单条验证（LLM 真，search 仍是 mock）：接入真实 usage 后重跑同一条命令成功。运行记录：latency `18524.693ms`，raw_search_result_count `6`，deduped_source_count `7`，citation_retention_rate `1.0`，supported_claims `6/6`。真实 usage：input_tokens `1842`，output_tokens `1310`，total_tokens `3152`，estimated_cost_usd `0.00193834`。分阶段成本：brief_generation `118 + 140 tokens / $0.00018586`，planning `277 + 226 tokens / $0.00032339`，synthesis `1447 + 944 tokens / $0.00142909`。注意：search 仍是 mock，所以这还不是“LLM + search 全真实”的 benchmark；步骤 4 再切 Wikipedia。

Embedding provider 验证：`py -3.11 -m deepresearch_agent.validate_embeddings --provider local --text "混合检索需要关键词召回和向量召回一起融合。"` 成功。本地模型 `BAAI/bge-small-zh-v1.5` 返回维度 `512`。`DASHSCOPE_API_KEY` 当前未配置，所以百炼 embedding 没有做真实 API 调用；代码层用本地 HTTP stub 测过 DashScope-compatible response parsing。

Reranker smoke：`py -3.11 -m deepresearch_agent.cli "How does citation checking reduce hallucination in agentic RAG?" --search-provider mock --llm-provider mock --max-researchers 1 --max-results 2 --local-retrieval-mode hybrid --rerank-enabled --rerank-provider local` 成功。因为首次下载/加载 `BAAI/bge-reranker-base`，latency 达到 `279692.721ms`。这只证明可选 rerank provider 能接入，不作为常规性能指标。

Web search / crawler provider smoke：`py -3.11 -m pytest tests/test_web_search_providers.py tests/test_failure_handling.py -q` 成功，`10 passed in 0.36s`，覆盖 SearxNG JSON parsing、crawler 内容替换、Jina Reader URL prefix、Jina Search JSON parsing、provider registry 和 unknown provider fail-fast。真实外网 smoke：`https://r.jina.ai/https://example.com` 返回 `200` 和 clean text，说明 Jina Reader crawler 这层能 live 访问；`--search-provider jina` 的 CLI run 成功但触发 fallback，trace error 是 `HTTP Error 401: Unauthorized`，所以 Jina Search 真实检索在当前环境未通过。SearxNG 需要 `SEARXNG_BASE_URL`，当前没有自建实例，未做 live search。

Claim-level evidence grounding smoke：`py -3.11 -m pytest tests/test_quality_and_citations.py -q` 成功，`6 passed in 0.26s`；新增测试覆盖 supported claim 提取 evidence quote，以及 missing source 标成 `unverifiable`。`py -3.11 -m deepresearch_agent.cli "How does citation checking reduce hallucination in agentic RAG?" --search-provider mock --llm-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --json` 成功，输出里的 citation assessment 已包含 `support_level="supported"` 和 `evidence_quotes`，quote 示例来自本地 source 句子 “Normal RAG retrieves context once for a single answer...”。这只证明 evidence quote plumbing 和 lexical grounding 生效，不代表语义事实校验完成。

Reflection loop smoke：`py -3.11 -m pytest tests/test_reflection_loop.py tests/test_run_control.py -q` 成功，`9 passed, 1 warning in 4.49s`。`tests/test_reflection_loop.py` 覆盖 orchestrator 在 `reflection_enabled=True`、`max_reflection_rounds=1`、`reflection_min_sources=4` 时追加 `R1` follow-up question，并写入 `compression.round1` / `reflection.round1` trace。`tests/test_run_control.py` 也覆盖了 `/runs` approve 后 result_json 的 plan 包含 `R1`，trace_events 包含 `reflection.round1`。CLI smoke：`py -3.11 -m deepresearch_agent.cli "How should citation grounding work in a research agent?" --search-provider mock --llm-provider mock --local-retrieval-mode keyword --max-researchers 1 --max-results 1 --reflection-enabled --max-reflection-rounds 1 --reflection-min-sources 4 --json` 成功，输出可见 `id="R1"`、`compression.round1`、`reflection.round1` 和 `should_add_question=true`。

MCP adapter smoke：`py -3.11 -m pytest tests/test_mcp_tools.py tests/test_web_search_providers.py tests/test_failure_handling.py -q` 成功，`14 passed in 0.37s`。覆盖 MCP result 的 `sources` array 转 `Source`、`content[type=text]` JSON 转 `Source`、`McpToolSearchAdapter` 调 fake client、`build_search_adapter(..., "mcp")` 构造 provider、以及缺少 `MCP_SEARCH_TOOL` 时 fail-fast。当前没有配置真实 MCP server，所以 stdio/http live call 未实测。

Run review UI smoke：`py -3.11 -m pytest tests/test_run_control.py tests/test_api.py -q` 成功，`11 passed, 2 warnings in 45.22s`，覆盖 `/ui` 返回页面、`GET /runs` 列出刚创建的 run、原 approval/edit/cancel/retry/SSE replay 仍可用。临时启动 `py -3.11 -m deepresearch_agent.api --host 127.0.0.1 --port 8010` 后，`/health` 返回 `ok`，`/ui` HTML 包含 `DeepResearch Run Review`、`planEditor`、`EventSource`，`POST /runs` 创建 run 后 `GET /runs` 能看到它。`POST /runs/{run_id}/approve` 第一次 30s probe 超时，但 20s 后查询 run 已 `succeeded`，metrics latency 约 `36307.56ms`；原因是临时服务没有设置 `LOCAL_RETRIEVAL_MODE=keyword`，默认 hybrid 冷启动加载本地模型。Node 环境没有 `playwright` 包，所以没有做截图验证。

Worker lease smoke：`py -3.11 -m pytest tests/test_run_control.py -q` 成功，`12 passed, 1 warning in 6.97s`。新增测试覆盖 `RunStore.acquire_lease()` 只能让一个 worker 获得 lease、同 worker heartbeat、release 后另一个 worker 可重新 acquire、同一 worker 即使跨过 TTL 也能在未被别人接管时续租；API 测试覆盖 `/runs/{run_id}/lease`、竞争 worker 返回 `409`、`/heartbeat` 更新 `heartbeat_at`、`/runs/stale` 能列出过期 running run、`/runs/recover-stale` 会把 stale run 标记为 `failed` 并清空 `leased_by`；migration 测试覆盖旧版 `agent_runs` 表缺少 lease 列时，`RunStore` 会自动补 `leased_by`、`heartbeat_at`、`lease_expires_at`。

Persistent vector index smoke：`py -3.11 -m pytest tests/test_hybrid_retrieval.py -q` 成功，`3 passed, 1 warning in 2.77s`。新增测试用静态 embedding provider 和临时 Chroma PersistentClient 验证：第一次检索会 embed 2 个 corpus chunk 和 1 个 query，第二个 `LocalRagRetriever` 指向同一 `LOCAL_VECTOR_INDEX_PATH` 时只 embed query，不重新 embed corpus；`ChromaVectorIndex.reused_existing` 为 `True`。这只验证索引复用语义，没有做真实 BGE 冷/热启动延迟 benchmark。

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

公开 Deep Research 端到端 artifact 评测（LiveDRBench preview）：这组是公开任务驱动完整 orchestrator，不是只测 retriever。脚本是 `src/deepresearch_agent/deep_research_eval.py`，默认从 Hugging Face datasets-server 拉 `microsoft/LiveDRBench` 的 `preview/test` 行，输出 summary、raw JSONL 和 LiveDRBench-style predictions。当前还没有接官方 judge，所以 `official_judge_score=not_run`；下面的 success 仍是本项目现有 citation retention 阈值口径，不是官方排行榜分数。

| 口径 | summary | raw log | case_count | success_rate | citation_retention_rate_avg | deduped_source_count_avg | latency p50 | total_tokens | estimated_cost_usd_total | fallback_count_total |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LiveDRBench preview + mock/mock plumbing | `results/deep_research_eval_livedrbench_mock_summary.json` | `logs/deep-research-eval-20260607T133222828802Z.jsonl` | 1 | 1.0 | 1.0 | 3.0 | 5.785ms | 2624 | 0.0 | 0 |
| LiveDRBench preview + DeepSeek v4-flash + Wikipedia | `results/deep_research_eval_livedrbench_deepseek_wikipedia_summary.json` | `logs/deep-research-eval-20260607T133237936766Z.jsonl` | 1 | 0.0 | 0.5 | 3.0 | 20910.052ms | 3434 | 0.00072332 | 0 |

这条公开真实 case 的 query 是让系统根据 `American Community Survey / FEMA Harvey flood depths / USDA Food Access Research Atlas / Streetlight / SafeGraph POI` 找使用全部数据集的论文，并按 JSON 返回 `paper_title`。真实组没有报错，也没有 fallback，但 success 为 0，说明当前 DeepSeek + Wikipedia + 本地 keyword RAG 没有解决这类公开精确查证任务；citation retention 只有 `0.5`，也说明 lexical citation check 已经暴露支撑不足。面试里我会把它讲成“公开评测入口已经打通，但质量短板被暴露出来”，不会把 mock 组的 `1.0` 当质量成果。

未实测：LiveDRBench/Deep Research Bench 官方 judge 分数、真实搜索 API 高并发限流、语义级 citation faithfulness、Redis/PostgreSQL 缓存、OpenTelemetry/LangSmith tracing、真实用户流量、DashScope 真实 embedding/rerank、rerank 5 case 全量 benchmark。

# 8 评测设计

answer completeness：当前未做 LLM judge 或官方 Deep Research judge，只用 case success 间接衡量。LiveDRBench preview 已经能驱动完整 orchestrator 产 artifact，但 answer-quality 官方分数仍未实测。
citation faithfulness：当前实测指标仍以 claim/source lexical overlap 为基础，但已经从单一分数升级到 `support_level` 和 `evidence_quotes`。mock plumbing run 平均 retention 是 `1.0`，只能说明 mock 引用链路没断；最新 DeepSeek v4-flash + Wikipedia 对比里，keyword baseline 平均 retention 是 `0.8867`，local hybrid 是 `0.8929`，但 hybrid success_rate 更低，说明不能只看均值。下一步才是 LLM judge / NLI entailment。
retrieval quality：端到端 benchmark 里的 citation_retention 会受 LLM 和 search 波动影响，所以我新增了 BEIR/scifact 独立检索评测，直接用 qrels 计算 Recall@10、nDCG@10、MRR。当前真实结果是 keyword `0.6000/0.4823/0.4548`，hybrid `0.8239/0.6597/0.6114`，hybrid+rerank `0.8239/0.7307/0.7083`。
source diversity：当前记录 deduped_source_count，也记录 local retrieval metadata 里的 keyword/vector/rerank rank；但还没有按 domain/provider 多样性和人工相关性打分。
hallucination rate：当前用 unsupported citation count 作为 proxy，不能覆盖无引用幻觉。
latency：benchmark 记录每 case latency_ms，并计算 P50/P90/max；mock latency 只能作为 plumbing 回归信号，DeepSeek + Wikipedia latency 包含真实网络/API 时间，也不能当线上 SLA。local hybrid 比 keyword baseline p50 多 `6151.778ms`；独立检索评测里 keyword 平均每 query `0.3070s`，hybrid `0.5781s`，hybrid+rerank `3.4431s`，这些都需要如实讲。
cost：mock provider 成本为 0，token 用字符估算；DeepSeek provider 已接真实 usage，并按当前实现里的 v4-flash 价格常量估算成本，价格核对日期 `2026-06-07`。BEIR/scifact 独立检索评测不调用 LLM，LLM token 和 API cost 都是 0；本地 embedding/rerank 不产生 API 成本，但会产生本机 CPU/GPU 时间；DashScope 成本未实测。
工具失败恢复：有 unit test 覆盖 primary failure fallback 和 circuit breaker open；第一次 DeepSeek + Wikipedia benchmark 出现过 fallback，修复 Wikipedia 长查询压缩后 fallback 曾降到 0，但最新 keyword/hybrid 对比里仍分别出现 `fallback_count_total=1` 和 `2`，已在第 7 节如实记录。
multi-hop / reflection 成功率：当前没有真实 multi-hop 标注集，未实测质量提升；但 reflection loop 的控制流已用 mock/keyword smoke 验证，会在证据不足启发式触发时追加 `R<N>` follow-up question，并把 compression/reflection payload 写入 trace。下一步需要公开多跳任务或人工标注集来评估是否真的提升答案质量。

评测集构造方式：端到端 5 条 case 是围绕本项目核心能力手写的 smoke benchmark，覆盖 supervisor-researcher、citation faithfulness、tool failure、cost tracking、benchmark reproducibility。公开检索标准口径采用 BEIR/scifact test qrels，只评测 local retriever，不评测 LLM 回答质量。公开 Deep Research 端到端口径新增 LiveDRBench preview runner，会跑完整 orchestrator 并保存 answer/source/trace/cost/predictions artifacts；当前只跑了 1 条 mock 和 1 条 DeepSeek/Wikipedia 样本，官方 judge 分数尚未接入。

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
我做了哪些改造：砍掉内容生产和平台能力，只保留 deep research 主干和后端可观测部分；这次补了 SearxNG/Jina 这种搜索与正文抽取边界，但仍保持统一 `Source` 输出，不把 DeerFlow 的完整前端和工具生态搬进来。
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

Hybrid retrieval 还没有证明质量稳定提升：当前已经实现 keyword + vector + RRF、可选持久化 Chroma index 和可选 rerank，但 5 case 小样本里 hybrid success_rate 反而低于 keyword baseline。可行方案是扩大本地语料、补人工相关性标注、调 RRF 权重、引入 Qdrant/Milvus 这类外部向量库，并把 rerank 纳入全量 benchmark。工程代价是索引生命周期、模型加载时间、评测集标注和更多运行成本。面试怎么讲：我会说我完成了检索结构升级，但不会把一次小样本结果包装成质量提升。

Run control 还不是分布式调度：当前已经有 SQLite run store、planner checkpoint、approval/resume/cancel/retry、SSE replay 和单机 worker lease/heartbeat，但它仍是轻量实现。可行方案是引入真正的 worker queue、PostgreSQL/Redis、阶段幂等和更细粒度 checkpoint。工程代价是并发一致性、任务抢占、schema migration 和运维复杂度。面试怎么讲：我会说我先把长任务控制平面闭环和 worker ownership 语义做出来，生产化再升级存储和调度，不把 SQLite 版本包装成高并发任务系统。

没有 OpenTelemetry/LangSmith：当前问题是 trace 只写本地 JSONL。可行方案是接 OTel exporter 或 LangSmith。工程代价是外部账号、采样、隐私和成本。面试怎么讲：我会说本地 JSONL 先保证无外部依赖，后续可以从同一 trace event 结构导出。
