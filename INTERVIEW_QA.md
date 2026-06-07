# 架构与编排

## Q：这个项目一句话怎么介绍？
[状态: 待消化]
标签：Agent / 架构 / DeepResearch
检索关键词：DeepResearch, supervisor, researcher, citation
回答：我做的是一个收窄版 DeepResearch Agent，不追求大而全，而是把用户问题到结构化带引用报告这条链路做扎实。核心流程在 `src/deepresearch_agent/orchestrator.py`：normalize、生成 brief、planner 拆子问题、并发 researcher 检索、source dedup、synthesis、citation check，最后返回报告、trace 和成本统计。
关联模块：`orchestrator.py`, `schemas.py`, `api.py`
可追问：
1. 为什么要收窄范围？
2. 和 open_deep_research 的关系是什么？
3. 当前版本哪些能力还没有做？

## Q：为什么不是单 Agent？
[状态: 待消化]
标签：Agent / 并发 / 架构
检索关键词：multi-agent, supervisor-researcher, concurrency
回答：我没有为了形式上多 Agent，而是因为 research 问题天然可以拆成背景、证据、风险这些子问题。单 Agent 顺序做也能跑，但很难把每个子问题的来源、耗时、失败单独观测出来。当前用 `asyncio.gather` 和 `Semaphore` 跑多个 researcher，每个 researcher 都有自己的 trace event。
关联模块：`orchestrator.py`
可追问：
1. 并发数怎么控制？
2. 多 Agent 会不会增加成本？
3. 什么场景单 Agent 更合适？

## Q：为什么没有直接用 LangGraph？
[状态: 待消化]
标签：LangGraph / 架构取舍
检索关键词：LangGraph, lightweight orchestrator, checkpoint
回答：我参考了 LangGraph 系 deep research 的思路，但 MVP 里选择了自定义轻量 orchestrator。原因是我想先把 citation、fallback、trace、benchmark 这些工程点做清楚，避免项目被框架配置淹没。代价是没有 LangGraph Studio 和 checkpoint，后续可以把当前阶段函数映射成 graph node。
关联模块：`orchestrator.py`, `KNOWLEDGE_BASE.md`
可追问：
1. 什么时候会迁移到 LangGraph？
2. 现在的 state 怎么表达？
3. durable execution 怎么补？

## Q：这个系统的输入输出是什么？
[状态: 待消化]
标签：后端 / API / 数据结构
检索关键词：ResearchRequest, StructuredReport
回答：输入是 `ResearchRequest`，主要包含 query、max_researchers、max_results_per_researcher、llm_provider、search_provider 和 seed。输出是 `StructuredReport`，里面有 brief、plan、answer、claims、findings、sources、citation_check、cost、metrics 和 trace_events。这样做是为了让报告可展示，过程也可调试。
关联模块：`schemas.py`, `api.py`
可追问：
1. 为什么输出 trace_events？
2. metrics 里有哪些关键字段？
3. API 和 CLI 共用同一个编排器吗？

## Q：如何保证陌生人 clone 后能跑？
[状态: 待消化]
标签：工程化 / 可复现
检索关键词：Quickstart, mock provider, benchmark
回答：默认 LLM 和 search 都是 mock，不需要 API key；README 用 `py -3.11 -m deepresearch_agent.cli` 这种不依赖 PATH 的命令。测试用 `py -3.11 -m pytest -q`，benchmark 用固定 seed 和 `data/benchmark_cases.jsonl`，输出到 `logs/` 和 `results/`。如果要跑真实 LLM，就显式传 `--llm-provider deepseek`，并只从环境变量 `DEEPSEEK_API_KEY` 读取 key。
关联模块：`README.md`, `benchmark.py`, `data/benchmark_cases.jsonl`
可追问：
1. 为什么不用默认 Python 3.14？
2. mock 会不会让项目太假？
3. 怎么切到真实检索？

## Q：为什么第一版真实 LLM 选 DeepSeek？
[状态: 待消化]
标签：LLM Provider / 选型
检索关键词：DeepSeek, JSON mode, provider decision
回答：我需要一个真实 provider 来验证 structured output、usage 解析和成本归因，但又不能让默认运行依赖 API key。DeepSeek 的 API 兼容 OpenAI Chat Completions，支持 JSON Output，足够验证 brief、planner、synthesis 这些结构化输出；默认仍保留 mock，只有显式传 `--llm-provider deepseek` 才走真实模型。现在默认真实模型已切到显式 `deepseek-v4-flash`，legacy alias 只为兼容旧配置保留。
关联模块：`llm.py`, `config.py`, `KNOWLEDGE_BASE.md`
可追问：
1. 为什么不用 OpenAI？
2. tool calling 用了吗？
3. 模型价格变化怎么办？

# Planner 模块

## Q：Planner 在项目里做什么？
[状态: 待消化]
标签：Planner / Agent
检索关键词：planner, subquestion, research brief
回答：Planner 接收 `ResearchBrief`，输出一组 `SubQuestion`。默认 `MockLLMProvider.plan` 会稳定生成背景、实现证据、取舍风险三类问题；显式启用 DeepSeek 时，`DeepSeekLLMProvider.plan` 会用 JSON mode 输出同一套 Pydantic schema。这样后续 researcher 可以并发执行，并且每个子问题都有 rationale。
关联模块：`llm.py`, `schemas.py`
可追问：
1. 为什么 planner 输出要结构化？
2. planner 失败怎么办？
3. 真 LLM planner 怎么替换？

## Q：为什么先做 research brief？
[状态: 待消化]
标签：Brief / Scope
检索关键词：clarify, normalize, brief
回答：brief 是为了把原始 query 变成可执行约束。当前 `create_brief` 会 normalize query，写清 scope、constraints 和 assumptions。MVP 不做多轮澄清，因为用户要求全自主推进；模糊问题先归一化，不阻塞主线。
关联模块：`llm.py`
可追问：
1. 哪些情况应该追问用户？
2. brief 里为什么要保留 assumptions？
3. 如何评估 brief 好坏？

## Q：Planner 为什么默认拆 3 个问题？
[状态: 待消化]
标签：Planner / 并发控制
检索关键词：max_researchers, subquestions
回答：3 个问题是我在 MVP 里的默认取舍：足够覆盖背景、证据、风险，也不会让 trace 和 benchmark 太噪。`ResearchRequest.max_researchers` 和 settings 都会限制上限，最终在 orchestrator 里取较小值。
关联模块：`orchestrator.py`, `schemas.py`
可追问：
1. 如果问题很复杂怎么办？
2. 为什么不是 5 个 researcher？
3. 并发数和搜索成本有什么关系？

## Q：Planner 当前有什么局限？
[状态: 待消化]
标签：Planner / 局限
检索关键词：mock planner, limitation
回答：默认 Planner 是 deterministic mock，主要用于离线测试和 mock benchmark；DeepSeek Planner 已经实测能输出合法 JSON 并通过 Pydantic schema。当前局限不是“没有真实 LLM”，而是 planner 还不会根据搜索中间结果动态改 plan，也没有单独的 planner 质量评测集。
关联模块：`llm.py`
可追问：
1. 如何防止 LLM planner 输出非法 JSON？
2. planner 结果怎么评测？
3. 是否需要 human-in-the-loop？

## Q：Planner 和普通 RAG 的 query rewrite 有什么区别？
[状态: 待消化]
标签：RAG / Planner
检索关键词：query rewrite, agentic RAG
回答：普通 RAG 的 query rewrite 往往只是把一个问题改写成更适合检索的文本；这里 Planner 是把 research task 拆成多个可以独立执行的子问题。后续每个子问题都有独立来源、trace 和 citation，因此它更像任务分解，而不是单次检索优化。
关联模块：`llm.py`, `orchestrator.py`
可追问：
1. 子问题之间怎么去重？
2. 是否支持动态追加子问题？
3. 和 multi-hop QA 有什么关系？

# Researcher 与检索模块

## Q：Researcher 具体做了什么？
[状态: 待消化]
标签：Researcher / Tool
检索关键词：researcher, search, local RAG
回答：每个 researcher 接收一个 `SubQuestion`，调用 search adapter 和 local RAG，合并 source 后做去重、质量过滤和 summary。对应代码是 `orchestrator.py` 的 `_research_one`，它会给每个 researcher 写 trace，包括 provider、fallback_used、error 和 source_count。
关联模块：`orchestrator.py`, `search.py`, `rag.py`
可追问：
1. researcher 之间共享上下文吗？
2. 为什么要 context isolation？
3. 检索结果怎么合并？

## Q：真实 search adapter 是什么？
[状态: 待消化]
标签：Search / Adapter
检索关键词：WikipediaSearchAdapter, real search
回答：当前真实 adapter 是 `WikipediaSearchAdapter`，调用 Wikipedia Search API，不需要 API key。我实测过 DeepSeek v4-flash + Wikipedia benchmark，最新结果 `fallback_count_total=2`，没有把它包装成全真实无降级。中间踩过一个坑：LLM planner 生成的长自然语言子问题会让 Wikipedia 返回空结果或超时，所以我加了 query candidate 压缩；它能降低 fallback 风险，但不能保证 Wikipedia 永远不失败。
关联模块：`search.py`
可追问：
1. 为什么不用 Tavily？
2. Wikipedia 相关性有什么问题？
3. 怎么接第二个 provider？

## Q：mock search 的意义是什么？
[状态: 待消化]
标签：Mock / 可复现
检索关键词：MockSearchAdapter, offline
回答：mock search 不是为了假装真实搜索，而是为了无 key、离线、可复现地跑通主链路和 benchmark。它返回固定语料，方便测试 citation、fallback、dedup 和 cost 这些后端能力。
关联模块：`search.py`, `data/local_corpus.jsonl`
可追问：
1. mock benchmark 有什么可信度？
2. 怎么避免把 mock 指标包装成线上效果？
3. 是否应该把 mock 和真实指标分开？

## Q：工具失败怎么处理？
[状态: 待消化]
标签：Tool / 可靠性 / 降级
检索关键词：retry, timeout, circuit breaker, fallback
回答：`SearchService` 包了 timeout、bounded retry、circuit breaker 和 fallback。primary adapter 连续失败后 breaker 会打开，后续请求直接走 mock fallback。测试里用 `FailingSearchAdapter` 验证了 forced failure 和 circuit breaker open 两种情况。
关联模块：`search.py`, `tests/test_failure_handling.py`
可追问：
1. retry 次数怎么设？
2. fallback 会不会污染答案？
3. 如何向用户暴露 fallback？

## Q：RAG 在项目里怎么实现？
[状态: 待消化]
标签：RAG / Retriever
检索关键词：LocalRagRetriever, local_corpus, hybrid retrieval
回答：当前 RAG 是 `LocalRagRetriever`，读取 `data/local_corpus.jsonl` 后先分块。它支持两种模式：`keyword` 是旧基线，用 token overlap 排序；`hybrid` 是默认模式，用关键词召回和 BGE 向量召回两路并存，再用 RRF 融合。向量侧用 `sentence-transformers` 的 `BAAI/bge-small-zh-v1.5` 生成 embedding，用 Chroma 建本地 index。无论哪种模式，最后都返回统一的 `Source`，所以下游 dedup、verifier、synthesizer 不需要改。
关联模块：`rag.py`, `embeddings.py`, `data/local_corpus.jsonl`
可追问：
1. 为什么不用向量替换关键词？
2. embedding 成本和延迟怎么控制？
3. 本地 RAG 和 web search 冲突怎么办？

## Q：为什么要做混合检索，实测效果怎么样？
[状态: 待消化]
标签：RAG / Hybrid Retrieval / 评测
检索关键词：RRF, keyword, vector, benchmark
回答：我做混合检索不是为了说“向量一定更好”，而是因为 keyword 和 vector 解决的是不同问题。keyword 对精确术语和本项目里的固定工程词很稳，vector 对语义相近问题更友好；RRF 的好处是只融合排序名次，不要求 keyword 分数和 cosine 分数同尺度。最新 5 case benchmark 结果并不全好看：keyword baseline 的 success_rate 是 1.0，citation retention 平均 0.8867；local hybrid 的 success_rate 降到 0.6，citation retention 平均 0.8929，p50 latency 从 24595.506ms 升到 30747.284ms。所以我会把它讲成检索结构升级和可扩展性增强，不会包装成质量已经稳定提升。
关联模块：`rag.py`, `benchmark.py`, `results/retrieval_benchmark_comparison.json`
可追问：
1. 为什么 hybrid 反而变差？
2. RRF 的参数怎么选？
3. 下一步怎么证明相关性真的提升？

## Q：为什么还要做 BEIR/scifact 独立检索评测？
[状态: 待消化]
标签：RAG / 评测 / BEIR
检索关键词：BEIR, scifact, Recall@10, nDCG@10, MRR
回答：5 case 端到端 benchmark 会混在一起考 planner、Wikipedia、synthesis、citation checker 和网络波动，所以它不能单独证明 retriever 变好了。BEIR/scifact 是公开检索 benchmark，我只加载 corpus/query/qrels，复用 `LocalRagRetriever` 跑 keyword、hybrid、hybrid+rerank，不调用 LLM 和 Wikipedia。真实结果是 keyword Recall@10 `0.6000`、nDCG@10 `0.4823`、MRR `0.4548`；hybrid 到 `0.8239`、`0.6597`、`0.6114`；hybrid+rerank 到 `0.8239`、`0.7307`、`0.7083`。这里 `rerank_candidate_k=10`、`top_k=10`，所以 rerank 只是重排同一批 top10，Recall@10 不应该也没有变化；它主要改善 nDCG/MRR，但平均延迟从 `0.5781s` 到 `3.4431s`，不能默认开启。
关联模块：`retrieval_eval.py`, `rag.py`, `rerankers.py`, `results/retrieval_eval_scifact.json`
可追问：
1. SciFact 和中文知识库场景有什么差异？
2. 为什么不用完整 BEIR 框架？
3. rerank 为什么 Recall 提升不大但 nDCG 提升明显？

## Q：reranker 在项目里怎么接？
[状态: 待消化]
标签：RAG / Rerank
检索关键词：rerank, CrossEncoder, DashScope
回答：reranker 是可选后处理，不改变主链路。`LocalRagRetriever` 先拿 keyword/vector 融合候选，再在 `rerank_enabled=True` 时调用 `RerankProvider` 重排，最后仍返回 `Source`。默认 provider 是本地 `BAAI/bge-reranker-base`，不需要 API key；也实现了 DashScope rerank provider，只从 `DASHSCOPE_API_KEY` 读 key。默认关闭是因为本地 reranker 首次加载很慢，我实测单条 mock smoke 因下载/加载模型到了约 279 秒，不能默认压到所有运行路径上。
关联模块：`rag.py`, `rerankers.py`
可追问：
1. rerank 为什么默认关闭？
2. rerank 和 vector search 的区别是什么？
3. DashScope key 怎么管理？

# Verifier 与来源质量

## Q：Source Verifier 检查什么？
[状态: 待消化]
标签：Verifier / Source Quality
检索关键词：source verifier, quality score
回答：`SourceVerifier` 会按标题、正文长度、URL 稳定性、adapter 类型和低质量模式打 quality_score。它不是权威性判断模型，而是一道可解释的工程过滤层，避免空内容、明显低质量内容进入合成阶段。
关联模块：`verifier.py`
可追问：
1. quality_score 怎么定阈值？
2. 会不会误杀？
3. 怎么评估 source quality？

## Q：为什么要在 synthesis 前过滤来源？
[状态: 待消化]
标签：幻觉控制 / Verifier
检索关键词：source filtering, hallucination
回答：如果低质量 source 直接进入 synthesis，后面的 citation check 只能检查有没有支撑，很难弥补来源本身不可靠的问题。所以我在 researcher 阶段先做 verifier，再全局 dedup，把垃圾来源尽早挡住。
关联模块：`orchestrator.py`, `verifier.py`
可追问：
1. verifier 和 citation checker 有什么区别？
2. 来源质量和引用忠实度哪个更重要？
3. 是否需要 domain allowlist？

## Q：Dedup 怎么做？
[状态: 待消化]
标签：Dedup / Retrieval
检索关键词：SourceDeduplicator, normalized URL
回答：`SourceDeduplicator` 用规范化 URL 作为 key，去掉 query string 和尾部 slash，然后保留 quality_score 和 score 更高的来源。这样 researcher 局部和全局都可以复用同一套去重逻辑。
关联模块：`dedup.py`
可追问：
1. 同内容不同 URL 怎么办？
2. URL canonicalization 有哪些坑？
3. 什么时候需要内容指纹？

## Q：Verifier 当前最大局限是什么？
[状态: 待消化]
标签：Verifier / 局限
检索关键词：rule based verifier
回答：当前 verifier 是规则系统，能过滤空内容和明显低质量模式，但不能判断来源是否权威，也不能识别 subtle misinformation。后续可以加 domain reputation、发布时间、作者、交叉验证和 reranker。
关联模块：`verifier.py`
可追问：
1. 怎么处理论坛/博客？
2. 新闻时效性怎么判断？
3. 交叉验证怎么做？

## Q：Wikipedia adapter 的相关性 bug 说明了什么？
[状态: 待消化]
标签：Debug / Search Quality
检索关键词：Wikipedia score, page size, relevance
回答：我实测 Wikipedia provider 时发现一开始把 page size 当 score，会把 GPS/Data center 这种大页面排到前面。后来改成 query/source token overlap 加原始排序位置。这说明真实 adapter 不只是能连通，还要正确解释 provider 返回字段。
关联模块：`search.py`, `KNOWLEDGE_BASE.md`
可追问：
1. 这个修复彻底吗？
2. 为什么不用 BM25？
3. 怎么做 reranking？

# Synthesizer 模块

## Q：Synthesizer 负责什么？
[状态: 待消化]
标签：Synthesizer / Report
检索关键词：synthesis, cited report
回答：Synthesizer 接收 findings 和 sources，生成 markdown report、claims 和 sources 列表。默认 mock synthesis 用模板保证测试稳定；DeepSeek synthesis 已经接入 JSON mode，要求模型输出 answer 和结构化 claims，并且每条 factual claim 使用输入 sources 中已有的 `[Sx]`。
关联模块：`llm.py`
可追问：
1. 为什么 claim 要单独返回？
2. 真 LLM synthesis 怎么接？
3. 报告格式怎么扩展？

## Q：为什么 answer 和 claims 分开？
[状态: 待消化]
标签：Citation / 数据结构
检索关键词：claims, answer, citation check
回答：answer 是给用户看的完整报告，claims 是给 citation checker 的结构化单元。如果只解析 markdown，评测会很脆弱；把 claims 单独输出后，每条 claim 的支持情况都能进入 `CitationAssessment`。
关联模块：`schemas.py`, `citation.py`, `llm.py`
可追问：
1. claim 粒度怎么定？
2. 一条 claim 多个 citation 怎么办？
3. unsupported claim 怎么返给用户？

## Q：当前报告为什么比较模板化？
[状态: 待消化]
标签：Mock / 限制
检索关键词：mock synthesis, deterministic
回答：默认 mock 报告仍然比较模板化，这是为了保证离线测试和 mock plumbing benchmark 稳定。真实写作路径已经接入 DeepSeek，但我不会把它说成完全解决了幻觉，因为现在主要靠 prompt 约束和后置 lexical citation checker，还没有 LLM judge 或语义 entailment。
关联模块：`llm.py`
可追问：
1. 面试时会不会被认为太简单？
2. 怎么接真实 LLM？
3. 如何避免真实 LLM 改坏 citation？

## Q：Synthesizer 如何避免无来源结论？
[状态: 待消化]
标签：幻觉控制 / Citation
检索关键词：source_ids, cited claims
回答：当前 synthesizer 只基于 findings 生成 claim，并优先使用 finding 的第一个 source_id。后面 citation checker 会检查 citation ID 是否存在以及 claim 和 source 是否有支持关系。它不是绝对防幻觉，但把无引用结论变成可检测问题。
关联模块：`llm.py`, `citation.py`
可追问：
1. 如果 finding 没有 source 怎么办？
2. LLM 自己编 citation 怎么防？
3. 是否需要强制 source quote？

## Q：报告导出做了吗？
[状态: 待消化]
标签：Report / Optional
检索关键词：report exporter, optional
回答：当前只输出 markdown 字符串和 JSON 结构，没有做 PDF/Docx 导出。原因是交付标准优先要求主链路、SSE、trace、benchmark，而多格式导出被我放到 v2 optional。
关联模块：`schemas.py`, `cli.py`, `api.py`
可追问：
1. 怎么加 PDF？
2. 导出会影响 citation 吗？
3. 为什么不先做前端？

# Citation Checker 模块

## Q：Citation Checker 怎么工作？
[状态: 待消化]
标签：Citation / Faithfulness
检索关键词：CitationChecker, overlap
回答：`CitationChecker` 从 claim 里提取 `[S1]` 这种 citation ID，找到对应 source，然后计算 claim 和 source content 的 token overlap。overlap 达到阈值就认为第一层 supported，否则标 unsupported。
关联模块：`citation.py`
可追问：
1. 为什么不用 LLM judge？
2. overlap 阈值是多少？
3. 中文怎么处理？

## Q：citation_retention_rate 是什么？
[状态: 待消化]
标签：评测 / Citation
检索关键词：citation_retention_rate
回答：它是 supported_claims / total_claims。mock benchmark 的平均 retention 是 1.0，只说明 mock 引用链路没断；最新 DeepSeek v4-flash + Wikipedia 对比里，keyword baseline 平均 retention 是 0.8867，local hybrid 是 0.8929。但 hybrid 的 success_rate 反而从 1.0 降到 0.6，所以这个指标不能单独拿来证明检索变好。它仍是 lexical overlap，不等于完整语义事实校验。
关联模块：`citation.py`, `benchmark.py`, `results/retrieval_benchmark_comparison.json`
可追问：
1. 1.0 是否说明没有幻觉？
2. 真实 LLM 下会怎样？
3. unsupported claim 怎么处理？

## Q：citation checker 和 verifier 有什么区别？
[状态: 待消化]
标签：Verifier / Citation
检索关键词：source quality, faithfulness
回答：Verifier 看 source 本身能不能用，比如标题、正文长度、URL 和低质量模式；citation checker 看 claim 是否被它引用的 source 支撑。一个是来源质量，一个是论断忠实度，两个都需要。
关联模块：`verifier.py`, `citation.py`
可追问：
1. 哪个更靠前？
2. 低质量来源但 claim supported 怎么办？
3. 高质量来源但 claim unsupported 怎么办？

## Q：当前 citation 方法有什么风险？
[状态: 待消化]
标签：Citation / 局限
检索关键词：lexical overlap limitation
回答：最大风险是 lexical overlap 不等于语义支持。两个句子词重叠高也可能表达相反意思，词重叠低也可能语义等价。所以我把它定位成便宜、可 CI 化的第一层检查，不把它包装成最终事实验证。
关联模块：`citation.py`, `KNOWLEDGE_BASE.md`
可追问：
1. 怎么升级到语义级？
2. LLM judge 会不会也幻觉？
3. 如何构造人工标注集？

## Q：如何处理 missing citation？
[状态: 待消化]
标签：Citation / 错误处理
检索关键词：missing citation, unsupported
回答：如果 claim 没有 citation ID，或者 citation ID 找不到 source，`CitationChecker` 会给 overlap 0，并标成 unsupported。这样即使 synthesizer 出错，最终 report 里的 citation_check 也能暴露问题。
关联模块：`citation.py`
可追问：
1. API 是否应该直接失败？
2. 用户界面怎么显示？
3. 是否应该自动重写？

# Agent Run Control Plane

## Q：为什么 DeepResearch 是长任务？
[状态: 待消化]
标签：Run Control / Agent 后端
检索关键词：long running agent, run control
回答：DeepResearch 不是一次模型调用，它至少包含 planner、多个 researcher 并发检索、source dedup、synthesis、citation check。真实 provider 下每一步都可能耗时、失败、fallback 或方向跑偏，所以我给它加了 `run_id`、状态机、checkpoint 和 event stream，让它从脚本式调用变成可管理的后端长任务。
关联模块：`run_control.py`, `run_store.py`, `api.py`
可追问：
1. 哪些阶段最容易失败？
2. 为什么普通 request log 不够？
3. 长任务如何避免重复执行？

## Q：为什么需要 run_id？
[状态: 待消化]
标签：Run Control / 状态机
检索关键词：run_id, agent_runs
回答：`run_id` 是一次 Agent 任务的主键。没有它，客户端断线、服务重启、用户审核和失败重试都只能靠临时内存状态。现在 `agent_runs` 用 `run_id` 记录 status、current_stage、plan_json、result_json、token/cost 和 error，API 可以随时查 `/runs/{run_id}`、`/steps`、`/trace`。
关联模块：`run_models.py`, `run_store.py`
可追问：
1. run_id 怎么生成？
2. run 状态有哪些？
3. result_json 什么时候写入？

## Q：checkpoint 存什么？
[状态: 待消化]
标签：Checkpoint / 可恢复
检索关键词：SQLite checkpoint, agent_steps
回答：我把 checkpoint 分两层：`agent_runs.plan_json` 存 planner 稳定输出，包括 request、brief、subquestions 和 planner cost；`agent_steps` 记录每个阶段的 start/succeeded/failed、input_json、output_json、latency_ms、token_usage、cost、error、retry_count。这样失败后能看出是哪个阶段、什么输入、什么错误。
关联模块：`run_store.py`, `run_control.py`
可追问：
1. 为什么 JSON 字段不用 pickle？
2. 哪个 checkpoint 最关键？
3. step trace 和 event 有什么区别？

## Q：服务挂了怎么恢复？
[状态: 待消化]
标签：Durability / SQLite
检索关键词：restart recovery, SQLite
回答：当前不是分布式恢复，但服务重启后可以从 SQLite 读回 run、steps、events 和 planner checkpoint。比如 run 在 `waiting_approval`，重启后用户仍可以查 `/runs/{run_id}`，然后 approve/edit/reject；如果 run failed 且已有 plan_json，`/retry` 会复用 planner 输出从 researcher 阶段重跑。
关联模块：`run_store.py`, `run_control.py`, `api.py`
可追问：
1. running 中途宕机怎么办？
2. 为什么现在不做 worker lease？
3. SQLite 文件放在哪里？

## Q：cancel 和 retry 怎么设计？
[状态: 待消化]
标签：Run Control / Failure Recovery
检索关键词：cancel, retry, failed run
回答：`cancel` 会把非终态 run 标成 `cancelled`，后续 approve 会被拒绝；当前同步 mock 路径主要覆盖 waiting_approval 后取消。`retry` 只允许 failed run，优先复用 planner checkpoint，从 researcher 阶段重新跑，并在 step 里写 `retry_count=1`、event 里写 `retrying`。这不是精确恢复到某个 researcher 内部，而是可解释的稳定 checkpoint 重跑。
关联模块：`run_control.py`, `tests/test_run_control.py`
可追问：
1. 为什么 retry 不重跑 planner？
2. running 时取消有什么局限？
3. retry 怎么避免无限循环？

## Q：为什么 planner 后要 human-in-the-loop？
[状态: 待消化]
标签：HITL / Planner
检索关键词：human in the loop, approval gate
回答：planner 一旦方向错，后面的 researcher、search、synthesis 都会沿着错误方向花 token 和时间。我的 approval gate 放在 planner 后：`POST /runs` 先进入 `waiting_approval`，返回 plan/subquestions/estimated_researcher_count/risk_note，用户可以 approve、edit 或 reject。这样把人为判断放在成本放大之前。
关联模块：`run_control.py`, `run_models.py`
可追问：
1. 为什么不在 synthesis 后审核？
2. edit plan 保存在哪里？
3. 审核会不会拖慢系统？

## Q：HITL 怎么避免 token 浪费？
[状态: 待消化]
标签：成本控制 / HITL
检索关键词：approval, token cost
回答：HITL 不能让 planner 本身免费，但能阻止错误 plan 继续触发 researcher 检索和 synthesis。尤其真实 DeepSeek/Wikipedia 路径下，后续阶段才是主要延迟和成本来源。当前 `agent_runs` 会记录 total_tokens/total_cost，`agent_steps` 也记录阶段 token_usage/cost，所以能看到审核前后成本边界。
关联模块：`run_control.py`, `cost.py`, `run_store.py`
可追问：
1. planner 本身成本怎么控制？
2. 用户 edit 后是否复用 planner cost？
3. 如何给用户展示预计成本？

## Q：SSE 断线怎么续传？
[状态: 待消化]
标签：SSE / 可观测性
检索关键词：Last-Event-ID, SSE replay
回答：新 `/runs/{run_id}/events` 不是只读内存队列，而是从 `agent_events` 读取持久化 event。每个 event 有自增 `event_id`，SSE 输出里也写 `id:`。客户端断线后带 `Last-Event-ID`，服务端先补发这个 id 之后的历史事件；如果 run 还没到终态或 waiting_approval，再继续轮询新事件。
关联模块：`api.py`, `run_store.py`
可追问：
1. event_id 如何保证单调？
2. 和 `/research/stream` 有什么区别？
3. 多客户端订阅怎么办？

## Q：为什么不用 LangGraph 直接重写？
[状态: 待消化]
标签：架构取舍 / LangGraph
检索关键词：LangGraph durable execution, lightweight control plane
回答：我借鉴的是 LangGraph 的 durable execution、checkpoint、human-in-the-loop 思想，但没有把项目迁到 LangGraph。原因是现有 FastAPI + 自定义 orchestrator 已经能讲清主链路，这次目标是补 run 管理能力，而不是框架重构。轻量 SQLite control plane 更符合这个求职展示项目的范围。
关联模块：`run_control.py`, `orchestrator.py`, `KNOWLEDGE_BASE.md`
可追问：
1. 什么情况下会迁移 LangGraph？
2. 当前实现缺少 LangGraph 哪些能力？
3. checkpoint 怎么映射成 graph node？

## Q：SQLite 的局限是什么？
[状态: 待消化]
标签：存储 / 局限
检索关键词：SQLite, run store limitation
回答：SQLite 适合本地 demo、测试和单机轻量 checkpoint，但不适合高并发生产任务队列。它没有 worker lease、跨进程抢占、复杂锁调度和水平扩展。生产化我会把 `RunStore` 抽象后面换成 Postgres/Redis/队列，但当前不引入这些，是为了不把项目变成基础设施工程。
关联模块：`run_store.py`
可追问：
1. SQLite 会不会锁表？
2. schema migration 怎么做？
3. 为什么默认文件不提交？

## Q：怎么从 demo 变成生产系统？
[状态: 待消化]
标签：生产化 / Roadmap
检索关键词：worker queue, lease, production
回答：下一步不是再加 provider，而是把 run control 生产化：任务队列、worker lease/heartbeat、幂等阶段执行、Postgres 持久化、对象存储保存大结果、权限和审计、前端 approval 页面、provider 级限流。现在版本已经把状态机和 checkpoint 边界打出来，后面替换存储和调度不会影响 Agent 主链路。
关联模块：`run_control.py`, `run_store.py`, `api.py`
可追问：
1. 哪一步最该先做？
2. 如何处理重复 approve？
3. 大 result_json 怎么存？

## Q：这和普通日志有什么区别？
[状态: 待消化]
标签：可观测性 / Checkpoint
检索关键词：step trace, checkpoint, audit
回答：普通日志主要是事后排查文本；这里的 step trace 是结构化 checkpoint，能驱动 API 查询、retry、SSE replay 和审计。比如 planner step 的 output_json 可以直接作为 approval payload，failed step 的 input_json/error 可以定位重试边界，agent_events 可以按 Last-Event-ID 续传。这已经不只是日志，而是 run lifecycle 的数据模型。
关联模块：`run_store.py`, `api.py`
可追问：
1. step trace 会不会太大？
2. 哪些字段不能记录？
3. 如何做隐私脱敏？

# Cost Tracker 与可观测性

## Q：Cost Tracker 记录什么？
[状态: 待消化]
标签：成本控制 / Token
检索关键词：CostTracker, token accounting
回答：`CostTracker` 按阶段记录 input_tokens、output_tokens 和 estimated_cost_usd。当前阶段包括 brief_generation、planning、synthesis。mock provider 成本是 0，token 用字符数近似估算；DeepSeek provider 会读取 API 返回的 `prompt_tokens` / `completion_tokens`，并按当前实现里的价格常量估算成本。模型或价格页变化时，这个常量要同步更新。
关联模块：`cost.py`, `llm.py`
可追问：
1. 真实 provider 的 usage 怎么接？
2. token 估算准吗？
3. 为什么按阶段记录？

## Q：为什么要按阶段做成本归因？
[状态: 待消化]
标签：成本控制 / 可观测性
检索关键词：stage cost, attribution
回答：多 Agent 系统里只知道总 token 没有太大调优价值。按阶段归因后，才能知道成本主要花在 brief、planning 还是 synthesis。最新 retrieval 对比里，keyword baseline 总 token 是 17740，总成本按当前实现价格常量估算是 0.00368858 美元；local hybrid 总 token 是 18842，总成本是 0.00399994 美元。本地 embedding 不产生 API 费用，但会增加本机延迟；DeepSeek 价格取自官方 Models & Pricing 页，核对日期是 2026-06-07。
关联模块：`cost.py`, `benchmark.py`
可追问：
1. researcher 的搜索成本怎么计？
2. prompt caching 怎么利用？
3. 如何做成本预算？

## Q：Trace Logger 记录什么？
[状态: 待消化]
标签：可观测性 / Trace
检索关键词：TraceLogger, JSONL trace
回答：`TraceLogger` 每个 run 写 JSONL event，包含 run_id、stage、status、timestamp、duration_ms 和 payload。researcher 阶段会记录 provider、fallback_used、error、source_count。这样即使没有 LangSmith，也能本地排查每一步。
关联模块：`tracing.py`, `orchestrator.py`
可追问：
1. 为什么用 JSONL？
2. trace 文件是否提交？
3. 怎么接 OpenTelemetry？

## Q：SSE 流式输出怎么实现？
[状态: 待消化]
标签：SSE / FastAPI
检索关键词：StreamingResponse, SSE
回答：`api.py` 的 `/research/stream` 用 `StreamingResponse` 返回 `text/event-stream`。orchestrator 每完成一个 stage 都通过 emit callback 放进 `asyncio.Queue`，API generator 再把 stage 和 final report 转成 SSE event。
关联模块：`api.py`, `orchestrator.py`
可追问：
1. 为什么不用 WebSocket？
2. SSE 断线怎么办？
3. 如何做前端进度条？

## Q：Benchmark 记录哪些指标？
[状态: 待消化]
标签：Benchmark / 评测
检索关键词：benchmark, latency, token, citation
回答：`benchmark.py` 记录 seed、配置快照、case_id、query、latency_ms、total_tokens、estimated_cost_usd、deduped_source_count、raw_search_result_count、citation_retention_rate、success、fallback_count 和 output_summary。现在有四类口径：mock plumbing run 只证明离线路径；DeepSeek + Wikipedia + keyword baseline 是端到端检索基线；DeepSeek + Wikipedia + local hybrid 是混合检索小样本；BEIR/scifact 是独立检索模块评测，只看 Recall@10、nDCG@10、MRR，不调用 LLM。对比结果分别在 `results/retrieval_benchmark_comparison.json` 和 `results/retrieval_eval_scifact.json`，我不会只挑好看的数字说。
关联模块：`benchmark.py`, `retrieval_eval.py`, `results/benchmark_summary.json`, `results/retrieval_benchmark_comparison.json`, `results/retrieval_eval_scifact.json`
可追问：
1. 为什么只 5 条 case？
2. success 怎么定义？
3. 如何扩成公开 benchmark？

# 对比与岗位表达

## Q：和普通 RAG 最大区别是什么？
[状态: 待消化]
标签：RAG / Agent
检索关键词：normal RAG, agentic RAG
回答：普通 RAG 通常是一次 retrieve 后生成答案，而这个项目把 research 拆成多个子问题，并发检索，每个来源经过 verifier 和 dedup，最后还要做 citation check。它不是只回答，而是把研究过程变成可追踪、可失败恢复、可评测的 pipeline。
关联模块：`orchestrator.py`, `rag.py`, `citation.py`
可追问：
1. 普通 RAG 什么时候更好？
2. agentic RAG 成本更高怎么办？
3. 多步检索如何避免漂移？

## Q：和 ChatBI / Data Agent 有什么区别？
[状态: 待消化]
标签：Agent / 场景对比
检索关键词：ChatBI, Data Agent, DeepResearch
回答：ChatBI / Data Agent 更强调把自然语言转成数据查询、指标解释和业务分析，核心风险是 schema grounding 和 SQL/工具执行正确性。这个 DeepResearch Agent 更强调开放式资料检索、来源质量、引用忠实度和多阶段报告生成。两者都需要工具调用和可观测性，但 grounding 对象不同。
关联模块：`orchestrator.py`, `search.py`, `citation.py`
可追问：
1. 两者能共用什么架构？
2. Data Agent 如何做 citation？
3. 哪个更难评测？

## Q：你在参考项目上做了哪些自己的东西？
[状态: 待消化]
标签：项目差异 / 简历
检索关键词：open_deep_research, verifier, citation
回答：我参考 open_deep_research 的 supervisor-researcher 和并发思路，但没有 fork 或复制它的代码。我自己加厚的是 verifier、citation faithfulness、工具失败 fallback、结构化 trace、阶段级成本归因和本地 benchmark harness。这些都是 Agent 后端岗更容易追问的工程点。
关联模块：`KNOWLEDGE_BASE.md`, `verifier.py`, `citation.py`, `search.py`
可追问：
1. 为什么说这些算自己的？
2. 参考项目哪些能力你没做？
3. 如何证明没造假？

## Q：这个项目最能体现后端能力的地方是什么？
[状态: 待消化]
标签：Agent 后端 / 工程能力
检索关键词：fallback, observability, benchmark
回答：我觉得最能体现后端能力的是失败边界和观测闭环。不是只把 LLM 接起来，而是工具超时能 fallback，每个阶段有 trace，token/cost 能归因，benchmark 能复现，citation 能检查。这些东西让 Agent 系统从 demo 变成可调试工程。
关联模块：`search.py`, `tracing.py`, `cost.py`, `benchmark.py`
可追问：
1. 哪个模块最有扩展价值？
2. 生产环境还缺什么？
3. 怎么做 SLA？

## Q：当前项目最大的短板是什么？
[状态: 待消化]
标签：局限 / 诚实表达
检索关键词：limitations, real LLM, semantic evaluation
回答：最大短板已经不是“完全没有真实 LLM”，而是只接了 DeepSeek v4-flash 一个 provider，真实 benchmark 也只有 5 条小样本；Wikipedia 不是生产级搜索，citation checker 只是 lexical overlap，不能代表完整事实校验。我会在面试里主动说清楚：真实 provider 路径和 cost usage 已经跑通，但质量评测和生产化还没有完成。
关联模块：`llm.py`, `citation.py`, `KNOWLEDGE_BASE.md`
可追问：
1. 下一步先补哪个？
2. 怎么接 OpenAI？
3. 怎么评估真实输出质量？
