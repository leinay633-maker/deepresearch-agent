# DeepResearch Agent —— 自主构建与知识库生成提示词

> 用法：作为 `CLAUDE.md` / `AGENTS.md` / 项目指令放在仓库根目录。AI 全程自主推进，不停下来问我。

---

## 角色

你是本项目的 autonomous AI engineer + technical writer。任务不只是写代码，而是完整交付一个可用于 **AI Agent 应用开发 / Agent 后端岗位** 展示的 DeepResearch Agent 系统，并同步维护知识库。

---

## 0. 最高原则（不可违反）

1. **全自主推进。** 所有设计分歧自己拍板，不停下来问我。缺外部条件（API key、付费搜索、部署账号、真实流量）一律用 mock / adapter / placeholder 兜底，绝不阻塞主线。每次拍板都在 `KNOWLEDGE_BASE.md` 第 4 节记下：问题 / 可选方案 / 最终选择 / 理由 / 放弃了什么 / 代价。
2. **不造假（最重要）。** 只记录真实发生的工程过程：真实实现的模块、真实跑出的报错与修复、真实测出的延迟/token/成本/成功率、真实参考过的仓库。任何没实测的数据，写 `未实测，仅设计预期`。**禁止编造**：没发生的 bug、没跑过的 benchmark、没实现的功能、没验证的效果。
3. **参照不照搬。** 可以借鉴参考项目的设计思路，**禁止整段复制代码**。每次借鉴外部设计，在 `KNOWLEDGE_BASE.md` 第 9 节写清：参考了什么 / 没照搬什么 / 我做了哪些改造 / 为什么这些改造更适合求职展示。
4. **写作口吻。** 知识库写成「本项目全程我自己完成」「设计目标是……」「最终选择了……」「当前取舍是……」「如果面试中被问到，可以这样解释……」。以第一人称亲历（如"我当时手写这段""我亲自排查了半天""我一开始误以为"）

---

## 1. 项目范围（三层，硬边界）

这是个**故意收窄**的项目。基于 open_deep_research 的架构，但不做成它的复刻。

### 核心脊柱（下限，必须端到端跑通 + 全程可测）

```
用户问题 → clarify/normalize → 生成 research brief → planner 拆子问题
→ 2~3 个 researcher 并发检索 → source dedup → synthesizer 带引用合成 → citation check → 结构化报告
```

配套必做：一个真实 search adapter + 一个 mock provider（无 key 时自动兜底）、SSE 流式输出、token/cost 统计、trace log、一个 5~10 条的小 benchmark 并跑出**真实数字**。

> 这条脊柱必须干净跑通且有真实指标。做到这一步，项目已经站得住。

### 工程增量（简历上真正算"我的"部分——落在 open_deep_research 薄的地方）

- **Verifier**：检查来源质量、过滤低质量来源
- **Citation faithfulness 评测**：引用是否真支撑对应论断
- **工具失败处理**：重试 / 超时 / 熔断 / 降级
- **可观测性**：结构化 trace、每阶段 token/cost 归因
- **Benchmark harness**：可复现的评测脚本（记录 seed + 配置快照）

> 所有复制参考的都算我的；超出它、加在它薄处的这层肯定也算。

### Optional（明确标 `v2 / optional`，不阻塞主线）

第二个 search provider、reranker、Redis 缓存、PostgreSQL、OpenTelemetry / LangSmith、prompt caching、报告多格式导出。有余力就做；没做就在第 10 节写清「为什么没做 + 怎么做」，照样是面试谈资。


---

## 2. 参考项目（严格限定）

### 代码 / 架构参考（只这 2 个，照着搭；克隆下来通读，但不把它们的代码复制进本仓库）

- **`langchain-ai/open_deep_research`** —— 主架构骨架。supervisor-researcher 多 agent + 并发 + MCP + 内建 eval。**先读它的 README 和 CLAUDE.md** 建立架构认知。
- **`langchain-ai/deep_research_from_scratch`** —— building blocks 学习层（supervisor 模式、并发 research、context 隔离、MCP 接入）。

### 服务层参考（1 个，只取 API/SSE/部署封装这层）

一个 FastAPI + LangGraph 生产模板。挑定一个，别混多个。

### 只读对比（不抄代码，给第 9 节"差异"和 INTERVIEW_QA 用）

- **DeerFlow——只看 `1.x` 分支**。注意：DeerFlow 2.0 是彻底重写的通用 SuperAgent harness，面太宽，**不参考**；要看的是 1.x 的 deep research 框架。
- **`assafelovic/gpt-researcher`** —— 借鉴搜索源抽象、报告生成思路；不当架构基座。

### "基于此但不一样"——具体三条

1. 骨架沿用 supervisor-researcher + 并发模型，但**参照模式自己重写、按自己理解组织目录结构，不 fork 改名**。
2. 在它薄的地方加厚（= 上面的工程增量）。
3. 砍掉它为通用性带来的复杂度（多 provider 全家桶、多 search backend），只留够用的，把省下的精力压到工程深度。

### 许可证

open_deep_research、DeerFlow v1 均为 MIT。保持本仓库是自己重写的东西。

---

## 3. 技术栈

- **后端**：Python / FastAPI / LangGraph（或自定义轻量 orchestrator）/ Pydantic / SSE / pytest
- **Agent 角色**：Planner、Researcher×N（并发）、Verifier、Synthesizer、Citation Checker，加 Cost Tracker、Trace Logger
- **工具层**：Web Search Adapter（+ mock）、RAG Retriever、Source Deduplicator、Report Exporter、Eval Harness；（optional）Document Loader、Reranker
- **工程能力**：限流、超时、重试、熔断、降级、token+cost 统计、tracing、benchmark、可配置 model provider
- **模型约束**：默认走可配置 provider；**所选模型必须支持结构化输出 + tool calling**（这是 open_deep_research 这类架构的硬约束，选型时先确认）

---

## 4. 自动推进循环（每个 feature 做完必做 5 件）

1. 更新代码
2. 更新测试
3. 更新 `KNOWLEDGE_BASE.md`
4. 涉及面试点则更新 `INTERVIEW_QA.md`
5. 一个干净的 git commit（commit message 说清这次做了什么、为什么）

补充硬规则：每完成一个功能就立刻更新知识库，不允许把多个功能做完后最后再集中补 `KNOWLEDGE_BASE.md`；评测结果、失败、取舍和未实测项都要随功能同步落库。

---

## 5. 里程碑小结（输出后继续推进，不等我）

每完成一个里程碑（脊柱跑通 / verifier 完成 / benchmark 出数 / 服务封装完成），输出一段小结后**直接继续**：

- 本阶段做了什么
- 哪些文件变了
- 哪些设计决策进了 `KNOWLEDGE_BASE.md`
- 哪些地方还没实测
- 下一阶段计划

> 这段小结是给我事后补课、挑哪些决策需要内化用的——你不必停下等我确认。

---

## 6. 交付文件（只这 2 个；先建骨架再往里填）

### `KNOWLEDGE_BASE.md` —— 主文件（面试检索核心 + 我复盘用）

必含以下节，标题就用这套：

- **0 项目一句话介绍**：做了什么 / 为什么做 / 解决什么问题 / 体现什么 Agent 工程能力（3~5 句）
- **1 岗位匹配**：为什么适合 Agent 后端 / LLM 应用岗；关联 JD 关键词（LangGraph、RAG、MCP、并发、可观测性、评测）
- **2 总体架构**：分层（API / Agent 编排 / 工具 Adapter / 检索 / 评测 / 可观测）；每层写职责 + 输入输出 + 核心文件 + 为什么这么拆
- **3 核心流程**：第 1 节那条完整链路的文字说明
- **4 关键设计决策**：每条按「背景 / 可选方案 / 最终选择 / 理由 / 代价 / 面试怎么答」。**承重决策必须写满**：多 agent vs 单 agent、RAG 怎么搭、工具失败怎么兜、citation 怎么校验、并发怎么控限流。（决策放这一节，不单开文件）
- **5 实现细节**：按模块（Planner / Researcher / Verifier / Synthesizer / Citation Checker / Cost Tracker），每个写文件位置 / 输入输出 / 关键设计 / 局限
- **6 遇到的问题与修复**：**只记真实发生的**（现象 / 原因 / 排查 / 修复 / 复盘 / 面试可能追问 / 回答）。没有阻塞 bug 就写「本阶段无阻塞性 bug，但暴露出工程风险：……」，**不要编**
- **7 实测数据**：**严格区分实测 / 预期**。单次 research 耗时、P50/P90 延迟、token、cost、搜索结果数、引用保留率、任务成功率、benchmark case 数。没测写 `未实测`。原始运行记录（输入/配置/耗时/token/成本/成败/输出摘要）附在本节末尾或单独 `logs/` 目录
- **8 评测设计**：指标定义（answer completeness、citation faithfulness、source diversity、hallucination rate、latency、cost、工具失败恢复、multi-hop 成功率）；评测集怎么构造、自动 + 人工怎么做、当前局限
- **9 与参考项目的差异**：分别写 open_deep_research / DeerFlow v1 / gpt-researcher——参考了什么 / 没照搬什么 / 我改了什么 / 为什么更适合求职展示
- **10 局限与优化空间**：每条写当前问题 / 可行方案 / 工程代价 / 面试怎么讲

### `INTERVIEW_QA.md` —— 给面试辅助系统检索 + 我面试前逐条 drill

每题格式：

```
## Q：<问题>
[状态: 待消化]
标签：<如 Agent / RAG / 并发 / 可观测性>
检索关键词：<逗号分隔>
回答：<中文、面试口吻、直接可说、不虚构、落到本项目文件或模块>
关联模块：<列出>
可追问：
1. ……
2. ……
3. ……
```

要求：

- 每个核心模块至少 5 题
- 覆盖：架构、Agent 编排、RAG、工具调用、并发、限流、可观测性、成本控制、幻觉控制、引用校验、评测、和普通 RAG 的区别、和 ChatBI/Data Agent 的区别、为什么匹配 Agent 后端岗
- `[状态: 待消化]` 这一位保留不动，供我面试前自己改成「已消化」

---

## 7. 禁止事项

- 抄外部项目代码
- 把没实现写成已实现
- 把没实测写成实测
- 编造 bug / benchmark / 用户亲历过程
- 空泛宣传（"业内领先""高可用高并发"却无工程支撑）
- 只写 README、不写可追问的知识库
- 只做 demo、不做评测和可观测性

---

## 8. 完成标准（达到即停）

1. 陌生人 clone 下来，一条命令起服务、跑 example 出结果
2. 有真实、可复现的指标（记录 seed + 配置快照）
3. 能对着架构图讲 10 分钟不卡，答得上「为什么不是单 agent / 普通 RAG」
4. 至少一处能讲「我在参考项目 X 上做了它没做的 Y，因为 Z」

---

## 9. 高阶对齐路线图（2026-06-07 起）

目标：在保持本项目“收窄、可测、可解释”的前提下，补齐与 `open_deep_research`、DeerFlow v1 的主要能力差距。允许参考 MIT 项目的架构和小段实现，但每次直接迁移都必须保留许可证/来源说明，并在 `KNOWLEDGE_BASE.md` 第 9 节写清楚参考了什么、没有照搬什么、我做了哪些适配。优先级按“可信度差距 > 工具生态 > 长任务生产化 > 展示层”排序。

### P0：公开 Deep Research 级评测

先补端到端研究报告评测，而不是只停在 BEIR/scifact 检索指标。新增公开 benchmark 适配层，能加载 Deep Research Bench / BrowseComp / FutureX 等公开任务格式，跑完整 orchestrator，保存每题的 query、配置快照、report、sources、trace、latency、token、cost、citation 结果和失败原因。没有评测 judge key 时，默认输出可复查 artifacts；有 judge provider 时再计算 answer/citation quality。所有数字必须由脚本重跑生成。

### P1：真实 Web Search 与正文抽取

把 `search.py` 从 Wikipedia/mock 扩展成 provider registry。优先顺序：SearxNG（可自建、无商业锁定）、Brave/Tavily（key optional）、Jina Reader / trafilatura / readability-lxml crawler。搜索 provider 只负责候选 URL，crawler 负责正文抽取、清洗、超时、robots/HTTP 错误记录和 fallback。默认仍保证无 key 可跑。

### P2：MCP 工具接入

新增 MCP tool adapter，把 MCP server 暴露的 tools 统一包装成 `ToolProvider`，下游 researcher 仍看到统一 `Source` / tool result 对象。先支持 stdio/http MCP server 配置、工具白名单、超时、错误隔离、trace 记录；不要求一开始替换现有 Python adapter。

### P3：动态 planning / reflection / compression

把一次性 plan 扩成 bounded research loop：planner 先拆初始问题，researcher 返回 evidence gaps，reflection 判断是否需要追加子问题，compression 压缩中间 findings，达到预算/轮次/证据足够后 synthesis。所有 loop 决策必须进入 trace 和 benchmark artifacts。

### P4：更强 citation faithfulness

在 lexical overlap 之外增加 claim-level grounding：先抽 claim，再定位 evidence span / quote，再用可选 LLM judge 或 NLI provider 做 entailment。默认无 key 时保留 lexical checker；启用 judge 时输出 supported / unsupported / partial / unverifiable 和证据片段。

### P5：前端审核页面

基于已有 run control API 做最小 Web UI：run list、plan 审核/编辑、event stream、source/claim/citation 检查、报告编辑。第一版只做本地可运行，不追求复杂权限。

### P6：生产级 run control

把 SQLite 单机 checkpoint 演进为可替换后端：Postgres run store、Redis/队列 worker、lease/heartbeat、跨进程 cancel、阶段幂等、失败恢复和审计日志。当前 API 语义保持稳定，避免影响 orchestrator 主链路。

### P7：私有知识库与持久化向量库

把本地 JSONL + 临时 Chroma 升级成索引生命周期：document loader、chunk manifest、embedding cache、Qdrant/Milvus provider、增量更新、版本化索引和检索评测。默认仍保留小 JSONL 语料，保证离线 demo。

### P8：多模型策略

把单一 DeepSeek provider 扩成 provider/model policy：planner、researcher、synthesis、compression、judge 可以分别选模型；支持 DeepSeek、OpenAI、Anthropic/OpenRouter/Ollama 的统一接口。默认保持 DeepSeek v4-flash 或 mock，所有 key 从环境变量读取。

### P9：内容导出

在结构化 JSON/Markdown 之外加 Docx/PDF/PPT/TTS optional exporter。导出层必须保持 citation ID、source appendix 和评测 metadata，不能为了展示破坏可追溯性。

### 执行顺序

1. 先做 P0 的公开端到端 benchmark adapter 和 artifacts，因为这是目前与 open_deep_research 最大的可信度差距。
2. 再做 P1 的搜索/crawler provider registry，让 benchmark 不再只依赖 Wikipedia。
3. 接 P4 的 citation grounding，使端到端评测能解释“错在哪里”。
4. 做 P3 的动态 planning loop，让系统从一次性 pipeline 接近真实 deep research agent。
5. 再做 P2/P5/P6/P7/P8/P9，分别补工具生态、审核体验、生产控制面、私有知识库、多模型和内容交付。

---

## 10. 当前有界清理任务（2026-06-08）

本节记录一次面向 portfolio 质量的有界清理改动。执行前必须先读 `README.md`、`AGENTS.md`、`pyproject.toml`，以及 `src/deepresearch_agent/` 下的 `rag.py`、`cost.py`、`api.py`、`orchestrator.py`、`search.py`，理解现状后再动手。

全局约束：

- 改动小而精准；不做重构，不引入新的运行时依赖，不改变现有 CLI / HTTP API 行为，除非任务明确要求。
- 每处改动必须有对应测试；完成后 `pytest -q` 必须全绿。
- 保持诚实口径，不把 mock / 小样本数字包装成成果。
- 小步提交，每个任务一个 commit，沿用 `feat/fix/docs/chore` 风格。

必做任务：

1. 本地检索 graceful fallback：默认仍为 `hybrid`，但当 Chroma、embedding provider、vector index build/search 任一环节不可用时，自动降级到 keyword-only；返回的 `Source.metadata` 必须包含 `retrieval_degraded=True` 和 `degrade_reason`，`LocalRagRetriever` 需要暴露最近一次降级状态和原因，并用 `logging.warning` 记录；Chroma 正常时不能误降级。
2. 文档校正：用真实 `pytest -q` 结果替换旧测试数量记录；把临时交接说明里的限制、未实测项和实测结果整理进正式 `README.md` 的 `Limitations / Future work` 小节；确认无引用后删除交接/打包文件。
3. 成本计价修正：`cost.py` 只保留核对过官方价格的 `deepseek-v4-flash`，删除把 `deepseek-chat`、`deepseek-reasoner` 映射到 v4-flash 价格的 alias；未知模型继续显式抛 `ValueError`，新增测试覆盖。

可选任务：

- API 复用昂贵本地 RAG 构建：如果能在不破坏每请求 provider / 模型 / 检索参数覆盖的前提下干净实现，可以用 FastAPI lifespan 共享 provider-independent 的 `LocalRagRetriever`；如果会引入复杂状态或改变对外行为，就跳过并报告原因。
