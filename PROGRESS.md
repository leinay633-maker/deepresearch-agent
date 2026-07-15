# DeepResearch Agent 真实模型与质量优化进度

## 总目标

使用公司 LLM Gateway 已授权模型，把当前"历史真实运行、最新版仅离线模拟"的状态升级为：最新版能够稳定调用真实模型和真实搜索，使用固定题集与独立裁判反复评测，并针对失败样本迭代到可诚实展示的高质量水平。

## 阶段拆解

- ✅ 阶段一：网关与模型协议核对，安全接入凭证，建立真实请求探针。
- ✅ 阶段二：实现公司网关模型提供方和可用的真实搜索路径，补安全边界与测试。
- ✅ 阶段三：冻结真实质量基线，使用多模型独立裁判和人工可审计产物评估。
- 🚧 阶段四：针对检索、结构化输出、引用、上下文和失败恢复问题迭代。
- ⬜ 阶段五：完整复测、独立代码审查、中文提交、清理本进度文件与指针。

## 当前状态

### v3 四模型原始评测问题诊断（2026-07-14）

v3 运行目录：`~/.deepresearch-agent-eval/runs/single-model-dev-v3-20260714T025449`

四模型失败分类（共 32 次调用，仅 14 次执行成功，0 次答案评分）：

| 失败模式 | 次数 | 涉及模型 | 根因 |
|---------|------|---------|------|
| synthesis fallback (fallback_policy=fail) | 9 | Kimi×3, Opus4.8×4, Opus4.6×1, GLM×1 | **所有 research 子问题搜索全部失败**，0 sources → 合成无法产出有引用答案 |
| 规划校验误杀 (entity anchors) | 3 | Opus4.6×1, GLM×1, Kimi×1 | `_planner_entity_anchors` 只匹配 `[A-Z][a-z]{2,}` 格式，漏掉全大写缩写和数字实体；要求 ≥2 交集过严 |
| Opus 结构化输出类型错误 | 2 | Opus4.8×2 | constraints 输出为字符串而非数组；scope 为空字符串 |
| 搜索/抓取不可达 | 5 | Kimi×3, Opus4.6×1, Opus4.8×1 | DNS 解析失败 / SSL 超时 / circuit breaker 被 2 次失败触发后阻塞所有后续搜索 |

**核心发现**：synthesis fallback 9 次全部源于搜索层全部失败（sources=0），而非合成格式校验问题。

### v4/v5 修复（2026-07-14 已合并，commit 4fca458）

1. **`_string_array` 容忍字符串输入**：Opus 输出单字符串时按分号拆分为数组
2. **`_brief_from_payload` scope 默认值**：空/缺失 scope 不再抛错，用默认研究范围
3. **规划实体检查放宽**：实体锚点增加全大写缩写（`[A-Z]{2,}`）和数字实体（`\w*\d+\w*`）；交集门槛从 ≥2 降为 ≥1；增加通用关键词 fallback（≥1 非停用词交集），解决 TPLF 等缩写/同义词误杀
4. **Circuit breaker 阈值**：failure_threshold 2→4，cooldown 30s→10s
5. **Gateway-web snippet 降级证据**：当所有 crawl 失败但有 snippet 内容时，标记为 `evidence_grade=snippet` 作为降级证据而非直接抛错
6. **主搜索方法适配**：crawl_errors 分支识别 snippet evidence，不再对有内容的降级证据抛 SearchError

新增 7 个专项测试，总测试 267 passed。

### 三代评测执行成功率对比（8题公开开发集）

| 模型 | v3 | v4 | v5(最新代码) |
|------|-----|-----|-----|
| claude-4.6-opus | 5/8 | 8/8 | 6/8 |
| claude-opus-4-8 | 1/8 | 5/8 | 4/8 |
| glm-5.2 | 6/8 | 7/8 | 6/8 |
| kimi-k2.7-code-highspeed | 2/8 | 4/8 | 5/8 |

- v5 目录：`~/.deepresearch-agent-eval/runs/single-model-dev-v5-20260714T042926`
- v5 确认：规划校验误杀已彻底解决（Kimi Q3 TPLF 题从失败转为成功）
- v5 剩余失败全部是 synthesis fallback（sources=0），根因是 live search 网络非确定性（DNS/SSL/timeout），非代码 bug
- 注意：执行成功率波动是 live search 固有非确定性，不等于答案正确率——需双裁判评分判定

### 🚧 双裁判独立评分进行中

- Kimi 全量评分四组 v5 raw.jsonl
- Claude Opus 4.8 再全量独立评分四组
- 两裁判直接读 v5 原始 raw.jsonl，互不读取对方产物
- 用 --rejudge-replay，不带 --single-model-run
- 评分结果写到全新独立目录，不覆盖原始 raw.jsonl

### v5 双裁判评分完成（2026-07-14）

Kimi 裁判目录：`~/.deepresearch-agent-eval/runs/v5-judge-kimi-20260714T045902`
Opus 4.8 裁判目录：`~/.deepresearch-agent-eval/runs/v5-judge-opus48-20260714T045910`

**总体指标（答案正确率，非执行成功率）：**

| 模型 | 执行成功 | 诚实弃答 | Kimi判答对 | Opus判答对 | 一致答对 | 分歧 |
|------|---------|---------|-----------|-----------|---------|------|
| claude-4.6-opus | 6/8 | 5 | 2 | 1 | 1 | 1 |
| claude-opus-4-8 | 4/8 | 3 | 2 | 0 | 0 | 2 |
| kimi-k2.7-code-highspeed | 5/8 | 4 | 0 | 0 | 0 | 0 |
| glm-5.2 | 6/8 | 6 | 0 | 0 | 0 | 0 |

**关键结论：**
- 唯一两裁判一致答对：claude-4.6-opus 的 Q3（Kiyoshi Oka 学科题）——有证据时模型能正确回答
- 大量诚实弃答（sources=0 或证据不相关时模型不编造）——符合"没有证据的答案继续诚实弃答"硬约束
- 核心瓶颈是**搜索覆盖**：8题均为冷门事实题（17世纪法律记录、冷门游戏剧情、哥伦比亚小镇建市等），gateway-web+bing 难找到权威证据
- 两裁判分歧集中在 Q5（Project Firebreak）和 Q2（TPLF）——有部分证据但裁判对答对与否判断不一
- synthesis fallback 全部源于 sources=0（搜索层网络非确定性），非代码 bug

### ✅ 双裁判全量评分完成（v5b，60s 超时，完整）

Kimi 裁判目录：`~/.deepresearch-agent-eval/runs/v5b-judge-kimi-20260714T050344`
Opus 4.8 裁判目录：`~/.deepresearch-agent-eval/runs/v5b-judge-opus48-20260714T050349`
汇总报告：`~/.deepresearch-agent-eval/runs/v5-final-summary.txt`

初版评分用默认 4s 超时导致 Opus 4.8 评分不完整（仅 3-4 题）；重跑用 60s 超时后两裁判均接近全量（Kimi 全 8/8，Opus 6-8/8）。

**最终答案正确率（双裁判，非执行成功率）：**

| 模型 | 执行成功 | 诚实弃答 | Kimi判答对 | Opus判答对 | 两判一致答对 | 分歧 |
|------|---------|---------|-----------|-----------|------------|------|
| claude-4.6-opus | 6/8 | 5 | 1 | 1 | 1 (Q3) | 0 |
| claude-opus-4-8 | 4/8 | 3 | 1 | 1 | 1 (Q2) | 0 |
| kimi-k2.7-code-highspeed | 5/8 | 4 | 0 | 0 | 0 | 0 |
| glm-5.2 | 6/8 | 6 | 1 | 0 | 0 | 1 (Q6) |

**逐题可审计结论（不把执行成功当答对）：**
- claude-4.6-opus：Q3（Kiyoshi Oka 学科）两判一致答对；5题诚实弃答（证据不足不编造）；2题执行失败
- claude-opus-4-8：Q2（TPLF 移除）两判一致答对；3题诚实弃答；4题执行失败
- kimi：Q6（San Carlos 建市）已答但两判均判错；4题诚实弃答；3题执行失败
- glm-5.2：几乎全诚实弃答（搜索覆盖最弱）；Q6 Kimi判对/Opus判错（分歧，sources=0 弃答，疑 Kimi 误判）

### v6 宽搜索实验结论（已停）

尝试 max_results 3→5、max_researchers 2→3。Q2（TPLF）claude-4.6-opus 找到 6 个来源仍诚实弃答——**瓶颈是搜索结果相关性而非数量**，冷门事实题的权威答案页 gateway-web+bing 搜不到。更宽搜索未带来答对率提升，已停掉，以 v5 为最终结果。

## 阶段五：完成

### 🔥 答对率低的根因诊断（2026-07-14，v7→v8）

v7 整体答对率 4/32（12.5%），用户判断"上不了台面"。系统诊断**成绩低主要是 harness 损耗，不是题目难度**：

- **题目难度（次因）**：SimpleQA 是 OpenAI 对抗性收集的难题集，但 dev set 8 题**全部有 gold_urls、答案页都存在**（多数 Wikipedia/知名站）。kimi 修复 thinking 后 4/8（50%）证明可答。
- **Harness 损耗（主因）**，分层定位：
  1. **HTML 正文提取粗糙（最大损耗）**：`_HtmlTextParser` 只跳过 script/style，不剔除 nav/header/footer/aside，导航菜单占满 `crawler_max_chars=4000` 预算、正文（含答案）被截断。铁证：Q08 sherdog 找到正确页面但 4000 字符全是 "NEWS FEATURES FIGHT FINDER..."；Q03 找到 gold_url MacTutor 但 content 开头是导航。
  2. **context budget 偏小**：`crawler_max_chars=4000` + `per_source_tokens=650`，导航占满后正文两层截断丢失。
  3. **合成对已有证据利用不足**：Q03 nara-wu content 第 1137 字符就有 "Department of Physics"（在预算内），claude 仍弃答——合成对"Physics 次年转 Math"类表述判断过严。
  4. **crawl 失败率高**：Q08 tapology（标题就是 King Cobra）crawl 失败只剩 snippet。
  5. **搜索查询没命中 gold_url**：Q05 没找到 fandom、Q06 没找到 Wikipedia。
  6. **glm 不支持 server-side web_search 工具**（模型/网关侧限制）。

**修复（commit f334544，v8 验证中）：**
- `_HtmlTextParser` 扩展 skip 标签（nav/header/footer/aside/form/menu），优先 `<article>/<main>` 正文，无则回退去导航全文
- `crawler_max_chars` 4000→8000，`per_source_tokens` 650→1200
- 新增 3 个正文提取测试，总计 272 passed

### 🔥 关键 bug 发现与修复（commit 1a552af，v7）

v5 评分答对率低（1/8），深挖发现**根因不是"冷门题搜不到"，而是 kimi web search 100% 失败**：

- `GatewayWebSearchAdapter._request` 直接构造 httpx 请求，**没有像 `LLMGatewayClient` 那样对 kimi 模型添加 `thinking={type:enabled}` 参数**
- kimi 模型要求 thinking 必须 type=enabled，否则 HTTP 400 `"invalid thinking: only type=enabled is allowed for this model"`
- 这导致 kimi 的 web search 全部失败 → sources=0 → 全部 synthesis fallback 或诚实弃答 → 答对率 0

修复：复用 `llm_gateway._requires_thinking`，对 kimi 在 web search 请求体加 enabled thinking block。修复后直接测试 kimi 搜索从 400 变为返回 5 候选。

**附带诊断结论：**
- glm-5.2 调用 web_search 工具时只返回 text block（用自身知识），不返回 `web_search_tool_result`——glm 模型/网关侧不支持 server-side web_search 工具，非代码 bug，fallback 到 bing
- claude-4.6-opus 搜索间歇性超时（网络非确定性），直接测试多数成功
- claude-opus-4-8 web search 最稳定

### ✅ v7 双裁判评分完成（thinking 修复后，commit 1a552af）

v7 生成目录：`~/.deepresearch-agent-eval/runs/single-model-dev-v7-20260714T051613`
Kimi 裁判：`~/.deepresearch-agent-eval/runs/v7-judge-highspeed-20260714T054138`
Opus 4.8 裁判：`~/.deepresearch-agent-eval/runs/v7-judge-8-20260714T054138`
汇总报告：`~/.deepresearch-agent-eval/runs/v7-final-summary.txt`

**答案正确率（两裁判一致答对，非执行成功率）：**

| 模型 | v5 一致答对 | v7 一致答对 | 变化 |
|------|-----------|-----------|------|
| kimi-k2.7-code-highspeed | 0/8 | **4/8 (50%)** | +4 ⭐ |
| claude-4.6-opus | 1/8 | 0/8 | -1（搜索非确定性） |
| claude-opus-4-8 | 1/8 | 0/8 | -1（搜索非确定性） |
| glm-5.2 | 0/8 | 0/8 | — |
| **整体** | 2/32 (6.25%) | **4/32 (12.5%)** | **翻倍** |

**kimi 4 题两裁判一致答对**：Q01(Thomas Ballard 1637)、Q02(TPLF 移除)、Q03(Kiyoshi Oka 学科)、Q07(Meyrick 1886)。

**执行成功率（thinking 修复让 kimi 搜索恢复）：**

| 模型 | v5 | v7 |
|------|-----|-----|
| kimi | 5/8 (grounding 0.4) | 7/8 (grounding **0.7143**) |
| claude-4.6-opus | 6/8 | 7/8 |
| glm-5.2 | 6/8 | 7/8 |
| claude-opus-4-8 | 4/8 | 3/8（搜索间歇超时） |

**核心结论：**
- thinking 修复是决定性的：kimi web search 从 100% 失败恢复，答对率 0→50%，grounding 0.4→0.7143
- claude 模型 v7 退步是 live search 非确定性：v5 答对的 Q3 在 v7 找到来源但判断不相关、诚实弃答（非代码问题）
- glm 仍 0 答对：glm 不支持 server-side web_search 工具（只返回 text 用自身知识），靠 bing fallback 覆盖弱
- 剩余瓶颈：claude/glm 的搜索查询质量（找到来源但不含答案）+ 冷门事实题搜索引擎覆盖

---

### 历史验收基线（v5，thinking 修复前）

- ✅ pytest 269 passed、ruff、compileall、git diff --check 全通过
- ✅ 不再出现大面积结构化输出兼容失败（_string_array/scope 修复）
- ✅ 所有运行实际模型与目标模型一致（--single-model-run 校验）
- ✅ 没有证据的答案继续诚实弃答（sources=0 时不编造）
- ✅ 四模型公开开发集完成重新生成和双裁判全量评分
- ✅ 逐题可审计结论，区分执行成功/诚实弃答/答对

### 🚧 独立审计后的可信评测重建（2026-07-14）

此前“低分主要是 harness、HTML 正文提取是最大损耗、v8 已证明提升”的结论已被降级为待验证假设：v7/v8 都只有一次 live 生成，候选来源发生变化；v7 全局 24/32 执行成功，v8 为 23/32，不能把 Kimi 单次 6/8 外推为跨模型提升。v5/v7 还是 dirty worktree，v6 宽搜索也只完成了单题，历史归因不满足可复现门槛。

- ✅ 完成独立评测与工程链路审计，未触碰隐藏集。
- ✅ answer judge 重构为 `correct / incorrect / not_attempted / unscored`，答案正确性与 citation grounding 分开；裁判缺字段或实际模型漂移记 `unscored`。
- ✅ 修正弃答、实质答案、grounded answer、`final_result_usable` 与数据集内容哈希口径，新增正式运行 clean-worktree 校验。
- ✅ 从 OpenAI 官方公开 SimpleQA CSV 按固定种子生成独立 32 题主集及 manifest；现有 8 题保留为诊断集。
- ✅ 公开 32 题主集改为 topic 与 answer_type 双边近均衡配额：10 个 topic 每类 3–4 题，5 个 answer_type 每类 6–7 题；修复带括号、转义换行、重复和尾随引号的 gold URL 解析问题。当前 source SHA-256 为 `feee3f7e...5032`，case SHA-256 为 `9702de3e...bfc1`，另记录 8 条诊断集排除项的 hash。
- ✅ 完成检索候选/抓取错误审计、短暂错误单次重试、Gateway 工具能力探针和多实体相关性过滤；snippet/crawl-failed 候选只保留脱敏审计 hint，不进入最终 evidence、synthesis 或 citation。
- ✅ 真实 Gateway capability probe 已落盘到 `~/.deepresearch-agent-eval/runs/gateway-capability-20260714T173939/capabilities.json`：Claude 4.6、Opus 4.8、Kimi 均返回 `web_search_tool_result` 且实际模型匹配；GLM 5.2 仅返回 `text`，状态为 `text_only_no_tool`。这确认 GLM 的 server-side web search 缺失是当前网关能力差异，不是猜测；正式榜单仍保持严格单模型，不为 GLM 混入其他搜索模型。
- ✅ 固定历史 artifact 的离线分层分析已拆开正文与 snippet。v7→v8 的 answer-in-citable-source 变化：Claude 4.6 `2→1`、Opus 4.8 `1→0`、GLM `0→0`、Kimi `4→5`；只有 Kimi 增 1，且检索来源同时变化，进一步否定“HTML 修复已被单次 v8 证明是最大杠杆”。v7 Claude 4.6 的 650→1200 token 打包由 `1→2`，说明预算有个案收益，但不是跨模型普遍结论。
- ✅ 独立只读评审发现并关闭 5 项问题：空/失败输出被错误计 correct、不完整裁判 artifact 被合并器信任、自评一致未单列、正式 HTML crawler 丢失异常类型、gold URL 转义换行污染。复核确认 5 项全部关闭。
- ✅ 当前全量测试 `308 passed`；ruff、compileall、`git diff --check` 全部通过。
- ⬜ 下一步入口：建立干净中文提交后，执行四模型三波交错生成与 Kimi/Opus 4.8 事后双裁判。

### 🔄 评测目标再次纠偏：SimpleQA 不是 Deep Research 主评测（2026-07-14）

用户审阅实际题目与答案后指出：SimpleQA 单题通常只需要检索一个冷门事实并输出短答案，无法证明多问题规划、多源综合、冲突消解、报告完整性和研究结论质量。这一判断成立，继续完成 480 次 SimpleQA 生成会把“单事实 Web QA”测得更稳定，但不能支撑 DeepResearch Agent 的项目展示。

- ✅ 已停止 `48abd21` 的三波长跑，未启动 32 题主集；保留第一波诊断集的部分 artifact 供检索组件排障，不作为 Deep Research 质量结论。
- ✅ SimpleQA 重新定位为组件级回归：冷门事实召回、HTML 正文抽取、引用纪律、弃答与 judge 账本。
- 🚧 主质量评测改为公开 Deep Research 任务：问题必须需要多源证据、比较/归纳或冲突处理，并以报告完整性、事实正确性、citation grounding、来源质量/多样性、弃答、延迟和成本分层评分。
- ⬜ 下一步入口：审计现有 LiveDRBench 适配与公开任务内容，选出可复现的主评测集和 2–3 个可直接展示的报告样例；SimpleQA 只保留小规模 smoke，不再做三波主榜。

### ✅ DeepResearch Bench II 公开 12 题冻结（2026-07-14）

- ✅ 固定上游 commit `11d87de486ba7a9e10190be0afd66a9a0fc5d5da`，选择 idx `4,9,23,30,43,48,54,57,63,66,82,83`；6 中/6 英，6 个主题各 2 题，全部 CC BY 4.0。
- ✅ `evals/drb2_public12_v1.tasks.jsonl` 仅保留生成期字段，`evals/drb2_public12_v1.rubrics.jsonl` 与生成严格分离；manifest 锁定上游/产物 SHA、选择算法与许可。
- ✅ 从 132 行固定快照实际统计 rubric：`6983 info_recall + 1686 analysis + 746 presentation = 9415`；构建器对行数、分类数和总数 fail-closed，不采信 README 的过时数字。
- ✅ 主线复验 `tests/test_drb2_public_tools.py`：12 passed；定向 ruff 与 artifact hash 校验通过，未调用真实 LLM。
- ✅ concise/deep 双模式已实现：concise 保持 1 轮/4 结果/3 claims/2600 输出；deep 支持最多 5 分支、每分支有界 3 轮查询改写、8 结果、至少 2 证据项。
- ✅ deep 合成使用 48k 输入预算/12k 保留（可用 source context 严格不超过 36k）、最多 36 来源、单源 2400 tokens、72 claims 安全上限和 12k 输出上限。
- ✅ deep Markdown 保留标题、列表和比较表；事实句、列表项和表格数据行仍必须逐项与 claims 对齐并携带已知引用，不放宽弃答/引用/prompt injection 边界。
- ✅ 主线定向复验 33 passed，全量回归 325 passed / 1 个既有 warning。
- ✅ 动态 blocked-source denylist 已贯穿 `ResearchRequest → deep_research_eval → orchestrator → SearchService → HtmlTextCrawler`；canonical 比较规范 scheme/host/默认端口/path 并忽略 query/fragment。
- ✅ 搜索候选命中时 crawler 不会启动；HTML redirect Location 在下一跳请求打开前 fail-closed；自定义 crawler 的最终 URL/完整 redirect chain 仍会事后阻断，命中统一抛 `BenchmarkContaminationError` 且不 fallback。
- ✅ 静态污染 marker 增加 `deepresearch-bench` / `deepresearchbench` / `drb2` / `deep_research_bench`，仅屏蔽 GitHub/HF 的 benchmark 路径（含 HF datasets-server query），普通 GitHub 文档仍允许。
- ✅ 主线定向复验 117 passed；相关 ruff、compileall、`git diff --check` 通过，SSRF 和跨域重定向凭证隔离原测试未破坏。
- ✅ 新增独立 DRB II rubric evaluator，每次只运行 Kimi 或 Opus 4.8 一个裁判；按 rubric 保存 `1/0/-1`、reason、报告原文 quote、actual model、self-judge、prompt/response hash 和 append-only attempt。
- ✅ 裁判 JSON/字段/模型漂移/引用 quote 校验失败均保留为 `-1 unscored`，不静默记 0；Gateway 结构化 mismatch 异常保留 requested/actual model，断点恢复严格校验 generation/rubric SHA 与 manifest lineage。
- ✅ 新增 Kimi/Opus 双裁判汇总：仅 `1+1` 保守通过，分裁判/分题/`info_recall|analysis|presentation` 报告 agreement、disagreement、unscored 与 self-judge；citation grounding 与 generation execution status 均为独立账本，不混入 rubric 分。
- ✅ 裁判 prompt 保留 Markdown 标题/换行/表格结构，同时按行剔除明显 prompt injection 与控制字符；口径固定为“基于公开 DRB II rubric 的本地 Kimi/Opus 双裁判协议”，明确非官方榜单。
- ✅ 主线定向复验 81 passed，全量回归 358 passed / 1 个既有 warning；ruff、compileall、`git diff --check` 通过。
- ⚠️ 首次诊断性锚点已运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-20260715T110541`，`task2+ × claude-opus-4-8`；brief/planner 和 actual model 校验成功，但无报告产出。
- ⚠️ 失败根因不是报告结构/预算，而是 Q5 搜索候选命中该题禁止的 OECD 专家参考 URL；现策略在未 crawl 的候选阶段就终止整题，使 DRB II 近乎不可运行。
- ✅ denylist 粒度已修正：blocked 候选在 crawl 前剔除，blocked-only primary 不重试同一 provider 而直接转真实 fallback；HTML redirect 在下一跳 open 前阻断单个 source，干净 sibling 继续。
- ✅ `SearchOutcome.sources` 严格只含干净来源；安全命中记为 `denylist_enforcement_hit=true / benchmark_contamination=false`，并保存 `BenchmarkContaminationError` 类型、stage 与 URL identity SHA；若 blocked source 越过最终 evidence 边界仍立即硬失败。
- ✅ primary + 真实 fallback 全部被阻断时抛 `SearchEvidenceUnavailableError` 且附议定审计，不启用 mock；主线定向 94 passed，全量 360 passed / 1 个既有 warning，ruff/compileall/`git diff --check` 通过。
- ⚠️ 第二次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v2-20260715T112307`。denylist 已不再终止整题，11 个干净来源完成去重，禁用 URL 未进入最终 evidence，所有生成/搜索 actual model 均为 `claude-opus-4-8`。
- ⚠️ 仍无报告：合成返回约 9014 output tokens 的长草稿，但校验报 `LLM synthesis answer contains uncited factual text`；两次重试仍失败，由于 `fallback_policy=fail` 正确拒绝 deterministic fallback。
- ⚠️ 成本/延迟过高：单题 1136.8s、179263 tokens；主因是 5 分支强制至少 3 轮，多个分支在 crawl/真实 fallback 上跑满预算。
- ✅ deep Markdown 合成校验已修正：仅允许无数字/无引用/无结论且紧接表格或列表的通用结构引导语；真实无引用事实仍硬失败。
- ✅ answer 中已带合法 source ID 但 claims 漏列的句子/列表项/表格数据行会按去列表前缀、粗体和管道符的规范化结果补入 claims；补入后仍执行未知 citation 拒绝、72 claims 上限和 CitationChecker。
- ✅ deep 默认预算收敛为 2 轮/2 tool calls/单分支 300s 硬 deadline；调用方显式 3+ 或自定义 deadline 仍保留。校验失败仅记录截断 reason 和原始输出 SHA-256，不写长草稿。
- ✅ 主线定向 31 passed，全量 365 passed / 1 个既有 warning，ruff/compileall/`git diff --check` 通过。
- ⚠️ 第三次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v3-20260715T115624`；显式 2 轮/2 calls/300s deadline 已生效，但在 brief 阶段约 62s 后失败，未进入搜索。
- ⚠️ 错误为 `LLM JSON field constraints must be an array of strings`；长任务下 Opus 4.8 将 `constraints` 返回为非纯 `list[str]` 结构，现 `_string_array` 虽已兼容单字符串，仍不兼容带明确文本字段的对象项。actual model 仍严格匹配 `claude-opus-4-8`。
- ✅ `llm.py::_string_array` 已做最小兼容扩展：继续支持 `list[str]` 与分号字符串，并允许对象项中恰好一个 `text|constraint|assumption|value|description` 非空字符串；未知字段、嵌套值、非字符串和多个有效字段仍 fail-closed。
- ✅ 新增 brief 端到端正例及对象形状反例测试；主线复验定向 125 passed、全量 371 passed / 1 个既有 warning，ruff、compileall、`git diff --check` 全部通过。测试工具误将 `uv.lock` 从阿里云镜像重排为 PyPI 元数据，已只恢复该无关机械改写，工作树无锁文件变更。
- ⚠️ 第四次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v4-20260715T120836`。对象数组兼容修复有效，brief/planner/research/synthesis 全链路完成，评测进程本身正常写出 artifact；外层 zsh 包装脚本仅因误用只读变量名 `status` 在收尾时报错，不影响评测 artifact。
- ⚠️ 该次只能记为“报告生成流程执行成功 + 诚实弃答”，不能记为 Deep Research 报告成功：输出仅 77 字符 `The available sources are insufficient...`，无章节、无表格、0 claims、citation grounding/coverage 均为 0，`substantive_answer=false / grounded_answer=false / final_result_usable=false`。
- ⚠️ 本次耗时 587.6s、104,635 tokens；最终 7 个来源来自 7 个域名，但可见证据明显集中于印度尼西亚与越南，未覆盖七国、全部方案、DB benefit formula 与两张必答表。blocked URL 未进入最终 sources，`benchmark_contamination=false`。后续固定 artifact 审计确认：可见答案虽然是保守弃答，但不能把根因全归为“证据不足”，因为三次长 synthesis 实际是结构/引用校验失败后被代码误分类为证据弃答。
- ✅ v4 检索分层审计：71 个候选中 29 个安全正文抓取成功，研究语义过滤后只剩 7 个；42 个抓取损失以 `http_4xx=29 / timeout=7 / connection_failure=3` 为主，另有 22 个已验证页面在 branch relevance 阶段被过滤，Q3 为 14→0。denylist 无命中、circuit breaker 未打开，不能把本次覆盖失败归因于污染拦截。
- ✅ deep planner/research 新增可执行 coverage contract：`SubQuestion.required_entities / required_aspects` 默认空以保持历史兼容，deep planner 必须逐分支枚举命名目标和必答维度；Python executor 只有证据数量与 coverage 同时满足才接受 stop，缺口查询只聚焦未覆盖实体/方面。
- ✅ 复合多国分支不再要求单页命中多个国家：单国页面只有在通过原 lexical evidence 阈值且明确覆盖一个 required entity 时才可进入该分支，未命中任何目标实体的泛化页面继续拒绝；最终 claim 仍走原 CitationChecker，不放松引用。
- ✅ synthesis context packer 修复超长句段：每个 passage 最多 1200 字符并带重叠窗口，避免 9k 导航块靠绝对关键词数挤掉正文。用 v4 固定 artifact 重打包后，S7 已包含越南 `15 years` 与男女退休年龄，context 估算由 7,712 降至 5,958 tokens。
- ✅ 弃答分类收紧：缺 citation、answer 缺 citation 始终是 synthesis validation failure；有 verified sources 时 `no usable claims` 也不能伪装成证据不足成功。仅“没有 verified source + no usable claims”允许真正证据弃答；两条路径均只保留 bounded reason 与 output SHA，不保存失败长草稿。`fallback_policy=fail` 继续拒绝格式失败。
- ✅ 固定测试覆盖三国分支早停、缺失实体查询聚焦、单国权威页/泛化页分流、S7 导航噪声、空 claims 与 missing-citation 分类。独立审计随后发现 coverage 正文重复计数误用 set tokenizer，已改为真实词流/CJK 片段计数并补仅正文重复命中测试。
- ✅ 动态 denylist 再收紧：v4 曾出现被禁 OECD 专家报告的子章节 URL，旧 exact path 只因该页 4xx 才未污染。现在同 origin 下，被禁 `.html` 报告路径的后代章节也视为同一 blocked source；不屏蔽整个 host 或无关 sibling 报告，候选与重定向共用该匹配。新增 descendant/sibling/crawl-before-block 固定测试。
- ✅ 最新主线定向 43 passed、全量 380 passed / 1 个既有 warning，ruff、compileall、`git diff --check` 全部通过，`uv.lock` 无残留改写。
- 🚧 下一步入口：第五次复跑同一 `task2+ × claude-opus-4-8` 锚点，仍保持 2 轮/2 calls/300s/240s；重点检查 planner coverage contract、逐分支 coverage trace、Q3 单国页面保留、S7 正文打包、synthesis 是否产出带章节/两表/逐 claim 引用的局部或完整报告。若仍失败，按新 validation reason/SHA 定位；仍不直接启动四模型或 12×4。
- ⚠️ 第五次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v5-20260715T124632`，进程正常退出并写全 artifact，但整题 `execution_success=false / report_emitted=false / final_result_usable=false`，不能记报告成功。
- ✅ 新 coverage contract 在真实运行中生效：planner 为 5 个分支保存 required entities/aspects；所有分支 coverage 不完整时均未错误提前标记 `evidence_sufficient`。Q3 从 v4 的 14 个安全抓取→0 evidence 改善为 5 个 evidence，并识别出 Philippines SSS formula 覆盖；Q1 7 evidence，覆盖 Indonesia BPJS/Taspen 与 Philippines SSS/GSIS。
- ✅ descendant denylist 在真实运行中生效：Q1/Q5 各记录 1 次 `denylist_enforcement_hit=true / benchmark_contamination=false`，被禁 OECD 专家报告章节未进入 evidence；actual model 的 brief/planner/search/research_decision 均严格为 `claude-opus-4-8`。
- ⚠️ coverage 仍远未完成：Q1/Q3 均两轮结束且缺多数国家/字段；Q2 只有一轮、6 evidence 后打满 300s；Q4 首轮搜索未返回即 300s deadline、0 evidence；Q5 1 evidence 后 deadline。五分支全部 `budget_exhausted=true`。
- ⚠️ coverage follow-up query 没有按预期聚焦：Q1/Q3 第二轮 trace 被 `safe_follow_up_query` 回退为截断后的原始长 subquestion，而不是缺失实体/方面，说明当前 guardrail 把 coverage query 判成与原问题不相关；这会继续产生宽查询。
- ⚠️ synthesis 阶段最终报 `LLM Gateway synthesis JSON validation failed: The read operation timed out`，随后 deterministic fallback 被 `fallback_policy=fail` 正确拒绝。整题耗时 1104.3s、77,623 tokens；没有落无引用报告，也没有把格式/超时失败记成诚实弃答成功。
- 🚧 下一步入口：并行只读审计三个点后再改：① `safe_follow_up_query` 为什么拒绝 coverage-focused query，如何在保留 prompt-injection/SSRF 防护下让缺失实体查询通过；② gateway-web + HTML crawler 在 300s 内为何 Q4 0 返回，候选抓取是否串行/超时预算是否放大；③ 12k synthesis 在 240s 下的重试/超时策略，如何避免三次整段长生成并保留 strict actual-model/fallback 语义。固定测试与离线 v5 trace 审计通过后才第六次复跑，不启动四模型。
- ✅ coverage query 根因与修复：Q1/Q3 focused candidate 分别 448/560 字符，完整内容无 URL/注入且与原问题高度相关；旧 guardrail 因 `len(raw)>240` 直接回退。现在完整 raw 先做 newline/control/URL/role/injection 检查，再按词边界截到 240，并对实际发送的 bounded query 做相关性校验；相关词只在截断后仍不授权，unsafe suffix 仍拦截。
- ✅ 检索绝对预算与部分成功：researcher 建立 absolute deadline 并按剩余轮次切 slice，每轮搜索只使用 80%，为 RAG/decision 留预算；SearchService primary/retry/crawl/Bing fallback 均使用 `min(config, remaining)`，fallback 不再重获 240s。redirect 链共享总 timeout、逐跳递减，SSRF→denylist guard→open 与跨源凭证剥离顺序不变。
- ✅ crawl batch 默认每 search 并发 2（`CRAWLER_CONCURRENCY_PER_SEARCH` 可配）；batch deadline 到时保留已完成 clean evidence，慢项记 `batch_timeout` 和 partial audit，不再因最慢候选整批归零。原 blocked candidate/redirect/final URL、SSRF 和凭证隔离测试全部保留。
- ✅ deep synthesis 预算与重试分类：输出上限 12k→10k（仍在既定 10k–12k 范围），72 claims/多章节/列表/表格/逐 claim 引用不变；独立 socket timeout 默认 360s（`LLM_SYNTHESIS_TIMEOUT_SECONDS` / CLI 可配并进入 manifest），其他阶段仍使用 request timeout。
- ✅ synthesis transport timeout、无 content 失败、model mismatch 和确定性 4xx 均只请求一次并 fail-closed；仅收到完整 content 后的 JSON/schema/citation 失败允许一次定向 repair。repair 从完整原始 evidence context 重生成，不回灌半截长草稿；失败 trace 保留 bounded packed context 与最多两条 attempt ledger，不含正文或 key。
- ✅ 固定测试覆盖 artifact-like 长 query、安全后缀、round slice、fallback remaining、2 并发+partial crawl、redirect timeout 递减、single-call transport timeout、一次定向 repair、model mismatch actual model、10k/360 配置和失败 trace。主线全量 392 passed / 1 个既有 warning，ruff、compileall、`git diff --check` 全部通过，`uv.lock` 无残留改写。
- 🚧 下一步入口：第六次复跑同一 `task2+ × claude-opus-4-8` 锚点，参数仍为 2 rounds / 2 calls / researcher deadline 300s / request timeout 240s，新增 synthesis timeout 360s（默认）和 crawler concurrency 2。审计第二轮 focused query、partial crawl audit、research latency、packed context/attempt ledger、报告结构/两表/逐 claim 引用；合格前不启动四模型。
- ⚠️ 第六次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v6-20260715T133250`。进程正常退出但整题 `execution_success=false / report_emitted=false / final_result_usable=false`；没有无引用草稿或伪弃答。
- ✅ 组合超时显著收敛：整题 631.8s，较 v5 1104.3s 下降约 42.8%；researcher 均在 267–300s 内返回，trace 保存 `batch_timeout` 和 partial audit。Q4 从 v5 的 0 evidence 恢复为 6 evidence，Q1/Q2/Q4 各 4/6/6 evidence，最终研究阶段形成 17 个去重来源。
- ✅ coverage follow-up 生效：Q1/Q2/Q3/Q4/Q5 第二轮 query 均以缺失实体/方面开头，不再退回原 subquestion 前 240 字符；例如 Q1 第二轮以 `Malaysia Taspen... EPF/KWSP... KWAP...` 开头，Q5 枚举七国与缺失 gap drivers。
- ✅ absolute budget/partial crawl 生效：各轮审计出现 `batch_timeout`，但已完成来源保留；redirect/fallback 未再造成 v5 式整批 0 outcome。Q3 仍因 Sri Lanka/Thailand 候选大量 non_html/dns/batch timeout 最终 0 evidence，说明调度修复不能替代可抓取来源覆盖。
- ✅ synthesis 仅发起 1 次 10k/360s 请求，227.6s 后返回 `LLM Gateway returned no text content`；未重复三次，失败 trace 保存 estimated context 15,881 tokens、17 kept sources、socket timeout、max output 和单条 attempt ledger。actual model 因未解析出 text 响应仍为 null；`fallback_policy=fail` 正确拒绝 deterministic fallback。
- ⚠️ 当前新阻塞不是 context budget：17 个来源全部 kept、15,881 < 36,000。需要确认 Gateway 返回的是 thinking-only、其他 content block、空 content，还是 stop_reason/max_tokens 协议问题；当前异常未保存 aggregate-safe block types/stop reason/response model/usage，证据不足以决定调大/调小输出或改变解析。
- ✅ Gateway 无 text 审计已补齐：新增专用 `LLMGatewayNoTextContentError`，仅保留 requested/actual model、stop reason、规范化 content block types、usage、响应字节数和 SHA-256；不保存原始响应或 thinking 正文，strict actual-model 校验仍先于 no-text 分类。
- ✅ synthesis attempt ledger 新增 `no_text_content` 失败类并记录上述安全元数据；无文本响应的 usage 仍进入失败成本账本，thinking 绝不作为报告正文或 repair 输入。固定覆盖 thinking-only、空 content、模型漂移优先级、单次 fail-closed 和失败 usage；定向 `39 passed`，ruff、compileall、`git diff --check` 通过。
- ✅ Opus 4.8 安全结构 probe 未复现 v6：短输入在 `max_tokens=256` 与 `10,000` 下均约 2s 返回同一 11-byte 文本、`end_turn`；长合成输入/短输出在约 3.4s 返回文本，计费输入约 12.5k tokens；长输入 + 六章节/两表/60 claims 请求在约 102.2s 返回 5,736 bytes 文本、`output_tokens=4,522 / stop_reason=end_turn`。三组 actual model 均严格为 `claude-opus-4-8`，全程只输出长度、usage 与 SHA，不回显正文或 key。
- ⚠️ 现有证据排除了“只要 10k max_tokens、较长输入或长报告结构就稳定触发无 text”，但不能凭一次成功 probe 宣称 provider 已修复；v6 暂归类为不可稳定复现的 Gateway/provider 无文本响应。当前不应臆造协议参数修复，也不能把 thinking 当正文。
- ⚠️ 第七次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v7-20260715T140401`。brief 25.3s、planner 93.3s，五个 coverage 分支均在 254.8–300.0s 内按预算返回；去重后形成 14 个干净来源，packed context 13,736 tokens，全部来源 kept，未发生 benchmark contamination。
- ⚠️ v7 仍无报告，但失败已从 v6 的 no-text 改为明确 transport timeout：唯一一次 synthesis 在 360.0s socket 上限触发 `The read operation timed out`，actual model/usage 均因未收到完整响应而为空；整题 778.7s、128,030 tokens，`execution_success=false / report_emitted=false / evidence_abstention=false`，未落无引用草稿或伪弃答。
- 💡 新的结构性放大已确认：deep prompt 要求每个事实句、列表项和表格行既出现在完整 Markdown `answer`，又 verbatim 复制进 `claims`，会让七国长报告的 JSON 输出近乎翻倍；而 `_synthesis_from_payload(... preserve_markdown_structure=True)` 已能从 answer 的逐项引用内容确定性补齐并校验 claims，重复生成没有安全收益。
- ✅ deep synthesis 去重已实现：初始生成、citation repair 与结构化 repair 都要求完整带逐项引用的报告只写一次到 `answer`，`claims` 固定为空数组；Python 从答案确定性提取事实句/列表项/表格数据行。concise 行为、72 条上限、unknown citation、未引用事实、CitationChecker 与 fail-closed 均未放松。
- ✅ 固定测试覆盖 deep prompt 空 claims 契约、两类 repair、空 claims 安全提取、未知引用与未引用表格事实拒绝；主线定向 89 passed，全量 397 passed / 1 个既有 warning，ruff、compileall、`git diff --check` 全部通过，`uv.lock` 干净。
- ⚠️ 第八次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v8-20260715T142743`。检索 18→17 个干净来源，packed context 14,385 tokens，全部 kept；Q1/Q2/Q4/Q5 分别保留 5/8/4/1 evidence，Q3 本轮 deadline 为 0，denylist 命中均为成功拦截且 `benchmark_contamination=false`。
- ✅ 去重修复消除了 v7 的 360s transport timeout：synthesis 两次完整响应分别约 116.8s / 104.4s，actual model 均严格为 `claude-opus-4-8`；首轮 output 4,338 tokens，repair 3,353 tokens，整题 576.2s / 106,132 tokens，较 v7 778.7s / 128,030 tokens 均下降。
- ⚠️ v8 仍无报告：初始响应不是可解析 JSON（`Expecting value: line 1 column 1`），唯一一次定向 repair 返回完整 JSON 后因 `LLM synthesis answer contains uncited factual text` 被 fail-closed 拒绝；`execution_success=false / report_emitted=false / evidence_abstention=false`，没有把未引用草稿记为成功。
- ✅ deep Markdown citation sanitizer 已实现：claims 提取前保留标题/水平线/表头/分隔、严格 generic lead-in 与所有带引用句/列表项/表格数据行；未引用事实句或数据行被删除，混合段落只保留带引用句，绝不自动补 citation。deep 忽略模型平行 claims，只从清洗后的可见答案提取。
- ✅ `last_synthesis_context.synthesis_sanitization` 在成功/失败均记录纯聚合计数（dropped uncited sentence/line/table-row），不保存正文；unknown citation、72 claims、无 claims 失败、alignment、CitationChecker 和 concise 行为保持 fail-closed。主线定向 92 passed，全量 400 passed / 1 个既有 warning，ruff、compileall、`git diff --check` 全部通过。
- ⚠️ 第九次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v9-20260715T144833`。本轮 live search 形成 22 个干净来源，Q1/Q3/Q4/Q5 分别 8/4/8/2 evidence，Q2 为 0；packed context 增至 20,947 tokens，全部 kept，无 injection flag/benchmark contamination。
- ⚠️ v9 唯一一次 synthesis 在 360.0s 再次 transport timeout，未收到完整响应，sanitizer 因无 payload 未执行；整题 715.7s / 74,289 tokens。结合 v7 13,736-token context 超时、v8 14,385-token context 两次各约 104–117s 完成，说明 Gateway 长报告时延高度波动，360s 对复杂报告不是稳定上限，不能把一次 v8 成功外推为已关闭。
- ⚠️ 第十次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v10-20260715T150252`，仅将 synthesis timeout 提至 600s。检索形成 19 个干净去重来源，context packer kept 17 / dropped 2，estimated 15,164 tokens；无 benchmark contamination，sanitizer 因未收到正文未执行。
- 🔍 v10 获得了此前缺失的 Gateway 铁证：synthesis 约 154.5s 后返回 actual model `claude-opus-4-8`、`stop_reason=end_turn`、`content_block_types=["text"]`、response 211 bytes、usage 只有 `input_tokens=19,422` 且无 output tokens；text block 为空或纯空白，故触发 `no_text_content`。这排除了 thinking-only、max_tokens、模型漂移和 360s timeout，600s 不能解决该 provider 空文本响应。
- ⚠️ 当前 no-text 策略仍将所有无文本响应视为单次 fail-closed；v10 表明“actual model 匹配 + end_turn + 只有空 text block + output_tokens=0”更像可重试的瞬时 provider 空响应，而 thinking-only/max_tokens 等其它 no-text 形状仍不应重试或当正文。
- ✅ 精确 empty-text 瞬时重试已实现：只有 strict actual-model 校验已开启、requested/actual model 同族匹配、`stop_reason=end_turn`、唯一 block type 为单个 `text`、`input_tokens>0` 且 `output_tokens=0` 时，synthesis 才允许至多一次完整原请求 `retry`。空 content、重复/混合 block、thinking-only、`max_tokens`、无 usage、model mismatch、transport timeout 和确定性 4xx 均不触发该重试。
- ✅ synthesis 请求状态机明确区分 `initial / retry / repair` 并服从原 `max_retries` 总预算：0 次重试预算最多 1 请求，1 次预算最多 `initial+retry`，2 次以上最多 `initial+retry+repair`。连续两次空文本 fail-closed；retry 后完整内容校验失败才可 repair；repair 无文本或 retry transport timeout 均不会再开新分支。retry 复用冻结的原 model/messages/max_tokens/timeout，不回灌空响应、thinking 或失败长草稿。
- ✅ attempt ledger 与成本账本补齐：失败路径最多保存 3 条安全 ledger；成功恢复路径保存先前失败 ledger 和 `final_request_kind=retry|repair`，因此 artifact 可审计重试而不保存成功正文。所有携带安全 usage 的 empty-text、validation 和 strict model mismatch 失败均计费；Gateway mismatch/no-text 异常在删除 raw payload 后仅保留规范化 usage 与聚合元数据。
- ✅ content block type 聚合改为有界保序并保留重复，超过 32 块追加 `truncated` 哨兵；因此只有精确 `("text",)` 才符合重试形状，两个空 text 块不会被集合去重误判。固定 fake-response 覆盖 empty→成功、连续 empty、empty→invalid→repair、预算 0/1、thinking/max_tokens/empty/mixed/duplicate blocks、timeout、model mismatch、4xx、失败 usage 和 ledger 无正文。
- ✅ 本阶段主线验收：定向相关测试 `145 passed`；全量 `421 passed / 1 个既有 Starlette warning`。仓库级 ruff、compileall、`git diff --check` 全部通过，`uv.lock` 无 diff。
- ⚠️ 全新独立只读评审发现 3 项 commit 前阻塞：① deep Markdown 无条件保留标题/表头，事实型结构行可绕过 claims/CitationChecker；② `deep + text|json` 被 schema 允许，但 prompt 固定 `claims=[]` 且确定提取只支持 Markdown，契约自相矛盾；③ Gateway 任意 `stop_reason` 会原样进入安全 ledger，畸形响应可借此夹带正文。精确 empty retry 状态机、失败 usage、denylist 与 DRB II 隔离本身评审通过。
- ✅ 三项评审问题已完成修复：事实型标题（含带 citation）会被删除，事实/带引事实表头直接 fail-closed，普通章节名与 DRB II 两张必答表的完整列名保留；`ResearchRequest`/`ResearchBrief`/评测 case 均明确 deep 只允许 Markdown，concise 继续兼容 text/markdown/json，deep synthesis/repair prompt 同步；`stop_reason` 仅保留协议白名单、未知记 `unknown`，actual model 只保留有界协议标识，原始私密文本不进异常/repr/ledger。另补非 synthesis brief exact-empty 原请求 retry 测试。
- ✅ 评审修复定向验收 `149 passed`；修复后最终全量 `435 passed / 1 个既有 Starlette warning`。仓库级 ruff、compileall、`git diff --check` 全绿，`uv.lock` 无 diff，且无残留评测进程。知识库与面试问答已同步 deep Markdown-only、结构行引用边界和安全 stop-reason 口径。
- ⚠️ 第二次独立聚焦复核确认 deep Markdown-only 已关闭，但发现 2 个剩余绕过：名词化事实标题/表头（如 `Vietnam: Universal private-sector pension coverage`）不含谓词黑名单，仍可当结构保留；`response_model_matches` 接受任意 `requested-model-*` 后缀，恶意同前缀文本可作为 actual model 进入 no-text ledger。
- ✅ 第二次聚焦复核的两个绕过已完成收口：deep Markdown 结构标签改为通用结构/字段、query 连续主题短语和 query 命名实体组合的正向授权；`Universal/full/complete/mandatory` 等断言修饰词只有在用户 query 明确连续声明时才能作为结构。实际攻击标题 `Vietnam: Universal private-sector pension coverage` 被删除，等价事实型表头 fail-closed；task2+ Table 1 八列、Table 2 三列与另一 DRB 主题自定义表头均保留。
- ✅ Gateway actual-model 匹配仅接受 exact 或经过真实年月校验的 `YYYYMM` / `YYYYMMDD` 后缀；恶意同前缀、非法年月/日期和日期后追加文本均 mismatch，安全异常/ledger 只记 `unknown`，合法 Opus 日期 alias 与 Kimi 月份 alias 保持兼容。
- ✅ 主线程复验定向相关测试 `189 passed`；相关 ruff、compileall、`git diff --check` 全绿，`uv.lock` 无 diff。全新独立只读复核已完成，明确未发现阻塞问题：两条结构攻击、五类结构正例、actual-model alias/metadata、unknown stop reason、initial→retry→repair 及不重试形状均独立复现通过；ledger 不含 response/thinking 正文，失败 usage 正确计费。
- ✅ 最终全量验收：`.venv/bin/python -m pytest -q` 为 `439 passed / 1 个既有 Starlette warning`；仓库级 ruff、compileall、`git diff --check` 全绿，`uv.lock` 无 diff。知识库与面试问答已同步正向结构授权和 exact/日期 alias 安全口径。尚未运行 v11、未 commit/push。
- 🚧 下一步入口：新建唯一 v11 run dir，按 `task2+ × claude-opus-4-8`、2 rounds / 2 calls / 300s researcher deadline / 240s request timeout / 600s synthesis timeout / 10k output / crawler concurrency 2 / strict single-model / no external judge 复跑并严格审计。只有 v11 结论与评审问题全部落盘后才中文 commit、push GitHub；报告合格前不启动四模型或 12×4。
- ⚠️ 第十一次锚点运行：`~/.deepresearch-agent-eval/runs/drb2-anchor-smoke-opus48-v11-20260716T040153-33779`。进程正常退出，但 `execution_success=false / report_emitted=false / substantive_answer=false / grounded_answer=false / final_result_usable=false / evidence_abstention=false`，没有报告、claims 或可执行结构/引用审计，不能记为 Deep Research 报告成功。
- ✅ v11 检索与安全边界生效：Q1/Q2/Q3/Q4/Q5 分别保留 4/3/7/2/1 条 evidence，去重后 17 个干净来源全部进入 synthesis context，estimated context 14,766 tokens、无 dropped/injection source；Q1/Q2/Q5 的 denylist 命中均为成功拦截，所有分支与最终 case `benchmark_contamination=false`。run trace 中 47 个 actual-model 审计值唯一为 `claude-opus-4-8`，strict single-model 未见漂移。
- 🔍 v11 没有复现 v10 的精确空 text：唯一 synthesis 约 505.5s 后返回 actual model 匹配、`stop_reason=tool_use`、`content_block_types=[thinking, tool_use]`，usage 为 input 22,570 / output 4,602 / cache creation 492 / cache read 23,097。它不满足 `end_turn + 单一空 text + output=0` 白名单，所以只保留一条 `request_kind=initial` ledger 并 fail-closed；未发起 retry/repair、未把 thinking/tool_use 当报告正文，sanitizer 因无正文未执行。
- ⚠️ v11 总耗时 892.4s、142,319 tokens。coverage 仍不完整，尤其 Q5 只覆盖 ADB、Q1/Q2/Q4 缺多数国家/方案/公式字段；但本次最终阻塞首先是 provider 返回 thinking+tool_use 而无 text，不是 context 超限、模型漂移、污染或 citation sanitizer。报告未产出，因此两张必答表、多章节、多来源比较与逐项 citation 均应记“未达到/不可审计”，不能给结构质量分。
- ✅ 独立只读 v11 artifact 审计未发现口径问题：raw/summary/stdout 状态一致；35 条成本记录与 142,319 tokens 对平；trace 47 次、成本账本 35 次 actual model 唯一都是 `claude-opus-4-8`；3 个 denylist 命中、5 个实际拦截候选均未越过 evidence 边界，ledger 无 response/thinking 正文泄漏。
- 🚧 下一步入口：重新跑最终静态验收后中文 commit 并 push。当前仍禁止四模型同锚点和 12×4；若继续恢复 provider no-text，必须先取得 `tool_use` 无文本响应的协议证据，不能扩大 empty-text retry 或读取 thinking/tool body。
- ✅ 最终代码与评测基线已以中文 commit `bcbff65 重构：完善 DRB II 深度报告评测与网关失败恢复` 提交并推送到 `origin/master`；工作树保持干净。v11 的 fail-closed 结论已同步到本文件、`KNOWLEDGE_BASE.md`、`INTERVIEW_QA.md` 与 AI 改动记录，后续若继续只围绕 provider `tool_use` 无文本证据推进，不启动四模型或 12×4。

## 已定决策及原因

- 凭证只进 Keychain，运行时注入环境变量；仓库仅保存环境变量名和非敏感配置。
- 真实质量不能继续使用旧 `success` 或单一引用重叠指标；必须保留标准答案、完整来源、论断、证据、独立裁判和失败分类。
- 不把同一个生成模型的自评当最终质量结论；至少使用另一个模型做独立裁判，并保留人工可复查产物。
- 优先修复搜索覆盖、结构化格式和引用/事实支撑，再考虑增加更多导出或基础设施功能。
- ~~snippet 降级证据保留了诚实性~~：该历史决策已撤销。搜索摘要不是可引用正文；现在只保留脱敏失败候选审计信息，snippet 不进入最终证据、合成上下文或引用。
- circuit breaker 阈值放宽但不取消：4 次连续失败仍触发熔断，10 秒冷却后重试——防止完全不降级同时避免过度敏感。

## 遗留问题

- 四模型最大上下文长度尚未做边界压测；本轮通过显式 context budget 避免依赖未经证实的上限。
- Kimi gateway-web 搜索偶尔返回 HTTP 400（可能是查询格式不兼容），需关注 v4 复现率。
- 引用裁判仅衡量"论断是否被给定证据支撑"，不等于事实核查；答案正确性必须由独立 answer judge 和人工抽查补充。
