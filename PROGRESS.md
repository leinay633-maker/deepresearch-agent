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
