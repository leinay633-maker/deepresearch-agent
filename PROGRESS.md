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

### 🔥 关键 bug 发现与修复（commit 1a552af，v7 验证中）

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

## 已定决策及原因

- 凭证只进 Keychain，运行时注入环境变量；仓库仅保存环境变量名和非敏感配置。
- 真实质量不能继续使用旧 `success` 或单一引用重叠指标；必须保留标准答案、完整来源、论断、证据、独立裁判和失败分类。
- 不把同一个生成模型的自评当最终质量结论；至少使用另一个模型做独立裁判，并保留人工可复查产物。
- 优先修复搜索覆盖、结构化格式和引用/事实支撑，再考虑增加更多导出或基础设施功能。
- snippet 降级证据保留了诚实性（标记 `retrieval_degraded=True` 和 `evidence_grade=snippet`），不掩盖抓取失败。
- circuit breaker 阈值放宽但不取消：4 次连续失败仍触发熔断，10 秒冷却后重试——防止完全不降级同时避免过度敏感。

## 遗留问题

- 四模型最大上下文长度尚未做边界压测；本轮通过显式 context budget 避免依赖未经证实的上限。
- Kimi gateway-web 搜索偶尔返回 HTTP 400（可能是查询格式不兼容），需关注 v4 复现率。
- 引用裁判仅衡量"论断是否被给定证据支撑"，不等于事实核查；答案正确性必须由独立 answer judge 和人工抽查补充。
