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

### v4 修复（2026-07-14 已合并）

1. **`_string_array` 容忍字符串输入**：Opus 输出单字符串时按分号拆分为数组
2. **`_brief_from_payload` scope 默认值**：空/缺失 scope 不再抛错，用默认研究范围
3. **规划实体检查放宽**：实体锚点增加全大写缩写（`[A-Z]{2,}`）和数字实体（`\w*\d+\w*`）；交集门槛从 ≥2 降为 ≥1
4. **Circuit breaker 阈值**：failure_threshold 2→4，cooldown 30s→10s
5. **Gateway-web snippet 降级证据**：当所有 crawl 失败但有 snippet 内容时，标记为 `evidence_grade=snippet` 作为降级证据而非直接抛错
6. **主搜索方法适配**：crawl_errors 分支识别 snippet evidence，不再对有内容的降级证据抛 SearchError

新增 7 个专项测试，总测试 267 passed。

### v4 评测进行中

- 🚧 四模型并行重跑 8 题公开开发集
- 输出到：`~/.deepresearch-agent-eval/runs/single-model-dev-v4-20260714T034005`
- 下一步：完成后用 Kimi + Claude Opus 4.8 双裁判独立评分

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
