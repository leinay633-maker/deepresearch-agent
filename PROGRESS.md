# DeepResearch Agent 真实模型与质量优化进度

## 总目标

使用公司 LLM Gateway 已授权模型，把当前“历史真实运行、最新版仅离线模拟”的状态升级为：最新版能够稳定调用真实模型和真实搜索，使用固定题集与独立裁判反复评测，并针对失败样本迭代到可诚实展示的高质量水平。

## 阶段拆解

- ✅ 阶段一：网关与模型协议核对，安全接入凭证，建立真实请求探针。
- 🚧 阶段二：实现公司网关模型提供方和可用的真实搜索路径，补安全边界与测试。
- 🚧 阶段三：冻结真实质量基线，使用多模型独立裁判和人工可审计产物评估。
- ⬜ 阶段四：针对检索、结构化输出、引用、上下文和失败恢复问题迭代。
- ⬜ 阶段五：完整复测、独立代码审查、中文提交、清理本进度文件与指针。

## 当前状态

- ✅ 用户提供的新 LLM Gateway Key 已安全存入 macOS Keychain 服务 `deepresearch-agent-llm-gateway`，未写入仓库或日志。
- ✅ App ID 为 `datacenter.ab.flow-config-online`，可用模型包括 `claude-4.6-opus`、`kimi-k2.7-code-highspeed`、`glm-5.2`、`claude-opus-4-8`。
- ✅ 本机网关 HTTPS 入口已确认：Anthropic Messages 线为 `https://llmapi.bilibili.co/v1/messages`；Codex Responses 线为 `https://llmapi.bilibili.co/v1/responses`。携带 Bearer 的非回环 HTTP 现已 fail closed。
- ✅ 历史证据已核对：旧版真实 DeepSeek + Wikipedia 共 7 轮 5 题，另有 1 条 LiveDRBench；2026-07-12 优化后的 24 题基线全部是 mock，最新版真实质量尚未验证。
- ✅ 四模型 Anthropic Messages 探针均返回 HTTP 200，且能按要求输出严格 JSON：`claude-4.6-opus`、`kimi-k2.7-code-highspeed`、`glm-5.2`、`claude-opus-4-8`。Claude 返回 text；Kimi/GLM 还会返回 thinking block，接入层只解析 text block。
- ✅ usage 字段实测存在，但不同模型字段形状不同；统一记录 input/output/cache token，不伪造网关价格。
- ✅ 新增 `llm-gateway` Anthropic Messages provider，四模型真实生成链路已跑通；结构修复、thinking block 解析和 token 记账已覆盖。
- ✅ 新增 `gateway-web` 搜索 provider，已真实命中 Python 官方发布页、官方文档和 PEP；Bing RSS 仅作 fallback 候选 URL 发现。
- ✅ 已补 URL/SSRF 策略、重定向重验、MIME/大小限制、间接 prompt injection 过滤、全局上下文预算和域名多样性。
- ✅ Python 官网最新版本题真实答对 `Python 3.14.6`，引用 grounding 为 `0.6667`；已收紧合成提示和裁判 source URL，待复跑确认。
- ⚠️ 换用 `gateway-web + html crawler` 复跑 Python 题（run `3447be69aca4`）：版本号仍正确，抓到 9 个官方来源，但 GLM 把简单事实扩写成 6 个论断，日期归属/页面位置/JavaScript 说明出现过度断言，grounding 降至 `0.5`。
- ✅ 已实现合成约束与验证后修复：最多 3 个原子论断、中文问题强制中文答案、部分支撑论断收窄重写、仍未完全支撑的论断不进最终答案；新增专项测试已通过。
- ✅ 安全搜索与合成修复后再复跑 Python 题（run `0a138b33b862`）：答案 `Python 3.14.6`，中文输出，3 个论断全部被 python.org 正文支撑，citation grounding / precision / coverage 均为 `1.0`，无 mock、无 Bing fallback。其中一个并发分支有两个候选页抓取超时/重置，但仍保留 3 份可用正文，属部分抓取降级而非答案降级。
- ⚠️ 独立搜索安全审查发现 4 个阻断项：默认 HTTP 携带密钥、全局搜索摘要被复制为每个 URL 的证据、`to_thread` 超时后后台请求仍继续并与重试重叠、Gateway 失败时未真正回退 Bing。隐藏集已暂停，正在修复。
- ✅ 仓库外 SimpleQA 隐藏验收集已冻结：16 题，SHA256 `af44c266401717fe7cc61ccad782474c0a17070b84100e1a65186cb71a7a40c6`；主会话未读题目内容。
- 🚧 下一步入口：准备与隐藏集零重叠的 8 题 SimpleQA 开发集，先跑公开可诊断基线并修普适问题；再跑 16 题隐藏集，用 Kimi 与 Claude/Opus 独立 replay 裁判。

## 已定决策及原因

- 凭证只进 Keychain，运行时注入环境变量；仓库仅保存环境变量名和非敏感配置。
- 真实质量不能继续使用旧 `success` 或单一引用重叠指标；必须保留标准答案、完整来源、论断、证据、独立裁判和失败分类。
- 不把同一个生成模型的自评当最终质量结论；至少使用另一个模型做独立裁判，并保留人工可复查产物。
- 优先修复搜索覆盖、结构化格式和引用/事实支撑，再考虑增加更多导出或基础设施功能。
- 将网关 server web search 封装成只负责发现候选 URL 的独立 adapter；模型生成、URL 发现和安全正文抓取仍解耦，便于限制工具循环并保存原始证据。
- 真实评测暂用 brief/decision/synthesis=`glm-5.2`、planner=`kimi-k2.7-code-highspeed`、citation judge=`glm-5.2`；answer judge 用 Kimi 与 Claude/Opus 交叉判分。

## 遗留问题

- 四模型最大上下文长度尚未做边界压测；本轮通过显式 context budget 避免依赖未经证实的上限。
- 广域真实搜索优先评估现有 Jina/Tavily/Brave 路径；本地 HTML crawler 在完整 SSRF 防护前不得用于 live profile。
- ✅ pytest 慢测试根因是默认 hybrid RAG 在 API 测试中并发下载/初始化 HuggingFace embedding；已将单测默认隔离为 keyword，不影响显式 hybrid 测试。cassette 两条失败由缺省 `search_query` 被意外改写引起，已修复为初始检索保持原子问题语义。
- 引用裁判仅衡量“论断是否被给定证据支撑”，不等于事实核查；答案正确性必须由独立 answer judge 和人工抽查补充。
