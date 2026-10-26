# LLM API 差异矩阵

> 用途：W1 Day3 写 adapter 的唯一依据。**每格必须有结论句**——JSON 只是证据，没有结论等于没填。
> 填写原则：**只记录稳定核心路径**。beta 特性单独标注、W1 一律不实现（如 Anthropic 的 mid-conversation system messages / inline tools）。
> 标记约定：✅ W1 必需 ｜ ⏳ 后续周次需要 ｜ ❌ 与本项目无关

---

## 文档索引

| 章节 | 文件 | 状态 |
|---|---|---|
| 0. 四家总览（横向对比） | 本文件 §0 | ✅ |
| 1. Anthropic | [`Anthropic.md`](./Anthropic.md) | ✅ |
| 2. OpenAI | [`OpenAi.md`](./OpenAi.md) | ✅ |
| 3. DeepSeek | [`Deepseek.md`](./Deepseek.md) | ✅ |
| 4. Qwen（百炼 compatible-mode） | [`Qwen.md`](./Qwen.md) | ✅ |
| 5. 对架构的回填 | 本文件 §5 | ✅ **Day2 前必须逐条确认** |

---

## 0. 四家总览（横向对比，结论级）

| 维度 | Anthropic | OpenAI | DeepSeek | Qwen |
|---|---|---|---|---|
| 兼容协议 | **自有协议，无 OpenAI 兼容端点** | 原生 | OpenAI 兼容（另有 `/anthropic` 端点） | OpenAI 兼容（compatible-mode） |
| base_url | `https://api.anthropic.com` | `https://api.openai.com/v1` | `https://api.deepseek.com`（`/v1` 为别名） | `https://dashscope.aliyuncs.com/compatible-mode/v1`（地域/工作空间三段式） |
| 模型命名 | 版本化 ID | 版本化 ID | **只有 `deepseek-flash` / `deepseek-v4-pro`** | 别名 + **日期快照**（评测须钉快照） |
| system 位置 | **顶层 `system` 参数** | messages[0] | messages[0] | messages[0] |
| 对话中间插 system | 支持（beta） | 不支持 | **支持**（Chat 接口） | 不支持 |
| tool 定义位置 | **顶层 `tools` 数组** | `tools`（function 包裹） | 同 OpenAI | 同 OpenAI |
| tool 结果回填 | **user 消息内 `tool_result` block** | `role: "tool"` + `tool_call_id` | 同 OpenAI | 同 OpenAI |
| tool 参数类型 | **`input` 已是对象** | `arguments` 是 JSON **字符串** | 同 OpenAI | 同 OpenAI |
| 流式协议 | SSE，`event:` + `data:` 双字段，事件有名 | SSE，仅 `data:` | 仅 `data:`，**另有 `: keep-alive` 注释行** | 仅 `data:` |
| tool 参数流式拼装 | 按 `content_block index` 累 `partial_json` | 按 `index` 累 `arguments` 字符串 | 同 OpenAI | 同 OpenAI（**须实测确认是否真增量**） |
| **流式 usage 位置** | input 在 `message_start`、output 在 `message_delta` | **独立 usage 片，`choices: []`** | **挂在最后内容块上**（`choices` 仍 1 元素、`finish_reason` 非 null） | 同 OpenAI（独立片） |
| usage 明细 | `cache_read_input_tokens` | `cached_tokens`、`reasoning_tokens` | **`hit`/`miss` 分价** + `reasoning_tokens` | 仅三项，**无缓存明细** |
| 必填差异 | **`max_tokens` 必填** | 可选 | 可选（默认 8K / 思考 64K） | 可选（建议显式设） |
| `tool_choice` 形态 | 对象 `{"type":"auto"}` | 字符串 `"auto"` | 字符串 + 指定函数 | 字符串 + 指定函数 |
| **`tool_choice: required`** | 支持（`"any"`） | 支持 | **思考模式下 400** | **Qwen 系列不支持** |
| 并行工具调用 | `disable_parallel_tool_use` | `parallel_tool_calls`，**默认 true** | **无此参数**（不可控） | `parallel_tool_calls`，**默认 false** |
| 思考模式开关 | `thinking: {type, budget_tokens}` | `reasoning_effort` | **`thinking` 默认 enabled**；`reasoning_effort` none/low/high/max | `enable_thinking`（**非流式 400**）、`thinking_budget` |
| 思考内容字段 / 回传 | `thinking` block + `signature`（须原样回传） | 不返回 | **`reasoning_content` 必须完整回传**（带 tools 时，否则 400） | `reasoning_content`；3.7 系列**不回传**，3.8 系列**必须回传** |
| `finish_reason` 额外值 | `pause_turn` / `refusal` | — | **`insufficient_system_resource` / `aborted`** | — |
| 限流口径 | RPM/TPM + 529 | RPM/TPM | **按并发数**（2500 / 500），无 `retry-after` | **主账号合并** RPM/TPM，429 三种语义 |
| **欠费/额度耗尽落在哪** | 400 `credit balance` | **429 `insufficient_quota`** | **402** | **400/403 `Arrearage`** |

**由此推出的 adapter 结论**：`openai_compat` 能覆盖三家（OpenAI/DeepSeek/Qwen），Anthropic 必须单写——不是因为风格不同，而是协议层结构不同（system 顶层、tool_result 走 user、参数已是对象）。

**但"能覆盖"≠"没有分支"**。三家差异集中在四处，全部要在 `openai_compat` 里做成**数据驱动**（`models.yaml` 能力标记），而不是 `if provider == "qwen"`：

| 差异点 | 三家的取值 | 承载方式 |
|---|---|---|
| 流式 usage 形状 | 独立片（OpenAI/Qwen） vs 挂末块（DeepSeek） | **统一规则：任何 chunk `usage` 非空就覆盖式记录**，不看 `choices` |
| `parallel_tool_calls` 默认值 | true（OpenAI）/ 无（DeepSeek）/ **false（Qwen）** | `models.yaml` 显式声明，**永远显式发送** |
| `tool_choice` 支持度 | 全支持 / 思考模式受限 / `required` 不支持 | 能力标记 + **降级并记日志**，不抛异常 |
| 欠费状态码 | 429 / 402 / 400-403 | 按 **`code` 字符串**白名单，不按状态码 |



## 5. 对架构文档的回填（写 adapter 前必须确认）

| 契约 | 本次核实结果 |
|---|---|
| `ToolCall.arguments` 恒为 dict（ADR-001） | ✅ 成立：Anthropic 原生是对象，OpenAI 系是字符串 → 统一在 adapter 内 `json.loads` |
| tool_call 由 adapter 组装完整后单次发出（ADR-002） | ✅ 成立：Anthropic 的 `partial_json` 与 OpenAI 的 `index` 累加是两套逻辑，只能关在 adapter 里 |
| `Message.role` 用扁平四值 | ⚠️ 需要一条隐含约定：**多条连续 `tool` 消息 → 一条 Anthropic user 消息**。已在 1.4 明确，W1 实现时必须写进 types 的文档字符串 |
| `finish_reason` 枚举 | ⚠️ **必须扩展，不能只映射四值**：Anthropic 多 `pause_turn` / `refusal`；DeepSeek 多 **`insufficient_system_resource`**（资源不足，**不可当 `stop`**）与 **`aborted`**（中断）。建议内部值：`stop / length / tool_calls / content_filter / aborted / upstream_error`，`pause_turn → stop`、`refusal → content_filter`、`insufficient_system_resource → upstream_error` |
| 内部 tool 结果消息形态 | ✅ 已定：采用 OpenAI 形态（**每条结果一条 `role:"tool"` 消息**），Anthropic adapter 负责合并成一条 user 消息。合并逻辑只放 `providers/anthropic.py`，types 层保持 1:1 |
| **ADR-006（建议新增）** | 协议基准选 `/chat/completions`，**不实现 Responses API**。理由：四家兼容层全部对齐该形态。**已知限制**：OpenAI 自 gpt-5.4 起在 `reasoning:"none"` 下 Chat Completions 不支持 tool calling，gpt-6-astra 的 tool calling 仅 Responses 支持 → 选模型需避开这类型号；W9 若确有需要，新增 `providers/responses.py`，client/router/types 零改动 |
| **ADR-007（建议新增，本周必须落）** | `Message` 增加 **`provider_meta: dict \| None`** 透传字段，承载"框架不解释、但必须原样还给上游"的数据：DeepSeek 的 `reasoning_content`、Anthropic 的 `thinking` block + `signature`、Qwen 3.8 的 `preserve_thinking` 输入。**约束**：业务层只整体传递、不得构造或解读；由 adapter 写入与读取；不参与路由与评测断言 |
| **流式 usage 统一规则（写进 adapter 基类）** | 四家三种形状（独立片 / 挂末块 / 事件内嵌）→ **任何 chunk 的 `usage` 非空就覆盖式记录**，禁止用 `choices == []` 作为读取条件。否则 DeepSeek 永远读不到 usage，成本埋点静默归零 |
| **`tool_choice` 与并行能力** | 必须在 `models.yaml` 声明 `supports_tool_choice_required` / `parallel_tool_calls`，**不支持时降级为 `auto` 并记 `tool_choice_downgraded`，不抛异常**。三种默认值（true/无/false）不可假设 |
| **错误映射** | **按 `code` 字符串白名单，不按状态码**：`insufficient_quota`(429) / `Arrearage`(400,403) / `402` / DeepSeek `422`(参数错误) 四类全部要显式收口，否则"余额耗尽"会被当参数错误或未知 5xx 重试 |
| **模型版本策略（评测前置条件）** | W3 起**必须钉日期快照**（Qwen `qwen3.7-plus-2026-05-26` 这类）。滚动别名会让第 6 周的分数与第 2 周不可比，12 周指标曲线失去因果性 |
