## 1. Anthropic

### 1.1 鉴权 ✅

| 项 | 值 |
|---|---|
| 主用 header | `x-api-key: <ANTHROPIC_API_KEY>` |
| 版本 header | `anthropic-version: 2023-06-01`（**必填**，缺失直接 400） |
| 内容类型 | `content-type: application/json` |
| beta 开关 | `anthropic-beta: <flag>`，多个用逗号分隔（**W1 不用**） |
| 备用凭证 | `Authorization: Bearer <token>`，用于 OAuth token（Claude Code 订阅态），**不是 API key 的"旧写法"** |

**结论**：adapter 固定注入 `x-api-key` + `anthropic-version` 两个 header；不实现 beta 开关，但请求构造处留一个 `extra_headers` 口子，W8 接 Skills/MCP beta 时不用改签名。

**坑**：漏 `anthropic-version` 的报错信息是 `invalid_request_error`，很容易被误判成 key 错误。

### 1.2 端点 ✅

| 端点 | 用途 | 标记 |
|---|---|---|
| `POST /v1/messages` | 对话 + 工具调用（**唯一需要**） | ✅ |
| `POST /v1/messages/count_tokens` | 预计算 token，做预算控制 | ⏳ W7 |
| `POST /v1/messages/batches` | 批量异步，成本 50% | ⏳ W9（离线批量评测） |
| `GET /v1/models` | 列模型，用来生成 `models.yaml` | ⏳ |
| `/v1/files`、`/v1/skills` | 文件与技能管理（beta） | ❌ |

**结论**：W1 只实现 `/v1/messages` 一个端点；**没有 `/chat/completions` 兼容端点**，这就是 `providers/anthropic.py` 必须独立存在的根本原因。

### 1.3 系统消息 ✅

- **顶层 `system` 参数**，类型为 string 或 block 数组；block 数组可挂 `cache_control`（prompt caching 用）
- `messages` 里 `role` 只有 `user` / `assistant`，**放 `system` 会 400**
- 请求体必填：`model`、`messages`、`max_tokens`（Anthropic 独有，**其他三家可选**）

```json
{
  "model": "claude-sonnet-4-5",
  "max_tokens": 1024,
  "system": "你是一个严谨的技术编辑。",
  "messages": [{"role": "user", "content": "用三句话解释 RRF 融合。"}]
}
```

**结论**：内部 system 消息 → 顶层 `system`；多条 system 合并（用 `\n\n` 拼字符串，或合并为 block 数组，二选一后固定下来）。

**坑**：`openai_compat` 里"messages[0] 是 system"的假设**不能带进本 adapter**，否则 system 会被当成普通用户消息，行为静默错乱（不报错，但指令失效——比报错更难查）。

> beta 特性（**不实现**）：`mid-conversation-system-messages` 允许在 `messages` 里插入 `role: "system"` 做对话中途指令注入。与本维度不是一回事。
> 文档：https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages

### 1.4 多轮中 tool 结果 ✅（**差异最大，最容易写错**）

流程：assistant 返回 `tool_use` block → **下一条 `user` 消息**的 content 数组里放 `tool_result` block。

```json
{
  "role": "user",
  "content": [
    {"type": "tool_result", "tool_use_id": "toolu_01A09q90qw90lq917835lq9", "content": "{\"count\": 1284}"},
    {"type": "tool_result", "tool_use_id": "toolu_01Bxx...", "content": "timeout", "is_error": true}
  ]
}
```

**硬约束（违反即 400）**：

1. 上一条 assistant 消息里**每个** `tool_use` 都必须在下一条 user 消息里有**对应** `tool_result` —— 少一个就报错
2. 多个 `tool_use` 的 `tool_result` **必须一次性给齐**，不能分两条消息补
3. `tool_use_id` 必须来自上一条 assistant 的返回，不能自造
4. 失败用 `is_error: true`，`content` 写错误信息（模型能据此调整策略）
5. `tool_result` 可与 `text` block 混排在同一条 user 消息里

**结论**：内部 `role: "tool"` 消息**不能一对一映射**成一条 Anthropic 消息，必须**先把连续的多条 tool 消息合并成一条 user 消息**再发。这是归一化里唯一需要"改写消息结构"的地方，不是简单字段改名。

**坑**：把 3 个 tool 结果拆成 3 条 user 消息 —— OpenAI 思维下的自然写法，Anthropic 直接 400。

### 1.5 tool 定义 ✅

**顶层 `tools` 数组**（不是放进 system，也不是放进 messages）：

```json
{
  "tools": [{
    "name": "db_query",
    "description": "对分析库执行只读 SQL。当用户询问历史数据统计时使用；不要用于写入。",
    "input_schema": {
      "type": "object",
      "properties": {"sql": {"type": "string"}},
      "required": ["sql"],
      "additionalProperties": false
    }
  }],
  "tool_choice": {"type": "auto"}
}
```

| 字段 | 说明 |
|---|---|
| `name` | snake_case，全局唯一 |
| `description` | **写"什么时候用/什么时候别用"**，直接决定选择准确率 |
| `input_schema` | JSON Schema，顶层必须 `type: object` |
| `tool_choice` | `{"type":"auto"}` / `{"type":"any"}` / `{"type":"tool","name":"x"}`；**OpenAI 的同名字段是字符串，必须映射** |
| `disable_parallel_tool_use` | 是否禁并行（少量场景需要，默认开） |

**结论**：`ToolSpec.parameters` → `input_schema`，字段名改一个；`tool_choice` 形态两家不同，需在 adapter 内转换。

> beta（**不实现**）：`inline-tools-2026-09-15` 允许在 `tool_addition` block 里按值定义工具，`mid-conversation-tool-changes-2026-07-01` 允许按引用增删工具。两者都要求工具**已在顶层 tools 声明**，否则 400 `tool_reference_unresolved`。

### 1.6 tool 调用响应 ✅

```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "stop_reason": "tool_use",
  "content": [
    {"type": "text", "text": "我先查一下昨天的订单量。"},
    {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9",
     "name": "db_query",
     "input": {"sql": "SELECT COUNT(*) FROM orders WHERE shipped_at::date = CURRENT_DATE - 1"}}
  ],
  "usage": {"input_tokens": 812, "output_tokens": 64}
}
```

**结论（W1 最关键的一条）**：

- **`input` 已经是 dict**，不需要 `json.loads`；而 OpenAI 的 `arguments` 是 JSON **字符串**。→ 这就是 ADR-001「`ToolCall.arguments` 恒为 dict」的由来，也是 conformance 断言的核心
- 工具调用时 `stop_reason == "tool_use"` → 映射内部 `finish_reason = tool_calls`
- `content` 是数组，**可能同时含 text 和 tool_use**（模型先说话再调工具）→ 内部 `LLMResponse.text` 要把所有 text block 拼起来，不能只取第一个
- `id` 形如 `toolu_*`，直接作内部 `ToolCall.id`

### 1.7 流式 ✅

请求加 `"stream": true`，响应为 SSE，**每条事件有 `event:` 名和 `data:` 体**（OpenAI 只有 `data:`，事件类型藏在 JSON 的 `object` 字段里）。

```text
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","usage":{"input_tokens":812,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_...","name":"db_query","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"sql\":\"SELECT COUNT"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":64}}

event: message_stop
data: {"type":"message_stop"}
```

事件全集：`message_start`、`content_block_start`、`content_block_delta`、`content_block_stop`、`message_delta`、`message_stop`，另有 `ping`（保活，**必须忽略**）和 `error`（流内错误）。

| 内部 StreamEvent | 来源 |
|---|---|
| `text_delta` | `content_block_delta` 且 `delta.type == "text_delta"` → 取 `delta.text` |
| `tool_call`（完整） | 同一 `index` 的 `input_json_delta.partial_json` **逐段字符串拼接** → `content_block_stop` 时 `json.loads` → 发一次完整 ToolCall |
| `usage` | `message_start.message.usage.input_tokens` + `message_delta.usage.output_tokens` |
| `done` | `message_stop`（`stop_reason` 取自 `message_delta.delta.stop_reason`） |
| `error` | `event: error` |

**结论**：必须维护 `index → {tool_use_id, name, buffer}` 的累加表，`content_block_stop` 才收口。`partial_json` 是**碎片字符串**，拼接前不能解析（会得到不完整 JSON）。

**坑**：

- 忽略 `ping` 会导致解析器把心跳当业务事件
- `message_start` 里 `output_tokens` 是 1（占位），不是真实值，别拿去计费
- 把 `partial_json` 每段单独 `json.loads` —— 必然解析失败

### 1.8 usage ✅

| 场景 | 字段位置 |
|---|---|
| 一次性 | `response.usage = {input_tokens, output_tokens, cache_creation_input_tokens?, cache_read_input_tokens?}` |
| 流式 | `input_tokens` 在 `message_start`；`output_tokens` 在 `message_delta`（**累计值，取最后一次**） |

**结论**：`cost = input_tokens × price_in + output_tokens × price_out`。
缓存字段（`cache_creation_input_tokens` / `cache_read_input_tokens`）W1 **只记录不参与计价**——四个字段的计价关系容易算错，等 W7 做成本优化时再精算。

**坑**：Anthropic 的 `input_tokens` **不含**缓存命中部分，直接乘单价会低估成本结构；W7 要用 `cache_read` 单独计价。

### 1.9 限流 / 错误 ✅

**限流 header**（响应里返回，直接用于背压）：

`anthropic-ratelimit-requests-limit` / `-remaining` / `-reset`，以及 `-tokens-*`、`-input-tokens-*`、`-output-tokens-*`，另有 `retry-after`。

**错误体**：

```json
{"type": "error", "error": {"type": "rate_limit_error", "message": "..."}}
```

| HTTP | `error.type` | 内部异常 | 可重试 |
|---|---|---|---|
| 400 | `invalid_request_error` | `BadRequestError` | ❌ |
| 401 | `authentication_error` | `UpstreamAuthError` | ❌ |
| 403 | `permission_error` | `UpstreamAuthError`（或权限类） | ❌ |
| 404 | `not_found_error` | `BadRequestError` | ❌ |
| 413 | `request_too_large` | `BadRequestError` | ❌ |
| 429 | `rate_limit_error` | `RateLimitError` | ✅ |
| 500 | `api_error` | `UpstreamServerError` | ✅ |
| **529** | **`overloaded_error`** | `UpstreamServerError` | ✅ |

**结论**：**529 是 Anthropic 独有的"过载"状态码**，不在标准 HTTP 语义里，必须显式归为可重试，否则高峰期会被当成致命错误直接失败。**且 529 常不带 `retry-after`**，退避只能走指数 + 抖动（这条正好验证 ADR-003 的退避策略不能只依赖 header）。

**坑**：403 `permission_error` 常出现在"账号无该模型权限 / 地区受限"，重试无用，要能提示到配置层。

### 1.10 特有能力

| 能力 | 说明 | 标记 |
|---|---|---|
| Prompt caching | `cache_control` 断点，前缀哈希顺序为 **tools → system → messages**；改动 tools 会让整段前缀失效 | ⏳ W7（降本主力） |
| Batch API | 异步批处理，成本 50%，结果 24h 内 | ⏳ W9 |
| Extended thinking | 返回 `thinking` block，与流式和 usage 都耦合 | ⏳ W9 评估 |
| Files / Skills API | 文件与技能托管（beta） | ❌ |
| 服务端 MCP connector | 由 Anthropic 侧代连 MCP server | ⏳ W8 作对比参考（我们自建） |

**结论**：唯一影响成本结构的是 prompt caching，且它的**前缀顺序约束直接决定了我们工具列表必须稳定排序**——这条要在 W8 加 Skills 时回头看（工具顺序抖动会让缓存全失效）。

### 1.11 Claude 侧「最容易归一化出错的三处」

1. **system 位置**：内部 system 必须提到顶层；漏了不报错，只是指令静默失效
2. **tool_result 成组回填**：多条 tool 消息必须合并成一条 user 消息，少一个结果就 400
3. **`input_json_delta` 拼装**：`partial_json` 是碎片，必须按 `index` 累加、`content_block_stop` 时一次解析

> 前两条会在 Day3 立刻暴露，第三条会拖到 Day4 流式才发现——建议 Day3 就把流式夹具录下来。