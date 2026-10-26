## 2. OpenAI

### 2.1 鉴权 ✅

| 项 | 值 |
|---|---|
| 主用 header | `Authorization: Bearer <OPENAI_API_KEY>`（**兼容协议唯一必需的 header**） |
| 内容类型 | `Content-Type: application/json` |
| 可选 header | `OpenAI-Organization`、`OpenAI-Project`（按项目计量，团队场景） |
| base_url | `https://api.openai.com/v1`，可换成任意兼容网关 / 代理 |

**结论**：adapter 固定注入 `Authorization` + `Content-Type`；组织/项目 header 走 `extra_headers` 口子，不进共用逻辑。

**坑**：`Bearer` 是本协议族唯一的鉴权公约数——`openai_compat` 里不要出现任何 `x-api-key` 之类的判断，那是 Anthropic 的事。四家共用同一个类时，header 注入点必须只有一个。

### 2.2 端点 ✅

| 端点 | 用途 | 标记 |
|---|---|---|
| `POST /v1/chat/completions` | 对话 + 工具调用（**协议基准**） | ✅ |
| `POST /v1/responses` | Responses API，功能超集，OpenAI 推荐新集成使用 | ❌ W1 不实现 |
| `GET /v1/models` | 列模型，生成 `models.yaml` | ⏳ |

**决策**：`openai_compat` 以 **`/chat/completions` 为协议基准**——DeepSeek / Qwen / Ollama / vLLM 全部对齐这一形态，这是"一个类覆盖四家"的前提。

**风险（影响 capstone 选型，必须记住）**：OpenAI 自 **gpt-5.4** 起，`reasoning: "none"` 时 Chat Completions **不支持 tool calling**；**gpt-6-astra 的工具调用必须走 Responses API**。我们的 `openai_compat` 只讲 chat/completions 协议，因此**最新一代模型可能无法用它做工具调用**。

→ 处理方式：W1 不实现 Responses adapter，但记为 **ADR-006 候选**（见 `llm-api-matrix.md` §5）。选模型时避开"必须走 Responses 才能调工具"的型号；W9 若确有需要，再加 `providers/responses.py`，届时 client / router / types 三层零改动（这就是 ADR-004 分层的意义）。

> 附：Assistants API 已于 2026-08-26 正式下线，不必再考虑。

### 2.3 系统消息 ✅

- `messages[0]` 为 `{"role": "system", "content": "..."}`；content 也可以是 content parts 数组
- 新模型（o 系列起）支持用 `role: "developer"` **替代** `system`；兼容层普遍仍接受 `system`
- 请求体**不要求** `max_tokens`（与 Anthropic 相反——Anthropic 必填）

**结论**：内部 system → `{"role": "system"}` 放 messages 最前。
若目标模型要求 `developer`，**做成 `models.yaml` 里的 capability 标记**（如 `system_role: developer`），**禁止在 adapter 里按模型名写 if 分支**。

**坑**：system 放在 messages 中间，多数实现会报错或静默忽略——后者更麻烦，指令失效但不报错，和 Anthropic 那条坑是同一个病。

### 2.4 多轮中 tool 结果 ✅（**与 Anthropic 的结构性差异**）

assistant 返回 `tool_calls` 后，**每个工具结果各占一条消息**：

```json
[
  {"role": "assistant", "content": null, "tool_calls": [
    {"id": "call_abc", "type": "function",
     "function": {"name": "db_query", "arguments": "{\"sql\":\"SELECT 1\"}"}},
    {"id": "call_def", "type": "function",
     "function": {"name": "db_query", "arguments": "{\"sql\":\"SELECT 2\"}"}}
  ]},
  {"role": "tool", "tool_call_id": "call_abc", "content": "{\"rows\": 1}"},
  {"role": "tool", "tool_call_id": "call_def", "content": "timeout"}
]
```

**约束**：

1. `tool_call_id` 必须匹配上一条 assistant 里某个 `id`，不能自造
2. `role: "tool"` 消息必须紧跟在含 `tool_calls` 的 assistant 消息之后
3. `content` 必须是**字符串**（结构化结果要 `json.dumps`），传对象直接 400
4. 可拆多条、顺序不限（与 Anthropic "必须一次性成组"正好相反）

**结论（这条决定内部 Message 的形态）**：内部模型采用 **OpenAI 形态——每条 tool 结果一条 `role: "tool"` 消息**，因为这是唯一 1:1 映射、无需合并的一侧；**Anthropic adapter 负责把连续多条 `tool` 消息合并成一条 user 消息**。合并逻辑只放 `providers/anthropic.py`，**不放 types 层**（放 types 会让非 Anthropic provider 反向付费）。

**坑**：在 `anthropic.py` 里逐条发 `role: "tool"`（忘了合并）→ 400，报错信息说 `tool_result` 缺失，很容易误判成 id 不匹配。

### 2.5 tool 定义 ✅

```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "db_query",
      "description": "对分析库执行只读 SQL。用户询问历史数据统计时使用；不要用于写入。",
      "parameters": {
        "type": "object",
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"],
        "additionalProperties": false
      },
      "strict": true
    }
  }],
  "tool_choice": "auto",
  "parallel_tool_calls": true
}
```

| 差异点 | OpenAI | Anthropic |
|---|---|---|
| 结构 | **多包一层** `type:"function"` + `function` 对象 | 扁平 `{name, description, input_schema}` |
| schema 字段名 | `parameters` | `input_schema` |
| `tool_choice` | **字符串** `"auto" / "none" / "required"`，或 `{"type":"function","function":{"name":...}}` | **对象** `{"type":"auto"/"any"/"tool"}` |
| 并行开关 | `parallel_tool_calls: true` | `disable_parallel_tool_use: false`（**语义相反**） |

**结论**：`ToolSpec` → OpenAI 需"包一层 + 改字段名"；`tool_choice` 写双向映射函数放 `providers/_common.py`。
**注意语义反转**：直接复制字段名会把并行能力关掉，**且不报错**——这是 conformance 断言要覆盖的一项。

### 2.6 tool 调用响应 ✅

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc", "type": "function",
        "function": {"name": "db_query", "arguments": "{\"sql\":\"SELECT 1\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 812, "completion_tokens": 64, "total_tokens": 876}
}
```

**结论**：

- **`arguments` 是 JSON 字符串**，必须 `json.loads`；解析失败 → `arguments={}` + `raw_arguments` 保留 + `arguments_error`（不抛异常，否则整轮对话断掉）
- **`content` 可能为 `null`**（纯工具调用），直接做字符串拼接会 `TypeError`
- 工具调用时 `finish_reason == "tool_calls"` → 内部 `tool_calls`
- 可能有**多个** `tool_calls`（并行），内部 `list[ToolCall]` 直接承接；注意 id 与 index 是两套标识，回填时必须用 `id`
- 多模态模型的 `content` 可能是 content parts 数组 → 内部需统一拼成 text

### 2.7 流式 ✅

`stream: true` + `stream_options: {"include_usage": true}`。SSE **只有 `data:` 行、没有 `event:` 名**（事件类型藏在 payload 的 `object` 字段里），结束哨兵为 `data: [DONE]`。

```text
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"db_query","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"sql\":\"SELECT"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":812,"completion_tokens":64,"total_tokens":876}}

data: [DONE]
```

| 内部 StreamEvent | 来源 |
|---|---|
| `text_delta` | `choices[0].delta.content`（可能是空字符串） |
| `tool_call`（完整） | 按 `delta.tool_calls[].index` 累加 `function.arguments` 字符串，流结束时 `json.loads` 后发一次 |
| `usage` | 末尾 `choices: []` 且 `usage` 非空的那一片 |
| `done` | 收到 `[DONE]`（`finish_reason` 取自前一片） |

**结论与五个坑**：

1. **`stream_options.include_usage` 不开，全程没有 usage**——成本埋点直接归零，这是最容易被忽略的一条
2. **末尾 usage 那一片的 `choices` 是空数组** → 写死 `choices[0]` 必 `IndexError`
3. 非末尾片的 `usage` 字段存在但为 `null`，不能用"字段是否存在"判断
4. 默认 `include_obfuscation: true`，delta 里会多出 `obfuscation` 字段（填充随机字符以掩盖包大小）→ **解析器不能断言字段集合固定**，未知字段一律忽略
5. `[DONE]` 不是 JSON，`json.loads("[DONE]")` 必崩，要先判等

> ADR-002 印证：OpenAI 按 `index` 累加、Anthropic 按 `content_block index` + `partial_json` 累加，两套逻辑完全不同，只能关在 adapter 内。

### 2.8 usage ✅

| 场景 | 字段 |
|---|---|
| 一次性 | `usage = {prompt_tokens, completion_tokens, total_tokens}` |
| 明细 | `prompt_tokens_details.cached_tokens`、`completion_tokens_details.reasoning_tokens` / `audio_tokens` |
| 流式 | **必须 `stream_options.include_usage = true`**；末尾单独一片；流中断可能收不到 → 标 `usage_estimated` |
| 计价 | 前缀**自动**缓存，`cached_tokens` 单价更低 |

**结论**：W1 统一按 `cost = prompt_tokens × p_in + completion_tokens × p_out` 计算，`cached_tokens` **只记录不参与**；W7 引入缓存单价后精算——与 Anthropic 保持同一口径，两家成本才可横向比较。

**坑**：`total_tokens` 不能替代分项求和（含 reasoning / audio 明细时对不上），计价一律用分项。

### 2.9 限流 / 错误 ✅

**限流 header**，直接用于背压：

`x-ratelimit-limit-requests` / `-remaining-requests` / `-reset-requests`，以及对应的 `-tokens-` 三项；另有 `retry-after`（不保证存在）。

**错误体**：

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}
```

| HTTP | `error.type` / `code` | 内部异常 | 可重试 |
|---|---|---|---|
| 400 | `invalid_request_error`（含 `context_length_exceeded`） | `BadRequestError` | ❌ |
| 401 | `invalid_api_key` / `authentication_error` | `UpstreamAuthError` | ❌ |
| 403 | `permission_error` | `UpstreamAuthError` | ❌ |
| 404 | `invalid_request_error`（模型不存在） | `BadRequestError` | ❌ |
| 429 | `rate_limit_error`（`rate_limit_exceeded`） | `RateLimitError` | ✅ |
| 429 | `insufficient_quota` | `UpstreamAuthError` | **❌** |
| 500 | `server_error` | `UpstreamServerError` | ✅ |
| 503 | `engine_overloaded` | `UpstreamServerError` | ✅ |

**结论**：

- **429 必须分两类**：`rate_limit_exceeded`（等 `retry-after` 重试）vs `insufficient_quota`（余额/配额耗尽，重试无用且会烧更多钱）——同为 429，只能靠 `code` 区分
- **503 `engine_overloaded` 必须可重试**，它和 Anthropic 的 529 是同一语义、不同编号 → 内部统一归 `UpstreamServerError`
- 400 的 `context_length_exceeded` 是**上下文超限**，不能笼统当参数错误：W7 上下文治理要能识别它并触发裁剪重试

**坑**：错误 `message` 常回显用户输入，写日志前必须脱敏，否则等于把用户内容落盘。

### 2.10 特有能力

| 能力 | 说明 | 标记 |
|---|---|---|
| Prompt caching | **自动**前缀缓存，命中量见 `cached_tokens`，单价更低；可用 `prompt_cache_key` 提升命中率 | ⏳ W7（降本主力） |
| Structured Outputs | `response_format: {"type":"json_schema","json_schema":{...}}` + `strict` | ⏳ W9（调研报告结构化输出） |
| 并行工具调用 | `parallel_tool_calls`（默认开） | ⏳ W9（多源检索并行） |
| Reasoning 系列 | `reasoning_effort`；注意与 tool calling 的兼容限制（见 2.2） | ⏳ |
| Responses 内置工具 | `web_search` / `file_search` / `computer_use`，OpenAI 侧托管 | ⏳ W8 作对比（我们自建 MCP） |
| Responses 有状态 | `store` / `previous_response_id` / Conversations | ⏳ W7 参考（与记忆模块职责重叠） |

**结论**：对 capstone 有实质影响的只有两位——**prompt caching（降本）** 和 **Structured Outputs（报告结构化）**。Responses 的内置工具与我们的 MCP 方案是"买 vs 建"的关系，W8 做 MCP 时正好拿来做对比论证。

### 2.11 OpenAI 侧「最容易归一化出错的三处」

1. **`arguments` 是字符串 + `content` 可能为 `null`**：前者要 `json.loads`（失败要降级不崩），后者直接拼接会 `TypeError`
2. **流式 usage 要显式开启，且末尾片 `choices` 为空**：不开就没有成本数据；写死 `choices[0]` 会 `IndexError`
3. **tool 结果是 N 条消息、`content` 必须是字符串**：与 Anthropic 的"一条消息成组回填"相反，方向搞反会同时踩两家
