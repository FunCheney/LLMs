## 3. DeepSeek

> **一句话定位**：协议上是 **OpenAI 的子集 + 三个私有扩展**（`thinking` / `reasoning_effort`、`reasoning_content`、`user_id`），
> 但它是四家里**唯一一个「照抄 OpenAI 代码会在第二轮直接 400」**的供应商 —— 因为它**默认开启思考模式**。
> 标记：✅ W1 必需 ｜ ⏳ 后续周次 ｜ ❌ 与本项目无关

### 3.1 鉴权 ✅

| 项 | 值 |
|---|---|
| header | `Authorization: Bearer <DEEPSEEK_API_KEY>` + `Content-Type: application/json` |
| base_url（OpenAI 格式） | `https://api.deepseek.com`（`/v1` 是兼容别名，二者等价） |
| base_url（Anthropic 格式） | `https://api.deepseek.com/anthropic` |
| base_url（Beta） | `https://api.deepseek.com/beta`（`strict` 模式用） |
| 模型名 | 只有两个：`deepseek-flash`、`deepseek-v4-pro` |

**结论**：`openai_compat.py` 里 DeepSeek 只需要 `base_url` + `api_key` 两个配置项，**鉴权零差异**。

**注意**：旧模型名 `deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 下线；`deepseek-v4-flash` 仍可调用但会静默路由到 V4.1-Flash 并按 Flash 计费 —— **`models.yaml` 里不要写旧名**，否则你以为在跑 A 模型，实际跑的是 B。

### 3.2 端点 ✅

| 端点 | 用途 | 标记 |
|---|---|---|
| `POST /chat/completions` | 对话 + 工具调用（**协议基准**） | ✅ |
| `POST /anthropic/v1/messages` | Anthropic 协议兼容层 | ⏳ W1 不用，但**极有用**：可用它验证 `anthropic.py` 的正确性（第二套夹具零成本） |
| `POST /beta/chat/completions` | `strict` 模式工具调用（Beta） | ❌ |
| `POST /responses` | Responses API | ❌ W1 不做 |

**结论**：W1 只实现 `/chat/completions`。`/anthropic` 端点记在文档里，Day3 写完 `anthropic.py` 后可以拿它当**免费回归目标**——同一份代码打两家，是最便宜的 conformance 证据。

### 3.3 系统消息 ✅

- `messages[0] = {"role": "system", "content": "..."}`，**与 OpenAI 完全一致**
- DeepSeek 额外支持**在对话中间插入 system 消息**（Anthropic 也支持，OpenAI 不保证）
- 但**不支持在对话中间插入工具调用**（该能力只在 `/anthropic` 与 `/responses` 上提供）

**结论**：W1 只支持「system 在首部」。中间的 system 消息属于 Agent 动态换指令的场景，W9 再说；中间插 tool 调用对我们没用（我们是模型驱动，不是人工编排）。

### 3.4 多轮中 tool 结果 ✅（**思考模式下有硬约束**）

结构同 OpenAI：**每个工具结果一条 `role: "tool"` 消息**，用 `tool_call_id` 关联。

```json
[
  {"role": "assistant", "content": null,
   "reasoning_content": "需要先查日期…",
   "tool_calls": [{"id": "call_1", "type": "function",
                   "function": {"name": "get_date", "arguments": "{}"}}]},
  {"role": "tool", "tool_call_id": "call_1", "content": "2026-10-25"}
]
```

**硬约束（本家最重要的一条）**：请求里一旦带 `tools`，**`reasoning_content` 必须在后续所有请求中完整回传**，包括没发生工具调用的轮次和跨轮次的新对话。不回传 → **400：`The reasoning_content in the thinking mode must be passed back to the API`**。

**结论（直接影响架构）**：

1. 内部 `Message` 必须有一个 **`provider_meta: dict` / `raw` 透传字段**，用来携带 `reasoning_content` 这类**框架不解释、但必须原样还给上游**的字段
2. 该字段**不可由业务层构造**，只能由 adapter 写入、adapter 读回；业务层只做整体传递
3. Anthropic 的 `thinking` block + `signature` 是同一类问题（签名必须原样回传）→ 两者共用同一个机制

> 这条是 ADR-007 的由来（见 `llm-api-matrix.md` §5）。没有这个字段，你的 DeepSeek adapter 只能工作在 `thinking: disabled` 模式下——那就等于放弃它一半的能力。

**坑**：`messages.append(response.choices[0].message)` 之所以能跑通，是因为它把 `reasoning_content` 一起带上了；一旦你手写 `{"role":"assistant","content":...,"tool_calls":...}` 就会丢掉它 → 400。**adapter 里禁止手工重建 assistant 消息**。

### 3.5 tool 定义 ✅

与 OpenAI 同构：`tools[]{type:"function", function:{name, description, parameters, strict}}`。

| 差异点 | DeepSeek | OpenAI | 影响 |
|---|---|---|---|
| `tool_choice` 取值 | `none` / `auto` / `required` / `{"type":"function",...}` | 同 | 形态一致 |
| **思考模式下 `required` 与指定函数** | **不支持，返回 400** | 支持 | ⚠️ 必须按能力降级 |
| `parallel_tool_calls` | **没有这个参数** | 有（默认开） | 无法显式开关 |
| `strict` | 支持，但需 `/beta` base_url 且所有 function 都要 `strict:true` | `strict` 直接在 function 内 | W1 不开 |
| 函数名长度 | ≤ 128 字符 | ≤ 64 字符 | 取交集 64 |

**结论**：

- `tool_choice` 写映射时，**按 `capabilities` 降级**：模型不支持 `required` 时，把 `required` 降为 `auto` 并在日志里打 `tool_choice_downgraded`，**不要抛异常**（抛了会让上层 Agent 循环无缘无故崩掉）
- **不要假设并行可控**：没有 `parallel_tool_calls` 参数，但响应里 `tool_calls` 仍可能是数组 → 内部 `list[ToolCall]` 照常承接，只是不能"要求它并行"

**坑**：思考模式是**默认开启**的。也就是说，你按 OpenAI 写的 `tool_choice: "required"` 用例，第一次跑就会 400 —— 而错误信息只说"参数不支持"，不会提示你"因为思考模式开着"。排查时先看 `thinking.type`。

### 3.6 tool 调用响应 ✅

```json
{
  "choices": [{
    "message": {"role": "assistant", "content": null,
      "reasoning_content": "…",
      "tool_calls": [{"id": "call_1", "type": "function",
                      "function": {"name": "get_date", "arguments": "{}"}}]},
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160,
            "prompt_cache_hit_tokens": 96, "prompt_cache_miss_tokens": 24,
            "completion_tokens_details": {"reasoning_tokens": 30}}
}
```

**结论**：

- `arguments` 是 **JSON 字符串**，`content` 可能为 `null` —— 与 OpenAI 完全同构，共用同一段解析代码
- `finish_reason` **比 OpenAI 多两个值**（见下表），内部枚举必须覆盖

| DeepSeek finish_reason | 内部映射 | 处理方式 |
|---|---|---|
| `stop` / `length` | `stop` / `length` | 正常 |
| `tool_calls` | `tool_calls` | 正常 |
| `content_filter` | `content_filter` | 正常结束，不重试 |
| **`insufficient_system_resource`** | `upstream_error` | ⚠️ **不是正常结束**：生成被资源不足打断 → 可重试（但要带退避，别立刻重打） |
| **`aborted`** | `aborted` | 调用方中断/连接断开 → 不重试，直接向上抛 |

**坑**：把 `insufficient_system_resource` 当 `stop` 处理，会得到一条**半截但"看起来成功"的回答**，而且能通过所有非严格断言 —— 属于最难发现的一类 bug。

### 3.7 流式 ✅

`stream: true` + `stream_options: {"include_usage": true}`。SSE 只有 `data:` 行，结束哨兵 `data: [DONE]`。

**三个与 OpenAI 不同、必须单独处理的点**：

**① 有 SSE 注释行心跳**

```text
: keep-alive

data: {"choices":[{"index":0,"delta":{"reasoning_content":"需要"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"content":"杭州"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}

data: [DONE]
```

以 `:` 开头的行是注释，**不是 JSON**。自己用 httpx 逐行解析时，必须先跳过 `:` 开头、空行、`event:` 行，再判 `data: [DONE]`，最后才 `json.loads`。写漏这一条，长思考请求（>60s 无事发生）时解析器会崩在心跳上。

**② usage 挂在最后一个内容块上，不是单独一片**

| | OpenAI | DeepSeek |
|---|---|---|
| usage 所在片 | **单独一片**，`choices: []` | **最后一个内容块**，`choices` 仍有一个元素、`delta` 为空、`finish_reason` 非 null |
| 判断条件 | `if not chunk.choices` | `if chunk.usage`（或 `finish_reason` 非 null） |

**统一规则（写进 adapter 基类，不要每家写一遍）**：**任何 chunk 只要 `usage` 非空就记录，后来覆盖先来**；不要用「`choices` 是否为空」作为读 usage 的条件。这一条同时兼容 OpenAI / Qwen / DeepSeek 三种形状。

**③ 思考内容与答案在不同字段**

`delta.reasoning_content` 是思考，`delta.content` 是答案。内部 `StreamEvent` 必须能区分 —— 若把 `reasoning_content` 混进 `text_delta`，用户会在界面上看到模型的内心独白（W10 的"内容泄漏"红线就是这么来的）。

| 内部 StreamEvent | 来源 |
|---|---|
| `reasoning_delta` | `delta.reasoning_content` |
| `text_delta` | `delta.content` |
| `tool_call`（完整） | 按 `delta.tool_calls[].index` 累 `arguments`，流末 `json.loads` 后单次发出 |
| `usage` | 任意 `chunk.usage` 非空（覆盖式） |
| `done` | `[DONE]`；`finish_reason` 取最后一片；为 `insufficient_system_resource` 时改抛 `UpstreamServerError` |

**另外两个坑**：

- **10 分钟规则**：若 10 分钟内模型仍未开始推理，服务端直接关连接。`first_token_timeout` 要设在这个之内（建议 120s 上限 + 明确文案），不要指望 TCP 会给你一个明确的错误
- **参数在思考模式下失效**：`temperature` 不生效；`top_p` 仅在思考模式生效（<0.95 按 0.95 处理，非思考模式恒为 1.0）。`frequency_penalty` / `presence_penalty` **已废弃**，传了也没用 → `models.yaml` 不要给 DeepSeek 配这两项

### 3.8 usage ✅

| 字段 | 说明 |
|---|---|
| `prompt_tokens` | = `prompt_cache_hit_tokens + prompt_cache_miss_tokens` |
| `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` | **缓存命中/未命中分开计费**，单价差 25~50 倍 |
| `prompt_tokens_details.cached_tokens` | 与 `prompt_cache_hit_tokens` 同值 |
| `completion_tokens_details.reasoning_tokens` | 思考 token，**计入输出、按输出价收费** |
| `total_tokens` | prompt + completion |

**结论**：W1 的成本公式在 DeepSeek 上要写成

```text
cost = hit_tokens × p_in_cache + miss_tokens × p_in + completion_tokens × p_out
```

但 W1 **先按统一口径**（`prompt × p_in + completion × p_out`）记录，把 `hit/miss` 明细存进结构化日志，**W7 再精算** —— 否则四家的成本无法横向比较。

**坑**：`reasoning_tokens` 是**要付费的输出 token**。思考模式默认开启 + 默认 `reasoning_effort: high`，意味着你可能在"只问一句话"上花掉几百个输出 token。这正是 W10 成本熔断要拦的东西。

### 3.9 限流 / 错误 ✅

**限流按"并发"计，不是 QPS/RPM**：`deepseek-flash` 2500 并发、`deepseek-v4-pro` 500 并发，账号粒度（与 API Key 无关），超限返回 **429**。**不承诺返回 `retry-after`**。

| HTTP | 含义 | 内部异常 | 可重试 |
|---|---|---|---|
| 400 | 请求体格式错误（含 `reasoning_content` 未回传） | `BadRequestError` | ❌ |
| 401 | API Key 错误 | `UpstreamAuthError` | ❌ |
| **402** | **余额不足** | `UpstreamQuotaError` | **❌** |
| **422** | **参数错误** | `BadRequestError` | ❌ |
| 429 | 并发/速率超限 | `RateLimitError` | ✅ |
| 500 | 服务内部故障 | `UpstreamServerError` | ✅ |
| 503 | 服务器繁忙 | `UpstreamServerError` | ✅ |

**结论（三条都是归一化必改的）**：

1. **`402` 是独立语义**：OpenAI 把"余额耗尽"塞进 429（`insufficient_quota`），DeepSeek 单独给 402。→ 内部统一 `UpstreamQuotaError`，**HTTP 层映射 402 给调用方**，**不可重试**
2. **`422` 是参数错误**：OpenAI 用 400、DeepSeek 用 422。→ `openai_compat` 的错误映射表必须同时收 400 和 422，否则参数错误会被当成"未知 5xx"重试三次
3. **429 没有 `retry-after`**：这就是 ADR-003「退避不能只依赖 header」的第二个证据（第一个是 Anthropic 的 529）

**坑**：并发超限的表现是"请求排队很久然后 429"，而不是立刻 429。所以**客户端要有自己的并发信号量**（W2 做），别把 429 当唯一背压信号。

### 3.10 特有能力

| 能力 | 说明 | 标记 |
|---|---|---|
| **上下文硬盘缓存** | 自动命中，`hit/miss` 分价；**命中价是未命中的 1/25 ~ 1/50** | ⏳ W5–W7 降本主力 |
| **峰谷定价** | 空闲时段（含周末/节假日全天）= 高峰价的 **50%**；高峰为工作日 9–12、14–18 | ⏳ 批处理排夜间，直接砍半 |
| Json Output | `response_format: {"type":"json_object"}`，且**必须在 prompt 里明确要求 JSON**，否则可能一直吐空白直到超限 | ⏳ W10 |
| `strict`（Beta） | 需 `/beta` base_url，schema 限制严格（object 所有属性必须 required + `additionalProperties:false`） | ❌ |
| Anthropic 兼容端点 | `/anthropic` | ⏳ 可作 `anthropic.py` 的免费回归目标 |
| `user_id` | 传 `user_id` 可做**KVCache 隔离 + 内容安全隔离 + 调度隔离** | ⏳ W7（多租户隐私隔离，正好是记忆库需要的） |
| 网页搜索 / 抓取 | 独立计费（¥5/千次），2026-11-01 起收紧计费 | ❌ W8 自建抓取，不用它 |

**结论**：对 capstone 有实质影响的三项 —— **缓存分价**、**峰谷定价**、**`user_id` 隔离**。前两项直接决定「单任务成本 ≤ ¥2」这条 DoD 能不能达成；第三项是 W7 多租户记忆的现成能力，比自己造隔离省事。

### 3.11 DeepSeek 侧「最容易归一化出错的三处」

1. **思考模式默认开启 + `reasoning_content` 必须回传**：不建 `provider_meta` 透传字段，第二轮必 400；这条会逼你改 `types.py`，**Day2 就要定好**
2. **流式 usage 挂在最后内容块上**（不是空 `choices` 片）+ **`: keep-alive` 注释行**：沿用 OpenAI 的解析假设，会同时丢成本数据和崩在心跳上
3. **`402` / `422` / `insufficient_system_resource`**：三个都是"OpenAI 没有、而 DeepSeek 有"的语义，错误映射表不补齐就会出现"余额不足被重试三次"和"半截回答被当成功"
