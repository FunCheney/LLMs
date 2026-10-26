## 4. Qwen（阿里云百炼 / compatible-mode）

> **一句话定位**：**兼容度最高、约定最多**的一家 —— 同一份 OpenAI payload 基本能跑通，但有四处会绊人：
> `tool_choice: required` 不支持、`parallel_tool_calls` **默认关**、`enable_thinking` 非标准参数且**非流式会 400**、**欠费返回 400/403 而不是 429**。
> 标记：✅ W1 必需 ｜ ⏳ 后续周次 ｜ ❌ 与本项目无关

### 4.1 鉴权 ✅

| 项 | 值 |
|---|---|
| header | `Authorization: Bearer <DASHSCOPE_API_KEY>` + `Content-Type: application/json` |
| base_url（北京） | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| base_url（新加坡） | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| base_url（工作空间） | `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` |

**结论**：鉴权零差异，但 **base_url 是"三段式"**（地域 / 工作空间 / 协议路径），必须完整配置化，**不要用字符串拼接**。`.env` 里给 `QWEN_BASE_URL` 一个完整的可覆盖值。

**坑**：百炼的 **API Key 分地域**，北京 key 打新加坡域名会返回 401。所以"我 key 是对的但一直 401"是最常见的第一现场 —— 排查顺序：地域 → key → 工作空间。

### 4.2 端点 ✅

| 端点 | 用途 | 标记 |
|---|---|---|
| `POST /compatible-mode/v1/chat/completions` | 对话 + 工具调用（**协议基准**） | ✅ |
| `POST /api/v1/services/aigc/text-generation/generation` | DashScope 原生协议 | ❌ 不用 |
| `X-DashScope-Async: enable` | 异步任务（提交 + 轮询） | ❌ W2 用 Temporal，不用它 |

**结论**：只用 compatible-mode 的 `/chat/completions`。原生协议与异步任务都**故意不碰** —— 我们 W2 会用 Temporal 做长任务，两套异步机制并存只会让排障翻倍。

### 4.3 系统消息 ✅

- `messages[0] = {"role": "system", "content": "..."}`，**同 OpenAI**
- **没有 `developer` 角色**（OpenAI 新模型有）→ `models.yaml` 的 `system_role` 能力标记对 Qwen 一律是 `system`

**结论**：system 处理与 OpenAI 完全共用代码，不做任何分支。

### 4.4 多轮中 tool 结果 ✅

**同 OpenAI：每个工具结果一条 `role: "tool"` 消息**，`tool_call_id` 关联，`content` 必须是字符串。

```json
[
  {"role": "assistant", "content": null,
   "tool_calls": [{"id": "call_1", "type": "function",
                   "function": {"name": "get_order", "arguments": "{\"order_id\":\"A-1042\"}"}}]},
  {"role": "tool", "tool_call_id": "call_1", "content": "{\"status\":\"shipped\"}"}
]
```

**思考模式的回传规则（与 DeepSeek 相反，务必分清）**：

| 模型 | 历史 `reasoning_content` 要不要回传 |
|---|---|
| Qwen3.7 系列及更早 | **不需要**，官方要求多轮时**只用 `content`，忽略 `reasoning_content`** |
| `qwen3.8-max` / `qwen3.8-flash` | **必须回传**（`preserve_thinking` 默认 `true`），且**不能拼进 `content`** |

**结论**：这是四家里**唯一"同一家不同型号规则相反"**的地方。所以 `provider_meta` 透传字段必须**按模型开关**：

- 默认：写入 `provider_meta.reasoning_content`，但**发送时不带**
- 当 `models.yaml` 标了 `preserve_thinking: true`：发送时带上

**坑**：把 `reasoning_content` 拼进 `content` 回传——官方明确说这会**降低模型表现**（它不是普通文本）。宁可丢，不可拼。

### 4.5 tool 定义 ✅

与 OpenAI 同构（`type:"function"` + `function{name, description, parameters}`），但**能力边界比 OpenAI 小一圈**：

| 参数 | Qwen | OpenAI | 处理方式 |
|---|---|---|---|
| `tool_choice: "auto"` | ✅ | ✅ | 直传 |
| `tool_choice: "none"` | ✅ | ✅ | 直传 |
| **`tool_choice: "required"`** | **❌ Qwen 系列不支持**（非思考模式无法保证一定调用；思考模式明确不支持） | ✅ | **降级为 `auto`** + 日志 `tool_choice_downgraded` |
| `tool_choice: {"type":"function",...}` 指定函数 | ✅ 非思考模式；**❌ 思考模式** | ✅ | 思考模式下降级为 `auto` |
| **`parallel_tool_calls`** | 支持，**默认 `false`** | 默认 `true` | **必须显式置 `true`** |
| `strict` | 不提供 | `strict` | 忽略 |

**结论（本家最容易踩的一条）**：`parallel_tool_calls` 的**默认值与 OpenAI 相反**。你从 OpenAI 切过来，并行工具调用会**静默变成串行**——不报错、不告警，只是 W9 的多源检索忽然慢了一倍。→ `models.yaml` 里显式写 `parallel_tool_calls: true`，并在 conformance 用例里断言"请求体里这个字段存在且为 true"。

**坑**：`required` 的降级必须**只降不抛**。理由和 DeepSeek 那节一样：Agent 循环里 `tool_choice` 是策略意图，不是硬保证，抛异常会让整个循环无谓失败。

### 4.6 tool 调用响应 ✅

```json
{
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": null,
      "reasoning_content": "…",
      "tool_calls": [{"id": "call_1", "type": "function",
                      "function": {"name": "get_order", "arguments": "{\"order_id\":\"A-1042\"}"}}]},
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 220, "completion_tokens": 35, "total_tokens": 255}
}
```

**结论**：

- `arguments` 是 **JSON 字符串**（与 OpenAI 同构，共用解析代码），`content` 在纯工具调用时为 `null`
- `finish_reason`：`stop` / `length` / `tool_calls`（与 OpenAI 同名，映射表无需扩展）
- `usage` 只有三项粗粒度字段，**没有 `cached_tokens` 明细** → 缓存省下的钱在这里看不见，只能靠账单对账

**坑**：Qwen 的 `content` 在思考模式下可能先给空字符串再给正文，别用 `if content:` 判断"是否是工具调用轮"——**判断依据永远是 `tool_calls` 是否存在**。

### 4.7 流式 ✅

`stream: true` + `stream_options: {"include_usage": true}`。SSE 只有 `data:` 行，`data: [DONE]` 结束。

```text
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {"choices":[{"index":0,"delta":{"reasoning_content":"先查订单"},"finish_reason":null}]}
data: {"choices":[{"index":0,"delta":{"content":"已发货"},"finish_reason":null}]}
data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: {"choices":[],"usage":{"prompt_tokens":220,"completion_tokens":35,"total_tokens":255}}
data: [DONE]
```

| 内部 StreamEvent | 来源 |
|---|---|
| `reasoning_delta` | `delta.reasoning_content`（思考模式） |
| `text_delta` | `delta.content` |
| `tool_call`（完整） | 按 `delta.tool_calls[].index` 累 `arguments` → 流末 `json.loads` 单发 |
| `usage` | 末尾 **`choices: []` 的 usage-only 片**（**同 OpenAI**） |
| `done` | `[DONE]` |

**结论 / 三个坑**：

1. **`enable_thinking: true` 时必须有 `stream: true`** —— 非流式调用思考模型直接 **400**（`enable_thinking only support stream call`）。→ 内部契约要写成：`thinking=True ⇒ stream=True`，**在 `LLMClient.chat()` 里做前置校验并抛 `BadRequestError`**，别让上游去猜
2. `enable_thinking` / `thinking_budget` / `preserve_thinking` 都**不是 OpenAI 标准参数**，用 SDK 要放 `extra_body`；**我们是裸 httpx，直接放请求体顶层即可**（这反而比 SDK 干净）
3. 流式 tool_call 的 `arguments` 是否**真增量**，不同型号/兼容层行为不完全一致 → **必须有一条 conformance 用例实测**：断言"累加后的 JSON 能被解析且等于期望对象"，而不是断言"分了几片"

> usage 的形状注意：Qwen 与 OpenAI **同形**（独立 usage 片 + 空 `choices`），DeepSeek **不同形**（挂在最后内容块）。
> → 这正是 §3.7 那条"**任何 chunk 的 `usage` 非空就覆盖式记录**"统一规则的第二个理由。

### 4.8 usage ✅

| 字段 | 说明 |
|---|---|
| `prompt_tokens` / `completion_tokens` / `total_tokens` | 三项，**无 hit/miss 明细** |
| 思考 token | 计入 `completion_tokens`（= 输出价，贵） |
| 缓存 | 隐式缓存自动生效 + 显式缓存可指定，但 **usage 里看不到命中量** |
| 流式 | 必须 `stream_options.include_usage = true`，否则无 usage |

**结论**：Qwen 的成本只能按 `prompt × p_in + completion × p_out` 估算，**缓存收益无法在请求级归因**。→ W7 对账时，Qwen 的成本曲线会系统性偏高（因为缓存折扣体现不出来）。这不是 bug，**要在成本报告里注明口径差异**，否则你会误判"Qwen 比 DeepSeek 贵"。

**坑**：始终显式设 `max_tokens`。Qwen3.7 系列最大输出 128K，默认值因型号而异；不设的话，一个失控的长输出就能吃掉当天的成本预算。

### 4.9 限流 / 错误 ✅

**限流按主账号合并计算**：同一主账号下所有 RAM 子账号、业务空间、API Key **合并计入** —— 换 key、拆空间都绕不开。旗舰 `qwen3.7-max` 公开额度约 30,000 RPM / 5,000,000 TPM；具体以控制台为准。

**429 有三种语义（同名状态码，三种原因）**：

| 429 的 message | 含义 | 应对 |
|---|---|---|
| `Requests rate limit exceeded` | RPM（每分钟请求数）超限 | 指数退避，降频 |
| `Allocated quota exceeded` | TPM（每分钟 token 数）超限 | 退避 + **减小单次输入**（长文档要先切分） |
| `Request rate increased too quickly` | 总量没超但**瞬时激增** | 短退避 + **客户端令牌桶削峰** |

三者通常 1 分钟内自动恢复。**不承诺 `retry-after`**。

| HTTP | code / 场景 | 内部异常 | 可重试 |
|---|---|---|---|
| 400 | `InvalidParameter` | `BadRequestError` | ❌ |
| **400 / 403** | **`Arrearage`（欠费）** | `UpstreamAuthError` | **❌** |
| 401 | `InvalidApiKey` | `UpstreamAuthError` | ❌ |
| 404 | `model_not_found` | `BadRequestError` | ❌ |
| 429 | `Throttling.*`（上表三兄弟） | `RateLimitError` | ✅ |
| 500 | `InternalError` | `UpstreamServerError` | ✅ |
| 502 / 503 | 网关/服务不可用 | `UpstreamServerError` | ✅ |

**结论（本家最反直觉的一条）**：**欠费不是 429，是 400/403 的 `Arrearage`**。OpenAI 是 429 `insufficient_quota`、DeepSeek 是 402 —— **三家三个位置**。→ 错误映射**必须按 `code` 字符串判断，不能按状态码**。这是 `openai_compat.py` 里唯一必须硬编码 `code` 白名单的地方：

```text
insufficient_quota | Arrearage | BIL.UserArrears  →  UpstreamAuthError（retryable=False，对外 502 UPSTREAM_AUTH）
```

**坑**：欠费/额度耗尽被当成普通 400 参数错误 → 上层会提示"你的请求参数有误"，用户改半天参数也没用。这类错误文案必须能区分。

### 4.10 特有能力

| 能力 | 说明 | 标记 |
|---|---|---|
| **上下文缓存** | 隐式（自动）+ 显式（可指定片段），命中按折扣计费 | ⏳ W5–W7 |
| `enable_search` | 平台侧联网搜索（非标准参数），可 `forced_search` 强制 | ❌ W8 自建抓取 MCP，不用（对比论证用） |
| `enable_thinking` / `thinking_budget` / `thinking_level` | 思考开关与长度上限 | ⏳ W7（长文分析控成本） |
| `preserve_thinking` | 历史思考回传（3.8 默认开） | ⏳ |
| Batch 调用 | 半价，非实时任务 | ⏳ W11 评测批跑（可选） |
| 多模态（`qwen3.7-plus`） | 文本 + 图像 + 视频输入，同价 | ⏳ W10（竞品截图分析这类场景） |
| **模型快照钉版本** | `qwen3.7-plus-2026-05-26` 这类带日期 ID | ✅ **评测必须钉快照** |

**结论**：**评测（W3–W4）一定要钉日期快照**，不能用 `qwen3.7-plus` 这种滚动别名 —— 否则第 6 周的评测分数和第 2 周不可比，你那张"12 周指标曲线"就失去因果性。这是本家对 DoD 影响最大的一条。

**补充（选型）**：

| 模型 | 上下文 | 最大输出 | 价格（元/百万 tokens，进/出） | 用途 |
|---|---|---|---|---|
| `qwen3.7-plus` | 1M | 128K | 2 / 8 | **cheap tier 首选**，还带多模态 |
| `qwen3.7-max` | 1M | 128K | 12 / 36 | strong tier，纯文本 |
| `qwen3.7-flash` | 1M | — | 更低 | 分类/抽取 |

> `qwen3-max`、`qwen-turbo` 等旧模型已于 2026-10-10 下线，**不要写进 `models.yaml`**。

### 4.11 Qwen 侧「最容易归一化出错的三处」

1. **`parallel_tool_calls` 默认 `false`（OpenAI 默认 `true`）**：不显式开启，并行工具调用静默失效 —— 不报错、只是慢，最难发现
2. **`tool_choice: required` / 指定函数不被支持**：不降级就抛异常，会把 Agent 循环打断；降级要记日志
3. **欠费是 400/403 的 `Arrearage`，不是 429**：按状态码映射会把"余额耗尽"误判成"参数错误"，用户永远改不对
