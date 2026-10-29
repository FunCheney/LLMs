# 架构设计（W1 定稿 · 后续 11 周在此之上生长）

> 本文只定义**结构与契约**，不含实现代码。实现由你完成，我负责 review。
> 变更成本从低到高：命名 → 目录 → 数据契约 → 分层边界。所以本周必须把**数据契约**和**分层边界**一次定对。

---

## 0. 三条设计原则（后面 11 周不许破）

1. **边界只有一个：wire format ↔ internal format**。适配器只做翻译，不承载任何业务判断。
2. **业务层永远看不到厂商 SDK / 原始 JSON**。一旦泄漏，W5 之后每个模块都要为多厂商写分支。
3. **每个抽象必须有"第二实现"来证明它值得存在**。LLMClient 有 3 provider + mock，Router 有 2 种策略，后面 VectorStore 有 2 种后端。写不出第二实现的抽象就是过度设计。

---

## 1. 仓库布局（uv workspace，2 个成员）

```text
agent_framework/                         # 仓库根 = uv workspace
├── pyproject.toml                       # 只声明 workspace members + 工具配置（ruff/mypy/pytest）
├── packages/
│   └── agent_framework/                 # 框架包
│       ├── pyproject.toml
│       ├── src/agent_framework/
│       │   ├── config.py                # pydantic-settings + models.yaml 加载与校验
│       │   ├── observability.py         # 结构化日志 / request_id 上下文 / 成本计量
│       │   ├── llm/
│       │   │   ├── types.py             # 全部数据契约（只依赖 pydantic）
│       │   │   ├── errors.py            # 异常层次 + retryable 语义
│       │   │   ├── base.py              # LLMClient 抽象 + Capability 枚举
│       │   │   ├── client.py            # 装饰器链：重试 / 计量 / 日志 / trace
│       │   │   ├── registry.py          # name → adapter 工厂
│       │   │   ├── router.py            # task/tier → model + fallback 链
│       │   │   └── providers/
│       │   │       ├── openai_compat.py # OpenAI / Qwen / DeepSeek / Ollama / vLLM
│       │   │       ├── anthropic.py
│       │   │       └── mock.py          # 回放夹具，离线测试与 W3 评测量产用
│       │   └── server/
│       │       ├── app.py               # FastAPI 实例 + 依赖注入 + 异常处理器
│       │       ├── sse.py               # StreamEvent → SSE 帧编码
│       │       └── routes_chat.py       # POST /v1/chat
│       └── tests/                       # 单测 + 契约测试 + 夹具
├── apps/
│   └── growth_agent/                    # 产品包（W1 只放一个透传端点，证明框架可被消费）
│       ├── pyproject.toml               # 依赖 agent_framework（path 依赖）
│       └── src/growth_agent/api.py
├── evals/                               # golden / history / scripts（W2 起填充）
├── ops/                                 # Dockerfile / compose / runbook（W2 起）
└── docs/                                # architecture / w1-tasks / llm-api-matrix / adr
```

**依赖方向单向**：`growth_agent → agent_framework`，框架**永不** import 产品。CI 里加一条 import-linter 或简单脚本守住它（W12 再用工具，本周手工遵守）。

**迁移自当前仓库**：现有 `src/agent_framework/*` 整体移入 `packages/agent_framework/src/agent_framework/`，根 `pyproject.toml` 降级为 workspace 根（保留 dev 依赖与工具配置），`compose.yaml` 暂留根目录（W2 迁进 `ops/`）。
> 若 uv workspace 配置卡住超过 40 分钟，降级方案：不做 workspace，把 `growth_agent` 作为 `agent_framework` 同级的第二个顶层模块，`module-root = "src"`。结构不完美但本周不该为构建工具牺牲学习时间。

---

## 2. 分层与职责（含"禁止事项"，review 时逐条对照）

| 层 | 目录 | 职责 | **禁止** |
|---|---|---|---|
| 传输层 | `server/` | HTTP/SSE 编解码、请求校验、异常→HTTP 映射、request_id 注入 | 拼 prompt；直连厂商 HTTP；业务分支 |
| 编排层 | `apps/` + 未来 `planner/` | 决定"调谁、调几次、什么时候停" | W1 只允许一次透传，不许出现循环 |
| 客户端层 | `llm/client.py` | 能力协商、重试、计量、日志埋点、fallback | 感知任何厂商字段差异 |
| 适配层 | `llm/providers/` | wire ↔ internal 双向翻译 | 重试、日志、缓存、业务判断 |
| 契约层 | `llm/types.py`、`errors.py` | 纯数据结构与异常定义 | import 任何 provider / 网络库 |

> 自检口诀：**改一个 provider 的字段映射，只能动 `providers/` 下一个文件。**

---

## 3. 数据契约（本周最重要的产出）

### 3.1 Message（内部统一格式）

| 字段 | 类型 | 约定 |
|---|---|---|
| `role` | `system \| user \| assistant \| tool` | `system` 只允许出现在首条（多 system 由 adapter 自行合并，不向上暴露） |
| `content` | `str \| list[ContentBlock]` | 本周只实现 `str`；类型上留扩展位，W10 加图片/文件时不动契约 |
| `tool_calls` | `list[ToolCall] \| None` | 仅 `assistant` 可有 |
| `tool_call_id` | `str \| None` | 仅 `tool` 必填，且该消息必须紧跟对应的 `assistant(tool_calls)` |
| `name` | `str \| None` | 仅 `tool` 消息，标记来自哪个工具（部分厂商需要） |

**ADR-001：采用"扁平结构 + content 可扩展"，而非 Anthropic 式纯 block 列表。**
理由：扁平结构 5 家厂商都能一对一映射，适配器代码量最小；content 联合类型保留了多模态演进空间。代价：多模态混排（同一消息里文字+图片+工具结果交错）表达力弱——W10 若真需要，再加 `blocks` 分支，届时只影响 adapter。

### 3.2 ToolSpec

| 字段 | 说明 |
|---|---|
| `name` | `snake_case`，全局唯一；与 Skill 名同源（W8 直接复用） |
| `description` | **必填且要写成"什么时候该用/不该用"**。工具描述质量直接决定选择准确率，W9 会用评测数据证明这句话 |
| `parameters` | JSON Schema；顶层必须是 `object`，`additionalProperties: false` |
| `strict` | 是否要求厂商强约束（仅部分厂商支持，属 capability） |

### 3.3 ToolCall（统一后）

| 字段 | 说明 |
|---|---|
| `id` | 厂商返回则沿用；不返回则本地生成（`call_<uuid8>`），**必须**稳定贯穿到 `tool` 消息 |
| `name` | 工具名 |
| `arguments` | **一定是 `dict`**。厂商返回 JSON 字符串时由 adapter `json.loads`；解析失败 → `arguments={}` 且 `arguments_error` 非空，**不许抛异常打断整轮** |
| `raw_arguments` | 原始字符串，调试与审计用 |

### 3.4 LLMResponse

`text` / `tool_calls` / `finish_reason` / `usage` / `model` / `provider` / `response_id` / `latency_ms` / `raw_finish_reason`

`finish_reason` 统一枚举（映射表必须写进 adapter，不允许透传厂商字符串）：

| 内部值 | OpenAI | Anthropic | 语义 |
|---|---|---|---|
| `stop` | `stop` | `end_turn` / `stop_sequence` | 正常结束 |
| `length` | `length` | `max_tokens` | 触顶截断 |
| `tool_calls` | `tool_calls` / `function_call` | `tool_use` | 要调工具 |
| `content_filter` | `content_filter` | `refusal` | 被拦截 |
| `unknown` | 其他 | 其他 | 保留原值到 `raw_finish_reason` |

### 3.5 StreamEvent

| `kind` | 载荷 | 说明 |
|---|---|---|
| `text_delta` | `text` | 增量文本 |
| `tool_call` | `ToolCall` | **完整**工具调用（见 ADR-002） |
| `usage` | `Usage` | 通常在末尾出现，缺失时由 client 估算并标记 `estimated=True` |
| `error` | `LLMError` | 流中断 |
| `done` | `finish_reason` | 终止帧 |

**ADR-002：tool_call 增量由 adapter 内部组装，只在参数完整后发一个 `tool_call` 事件，不向上发 delta。**
理由：OpenAI 按 `index` 累加、Anthropic 按 `content_block_start/delta/stop` 事件流，两家拼装逻辑完全不同；把这套复杂度关在 adapter 里，W9 的 planner 只需消费完整 ToolCall。代价：工具参数大到一定程度时首字节延迟变高——可接受，因为工具调用不面向人眼。
**同时约定**：`text_delta` 结束后才可能出现 `tool_call`（顺序保证），消费方可安全按序处理。

### 3.6 错误模型与 HTTP 映射

| 内部异常 | 触发 | retryable | 对外 HTTP | 对外 code |
|---|---|---|---|---|
| `UpstreamAuthError` | 厂商 key 无效/欠费 | ❌ | **502** | `UPSTREAM_AUTH` |
| `RateLimitError` | 429 | ✅ | 429（带 `Retry-After`） | `RATE_LIMITED` |
| `UpstreamTimeoutError` | 连接/首 token 超时 | ✅ | 504 | `UPSTREAM_TIMEOUT` |
| `BadRequestError` | 400/参数非法 | ❌ | 400 | `BAD_REQUEST` |
| `ContentFilteredError` | 被拦截 | ❌ | 422 | `CONTENT_FILTERED` |
| `UpstreamServerError` | 5xx | ✅ | 502 | `UPSTREAM_ERROR` |
| `CapabilityError` | 模型不支持该能力 | ❌ | 400 | `CAPABILITY_UNSUPPORTED` |

**ADR-005：上游鉴权失败绝不映射为 401。** 我们的 401 只表示"调用方没通过我方鉴权"（W10 才引入 API Key）。把上游 key 问题映射成 401 会让调用方以为自己凭证错了——这是线上排障最常见的误导。

所有异常统一携带：`provider`、`model`、`status_code`、`request_id`、`retry_after`，日志与响应体都能追。

### 3.7 重试语义（本周最容易做错的地方）

- **只允许在首 token 之前重试。** 一旦有 `text_delta` 发出，任何错误都只能以 `error` 事件终止，绝不能重放。
- 可重试：`RateLimitError` / `UpstreamTimeoutError` / `UpstreamServerError` / 连接错误。
- 不可重试：4xx（除 429）、`ContentFilteredError`、`CapabilityError`、已开始输出。
- 退避：`0.5s × 2^n` + 抖动，最多 3 次，尊重 `Retry-After`。
- 流式的判定边界 = `first_token_timeout`（配置项，默认 10s），不是整条请求超时。

**ADR-003：重试只在 `client.py` 实现一次，adapter 内部禁止自带重试。**
理由：adapter 自带重试会导致"3 provider × 3 次重试 = 9 次真实调用"的成本黑洞，且日志重复计数。

---

## 4. 配置、路由与成本

### 4.1 配置分层

- **密钥**：只走环境变量（`OPENAI_API_KEY` / `DASHSCOPE_API_KEY` / `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY`），绝不进 yaml、绝不进日志（日志里 key 一律打码为 `sk-***last4`）。
- **模型与价格**：`models.yaml`（随包发布），结构如下（**你来落盘，字段按此定义**）：

```yaml
providers:
  openai:   { type: openai_compat, base_url: https://api.openai.com/v1, api_key_env: OPENAI_API_KEY }
  deepseek: { type: openai_compat, base_url: https://api.deepseek.com/v1, api_key_env: DEEPSEEK_API_KEY }
  qwen:     { type: openai_compat, base_url: https://dashscope.aliyuncs.com/compatible-mode/v1, api_key_env: DASHSCOPE_API_KEY }
  anthropic:{ type: anthropic,     base_url: https://api.anthropic.com,   api_key_env: ANTHROPIC_API_KEY }

models:
  gpt-4o-mini:
    provider: openai
    tier: cheap
    capabilities: [stream, tools, json_mode]
    pricing: { input: 0.15, output: 0.60, currency: USD, per: 1_million_tokens }
  # ... 每个模型都要有 tier / capabilities / pricing

routes:
  default:      { primary: gpt-4o-mini, fallback: [deepseek-chat] }
  cheap:        { primary: deepseek-chat, fallback: [qwen-plus] }
  strong:       { primary: claude-sonnet-4-6, fallback: [gpt-4o] }
tasks:
  chat: cheap
  research: strong
```

### 4.2 路由契约

`Router.route(task: str) -> list[str]`：返回**候选模型有序列表**（首个为主，其余为 fallback 链）。
- W1 只实现"按 task → tier → 候选列表"这一种策略。
- fallback 触发条件与重试条件一致，且**全链只允许绕一圈**（禁止 A→B→A）。
- 每次 fallback 必须打日志（`fallback_from` / `fallback_reason`），W11 的看板靠它统计"降级率"。

### 4.3 成本计量

`cost = input_tokens × price_in + output_tokens × price_out`，货币与单位取自 `pricing`。
- usage 缺失时按字符数粗估，并打 `usage_estimated=true`。
- 计量结果写入 `observability`，**同时**计入本次请求的日志行——W11 的成本熔断直接读这里。

---

## 5. 可观测（W1 最小版，但是全项目的根）

每条请求**必须**产出一条结构化日志（JSON），字段固定：

`request_id`、`task`、`tier`、`provider`、`model`、`fallback_from`、`prompt_tokens`、`completion_tokens`、`cost`、`currency`、`latency_ms`、`first_token_ms`、`retry_count`、`finish_reason`、`error_code`、`stream`

两条硬要求：
1. `request_id` 从 HTTP 入口注入，贯穿到每条 provider 日志（用 contextvar，不许层层传参）。
2. **`first_token_ms` 必测**——它是 DoD 里 p95 ≤ 1.5s 的唯一数据来源，W1 不埋，W11 补不上历史数据。

---

## 6. 测试策略（本周的"证据"来源）

| 类型 | 目标 | 是否联网 |
|---|---|---|
| 契约单测 | types / router / 错误映射 / SSE 编解码 | 否 |
| **Provider 一致性测试（conformance）** | **同一组断言在 3 provider 上参数化跑** | 否（用录制夹具） |
| 适配器回放测试 | 用真实录制响应验证解析（含 tool_call 拼接、流式事件序列） | 否 |
| 真机冒烟 | 3 家各跑 1 次，人工确认 + 记录延迟成本 | 是（手动，`-m live`） |

**conformance 套件是本周最有价值的资产**（面试可直接讲）：它把"归一化是否真的成立"变成可执行断言，而不是口头声明。断言至少覆盖：
1. 同一 `messages` + 同一 `ToolSpec`，三家解析出的 `ToolCall.name` / `arguments` 完全一致
2. `finish_reason` 在 tool_calls 场景下三家都归一为 `tool_calls`
3. 流式输出拼接后与一次性输出等价（同一夹具）
4. usage 字段三家都能映射（缺失时 `estimated` 标记）

夹具生成方式：真机跑一次 → 存原始 response body 到 `tests/fixtures/<provider>/`；`mock.py` 读夹具回放。**W9 之后所有评测都跑在这套夹具上，不烧钱、不受外部波动影响。**

---

## 7. W1 明确不做（review 时我会检查是否越界）

- ❌ 工具执行循环（只做归一化，不 `execute`）——那是 W8/W9
- ❌ RAG / 记忆 / Skills
- ❌ 我方 API 鉴权（W10）
- ❌ 语义缓存、熔断器（W11）
- ❌ 流式取消/断点续传（W9 再考虑，但要预留 `stream` 的 cancel 语义）
- ❌ LangChain / LangGraph / 任何 Agent 框架
- ❌ `/v1/chat` 走 Temporal（流式走直连，长任务才走工作流）
- ❌ 多模态、图片、文件

---

## 8. 为后续周预留的扩展点（本周只留缝，不实现）

| 扩展 | 插入位置 | 生效周 |
|---|---|---|
| 新 provider | 新增 `providers/x.py` + registry 注册，**不改 client** | 随时 |
| 新路由策略（按成本/延迟/负载） | 实现 `Router` 协议 | W11 |
| Guardrail 中间件（注入检测） | `client` 装饰器链 | W10 |
| 工具执行 / Skill / MCP | 新增 `skills/`，消费 `ToolCall` | W8 |
| Trace 上报 Langfuse | `observability` 增加 exporter，日志字段已就位 | W3 |
| 记忆与知识库 | 编排层注入 context，契约不变 | W5–W7 |

---

## 9. ADR 摘要（W12 拆成独立文件）

| 编号 | 决策 | 主要代价 |
|---|---|---|
| ADR-001 | 内部消息扁平结构 + content 可扩展 | 多模态混排表达力弱 |
| ADR-002 | tool_call 由 adapter 组装完整后单次发出 | 工具参数大时首字节延迟升高 |
| ADR-003 | 重试只在 client 实现，adapter 禁止重试 | client 需感知流式边界 |
| ADR-004 | 框架与业务同仓不同包、依赖单向 | 需守住 import 方向 |
| ADR-005 | 上游鉴权失败映射 502 而非 401 | 调用方需按 code 区分错误 |
| ADR-006 | 协议基准选 `/chat/completions`，不实现 Responses API（详见 `w1/llm-api-matrix.md` §5） | 避开"仅 Responses 支持 tool calling"的型号 |
| ADR-007 | `Message.provider_meta` 透传 `reasoning_content` / `thinking.signature`，业务层禁止解读 | 多一个不透明字段，需防泄漏到对外响应 |
| ADR-008 | 客户端层用编排方法而非装饰器链（详见 `w1/w1-day4.md` §7.1） | 单方法较长，靠私有方法控制可读性 |
| ADR-009 | alias 与 wire model 名分离；允许混币种但 `currency` 必进日志 | 排查时要同时看两个名字 |
| ADR-010 | 夹具分两级：`_synthetic/` 永久保留手写基线，真录制带 `synthetic=false + recorded_at`（详见 `w1/w1-day5.md` §5） | 夹具目录翻倍，需维护 A13 键路径比对 |
| ADR-011 | 录制走 transport 层 tee，不另写 HTTP 客户端，且不覆盖 `request.json` | Registry 多注入点；漂移需人工裁决 |
| ADR-012 | 日志在应用边界配置，指标读取方挂 handler 消费同一条 `RequestLog` | 每个可执行入口都要调一次，需幂等 |
