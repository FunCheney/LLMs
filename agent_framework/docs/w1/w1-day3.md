# Day 3 设计与验收（三个适配器 + conformance 套件）

> 角色不变：**我定结构与验收，你写代码**。本文只给契约、边界、断言与自查方法，不含实现。
> 依赖：`architecture.md`（数据契约）、`w1-tasks.md` T3.x（任务清单）、`llm-api-matrix.md`（四家差异）。

---

## 0. 开工前硬门槛（没满足就别开 Day3）

Day3 的全部代码都长在 Day2 的契约上。以下 4 项缺一项，今天下午必然返工：

| 前置 | 为什么 Day3 强依赖它 | 缺了会怎样 |
|---|---|---|
| `types.py` 有 **`Message.provider_meta`**（ADR-007） | DeepSeek 的 `reasoning_content`、Anthropic 的 `thinking.signature` 要有地方放 | 第二轮 400；或把它拼进 `content` → 污染评测 |
| `finish_reason` 枚举已扩到 `aborted` / `upstream_error` | DeepSeek 的 `insufficient_system_resource` 无处可落 | 半截回答被当成 `stop` 成功 |
| `errors.py` 有"状态码/错误体 → 内部异常"的**单一映射函数** | 三家欠费落在 429 / 402 / 400-403 三个位置 | 每个 adapter 各写一份 → 判重试逻辑不一致 |
| `base.py` 的 `LLMClient` 抽象 + `Capability` 枚举 | 三个 adapter 必须同签名，conformance 才能参数化 | 参数化写不出来，只能写三套测试 |

**修正一处我上轮的不严谨**：昨日文档里写过 `UpstreamQuotaError`，与 `architecture.md` §3.6 冲突。**以架构为准**：欠费（OpenAI `insufficient_quota` / DeepSeek 402 / Qwen `Arrearage`）统一归 **`UpstreamAuthError`**（`retryable=False`，对外 502 `UPSTREAM_AUTH`），**不新增异常类**。文档已同步改。

> 若 `provider_meta` 还没加：今天先花 30 分钟补，再开工。这是唯一"现在改 10 行、以后改 10 个文件"的项。

---

## 1. 今天的产出边界

**算完成**（Gate 2 只认这些）：

- `providers/openai_compat.py` / `anthropic.py` / `mock.py` 三个 client
- `providers/_common.py`（只放纯函数）
- 每家的 **ToolCall 组装器**（可离线单测的状态机）
- `tests/fixtures/<provider>/<case>/` 夹具目录约定 + 最小夹具集
- conformance 套件 **A1–A8 全绿**，**断网可跑、零真实调用**
- `pytest` / `ruff` / `mypy` 三项干净

**不算完成**（今天故意不做）：

- `client.py`（重试/计量/日志）、`router.py`、`models.yaml`、`/v1/chat` HTTP 接口 → **Day4**
- 真机录制夹具（今天先手写，**必须标 `synthetic: true`**）→ **Day5** 替换
- Anthropic 的 `pause_turn`/`refusal` 真实行为验证 → Day5 真机确认

---

## 2. 文件布局与行数预算

```text
packages/agent_framework/src/agent_framework/llm/providers/
├── __init__.py
├── _common.py            # 纯函数区（无状态、无 IO）
├── openai_compat.py      # 覆盖 openai / deepseek / qwen 三家
├── anthropic.py
└── mock.py               # 夹具回放器（复用真 adapter 的 parse）

packages/agent_framework/tests/
├── conftest.py           # 夹具加载器 + MockTransport 工厂
├── fixtures/
│   ├── openai/{tool_call_basic,text_basic,stream_tool_call,error_429,usage_missing}/
│   ├── deepseek/{...同上 + tool_call_thinking}/
│   ├── qwen/{...同上}/
│   └── anthropic/{tool_call_basic,text_basic,stream_tool_call,error_529,thinking_block}/
├── test_openai_compat.py     # 单家细节（宽容/容错/心跳/usage 形状）
├── test_anthropic.py         # 单家细节（tool_result 合并/ping/usage 分段）
└── test_conformance.py       # A1–A8，参数化跑全部 provider
```

| 文件 | 职责 | 行数预算 | 超了说明什么 |
|---|---|---|---|
| `openai_compat.py` | 三家 wire ↔ internal + 差异开关 | 250–320 | 超 350 = 把 client 的事写进来了 |
| `anthropic.py` | 同上（结构不同） | 200–260 | 超 300 = 复制粘贴了 openai 的分支 |
| `mock.py` | 读夹具 → 走真 parse | 80–120 | 超 150 = 自己在造 `LLMResponse`（见 §6.6） |
| `_common.py` | SSE 行解析 / arguments 解析 / 错误体规范化 / tool_choice 映射 | 100–140 | 超 180 = 混进了 provider 专属逻辑 |

> **`_common.py` 的准入标准**：函数必须"**三家以上都要用，且不含任何厂商专属字段名**"。只要出现 `reasoning_content`、`input_schema`、`enable_thinking` 任何一个词，就不许放进来。

---

## 3. 三条不可破的边界（我会逐条 grep 检查）

| 边界 | 违反的典型写法 | 自查命令（打到 0 结果才算过） |
|---|---|---|
| adapter 不做重试/退避 | `for attempt in range(3)` / `asyncio.sleep` | `grep -rn "sleep\|retry\|backoff" src/agent_framework/llm/providers/` |
| adapter 不打日志/不做计量 | `logger.` / `cost` / `price` 出现在 providers | `grep -rn "logger\.\|cost\|price" src/agent_framework/llm/providers/` |
| 依赖单向：providers 只依赖 types/errors/_common | `from ..client import` / `from ..config import` | `grep -rn "from \.\." src/agent_framework/llm/providers/` |

**唯一的例外**：`mock.py` 允许读 `tests/fixtures/`（通过构造参数传路径，不写死）。

**为什么这么严**：Day4 的 `client.py` 要对三个 adapter 无差别地加重试、计量、fallback。只要有一个 adapter 自带 `sleep`，你就会出现"3 provider × 3 retry = 9 次真实调用"的成本黑洞，而且日志会重复计数（ADR-003）。

---

## 4. 适配器的内部形状（今天最关键的设计决策）

### 4.1 每家都拆成三段，且三段都必须是"无 IO 的纯函数/纯类"

| 段 | 签名（伪代码，不含实现） | 输入 | 输出 | 能否离线单测 |
|---|---|---|---|---|
| 请求构造 | `build_request(messages, tools, **opts) -> dict` | 内部对象 | wire dict | ✅ 直接断言 dict |
| 响应解析 | `parse_response(body: dict) -> LLMResponse` | wire dict | 内部对象 | ✅ 直接喂夹具 |
| 流式组装 | `assembler.feed(event: dict) -> list[StreamEvent]` | **已 json 化的 SSE payload** | 事件列表 | ✅ 不用 async、不造流 |

**唯一碰网络的地方**是 `async def stream()/chat()` 里那一层薄壳：发请求、逐行读、把 `(event_name, payload)` 交给上面三段。**薄壳里不许有业务逻辑**——因为薄壳是唯一无法离线测的部分，逻辑写进去就等于不可测。

### 4.2 为什么要单独抽 `ToolCallAssembler`（而不是在 `async for` 里攒变量）

| 写法 | 后果 |
|---|---|
| 在 async generator 里用局部变量累 `arguments` | 只能靠"造一个假 async 流"来测；拼装 bug（丢片、错位、`[DONE]` 崩）只能在集成测试里偶然发现 |
| 抽成状态机类，只吃 dict | 可以拿 20 行夹具直接断言"喂 5 个 chunk，出 1 个完整 ToolCall"；断流、乱序、重复 index 全能覆盖 |

**两家状态机的最小状态**（这是本节的干货）：

| | OpenAI / DeepSeek / Qwen | Anthropic |
|---|---|---|
| 分片标识 | `delta.tool_calls[].index` | `content_block_start` 的 `index` |
| 累加字段 | `function.arguments`（字符串，可能一篇为空） | `input_json_delta.partial_json`（字符串分片） |
| id/name 来源 | **首片**才有，后续片没有 → 必须缓存 | `content_block_start` 一次给全 |
| 结束信号 | `finish_reason` 非 null 或 `[DONE]` | `content_block_stop`（**每个 block 各一次**） |
| 易错点 | 用 `id` 当 key（后续片没有 id）→ 第二个工具调用被并到第一个 | 一个 response 里可能有 text block + 2 个 tool_use block → 用 `index` 做 key，不能只用"当前块" |

**`feed()` 的返回约定**（写进 docstring，conformance 会验）：**只在参数完整时才产出 `tool_call` 事件**（ADR-002），且 `text_delta` 全部先于 `tool_call`（顺序保证）。

### 4.3 SSE 行解析统一收进 `_common.py`

输入是 `bytes` 行，输出是 `(event_name | None, payload | None)` 或"跳过"。必须处理的五种行：

| 行 | 处理 | 谁会产生 |
|---|---|---|
| 空行 | 跳过（SSE 事件分隔符） | 全部 |
| `: keep-alive` 等 **`:` 开头** | 跳过（注释，**不是 JSON**） | **DeepSeek**（长思考时） |
| `event: xxx` | 记为当前事件名 | **仅 Anthropic** |
| `data: [DONE]` | 终止哨兵，**先判等再 parse** | OpenAI / DeepSeek / Qwen |
| `data: {...}` | `json.loads` → payload | 全部 |

**这是 Day3 唯一允许"三家共用"的解析代码**，也是唯一值得写满边界用例的函数（`: ` 无空格、`data:` 无空格、一行多个 `data:`、超长行）。

### 4.4 `provider_meta` 的写入与读取位置（ADR-007 落地）

| 时机 | 谁写 | 写什么 |
|---|---|---|
| 解析响应/流结束时 | adapter 的 `parse_response` / assembler | DeepSeek & Qwen 3.8：`{"reasoning_content": ...}`；Anthropic：`{"thinking_blocks": [...含 signature...]}` |
| 构造请求时 | adapter 的 `build_request` | 按 `models.yaml` 的 flag 决定**是否回传**（Qwen 3.7 不回传、3.8 必须回传、DeepSeek 凡是带 tools 就必须回传） |
| 业务层 | **禁止读写** | 只做整体透传 |

**断言（进 conformance A7）**：`LLMResponse` 的 `text`、`tool_calls` 里**不得出现** `reasoning_content` 的内容；它只能在 `provider_meta` 里。

---

## 5. 三家今天必须做对的"最小集合"

### 5.1 `openai_compat.py`

**请求侧 4 件事**：`tools` 包一层 `type/function` 且 schema 字段叫 `parameters`；`tool_choice` 字符串↔对象双向映射；`parallel_tool_calls` **永远显式发送**；`stream` 时**必须带** `stream_options.include_usage=true`。

**差异开关（全部数据驱动，禁止 `if "qwen" in base_url`）**：

| flag | openai | deepseek | qwen | 不处理会怎样 |
|---|---|---|---|---|
| `send_parallel_tool_calls` | `true` | **不发**（无此参数） | **`true`**（默认 false） | Qwen 并行静默失效 |
| `supports_tool_choice_required` | `true` | `false`（思考模式 400） | `false` | 400 / 被打断 |
| `thinking_style` | `none` | `thinking:{type}`（**默认开**） | `enable_thinking`（**非流式 400**） | 400 / 成本暴涨 |
| `stream_keepalive_comment` | `false` | `true` | `false` | 解析器崩在心跳 |
| `preserve_reasoning_content` | `false` | `always` | `per_model`（3.7 否 / 3.8 是） | 400 或效果下降 |
| `error_code_map` | 标准 | +402/+422 | +`Arrearage` | 欠费被当参数错误重试 |

> flag 的**来源**是 `models.yaml`（Day4 落盘），今天先在 adapter 构造函数上留 `flags: dict` 注入点，测试里直接构造。**不要今天就去写 yaml**。

**响应侧 3 件事**：`content` 可能为 `null`；`arguments` 是 JSON **字符串**（失败降级不抛）；`tool_calls` 可能多个且 `id` 与 `index` 是两套标识。

**流式 3 件事**：按 `index` 累加、**usage 覆盖式记录**（不看 `choices` 是否为空）、跳过注释行。

### 5.2 `anthropic.py`

| # | 必须做对 | 不做的后果 |
|---|---|---|
| 1 | `system` 提到**顶层参数**，多条 system 合并成一条 | 400（messages 里不许有 system 角色） |
| 2 | **`max_tokens` 必填**（给出默认值 + 超限时明确报错） | 400，且报错信息不指向根因 |
| 3 | 连续多条 `tool` 消息 → **合并成一条 user 消息**，内含全部 `tool_result`，**一次给齐** | 400，报错说 `tool_result` 缺失，极易误判成 id 不匹配 |
| 4 | 工具失败要带 `is_error: true` | 模型会把错误当正常结果继续推理 |
| 5 | 工具定义 `input_schema`（且顶层 `type: object`）；`tool_choice` 是**对象** | 400 |
| 6 | 流式：`message_start` 取 `usage.input_tokens`；`message_delta` 取 `usage.output_tokens`（**累计值，取最后一次**）+ `stop_reason`；`ping` 忽略；`message_stop` 收尾 | 成本系统性偏低；`ping` 被当业务事件 |
| 7 | `thinking` block 的 `signature` 进 `provider_meta`，回传时原样带回 | 多轮思考场景 400 |
| 8 | `finish_reason` 映射：`end_turn/stop_sequence→stop`、`max_tokens→length`、`tool_use→tool_calls`、`refusal→content_filter`、`pause_turn→stop`，其余 → `unknown` + `raw_finish_reason` | 透传厂商字符串，上层判断失效 |

### 5.3 `mock.py`（今天最容易写错、也最容易写出"假绿"的一个）

**它不是一个假 LLM，而是一个 replay 引擎。** 核心设计决定：

| 做法 | 评价 |
|---|---|
| mock 自己拼一个 `LLMResponse` 返回 | ❌ **禁止**。等于"自己跟自己对答案"，真机的解析 bug 一个都测不出来 |
| mock 回放 **wire 格式**（真实 body / SSE 行），**复用真 adapter 的 `parse_response` / assembler** | ✅ 唯一正确做法 |

**必须能模拟的 5 类异常输入**（比 happy path 重要得多）：空 `content`、坏 JSON 的 `arguments`、缺 `usage`、流中断（有 `text_delta` 无 `done`）、错误响应体（429/402/529 各一）。

**夹具目录约定**（今天定死，Day5 只填内容）：

```text
fixtures/<provider>/<case>/
├── meta.json      # {"synthetic": true, "recorded_at": null, "note": "手写，Day5 替换"}
├── request.json   # 期望我们发出的请求体（用于断言 build_request）
├── response.json  # 非流式原始 body
└── stream.txt     # 原始 SSE 原文（含 data:/event:/注释行，逐字节保真）
```

`stream.txt` **必须是原文而不是解析后的 JSON 数组**——否则你测的是自己的解析结果，不是真实协议。

---

## 6. conformance 套件怎么设计（今天最值钱的资产）

### 6.1 骨架思路

一个"用例" = **同一份内部输入 + 每家一份 wire 夹具** → 断言"解析回内部后，三家结果语义相等"。

不是断言"三家 wire 相同"（那不可能），而是断言**经过各自 adapter 之后，内部表示一致**。

### 6.2 用 `httpx.MockTransport`，不要 monkeypatch

| 方式 | 测到了什么 |
|---|---|
| monkeypatch 掉 adapter 的 `_post` | 只测到 parse，**header、URL、body 序列化、SSE 逐行解析全没测** |
| `httpx.MockTransport(handler)` 按夹具返回原始 bytes | 真 HTTP 语义全走一遍，且**零网络**、可断言"发了几次请求" |

后者才能支撑 A8（单请求断言）。

### 6.3 八条断言（A1–A8）

| # | 断言 | 防的假绿 |
|---|---|---|
| **A1** | 同一 `ToolSpec` 经三家 `build_request` 后，**wire 形态各不相同但语义等价**（用各自的 `request.json` 精确比对，而不是"包含 name 即可"） | "只要字段出现就算过" |
| **A2** | 同一 `tool_call` 语义夹具 → 三家 `ToolCall.name` 相同、`arguments` **是 dict 且相等**、`id` 非空 | 只断言 `arguments` 不是 None |
| **A3** | **流式拼接结果 == 非流式结果**（text + tool_calls + finish_reason 三者全等） | 只比 text；tool_call 流式拼接 bug 逃逸 |
| **A4** | `usage` 三家都能映射；**缺失时** `estimated=True` 且 token 数 > 0 | 只测"有 usage"的正例（DeepSeek 形状差异测不到） |
| **A5** | `finish_reason` 归一：`tool_use`/`tool_calls`→`tool_calls`；`max_tokens`/`length`→`length`；**DeepSeek `insufficient_system_resource`→`upstream_error`**；`refusal`→`content_filter`；未知值 → `unknown` 且 `raw_finish_reason` 保留 | 只测 `stop` 一个值 |
| **A6** | 同一错误语义 → **同一个异常类 + 同一个 `retryable`**：429 限流(✅重试) / 欠费(❌不重试) / 5xx(✅) / 400(❌) 四组，覆盖 429、402、400-403 三种落点 | 只测 500 |
| **A7** | 内部对象**无厂商字段泄漏**：`text` 与 `tool_calls` 序列化后不含 `reasoning_content` / `signature` / `partial_json`；它们只在 `provider_meta` | 厂商字段混进 content 污染评测 |
| **A8** | **一次 `chat()` 只发 1 次 HTTP**（transport 计数 == 1），一次 `stream()` 也是 1 次 | adapter 偷偷加重试（ADR-003） |

> A8 是这周最容易被忽略、又最能证明"你真的懂架构"的一条：它把 ADR-003 从文档变成断言。

### 6.4 参数化的写法要求

- 维度：`provider × case`，`pytest.param(..., id="anthropic::tool_call_basic")` —— 失败时一眼知道是哪家哪个用例
- **断言里不许出现 `if provider == ...`**：出现即说明你断言的是"厂商差异"，而不是"归一化成立"。差异应该已被 adapter 吸收，测试层看不到
- 单家特有行为（心跳、`ping`、`input_json_delta`）放**各自的 `test_<provider>.py`**，不要污染 conformance

### 6.5 最小夹具集（今天先手写，Day5 换真录制）

| case | 覆盖断言 | 四家都要 |
|---|---|---|
| `text_basic` | A3、A4 | ✅ |
| `tool_call_basic` | A1、A2、A5 | ✅ |
| `stream_tool_call` | A2、A3、A8 | ✅ |
| `usage_missing` | A4 | ✅ |
| `error_rate_limit` / `error_quota` / `error_5xx` | A6 | ✅（每家的状态码不同，正好验证映射表） |
| `tool_call_thinking` | A7 | DeepSeek、Qwen、Anthropic |
| `stream_keepalive` | SSE 解析 | **仅 DeepSeek** |
| `stream_ping` | SSE 解析 | **仅 Anthropic** |

### 6.6 "假绿"的三大来源（今天必须主动规避）

1. **手写理想夹具**：自己编一份"完美响应"，等于自己跟自己对答案。→ 强制 `meta.json` 里 `synthetic: true`，**Gate 3 前必须全部换成 Day5 真机录制**，否则周验收不算过
2. **mock 自己造 `LLMResponse`**：mock 与真路径分裂，真机一跑全崩。→ mock 只回放 wire，复用真 parse
3. **只测 happy path**：坏 `arguments`、缺 usage、流中断、空 content，四个都必须有正例反例

---

## 7. 三小时时间盒（顺序不能换）

| 时间 | 做什么 | 完成标志 |
|---|---|---|
| 0:00–0:40 | `_common.py`（SSE 行解析 / arguments 解析 / 错误体规范化）+ `openai_compat` 请求侧与非流式解析 | `text_basic`、`tool_call_basic` 在 openai 上过 |
| 0:40–1:10 | OpenAI 组装器状态机 + 单测（喂 chunk 序列） | 5 个 chunk → 1 个完整 ToolCall |
| 1:10–1:50 | `anthropic.py` 非流式（system 顶层 / tool_result 合并 / max_tokens） | `tool_call_basic` 在 anthropic 上过 |
| 1:50–2:20 | Anthropic 组装器（`content_block_*` + 分段 usage + `ping`） | `stream_tool_call` 过 |
| 2:20–2:40 | `mock.py` + 夹具目录 + `conftest.py` | 一条用例能用 mock 跑通 |
| 2:40–3:00 | conformance A1–A8 参数化，全绿 | `pytest -q` 全绿且断网可跑 |

**超时降级**（按此顺序砍，不要自己发明）：先砍 Anthropic 的流式组装（推到 Day4 上午 30 分钟），再砍 DeepSeek/Qwen 的思考模式夹具；**绝不砍 A3/A7/A8 三条断言**——它们是这套套件的价值所在。

---

## 8. 验收（Gate 2）与我会打回的写法

### 你必须给我的东西

1. `pytest -q` 全绿输出（**先在断网状态跑一遍**：`networksetup -setairportpower en0 off` 或直接拔网线，证明零真实调用）
2. `tree -L 4 packages/agent_framework/tests/fixtures`
3. 三个 adapter 的文件路径 + 行数（`wc -l`）
4. `grep` 三条自查的**空结果**（见 §3）
5. 一条 A8 的失败示范：故意在 adapter 里加一次 `await self._post(...)`，看断言是否变红（证明断言真的有效）

### 八条验收

- [ ] 同一段业务代码（构造 client + chat）改 `flags` 即可在 openai / deepseek / qwen 三个配置上跑
- [ ] A1–A8 全部参数化通过，**测试里没有任何 `if provider ==`**
- [ ] 流式与非流式在**同一 case** 上结果全等（A3）
- [ ] 坏 `arguments` 不抛异常：`arguments={}` + `arguments_error` 有值 + 整轮不断
- [ ] 缺 usage 时 `estimated=True`；DeepSeek 的"末块挂 usage"形状能读到值
- [ ] 欠费 / 限流 / 5xx / 参数错误四组 → 异常类与 `retryable` 全对
- [ ] 断网 `pytest` 全绿；`ruff` / `mypy` 干净
- [ ] 每个夹具目录有 `meta.json`，手写夹具标了 `synthetic: true`

### 我会直接打回的 6 种写法

| 写法 | 为什么打回 |
|---|---|
| `if "qwen" in base_url` / `if provider == "deepseek"` | 差异必须走 flags，否则加第 5 家要改 4 个分支 |
| adapter 里 `asyncio.sleep` / `for attempt in range(3)` | 违反 ADR-003，成本黑洞 |
| 在 async generator 里攒 `arguments` | 拼装逻辑不可单测，A3 一定不稳 |
| `mock.py` 里 `return LLMResponse(text=...)` | 假绿源头 |
| `parse_response` 里 `body["choices"][0]` 直接取 | 不防御 → 一次异常响应打崩整个服务 |
| 把 `reasoning_content` 拼进 `text` | 污染评测与用户可见输出（W10 红线） |

---

## 9. 五个坑（提前避）

1. **Anthropic 的 `tool_result` 合并只放 adapter**：一旦你"顺手"在 `types.py` 里加个合并辅助，其他三家就为它付账（架构 §2 禁止事项）
2. **`max_tokens` 忘了给 Anthropic**：建议 adapter 构造时要求显式传，**没有默认值就直接抛 `BadRequestError`**——静默兜底会让"上游截断"变成"我们莫名截断"
3. **`stream_options` 不是三家都吃**：Anthropic 没有这个字段，传了不影响但别写进公共路径
4. **DeepSeek 思考模式默认开**：`tool_choice: required` 的用例在它上面必 400 → 这正是 A6/A5 要覆盖的"能力降级"，别把它当 bug 修
5. **夹具用"解析后的 JSON"而非原文**：测的是自己的解析器，等于自证；`stream.txt` 必须逐字节保真

---

## 10. 给 Day4 留的三个接口（今天先留缝，不实现）

| 接口 | 形态 | Day4 谁消费 |
|---|---|---|
| `LLMClient.chat(messages, tools, **opts) -> LLMResponse` | 三个 adapter 已同签名 | `client.py` 加重试/计量/日志 |
| `LLMClient.stream(...) -> AsyncIterator[StreamEvent]` | 首 token 前可重试、之后不可 | `client.py` 判 `first_token_timeout` |
| `flags: dict` 注入 | 构造函数可选参数 | `models.yaml` → `registry.py` 组装 |

> 今天结束时的判据：**换一家 provider，只改一个文件。** 如果为了接 Anthropic 你动了 `types.py` 或计划改 `client.py`，说明边界漏了。
