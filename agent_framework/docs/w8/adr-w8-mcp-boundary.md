# ADR-W8-001：MCP 与 Skill 分层，业务层不 import MCP

- 状态：已采纳（W8 D1）
- 日期：2026-11-11
- 相关：`skills/mcp/{types,client,inmemory}.py`、`tests/test_mcp_boundary.py`、ADR-003（重试只在 client）

---

## 背景

W8 要把外部工具接进来。市场 54% 的 JD 点名 MCP，所以「接得上」不是问题；
**接完之后边界还在不在**才是问题。W1 定下的 `ToolSpec` / `ToolCall` / `ToolResult`
是内部契约，`LLMGateway` 和产品层都按它写。如果 MCP 的 JSON-RPC 结构
直接流到上层，会立刻出现三件坏事：

1. 上层开始判断 `-32601` 是什么——错误语义从「业务失败」退化成「协议细节」；
2. 换一个 MCP 实现（stdio → SSE → HTTP）就要改业务代码；
3. 对外的 4xx/5xx 响应里带上厂商字段名和 server 路径。

---

## 决策

**MCP = 外部工具传输协议；Skill = 内部能力封装。两者不是一回事，中间必须有一层翻译。**

```
业务 / 产品层        ToolSpec / ToolCall / ToolOutcome      ← 只认这些
                          ↑
skills/mcp/client.py  翻译：命名空间、错误码、超时、成本
                          ↑
skills/mcp/types.py   wire：JSON-RPC result 体的受控解析
                          ↑
MCPTransport          list_tools() / call_tool() → 原始 dict
                          ↑
              stdio / SSE（W11）· InMemory（测试）· Recorded（离线档）
```

### 1. `types.py` 只做三件事

- **受控解析**：`parse_tool_list` / `parse_call_result` 把 wire 结构解析成 dataclass。
  畸形字段丢进 `warnings`，**不抛异常也不猜**。一个 server 少写 `inputSchema`
  不该让整次 `tools/list` 失败，但也绝不能凭空给它编一个 schema。
- **`MCPTransport` 协议**：返回原始 dict。真实实现（stdio/SSE）留到 W11，
  测试走 `InMemoryMCPTransport`，离线档走 `RecordedToolTransport`——
  三种实现的差别被这个协议吸收。
- **命名规则写在一处**：`internal_name(server, tool)`。

### 2. 命名空间用 `__`，不用 `.`

MCP 的 tool 名只在 server 内唯一。两个 server 都叫 `search_web` 时内部必须能区分。
而 OpenAI 与 Anthropic 的工具名都限制在 `^[a-zA-Z0-9_-]{1,64}$`——
用不了 `.` 和 `:` 这些更自然的分隔符。`__` 是唯一既合法、又不易与常见工具名撞车的选择。

超长时**截 server 前缀，不截 tool**：tool 名是模型用来选工具的，
截错一半会让模型拿着 `search_web_fetch_cont` 去猜；server 前缀截了只是少一截归属信息，
真正的归属还有 `ToolSpec.description` 兜着。

非法字符换成 `_` 而不是丢弃：`search.web` 与 `searchweb` 会撞成一个，
**撞名比名字难看严重得多**。

### 3. 错误规范化：只在 client 里做一次

`call_tool` 返回 `ToolOutcome` 而不是抛异常。抛异常会逼每个调用方写
`try/except`，而那个 except 迟早会顺手吞掉一个真 bug。

| 情况 | 内部码 | 上层要做什么 |
|---|---|---|
| 工具不存在 | `tool_not_found` | 模型选错了工具，回注错误让它换 |
| 超时 | `tool_timeout` | 不自动重试（ADR-003），回注让模型决定 |
| server 返回 error | `tool_error` | 回注错误摘要 |
| 响应体不是合法 JSON-RPC | `tool_protocol_error` | 当工具坏了处理，别重试 |
| 传输层异常 | `tool_error`（附原始异常摘要） | 不穿透 |

**不重试**是刻意的（ADR-003：重试只在 LLM client）。工具失败可能是「参数就错了」，
重试三次只是把同一个错误写三遍日志，还把延迟乘三。要不要重试是模型看着错误信息该做的决定。

### 4. 缺 `inputSchema` 给严格空 schema

缺 `inputSchema` 时给的是 `{"type":"object","properties":{},"additionalProperties":false}`
——**严格空**，不是宽松空。宽松空等于「什么参数都收」，那是在替 server 猜它的契约。

---

## 后果

### 好的

- 业务层 grep 不到 `import mcp`（`tests/test_mcp_boundary.py` 用 AST 扫源码断言，
  不是 grep 字符串——注释里提到 MCP 不算违反）。
- 换传输实现不改业务代码：测试里同一个 `MCPClient` 接 `InMemory`、`Recorded`、
  未来的 stdio，业务侧零改动。
- 对外响应里没有厂商痕迹：错误码是我们自己的字符串常量。

### 代价

- 每加一种 MCP 能力（resources、prompts、sampling）都要在 `types.py` 里加解析。
  这是想要的代价：**加一次就要想清楚它在我们这边对应什么**，
  而不是让 wire 结构自己长进来。
- `ToolOutcome` 比 `ToolResult` 多了一层。理由见 ADR-W8-002：
  回放标记、成本、耗时必须有地方放，塞进 `ToolResult` 会污染 W1 的契约。

### 被否决的方案

| 方案 | 否决理由 |
|---|---|
| 让业务层直接用 MCP SDK 类型 | 边界立刻破；换 SDK 版本就要改业务代码 |
| 在 `LLMGateway` 里做工具执行 | 违反 ADR-003 与职责分层；重试语义会混在一起 |
| 用 `.` 做命名空间分隔 | OpenAI/Anthropic 工具名正则都不允许 |
| 缺 `inputSchema` 时放宽为「任意参数」 | 等于替 server 猜契约，越权风险直接进执行层 |
