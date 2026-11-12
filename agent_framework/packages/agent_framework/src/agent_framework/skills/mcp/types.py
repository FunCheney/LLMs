"""MCP 的 wire 类型（W8 D1）：这一层只认 JSON-RPC 的 `result` 体，不认业务。

为什么把"传输"和"翻译"拆成两个文件：MCP 是**外部工具传输协议**，
我们的框架是**内部能力契约**（`ToolSpec` / `ToolCall` / `ToolResult`）。
两者的差别不是命名风格，而是失败语义——MCP 用 JSON-RPC 错误码，
我们用"错误码 + 是否可重试 + 是否要给模型看"。如果让 JSON-RPC 的错误体
直接流到 `LLMGateway`，上层就要开始判断 `-32601` 是什么，边界立刻破。

所以这里刻意只做三件事：

1. `parse_tool_list` / `parse_call_result`：把 wire 结构解析成**受控**的
   dataclass，畸形字段丢掉并记 `warnings`——**不抛异常也不猜**。
   一个 server 少写 `inputSchema` 不该让整次 `tools/list` 失败，
   但也绝不能凭空给它编一个 schema。
2. `MCPTransport` 协议：`list_tools()` / `call_tool()` 返回**原始 dict**。
   真实实现走 stdio/SSE（W11 才接），测试与离线档走 `InMemoryTransport`
   和 `RecordedTransport`——三种实现的差别被这个协议吸收。
3. 工具名命名规则写在一处（`internal_name`）：MCP 的 tool 名只在 server
   内唯一，两个 server 都叫 `search_web` 时内部必须能区分；而 OpenAI /
   Anthropic 的工具名都限制在 `^[a-zA-Z0-9_-]{1,64}$`，用不了 `.` 和 `:`
   这些"更自然"的分隔符。用 `__` 是唯一既合法又不易与常见工具名撞车的选择。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "MCP_JSONRPC_VERSION",
    "MCPCallResult",
    "MCPToolDef",
    "MCPTransport",
    "internal_name",
    "parse_call_result",
    "parse_rpc_error",
    "parse_tool_list",
    "tool_name_part",
]

MCP_JSONRPC_VERSION = "2.0"

# OpenAI 与 Anthropic 的工具名约束都是这个（Anthropic 明确写了
# `^[a-zA-Z0-9_-]{1,64}$`）。写在一处，改一处即可。
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
# 拆分内部名用的分隔符。`__` 而不是 `.`：点号在两家都非法。
NAMESPACE_SEPARATOR = "__"


def tool_name_part(raw: str) -> str:
    """把任意 server 侧名字压成合法片段。

    非法字符换成 `_` 而不是丢弃：`search.web` 与 `searchweb` 变一样，
    两个不同工具会撞成一个——撞名比名字难看严重得多。
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", raw or "")
    return cleaned or "unnamed"


def internal_name(server: str, tool: str) -> str:
    """`(server, tool)` → 内部工具名 `server__tool`，并保证 ≤ 64 字符。

    超长时**截 server 前缀**而不截 tool：tool 名是模型用来选工具的，
    截错一半会让模型拿着 `search_web_fetch_cont` 去猜。server 前缀截了
    只是少一截归属信息，且真正的归属还有 `ToolSpec.description` 兜着。
    """
    tail = tool_name_part(tool)
    head = tool_name_part(server)
    room = 64 - len(NAMESPACE_SEPARATOR) - len(tail)

    if room <= 0:
        return tail[:64]

    return f"{head[:room]}{NAMESPACE_SEPARATOR}{tail}"


@dataclass(frozen=True, slots=True)
class MCPToolDef:
    """一个 MCP 工具定义（已从 wire 解析，仍属 mcp/ 内部类型）。"""

    # server 侧原始工具名。调用时必须用这个名字回传给 server，
    # 不能用内部名——内部名是我们自己编的，server 不认识。
    name: str
    description: str = ""
    # MCP 的 `inputSchema`。缺失时为 None，**不填默认值**：
    # 空 dict 和"这个 server 没声明参数"在 A14 校验里含义不同。
    input_schema: dict[str, Any] | None = None
    # 解析时丢掉/修正了什么。进日志不进 prompt。
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MCPCallResult:
    """一次 `tools/call` 的结果。

    `content` 保留 MCP 的 content block 列表（`[{"type":"text","text":...}]`），
    由 client 决定怎么摊平成内部 `ToolResult.content`——摊平规则属于
    "翻译"，不属于 "wire"。
    """

    content: tuple[dict[str, Any], ...] = ()
    # MCP 的 `isError`。注意它表示"工具执行失败"，**不表示**协议失败：
    # 协议失败走 JSON-RPC error（`parse_rpc_error`），两者不能混成一个 is_error。
    is_error: bool = False
    # server 可能返回 `structuredContent`（新版规范），原样带上。
    structured: dict[str, Any] | None = None


def parse_tool_list(payload: Any) -> tuple[MCPToolDef, ...]:
    """`tools/list` 的 `result` → 工具定义列表。

    容错但不猜：`tools` 不是列表直接返回空（server 没工具是合法状态，
    不是错误）；单个条目缺 `name` 就跳过并记 warning。
    """
    if not isinstance(payload, dict):
        return ()

    raw_tools = payload.get("tools")
    if not isinstance(raw_tools, list):
        return ()

    parsed: list[MCPToolDef] = []
    for index, raw in enumerate(raw_tools):
        if not isinstance(raw, dict) or not raw.get("name"):
            # 没有名字的工具无法被调用，留着只会在模型侧变成一个黑洞。
            continue

        name = str(raw["name"])
        warnings: list[str] = []
        schema = raw.get("inputSchema")
        if schema is None:
            warnings.append("missing_input_schema")
        elif not isinstance(schema, dict):
            # 类型错就按"没声明"处理，同时记下来：这是 server 的 bug，
            # 我们不能替它修，但排障时必须看得见。
            warnings.append(f"invalid_input_schema:{type(schema).__name__}")
            schema = None

        parsed.append(
            MCPToolDef(
                name=name,
                description=str(raw.get("description") or ""),
                input_schema=schema,
                warnings=tuple(warnings),
            )
        )
        del index  # 序号只用于报错定位，当前不需要，显式声明"没忘"

    return tuple(parsed)


def parse_call_result(payload: Any) -> MCPCallResult:
    """`tools/call` 的 `result` → 受控结果。

    content block 只保留 `type` 与 `text`/`data`：MCP 允许 server 塞任意
    字段，全量透传等于把 server 的私有结构带进我们的 prompt。
    """
    if not isinstance(payload, dict):
        return MCPCallResult(content=(), is_error=True)

    blocks: list[dict[str, Any]] = []
    raw_content = payload.get("content")
    if isinstance(raw_content, list):
        for block in raw_content:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("type") or "")
            if kind == "text":
                blocks.append({"type": "text", "text": str(block.get("text") or "")})
            elif kind in {"image", "resource"}:
                # 二进制/资源块不摊平，只留类型标记：把它们当文本塞进
                # prompt 会变成一串 base64，既烧 token 又无意义。
                blocks.append({"type": kind, "note": "unsupported_block"})

    structured = payload.get("structuredContent")
    return MCPCallResult(
        content=tuple(blocks),
        is_error=bool(payload.get("isError")),
        structured=structured if isinstance(structured, dict) else None,
    )


def parse_rpc_error(payload: Any) -> tuple[int, str] | None:
    """JSON-RPC `error` → `(code, message)`；不是错误体则 None。

    message 只用于**日志与审计**，绝不进 `ToolResult`：MCP server 的
    错误文本可能包含内网地址、密钥片段、上游返回体。对外统一成我们
    自己的错误码（见 client 的 `ERROR_*`）。
    """
    if not isinstance(payload, dict):
        return None

    raw = payload.get("error")
    if not isinstance(raw, dict):
        return None

    code = raw.get("code")
    return (int(code) if isinstance(code, int) else -32000, str(raw.get("message") or ""))


@runtime_checkable
class MCPTransport(Protocol):
    """MCP 传输层契约：**只搬运 JSON-RPC 方法，不解释内容**。

    三个实现共用：真实 stdio/SSE（W11）、`InMemoryMCPTransport`（测试/离线）、
    `RecordedTransport`（回放）。传输层不认识 `ToolSpec`，
    这是"业务不依赖 MCP"的结构性保证——换传输不改上层。
    """

    async def list_tools(self) -> dict[str, Any]:
        """返回 `tools/list` 的原始 `result`。"""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """返回 `tools/call` 的原始 `result`；协议错误体原样返回（含 `error`）。"""
        ...


@dataclass(frozen=True, slots=True)
class MCPServerInfo:
    """握手结果。`list_changed` 用于决定要不要热重载工具集。"""

    name: str
    version: str = ""
    protocol_version: str = ""
    # `capabilities.tools.listChanged`：server 支持主动通知工具变更。
    list_changed: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
