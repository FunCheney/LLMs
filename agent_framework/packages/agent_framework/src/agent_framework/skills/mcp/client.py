"""MCP Client（W8 D1）：把外部工具**翻译**成内部契约，别的一概不做。

三条边界，每条都有对应测试：

1. **上层只认内部类型。** `list_tools()` 返回 `list[ToolSpec]`，
   `call_tool()` 返回内部 `ToolResult`。业务代码 grep 不到 `mcp` 这个名字
   （`test_mcp_boundary.py` 直接扫源码）。换掉 MCP 去接别的协议，
   上层一行不改——这不是洁癖，是因为 MCP 还在演进，绑定它等于绑定它的版本。
2. **超时是我们的事，不是传输层的事。** `asyncio.wait_for` 包在每次调用外面，
   超时后不重试（重试策略在 `LLMGateway`，工具侧重试会让"同一动作执行两次"
   变成默认行为，直接和 D5 的幂等冲突）。
3. **错误规范化，原始错误体不出这道门。** JSON-RPC 的 `-32601`、上游的
   HTML 错误页、内网地址，全部收敛成三个我们自己的码。原始文本只进日志
   （经 `redact.sanitize`），并挂 `request_id` 方便回查。

`call_tool` 收的是**内部名**（`server__tool`），回传给 server 前换回原始名。
换名只在这一处发生：模型看到的名字与 server 认识的名字必然不同，
如果两处各换一次，改名那天就会有一半调用打到不存在的工具。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from ...llm.types import ToolResult, ToolSpec
from ..contract import (
    CODE_NOT_FOUND,
    CODE_OK,
    CODE_TIMEOUT,
    CODE_TOOL_ERROR,
    ToolOutcome,
)
from ..redact import sanitize
from .types import (
    MCPCallResult,
    MCPToolDef,
    MCPTransport,
    internal_name,
    parse_call_result,
    parse_rpc_error,
    parse_tool_list,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ERROR_PROTOCOL",
    "ERROR_TOOL_ERROR",
    "ERROR_TOOL_NOT_FOUND",
    "ERROR_TOOL_TIMEOUT",
    "MCPClient",
]

# 工具侧错误码。故意只有三档：可归因、可重试、可上报，多一档就没人记得住。
ERROR_TOOL_NOT_FOUND = CODE_NOT_FOUND
ERROR_TOOL_TIMEOUT = CODE_TIMEOUT
ERROR_TOOL_ERROR = CODE_TOOL_ERROR
# 协议层失败（JSON-RPC error / 返回体畸形）。与 tool_error 分开：
# 前者是我们的集成问题，后者是工具自己的问题，值班的人不是同一个。
ERROR_PROTOCOL = "tool_protocol_error"


class MCPClient:
    """一个 MCP server 的门面。构造即绑定 server，调用时才连接。"""

    def __init__(
        self,
        transport: MCPTransport,
        *,
        server_name: str,
        timeout_s: float = 8.0,
        description_suffix: str | None = None,
    ) -> None:
        self._transport = transport
        # server 名进内部命名空间。它同时是审计里的"哪个 server"，
        # 所以必须稳定：改名会让历史审计记录与新记录对不上。
        self._server_name = server_name
        self._timeout_s = timeout_s
        # 给模型看的补充说明。MCP 的 description 往往写成"Search the web"，
        # 产品层想补"只用于公开信息"这类边界时挂这里，不改 server。
        self._description_suffix = description_suffix
        # internal name -> server 侧原始名。`list_tools` 填，`call_tool` 读。
        self._names: dict[str, str] = {}
        # 最近一次 `tools/list` 的解析告警（畸形 schema 等），进审计用。
        self._warnings: list[str] = []

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    async def list_tools(self) -> list[ToolSpec]:
        """发现工具。**动态**：server 加了工具，重新调用就能看到。

        这里不缓存过期时间。缓存的收益是省一次往返（本地 server 是微秒级），
        代价是"新工具要等缓存过期才可见"——而检查点要求的正是"加了就能用"。
        """
        try:
            payload = await asyncio.wait_for(
                self._transport.list_tools(), timeout=self._timeout_s
            )
        except TimeoutError:
            # 发现失败不抛：一个 server 挂了不该让整个工具集消失，
            # 调用方按"这次没发现到工具"处理，并由审计记下原因。
            self._warnings.append(f"{self._server_name}:list_timeout")
            logger.warning("mcp list_tools timeout: server=%s", self._server_name)
            return []
        except Exception as exc:  # noqa: BLE001 - 传输层异常一律不穿透
            self._warnings.append(f"{self._server_name}:list_failed")
            logger.warning(
                "mcp list_tools failed: server=%s err=%s",
                self._server_name,
                sanitize(f"{type(exc).__name__}: {exc}"),
            )
            return []

        defs = parse_tool_list(payload)
        self._names = {}
        specs: list[ToolSpec] = []
        for definition in defs:
            specs.append(self._to_spec(definition))

        return specs

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """按内部名调用。所有失败都返回带 `code` 的 `ToolOutcome`，**不抛异常**。

        不抛异常：工具失败是**正常路径**（搜索结果为空、页面 404 都是常态），
        抛异常会逼调用方写 `try/except`，而那个 except 迟早会顺手吞掉
        真正的编程错误。
        """
        started = time.perf_counter()
        server_tool = self._names.get(name)
        if server_tool is None:
            # 名字不在缓存里：可能是"发现之后 server 又加了工具"，
            # 也可能是模型编了一个不存在的工具名。刷新一次再判，
            # 这样才能区分两种情况——对审计来说这是完全不同的两件事。
            await self.list_tools()
            server_tool = self._names.get(name)

        if server_tool is None:
            return _error(
                name,
                CODE_NOT_FOUND,
                f"没有名为 {name!r} 的工具",
                started=started,
                server=self._server_name,
            )

        try:
            payload = await asyncio.wait_for(
                self._transport.call_tool(server_tool, arguments), timeout=self._timeout_s
            )
        except TimeoutError:
            return _error(
                name,
                CODE_TIMEOUT,
                f"工具 {name} 超时（{self._timeout_s:g}s）",
                started=started,
                server=self._server_name,
            )
        except Exception as exc:  # noqa: BLE001 - 见 docstring 的不抛约定
            logger.warning(
                "mcp call_tool failed: server=%s tool=%s err=%s",
                self._server_name,
                server_tool,
                sanitize(f"{type(exc).__name__}: {exc}"),
            )
            return _error(
                name,
                CODE_TOOL_ERROR,
                f"工具 {name} 调用失败",
                started=started,
                server=self._server_name,
            )

        rpc_error = parse_rpc_error(payload)
        if rpc_error is not None:
            code, message = rpc_error
            # 原始 message 只进日志：它可能带内网主机名或上游返回体。
            logger.warning(
                "mcp rpc error: server=%s tool=%s code=%s msg=%s",
                self._server_name,
                server_tool,
                code,
                sanitize(message),
            )
            return _error(
                name,
                ERROR_PROTOCOL,
                f"工具 {name} 协议错误（{code}）",
                started=started,
                server=self._server_name,
            )

        result = parse_call_result(payload)
        meta = _meta_of(payload)
        return _to_outcome(
            self._server_name, name, result, meta, started=started
        )

    # -- 内部 -------------------------------------------------------------- #

    def _to_spec(self, definition: MCPToolDef) -> ToolSpec:
        internal = internal_name(self._server_name, definition.name)
        self._names[internal] = definition.name

        description = definition.description or f"{definition.name}（来自 {self._server_name}）"
        if self._description_suffix:
            description = f"{description} {self._description_suffix}"
        if definition.warnings:
            self._warnings.extend(
                f"{self._server_name}/{definition.name}:{w}" for w in definition.warnings
            )

        return ToolSpec(
            name=internal,
            description=description.strip(),
            # 缺 schema 时给**空对象 schema**（合法且严格）：不给默认的
            # `{"type":
            # "object"}` 宽松版，否则 A14 会放行任意参数——"没声明"被
            # 误当成"随便传"是这类适配里最常见的一次放水。
            parameters=definition.input_schema
            if definition.input_schema is not None
            else {"type": "object", "properties": {}, "additionalProperties": False},
        )


def _error(
    tool: str, code: str, message: str, *, started: float, server: str = ""
) -> ToolOutcome:
    """错误结果。`content` 是 JSON 字符串而不是自然语言：
    验真层（D5）要按 `error` 字段判定，读自然语言等于让它做 NLP。"""
    return ToolOutcome(
        result=ToolResult(
            tool_call_id="",
            content=json.dumps({"error": code, "message": message}, ensure_ascii=False),
            is_error=True,
        ),
        code=code,
        latency_ms=(time.perf_counter() - started) * 1000,
        server=server,
        tool=tool,
    )


def _meta_of(payload: Any) -> dict[str, Any]:
    """取 MCP 的 `result._meta`（规范里留给实现方元数据的字段）。

    回放标记与成本走这里而不是自造顶层字段：`_meta` 是**协议预留**的位置，
    塞在顶层会被严格的 server 当非法字段拒掉，而我们的回放要能一对一流回。
    """
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("_meta")
    return meta if isinstance(meta, dict) else {}


def _decode_payload(content: str) -> dict[str, Any] | None:
    """文本结果是 JSON 对象时解出来给验真层用（解不出就当纯文本）。"""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return None
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _to_outcome(
    server: str,
    name: str,
    result: MCPCallResult,
    meta: dict[str, Any],
    *,
    started: float,
) -> ToolOutcome:
    """MCP content blocks → 内部 `ToolOutcome`。

    - 纯文本块：直接拼起来（模型最省 token 的形态）
    - 有 `structuredContent`：序列化成 JSON（结构信息不丢，验真要读它）
    - `isError=True`：转成我们的 `tool_error`，**不把 server 的错误文案
      原样回给模型**——它可能是上游 HTML 或堆栈
    """
    latency_ms = (time.perf_counter() - started) * 1000
    replayed = bool(meta.get("replayed"))
    try:
        cost = float(meta.get("cost") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0

    if result.is_error:
        logger.info("mcp tool returned isError: tool=%s", name)
        return ToolOutcome(
            result=ToolResult(
                tool_call_id="",
                content=json.dumps(
                    {"error": CODE_TOOL_ERROR, "message": f"工具 {name} 执行失败"},
                    ensure_ascii=False,
                ),
                is_error=True,
            ),
            code=CODE_TOOL_ERROR,
            replayed=replayed,
            cost=0.0 if replayed else cost,
            latency_ms=latency_ms,
            server=server,
            tool=name,
        )

    if result.structured is not None:
        content = json.dumps(result.structured, ensure_ascii=False)
        payload: dict[str, Any] | None = result.structured
    else:
        content = "\n".join(
            str(block.get("text") or "")
            for block in result.content
            if block.get("type") == "text"
        )
        payload = _decode_payload(content)

    return ToolOutcome(
        result=ToolResult(tool_call_id="", content=content, is_error=False),
        code=CODE_OK,
        replayed=replayed,
        cost=0.0 if replayed else cost,
        latency_ms=latency_ms,
        server=server,
        tool=name,
        payload=payload,
    )
