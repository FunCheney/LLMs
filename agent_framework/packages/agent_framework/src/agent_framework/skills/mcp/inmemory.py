"""内存版 MCP 传输（W8 D1/D2）：把 server 的函数当 wire 用。

它的存在理由不是"偷懒不接真 stdio"：真实 MCP 的 stdio/SSE 传输在 W11 才需要
（容器化 + 多进程），而**协议边界**从第一天就必须是对的。这个实现让
`MCPClient` 面对的是**真实形状的 JSON-RPC 返回体**（`{"tools": [...]}` /
`{"content": [{"type": "text", ...}]}` / `{"error": {...}}`），而不是我们
自己编的 dataclass。测试因此能抓到的 bug 和真连接时一样多：
字段名写错、schema 缺 `inputSchema`、返回 `isError` 而不是抛异常。

三个刻意的能力，都是为了让边界被测到：

1. `fail_next`：下一次调用返回 JSON-RPC error（测错误规范化）
2. `delay_s`：人为延迟（测超时，不用真的 sleep 5 秒）
3. `add_tool`：运行时加热工具（测动态发现——Go 检查点"不改主流程"）
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ..redact import sanitize

__all__ = ["InMemoryMCPTransport"]

Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class InMemoryMCPTransport:
    """把 `{method: handler}` 当传输层。handler 收 `params`，返回 JSON-RPC `result`。

    handler 返回 `{"error": {...}}` 时按协议错误处理（不抛异常）——
    真实 server 的错误也是这么回来的，抛异常反而和线上不一致。
    """

    def __init__(
        self,
        tools: list[dict[str, Any]] | None = None,
        *,
        server_name: str = "memory",
        handlers: dict[str, Handler] | None = None,
    ) -> None:
        self._tools: list[dict[str, Any]] = list(tools or [])
        self._handlers: dict[str, Handler] = dict(handlers or {})
        self.server_name = server_name
        # 人为延迟（秒）。0 表示不延迟。
        self.delay_s: float = 0.0
        # 下一次调用注入的错误体。用一次即清空，避免污染后续用例。
        self.fail_next: dict[str, Any] | None = None
        # 调用流水（测试断言"发了几次请求"用，等价于 W1 的 A8）。
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    # -- MCPTransport 协议 -------------------------------------------------- #

    async def list_tools(self) -> dict[str, Any]:
        self.calls.append(("tools/list", "", {}))
        await self._maybe_delay()
        failure = self._take_failure()
        if failure is not None:
            return failure
        return {"tools": [dict(tool) for tool in self._tools]}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("tools/call", name, sanitize(arguments)))
        await self._maybe_delay()
        failure = self._take_failure()
        if failure is not None:
            return failure

        handler = self._handlers.get(name)
        if handler is None:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Method not found: tools/call {name}",
                }
            }

        params = {"name": name, "arguments": arguments}
        return await handler(params)

    # -- 测试能力 ----------------------------------------------------------- #

    def add_tool(self, tool: dict[str, Any], handler: Handler) -> None:
        """运行时加热工具。`MCPClient.list_tools()` 再调一次就能看到。"""
        self._tools.append(dict(tool))
        self._handlers[str(tool["name"])] = handler

    def tool_names(self) -> list[str]:
        return [str(tool.get("name")) for tool in self._tools]

    # -- 内部 -------------------------------------------------------------- #

    async def _maybe_delay(self) -> None:
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

    def _take_failure(self) -> dict[str, Any] | None:
        """取出注入的错误体（用一次即清空）。

        直接**返回** `{"error": ...}` 而不是抛异常：真实 MCP server 的错误
        就是这么回来的（JSON-RPC 的错误是响应的一部分，不是连接异常）。
        用异常模拟会让 client 走进 "传输层异常" 分支，从而测不到
        `tool_protocol_error` 这条真正要测的路径。
        """
        if self.fail_next is None:
            return None
        payload = self.fail_next
        self.fail_next = None
        return payload
