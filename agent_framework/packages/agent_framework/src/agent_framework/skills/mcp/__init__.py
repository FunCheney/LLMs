"""MCP 子包：协议边界（W8 D1）。

三个文件各管一件事，别混：

- `types.py`  wire 结构与解析（只认 JSON-RPC，不认业务）
- `client.py` 翻译成内部 `ToolSpec` / `ToolOutcome`，负责超时与错误规范化
- `inmemory.py` 内存传输（测试/离线档用真实形状的返回体）

业务层只 import `skills`，不 import 这里——`test_mcp_boundary.py` 会扫源码。
"""

from .client import (
    ERROR_PROTOCOL,
    ERROR_TOOL_ERROR,
    ERROR_TOOL_NOT_FOUND,
    ERROR_TOOL_TIMEOUT,
    MCPClient,
)
from .inmemory import InMemoryMCPTransport
from .types import (
    MCPCallResult,
    MCPToolDef,
    MCPTransport,
    internal_name,
    parse_call_result,
    parse_rpc_error,
    parse_tool_list,
)

__all__ = [
    "ERROR_PROTOCOL",
    "ERROR_TOOL_ERROR",
    "ERROR_TOOL_NOT_FOUND",
    "ERROR_TOOL_TIMEOUT",
    "InMemoryMCPTransport",
    "MCPCallResult",
    "MCPClient",
    "MCPToolDef",
    "MCPTransport",
    "internal_name",
    "parse_call_result",
    "parse_rpc_error",
    "parse_tool_list",
]
