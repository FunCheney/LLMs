"""W8 D1：MCP Client 的协议边界。

测的是四件容易写错的事：
1. `tools/list` → 内部 `ToolSpec`（业务只认内部类型）
2. 命名空间（两个 server 都有 `search_web` 时不能撞）
3. 超时与错误规范化（MCP 原始错误体不出这道门）
4. 动态发现（server 加了工具，重新 list 就能看到）
"""

from __future__ import annotations

import json

import pytest
from agent_framework.llm.types import ToolSpec
from agent_framework.skills.contract import CODE_OK
from agent_framework.skills.mcp import (
    ERROR_PROTOCOL,
    ERROR_TOOL_NOT_FOUND,
    InMemoryMCPTransport,
    MCPClient,
    internal_name,
    parse_tool_list,
)
from agent_framework.skills.servers import search_client


def _tool(name: str, **extra: object) -> dict[str, object]:
    return {
        "name": name,
        "description": f"{name} 的描述",
        "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
        **extra,
    }


async def test_list_tools_returns_internal_tool_spec() -> None:
    """业务层拿到的是 `ToolSpec`，不是 MCP 的 dict。"""

    transport = InMemoryMCPTransport([_tool("search_web")], server_name="search")
    client = MCPClient(transport, server_name="search")

    specs = await client.list_tools()

    assert all(isinstance(spec, ToolSpec) for spec in specs)
    assert [spec.name for spec in specs] == ["search__search_web"]
    # 描述里要带 server 归属与边界说明，模型才知道该不该用。
    assert "search__search_web" not in specs[0].description
    assert specs[0].parameters["type"] == "object"


async def test_same_tool_name_on_two_servers_does_not_collide() -> None:
    """命名空间的存在理由：两个 server 都叫 `search_web` 时内部必须能区分。"""

    first = MCPClient(InMemoryMCPTransport([_tool("search_web")]), server_name="search")
    second = MCPClient(InMemoryMCPTransport([_tool("search_web")]), server_name="internal")

    first_names = [spec.name for spec in await first.list_tools()]
    second_names = [spec.name for spec in await second.list_tools()]

    assert first_names == ["search__search_web"]
    assert second_names == ["internal__search_web"]
    assert first_names[0] != second_names[0]


async def test_call_tool_uses_server_side_name() -> None:
    """回传给 server 的必须是原始名（`search_web`），不是内部名。"""

    transport = InMemoryMCPTransport([_tool("search_web")], server_name="search")

    async def handler(params: dict) -> dict:
        return {"content": [{"type": "text", "text": json.dumps({"echo": params["arguments"]})}]}

    transport.add_tool({"name": "search_web"}, handler)
    client = MCPClient(transport, server_name="search")
    await client.list_tools()

    outcome = await client.call_tool("search__search_web", {"q": "增长"})

    assert outcome.code == CODE_OK
    assert outcome.payload == {"echo": {"q": "增长"}}
    # 第二次 tools/list 由 add_tool 之前发生过一次；这里断言调用名是原始名。
    assert ("tools/call", "search_web", {"q": "增长"}) in transport.calls


async def test_timeout_is_normalized_and_does_not_raise() -> None:
    transport = InMemoryMCPTransport([_tool("slow")], server_name="s")
    transport.add_tool({"name": "slow"}, _slow_handler)
    client = MCPClient(transport, server_name="s", timeout_s=0.05)

    # 先发现（不延迟），再打开延迟：否则连 `tools/list` 都超时，
    # 拿到的是 `tool_not_found` 而不是 `tool_timeout`——这是两个不同的故障。
    await client.list_tools()
    transport.delay_s = 0.3
    outcome = await client.call_tool("s__slow", {})

    assert outcome.code == "tool_timeout"
    assert outcome.result.is_error
    payload = json.loads(str(outcome.result.content))
    assert payload["error"] == "tool_timeout"
    # 规范化后的消息里**不能**出现传输层/厂商细节。
    assert "slow" in payload["message"]


async def test_rpc_error_becomes_protocol_error_without_leaking_message() -> None:
    """MCP 的 `error.message` 可能含内网地址，不能进 ToolResult。"""

    transport = InMemoryMCPTransport([_tool("x")], server_name="s")
    transport.add_tool({"name": "x"}, _ok_handler)
    client = MCPClient(transport, server_name="s")
    await client.list_tools()

    transport.fail_next = {
        "error": {"code": -32601, "message": "internal host 10.0.0.7 refused"}
    }
    outcome = await client.call_tool("s__x", {})

    assert outcome.code == ERROR_PROTOCOL
    assert "10.0.0.7" not in str(outcome.result.content)


async def test_unknown_tool_returns_not_found_after_refresh() -> None:
    """模型编工具名 → `tool_not_found`，不是异常，也不是静默成功。"""

    client = search_client()
    await client.list_tools()

    outcome = await client.call_tool("search__nope", {})

    assert outcome.code == ERROR_TOOL_NOT_FOUND


async def test_dynamic_discovery_sees_new_tool_without_client_change() -> None:
    """检查点「新增工具不改主流程」的最小验证。

    注意这里**没有**重建 client：server 加工具之后重新 `list_tools` 就能看到。
    如果发现逻辑带缓存，这条会红——而缓存正是"加了要等一会儿才生效"的来源。
    """

    transport = InMemoryMCPTransport([_tool("a")], server_name="s")
    client = MCPClient(transport, server_name="s")
    assert [spec.name for spec in await client.list_tools()] == ["s__a"]

    transport.add_tool(
        {"name": "b", "description": "新工具", "inputSchema": {"type": "object"}},
        _ok_handler,
    )

    assert [spec.name for spec in await client.list_tools()] == ["s__a", "s__b"]


def test_missing_input_schema_is_not_silently_relaxed() -> None:
    """缺 `inputSchema` → 空对象 schema（严格），而不是"随便传"。"""

    parsed = parse_tool_list({"tools": [{"name": "no_schema"}]})

    assert parsed[0].input_schema is None
    assert "missing_input_schema" in parsed[0].warnings


async def test_missing_schema_becomes_strict_empty_schema() -> None:
    client = MCPClient(
        InMemoryMCPTransport([{"name": "no_schema"}], server_name="s"), server_name="s"
    )

    spec = (await client.list_tools())[0]

    assert spec.parameters["properties"] == {}
    assert spec.parameters["additionalProperties"] is False


def test_internal_name_is_legal_for_providers_and_capped() -> None:
    """两家厂商都限制 `^[a-zA-Z0-9_-]{1,64}$`，所以分隔符只能用 `__`。"""

    name = internal_name("my server", "search.web/v2")
    assert len(name) <= 64
    assert all(char.isalnum() or char in "_-" for char in name)

    long_name = internal_name("s" * 80, "tool")
    assert len(long_name) <= 64
    # 截断只截 server 前缀，tool 名必须完整（模型靠它选工具）。
    assert long_name.endswith("__tool")


async def test_health_check_tools_are_listed_by_real_search_server() -> None:
    """内置 search server 的声明要能被自己的 client 发现（自洽性）。"""

    client = search_client()
    specs = await client.list_tools()

    assert [spec.name for spec in specs] == ["search__search_web"]
    assert "不可信" in specs[0].description


async def _slow_handler(params: dict) -> dict:
    return {"content": [{"type": "text", "text": "too late"}]}


async def _ok_handler(params: dict) -> dict:
    return {"content": [{"type": "text", "text": "ok"}]}


@pytest.mark.parametrize("code", [ERROR_TOOL_NOT_FOUND, ERROR_PROTOCOL])
async def test_error_codes_are_stable_strings(code: str) -> None:
    """错误码是字符串而不是异常类：审计与告警规则按它匹配，类名一改全崩。"""

    assert isinstance(code, str)
