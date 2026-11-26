"""W8 D4：执行器五道门。

Gate 8 #5 的判据是「校验→权限→沙箱→执行→回注 顺序固定」。顺序不是文档
里的排法，而是"哪一步先返回"的实测结果——所以下面的测试全是**构造一个
只应该被某一道门拦下**的输入，然后断言拿到的那个 code。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from agent_framework.context import ContextKind, TrustLevel
from agent_framework.llm.types import MessageRole, ToolCall
from agent_framework.retrieval import Principal
from agent_framework.skills import (
    CODE_INVALID_ARGUMENTS,
    CODE_NOT_FOUND,
    CODE_OK,
    CODE_PERMISSION_DENIED,
    CODE_SANDBOX_DENIED,
    CODE_TOOL_NOT_VISIBLE,
    AuditLog,
    ResultTruncator,
    SandboxPolicy,
    SkillExecutor,
    SkillRegistry,
)
from agent_framework.skills.servers import (
    FetchServer,
    SearchServer,
    WorkspaceServer,
)

EDITOR = Principal(subject_id="e1", groups=frozenset({"editor"}))
VIEWER = Principal(subject_id="v1", groups=frozenset({"viewer"}))


@dataclass
class Env:
    executor: SkillExecutor
    workspace: WorkspaceServer
    root: Path
    audit: AuditLog


@pytest.fixture()
def env(tmp_path: Path) -> Env:
    root = tmp_path / "repo"
    root.mkdir()
    sandbox = SandboxPolicy(
        write_roots=(root,),
        command_prefixes=("echo",),
        network_allow=("example.com",),
    )
    workspace = WorkspaceServer(root=root, sandbox=sandbox)
    clients = {
        "workspace": workspace.client(),
        "search": SearchServer().client(),
        "fetch": FetchServer().client(),
    }
    audit = AuditLog()
    executor = SkillExecutor(
        registry=SkillRegistry(),
        clients=clients,
        sandbox=sandbox,
        audit=audit,
        truncator=ResultTruncator(max_chars=200),
    )
    return Env(executor=executor, workspace=workspace, root=root, audit=audit)


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=arguments)


async def test_unknown_skill_is_not_found(env: Env) -> None:
    result = await env.executor.execute(_call("no_such_skill"), principal=EDITOR)

    assert result.code == CODE_NOT_FOUND
    assert result.message is not None
    assert '"error": "tool_not_found"' in str(result.message.content)


async def test_tool_outside_surface_is_not_visible_not_denied(env: Env) -> None:
    """调研阶段调 publish_post → `tool_not_visible`（不是 `permission_denied`）。"""

    result = await env.executor.execute(
        _call("publish_post", title="t", body="b" * 60),
        principal=EDITOR,
        stage="research",
    )

    assert result.code == CODE_TOOL_NOT_VISIBLE
    assert "stage" in result.detail


async def test_invalid_arguments_are_caught_before_permission(env: Env) -> None:
    """参数不合法时连"要不要给权限"都不用问——顺序断言。"""

    result = await env.executor.execute(
        _call("write_brief", path="a.md"),  # 缺 content
        principal=VIEWER,  # 这个人也没写权限
        stage="draft",
    )

    assert result.code == CODE_INVALID_ARGUMENTS


async def test_extra_argument_is_rejected(env: Env) -> None:
    result = await env.executor.execute(
        _call("web_search", query="增长", site="internal.example.com"),
        principal=EDITOR,
        stage="research",
    )

    assert result.code == CODE_INVALID_ARGUMENTS
    assert "schema 之外" in result.detail


async def test_permission_denied_when_visible_but_unauthorized(env: Env) -> None:
    """`write_brief` 对 viewer 可见（write 不参与粗筛），执行时被权限拦下。

    这条测试同时保住了 `permission_denied` 这个码的可达性。
    """

    result = await env.executor.execute(
        _call("write_brief", path="briefs/a.md", content="内容" * 20),
        principal=VIEWER,
        stage="draft",
    )

    assert result.code == CODE_PERMISSION_DENIED
    assert "write" in result.detail


async def test_sandbox_denies_path_escape_before_side_effect(env: Env) -> None:
    """Gate 8 #9：`../outside.txt` 被拦，且文件没被创建。"""

    result = await env.executor.execute(
        _call("write_brief", path="../outside.txt", content="内容" * 20),
        principal=EDITOR,
        stage="draft",
    )

    assert result.code == CODE_SANDBOX_DENIED
    assert "path_denied" in result.detail
    assert not (env.root.parent / "outside.txt").exists()


async def test_sandbox_denies_command_and_network(env: Env) -> None:
    from agent_framework.skills.spec import SkillSpec

    shell_skill = SkillSpec(
        name="shell_probe",
        description="跑一条命令。",
        mission="负责跑命令；不负责别的。",
        instructions="1. 跑。",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        entrypoint="workspace.run_command",
        permissions=frozenset({"read", "shell"}),
        sandbox="subprocess",
        stages=frozenset({"maintenance"}),
    )
    env.executor.registry.register(shell_skill)
    ops = Principal(subject_id="o1", groups=frozenset({"ops"}))

    denied = await env.executor.execute(
        _call("shell_probe", command="rm -rf /tmp/x"), principal=ops, stage="maintenance"
    )
    assert denied.code == CODE_SANDBOX_DENIED
    assert "command_denied" in denied.detail

    env.workspace._sandbox = SandboxPolicy(  # noqa: SLF001 - 测试故意收紧策略
        write_roots=(env.root,), command_prefixes=("echo",)
    )
    blocked_net = await env.executor.execute(
        _call("web_fetch", url="https://evil.example.org/page"),
        principal=EDITOR,
        stage="draft",
    )
    assert blocked_net.code == CODE_SANDBOX_DENIED
    assert "network_denied" in blocked_net.detail


async def test_success_reinjects_tool_message_with_external_trust(env: Env) -> None:
    result = await env.executor.execute(
        _call("write_brief", path="briefs/a.md", content="# 简报\n待确认：转化数据" * 5),
        principal=EDITOR,
        stage="draft",
    )

    assert result.code == CODE_OK
    assert result.message is not None
    assert result.message.role == MessageRole.TOOL
    assert result.message.tool_call_id == "call-write_brief"
    assert result.message.name == "write_brief"

    assert result.item is not None
    assert result.item.kind == ContextKind.TOOL_RESULT
    # 工具返回永远不是指令：信任级必须是 external（W10 注入防护同一个原则）。
    assert result.item.trust == TrustLevel.EXTERNAL
    assert result.item.source == "tool:write_brief"
    assert result.item.message is result.message


async def test_side_effect_actually_happens(env: Env) -> None:
    """阳性对照：如果所有写入都被拦了，"拦住了越界"就没有任何意义。"""

    await env.executor.execute(
        _call("write_brief", path="briefs/a.md", content="内容" * 20),
        principal=EDITOR,
        stage="draft",
    )

    assert (env.root / "briefs" / "a.md").is_file()


async def test_large_result_is_truncated_with_handle(env: Env) -> None:
    big = "x" * 5000
    (env.root / "big.txt").write_text(big, "utf-8")

    from agent_framework.skills.spec import SkillSpec

    reader = SkillSpec(
        name="read_big",
        description="读一个大文件。",
        mission="负责读文件；不负责写。",
        instructions="1. 读。",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        entrypoint="workspace.read_file",
        permissions=frozenset({"read"}),
        sandbox="none",
        stages=frozenset({"research"}),
    )
    env.executor.registry.register(reader)

    result = await env.executor.execute(
        _call("read_big", path="big.txt"), principal=EDITOR, stage="research"
    )

    assert result.code == CODE_OK
    assert result.truncated
    assert result.handle
    assert "句柄" in str(result.message.content)
    # 句柄能取回完整内容：截断不能变成丢数据。
    full = env.executor._truncator.refetch(result.handle)  # noqa: SLF001
    assert full is not None and len(full) > 4000


async def test_audit_records_every_attempt_including_failures(env: Env) -> None:
    await env.executor.execute(_call("no_such_skill"), principal=EDITOR)
    await env.executor.execute(
        _call("write_brief", path="../outside.txt", content="内容" * 20),
        principal=EDITOR,
        stage="draft",
    )

    entries = env.audit.entries()

    assert [entry.status for entry in entries] == [CODE_NOT_FOUND, CODE_SANDBOX_DENIED]
    assert entries[0].principal == "e1"
    assert entries[0].stage == "research"


async def test_audit_does_not_store_argument_values(env: Env) -> None:
    """审计日志的读取方比业务数据宽得多：参数只记摘要。"""

    secret = "客户名单：张三 13800000000"
    await env.executor.execute(
        _call("write_brief", path="briefs/a.md", content=secret),
        principal=EDITOR,
        stage="draft",
    )

    payload = str([entry.as_dict() for entry in env.audit.entries()])

    assert "13800000000" not in payload
    assert "content" in payload  # 键名保留，值不留


async def test_replayed_flag_and_cost_flow_into_audit(env: Env, tmp_path: Path) -> None:
    """回放的成本记 0，但**标记**必须留在审计里（否则 W11 看板失真）。"""

    from agent_framework.skills import RecordedToolTransport
    from agent_framework.skills.servers.search_server import SearchServer as SS

    recorder = RecordedToolTransport(
        fixtures_root=tmp_path / "fx", record=True, upstream=SS().transport()
    )
    client = SS().client(transport=recorder)
    await client.list_tools()
    await client.call_tool("search__search_web", {"query": "增长"})

    player = RecordedToolTransport(fixtures_root=tmp_path / "fx")
    replayed_client = SS().client(transport=player)
    await replayed_client.list_tools()

    executor = SkillExecutor(
        registry=SkillRegistry(),
        # 顺序：先铺开原 clients，再用回放版覆盖 search —— 反过来会把
        # 回放版覆盖掉，测试就变成了"在测 live 调用"。
        clients={**env.executor.clients, "search": replayed_client},
        sandbox=env.executor.sandbox,
        audit=AuditLog(),
    )

    await executor.execute(
        _call("web_search", query="增长"), principal=EDITOR, stage="research"
    )

    entry = executor.audit.entries()[-1]
    assert entry.replayed is True
    assert entry.cost == 0.0
    assert executor.audit.summary()["replayed"] == 1


async def test_surface_helper_matches_execution_gate(env: Env) -> None:
    decision = env.executor.surface(principal=EDITOR, stage="research")

    assert "web_search" in decision.names()
    assert "publish_post" not in decision.names()
