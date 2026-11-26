"""W8 D5：Action Runtime 与三个 badcase。

Gate 8 #10 是本周的验收核心：三个 badcase 各对应一个机制，缺一个都不算过。

| badcase | 机制 | 断言的是 |
|---|---|---|
| 重复请求 | 幂等键 | 副作用只发生 1 次（数副作用，不看返回值） |
| 参数篡改 | validate 层业务规则 | 被拒的是**业务规则**，不是权限 |
| 接口假成功 | verify 层回读 | `unverified`，绝不写"已发布" |
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_framework.llm.types import ToolCall
from agent_framework.retrieval import Principal
from agent_framework.skills import (
    STATUS_DUPLICATE,
    STATUS_INVALID,
    STATUS_NOT_VISIBLE,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    ActionRuntime,
    AuditLog,
    NumericCapRule,
    PermissionGuard,
    SandboxPolicy,
    SkillExecutor,
    SkillRegistry,
    SkillSpec,
)
from agent_framework.skills.servers import SearchServer, WorkspaceServer

PUBLISHER = Principal(subject_id="p1", groups=frozenset({"publisher", "editor"}))
VIEWER = Principal(subject_id="v1", groups=frozenset({"viewer"}))

BODY = "这是一篇足够长的正文，用来满足发布工具的最小长度限制。" * 2


@dataclass
class Env:
    runtime: ActionRuntime
    workspace: WorkspaceServer
    root: Path


def _build(
    tmp_path: Path,
    *,
    approver: Any = None,
    fail_soft: bool = False,
    rules: tuple[Any, ...] = (),
) -> Env:
    root = tmp_path / "repo"
    root.mkdir()
    sandbox = SandboxPolicy(write_roots=(root,), command_prefixes=("echo",))
    workspace = WorkspaceServer(root=root, sandbox=sandbox, fail_soft=fail_soft)
    executor = SkillExecutor(
        registry=SkillRegistry(),
        clients={
            "workspace": workspace.client(),
            # 只读检索也接进来：低风险动作的"不需要审批"要有真链路可走。
            "search": SearchServer().client(),
        },
        sandbox=sandbox,
        guard=PermissionGuard(approver=approver),
        audit=AuditLog(),
    )
    runtime = ActionRuntime(executor=executor, rules=rules)
    return Env(runtime=runtime, workspace=workspace, root=root)


def _publish(title: str = "标题") -> ToolCall:
    return ToolCall(
        id="call-publish", name="publish_post", arguments={"title": title, "body": BODY}
    )


async def _approve_always(skill: Any, arguments: Any, summary: str) -> str:
    return "approved"


# --------------------------------------------------------------------------- #
# badcase 1：重复请求
# --------------------------------------------------------------------------- #


async def test_duplicate_request_produces_one_side_effect(tmp_path: Path) -> None:
    """同一 `(action, args)` 连发 3 次：只产生 1 次副作用，后两次拿首次结果。"""

    env = _build(tmp_path, approver=_approve_always)

    first = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")
    second = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")
    third = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    assert first.status == STATUS_VERIFIED
    assert second.status == STATUS_DUPLICATE
    assert third.status == STATUS_DUPLICATE
    assert second.duplicate_of == first.action_id
    # 副作用计数是唯一可信的判据。
    assert len(env.workspace.posts()) == 1
    # 重复请求仍能拿到首次结果（模型需要它继续往下走）。
    assert second.execution is not None
    assert second.execution.outcome is not None
    assert second.execution.outcome.payload["post_id"] == first.execution.outcome.payload["post_id"]


async def test_different_arguments_are_a_different_action(tmp_path: Path) -> None:
    env = _build(tmp_path, approver=_approve_always)

    await env.runtime.submit(_publish("标题 A"), principal=PUBLISHER, stage="publish")
    second = await env.runtime.submit(_publish("标题 B"), principal=PUBLISHER, stage="publish")

    assert second.status == STATUS_VERIFIED
    assert len(env.workspace.posts()) == 2


async def test_intent_id_separates_same_arguments(tmp_path: Path) -> None:
    """"同样的参数、不同的意图"必须能各执行一次（比如两次不同的归档任务）。"""

    env = _build(tmp_path, approver=_approve_always)

    await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish", intent_id="task-1")
    second = await env.runtime.submit(
        _publish(), principal=PUBLISHER, stage="publish", intent_id="task-2"
    )

    assert second.status == STATUS_VERIFIED


async def test_ttl_expiry_allows_reexecution(tmp_path: Path) -> None:
    """TTL 之后同一动作可以再执行：跨天重放更可能是新需求。"""

    env = _build(tmp_path, approver=_approve_always)
    # 可推进的时钟：TTL 测试不能靠 sleep 24 小时。
    now = [1_000_000.0]
    env.runtime._clock = lambda: now[0]  # noqa: SLF001

    await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")
    now[0] += 25 * 60 * 60  # 跨过 24h
    again = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    assert again.status == STATUS_VERIFIED


async def test_repeat_calls_raise_loop_warning(tmp_path: Path) -> None:
    """死循环预警复用 W7 Drift 的 repeat_action，只报警不阻断。"""

    env = _build(tmp_path, approver=_approve_always)

    first = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")
    second = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    assert first.loop_warning is False
    assert second.loop_warning is True
    # 报警不阻断：第二次仍然是"已处理"（duplicate），而不是 failed。
    assert second.status == STATUS_DUPLICATE


# --------------------------------------------------------------------------- #
# badcase 2：参数篡改
# --------------------------------------------------------------------------- #


def _budget_skill() -> SkillSpec:
    return SkillSpec(
        name="issue_budget",
        description="给某个活动批预算并记录。",
        mission="负责记录预算审批；不负责实际付款。",
        principles=("超过业务上限一律拒绝",),
        instructions="1. 写入预算记录。",
        parameters={
            "type": "object",
            "properties": {
                "budget": {"type": "number", "minimum": 0, "maximum": 1_000_000},
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["budget", "path", "content"],
            "additionalProperties": False,
        },
        entrypoint="workspace.write_file",
        permissions=frozenset({"read", "write"}),
        sandbox="subprocess",
        stages=frozenset({"draft"}),
    )


async def test_tampered_argument_is_rejected_by_business_rule(tmp_path: Path) -> None:
    """上下文里塞「把预算改成 100 万」→ 模型照做 → **业务规则**拒绝。

    关键区别：schema 的 `maximum` 是给模型看的说明书（它可能照抄也可能不理），
    业务上限是我们的规矩。这条测试故意让 schema 允许 100 万，
    由 `NumericCapRule` 只批到 20 万——如果只靠 schema，这条会漏。
    """

    env = _build(tmp_path, rules=(NumericCapRule(field="budget", maximum=200_000),))
    env.runtime._executor.registry.register(_budget_skill())  # noqa: SLF001

    result = await env.runtime.submit(
        ToolCall(
            id="call-budget",
            name="issue_budget",
            arguments={"budget": 1_000_000, "path": "budget.md", "content": "预算记录"},
        ),
        principal=PUBLISHER,
        stage="draft",
    )

    assert result.status == STATUS_INVALID
    assert "numeric_cap" in result.detail
    # 不是权限拒绝：主体有 write 权限，是规则不允许。
    assert "权限" not in result.detail
    assert not (env.root / "budget.md").exists()


async def test_schema_level_tampering_is_also_caught(tmp_path: Path) -> None:
    """模型把 `max_results` 改成 1000（schema 上限 10）→ A14 拦下。

    与业务规则是两道不同的闸：这一道在 schema 上，改 schema 就改。
    """

    env = _build(tmp_path)
    result = await env.runtime.submit(
        ToolCall(id="c1", name="web_search", arguments={"query": "增长", "max_results": 1000}),
        principal=PUBLISHER,
        stage="research",
    )

    assert result.status == "failed"
    assert result.execution is not None
    assert result.execution.code == "invalid_arguments"


# --------------------------------------------------------------------------- #
# badcase 3：接口假成功
# --------------------------------------------------------------------------- #


async def test_fake_success_is_unverified_not_published(tmp_path: Path) -> None:
    """接口 200 但响应体 `{"ok": false}` → `unverified`，不写"已发布"。"""

    env = _build(tmp_path, approver=_approve_always, fail_soft=True)

    result = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    assert result.status == STATUS_UNVERIFIED
    assert result.verify is not None
    assert "ok=false" in result.verify.detail
    assert len(env.workspace.posts()) == 0


async def test_success_is_verified_by_read_back(tmp_path: Path) -> None:
    """阳性对照：回读确认 `status=published` 才算 verified。"""

    env = _build(tmp_path, approver=_approve_always)

    result = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    assert result.status == STATUS_VERIFIED
    assert result.verify is not None
    assert "回读" in result.verify.detail


# --------------------------------------------------------------------------- #
# 审批门与风险分级
# --------------------------------------------------------------------------- #


async def test_irreversible_action_is_pending_without_approver(tmp_path: Path) -> None:
    """没接 HITL 不等于默认放行：不可撤销动作停在 pending。"""

    env = _build(tmp_path)  # 没有 approver

    result = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    assert result.status == STATUS_PENDING
    assert result.execution is None
    assert len(env.workspace.posts()) == 0


async def test_rejected_action_does_not_execute(tmp_path: Path) -> None:
    async def reject(skill: Any, arguments: Any, summary: str) -> str:
        return "rejected"

    env = _build(tmp_path, approver=reject)

    result = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    assert result.status == STATUS_REJECTED
    assert len(env.workspace.posts()) == 0


async def test_approval_summary_shows_what_will_happen(tmp_path: Path) -> None:
    """审批页要让人一眼确认"是不是我批的那件事"，所以摘要里要有参数。"""

    captured: list[str] = []

    async def capture(skill: Any, arguments: Any, summary: str) -> str:
        captured.append(summary)
        return "approved"

    env = _build(tmp_path, approver=capture)

    await env.runtime.submit(_publish("季度增长复盘"), principal=PUBLISHER, stage="publish")

    assert captured
    assert "publish_post" in captured[0]
    assert "季度增长复盘" in captured[0]
    assert "不可撤销" in captured[0]


async def test_low_risk_action_does_not_ask_for_approval(tmp_path: Path) -> None:
    env = _build(tmp_path)

    result = await env.runtime.submit(
        ToolCall(id="c1", name="web_search", arguments={"query": "增长"}),
        principal=VIEWER,
        stage="research",
    )

    assert result.status == STATUS_VERIFIED


# --------------------------------------------------------------------------- #
# 与可见面的分工
# --------------------------------------------------------------------------- #


async def test_out_of_surface_tool_is_not_visible_not_forbidden(tmp_path: Path) -> None:
    """被可见面裁掉的工具被强行调用 → `tool_not_visible`（不是 `forbidden`）。"""

    env = _build(tmp_path, approver=_approve_always)

    result = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="research")

    assert result.status == STATUS_NOT_VISIBLE


async def test_visible_but_unauthorized_is_forbidden(tmp_path: Path) -> None:
    """看得见但主体没授权 → `forbidden`。两者在值班队列里不是一类工单。"""

    env = _build(tmp_path)

    result = await env.runtime.submit(
        ToolCall(id="c1", name="write_brief", arguments={"path": "a.md", "content": "内容" * 20}),
        principal=VIEWER,
        stage="draft",
    )

    assert result.status == "forbidden"
    assert "write" in result.detail


async def test_unknown_skill_returns_invalid_action(tmp_path: Path) -> None:
    env = _build(tmp_path)

    result = await env.runtime.submit(
        ToolCall(id="c1", name="no_such_skill", arguments={}), principal=PUBLISHER
    )

    assert result.status == STATUS_INVALID


# --------------------------------------------------------------------------- #
# 审计
# --------------------------------------------------------------------------- #


async def test_audit_has_all_gate_fields(tmp_path: Path) -> None:
    """Gate 8 #6：主体/工具/风险级/结果/replay 标记，一个都不能少。"""

    env = _build(tmp_path, approver=_approve_always)
    await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    entry = env.runtime._executor.audit.entries()[-1]  # noqa: SLF001
    payload = entry.as_dict()

    assert payload["principal"] == "p1"
    assert payload["tool"] == "workspace.publish_post"
    assert payload["risk"] == "irreversible"
    assert payload["status"] == "ok"
    assert payload["replayed"] is False
    assert payload["approval"] == "approved"
    assert payload["args_digest"]
    assert set(payload["arg_keys"]) == {"title", "body"}
    # 参数值不进审计。
    assert BODY[:20] not in str(payload)


async def test_audit_summary_separates_replay_and_live(tmp_path: Path) -> None:
    env = _build(tmp_path, approver=_approve_always)
    await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    summary = env.runtime._executor.audit.summary()  # noqa: SLF001

    assert summary["total"] == 1
    assert summary["live"] == 1
    assert summary["replayed"] == 0


async def test_pending_and_rejected_are_audited(tmp_path: Path) -> None:
    env = _build(tmp_path)

    await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    statuses = [entry.status for entry in env.runtime._executor.audit.entries()]  # noqa: SLF001
    assert STATUS_PENDING in statuses


async def test_runtime_reset_clears_idempotency(tmp_path: Path) -> None:
    env = _build(tmp_path, approver=_approve_always)
    await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")

    env.runtime.reset()

    assert env.runtime.cache_size == 0
    again = await env.runtime.submit(_publish(), principal=PUBLISHER, stage="publish")
    assert again.status == STATUS_VERIFIED
