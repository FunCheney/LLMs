"""工具运行时装配（W8）：把框架的 skills 组件接成产品能用的一个对象。

为什么装配逻辑在 `apps/` 而不是框架里：**用什么目录、开哪些命令、谁能发布**
是产品决策。框架只提供机制（白名单、沙箱、审计），把它写进框架等于替
所有使用方做决定——而第一个"我们这个项目特殊"就会变成框架里的一个 if。

三个装配选择及其理由：

1. **工作区默认是 `./.agent_workspace`**，不是仓库根。沙箱默认只允许这个
   子树写入，配成 `.` 等于让模型能改源码——而"改源码"这个能力不该因为
   一条默认配置被顺带授予。
2. **命令与网络白名单默认空**（全拒）。放行清单必须是显式配置：
   `echo`、`pytest` 这些看起来无害，但白名单一旦非空，模型就能在
   前缀之后接任意参数（`pytest --rootdir=/` 之类）。所以清单要短、要评审。
3. **审批通道接的是 `ApprovalGate`（W10 改）**。W8 时这里留空，`irreversible`
   动作返回 `pending` 且不执行——那是对的默认值，但没有人能把它放行。
   现在接上真通道：第一次调用仍返回 `pending`（人还没批），批完之后同一动作
   返回 `approved`。**默认行为没有变化**（仍是 fail-closed），变化的是
   "pending 终于有出口了"。

`TOOLS=off` 时装配函数直接抛 `ConfigError`：`/v1/tools` 是运维/演示面，
"静默返回空工具集"会让人以为 Skill 没加载，而实际是没开。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_framework.config import ConfigError, Settings
from agent_framework.guardrails import (
    ApprovalGate,
    BehaviorMonitor,
)
from agent_framework.retrieval import Principal
from agent_framework.skills import (
    ActionRuntime,
    AuditLog,
    PermissionGuard,
    RecordedToolTransport,
    ResultTruncator,
    SandboxLimits,
    SandboxPolicy,
    SkillExecutor,
    SkillRegistry,
    TaskStage,
    VisibilityDecision,
)
from agent_framework.skills.servers import (
    FetchServer,
    SearchServer,
    WorkspaceServer,
)

logger = logging.getLogger(__name__)

__all__ = ["ToolRuntime", "build_tool_runtime"]

# 默认命令前缀：只读、无副作用、CI 里天天跑的那几条。
# 不含 `git push` / `git reset` / 任何写操作——参数级黑名单也会拦，
# 但"不放进白名单"比"放进来再拦"少一层依赖。
DEFAULT_COMMAND_PREFIXES = (
    ".venv/bin/ruff",
    ".venv/bin/pytest",
    "ruff",
    "pytest",
    "git status",
    "git diff",
    "git log",
    "echo",
    "ls",
)


@dataclass
class ToolRuntime:
    """工具链路的全部装配产物。路由只读它，不自己构造组件。"""

    registry: SkillRegistry
    executor: SkillExecutor
    actions: ActionRuntime
    audit: AuditLog
    workspace: WorkspaceServer
    settings: Settings
    # W10：审批门与行为监控。路由要用它们开审批端点、看频率告警。
    approvals: ApprovalGate
    behavior: BehaviorMonitor

    def surface(
        self, *, principal: Principal | None = None, stage: TaskStage | str = TaskStage.RESEARCH
    ) -> VisibilityDecision:
        return self.executor.surface(principal=principal, stage=stage)

    async def reload(self) -> tuple[str, ...]:
        """热重载 Skill 文件。返回全部 Skill 名（调用方可以直接对比前后差异）。"""
        await self.registry.reload()
        return self.registry.names()

    def close(self) -> None:
        """当前没有长连接；留这个方法是为了补齐 lifespan 的对称性
        （W11 接真 MCP stdio 时在这里关子进程）。"""


def build_tool_runtime(settings: Settings, *, gateway: Any = None) -> ToolRuntime:
    """按 Settings 装配。`gateway` 当前未用，占位给 W9 的执行循环。"""
    if settings.tools != "governed":
        raise ConfigError(
            f"工具链路未开启（TOOLS={settings.tools}）。要启用设 TOOLS=governed。"
        )

    root = Path(settings.tools_workspace_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    command_allow = _split(settings.tools_command_allow) or DEFAULT_COMMAND_PREFIXES
    network_allow = _split(settings.tools_network_allow)

    sandbox = SandboxPolicy(
        write_roots=(root,),
        command_prefixes=command_allow,
        network_allow=network_allow,
        limits=SandboxLimits(timeout_s=settings.tools_command_timeout_s),
    )

    workspace = WorkspaceServer(root=root, sandbox=sandbox)
    clients = {
        "search": SearchServer().client(),
        "fetch": FetchServer().client(),
        "workspace": workspace.client(timeout_s=settings.tools_command_timeout_s + 5),
    }

    registry = SkillRegistry()
    skills_dir = settings.tools_skills_dir.strip()
    if skills_dir:
        loaded = registry.discover([item for item in _split(skills_dir)])
        logger.info("已加载外部 Skill：%s", [spec.name for spec in loaded])

    audit = AuditLog()
    # W10：审批门。`PermissionGuard(approver=...)` 之后，W8 的
    # "irreversible → pending" 就接到了真通道上——但**默认仍然是 pending**，
    # 因为人还没批。这条接线不会让任何动作自动放行。
    approvals = ApprovalGate()
    behavior = BehaviorMonitor()
    executor = SkillExecutor(
        registry=registry,
        clients=clients,
        guard=PermissionGuard(approver=approvals.as_approver()),
        sandbox=sandbox,
        audit=audit,
        truncator=ResultTruncator(max_chars=settings.tools_max_result_chars),
    )
    actions = ActionRuntime(
        executor=executor, ttl_s=settings.tools_idempotency_ttl_s
    )

    return ToolRuntime(
        registry=registry,
        executor=executor,
        actions=actions,
        audit=audit,
        workspace=workspace,
        settings=settings,
        approvals=approvals,
        behavior=behavior,
    )


def _split(raw: str) -> tuple[str, ...]:
    """逗号分隔字符串 → 元组。空串 → 空元组（**不是** `("",)`）。

    用逗号串而不是 JSON 数组：`.env` 里写 JSON 数组要处理引号与转义，
    compose 文件里再包一层就基本没人写得对了。逗号串的代价是"值里不能有逗号"，
    而域名与命令前缀本来就不会有。
    """
    return tuple(item.strip() for item in (raw or "").split(",") if item.strip())


def replay_transport(fixtures_root: str | Path, *, server: Any) -> Any:
    """给离线评测用的回放传输（`TOOLS` 与 `record` 都不在这里管）。

    单独一个函数是为了让"怎么录"和"怎么装配"分开：录制是脚本的事
    （`evals/scripts/tool_boundary_check.py --record`），装配是产品的事。
    """
    return RecordedToolTransport(fixtures_root=fixtures_root, upstream=server)
