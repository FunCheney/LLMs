"""Skill 与工具执行（W8）：声明 → 受控执行 → 回注 → 审计。

模块分工（一条链路的四段，别混）：

| 文件 | 管什么 | 不管什么 |
|---|---|---|
| `mcp/` | 外部工具传输协议（wire）与翻译 | 权限、审计、验真 |
| `servers/` | 内置的搜索/抓取/工作区 server | 谁可以用 |
| `replay.py` | 录制回放（离线可复现） | 回放之外的重试/缓存 |
| `spec.py` | Skill 六层规范 + A14 校验 | 加载、执行 |
| `loader.py` | 发现 / 热重载 / 渐进式加载 | 加载之后的执行 |
| `sandbox.py` | 路径/命令/网络三道边界 + 超时 | 授权（那是 guard 的事） |
| `visibility.py` | 最小模型可见面 | 细粒度授权（那是 guard 的事） |
| `guard.py` | 白名单 + 风险分级 + 审批门 | 执行（那是 executor 的事） |
| `executor.py` | 五道门 + 回注 | 幂等/验真/审批编排 |
| `action.py` | 提议→校验→授权→审批→执行→验真 | 怎么执行（那是 executor 的事） |
| `audit.py` | 全审计（参数只记摘要） | 判定 |
| `validator.py` | 完成标准校验（第 6 层） | 重试策略之外的重生成 |

一个刻意的分界：**执行循环不在框架里**。`executor.execute()` 执行**一次**
工具调用并返回要回注的消息；"模型 → 工具 → 再问模型"这个循环属于产品层
（`apps/growth_agent/tools.py`）或 W9 的编排层。把它塞进 `LLMGateway`
会违反 ADR-003（重试只在 client），也让网关开始承担它不该承担的职责。
"""

from .action import (
    STATUS_DUPLICATE,
    STATUS_FAILED,
    STATUS_FORBIDDEN,
    STATUS_INVALID,
    STATUS_NOT_VISIBLE,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    Action,
    ActionResult,
    ActionRuntime,
    NumericCapRule,
    VerifyOutcome,
)
from .audit import AuditLog, ToolAudit, args_digest
from .contract import (
    CODE_INVALID_ARGUMENTS,
    CODE_NOT_FOUND,
    CODE_OK,
    CODE_PERMISSION_DENIED,
    CODE_SANDBOX_DENIED,
    CODE_TOOL_NOT_VISIBLE,
    CODE_UNVERIFIED,
    ToolOutcome,
)
from .executor import ResultTruncator, SkillExecutor, ToolExecution
from .guard import (
    ANY_PRINCIPAL,
    DEFAULT_POLICY,
    RISK_IRREVERSIBLE,
    RISK_READ,
    RISK_WRITE,
    ApprovalDecision,
    PermissionDecision,
    PermissionGuard,
    PermissionPolicy,
    risk_of,
)
from .library import BUILTIN_SKILLS, builtin_skills, get_builtin
from .loader import SkillRegistry, registry_snapshot
from .replay import (
    RecordedToolTransport,
    ReplayMissError,
    ReplayStats,
    canonical_args,
    fixture_key,
)
from .sandbox import (
    COMMAND_DENIED,
    NETWORK_DENIED,
    PATH_DENIED,
    CommandResult,
    SandboxDecision,
    SandboxLimits,
    SandboxPolicy,
)
from .spec import (
    CHECK_KINDS,
    KNOWN_PERMISSIONS,
    PERMISSION_NET,
    PERMISSION_PUBLISH,
    PERMISSION_READ,
    PERMISSION_SHELL,
    PERMISSION_WRITE,
    Check,
    SkillSpec,
    SkillSpecError,
    TaskStage,
    VerifySpec,
    validate_arguments,
)
from .validator import SkillValidator, Verification
from .visibility import (
    COARSE_PERMISSIONS,
    HIDDEN_BY_PRINCIPAL,
    HIDDEN_BY_STAGE,
    VisibilityDecision,
    VisibilityPolicy,
)

__all__ = [
    "ANY_PRINCIPAL",
    "BUILTIN_SKILLS",
    "CHECK_KINDS",
    "CODE_INVALID_ARGUMENTS",
    "CODE_NOT_FOUND",
    "CODE_OK",
    "CODE_PERMISSION_DENIED",
    "CODE_SANDBOX_DENIED",
    "CODE_TOOL_NOT_VISIBLE",
    "CODE_UNVERIFIED",
    "COMMAND_DENIED",
    "COARSE_PERMISSIONS",
    "DEFAULT_POLICY",
    "HIDDEN_BY_PRINCIPAL",
    "HIDDEN_BY_STAGE",
    "KNOWN_PERMISSIONS",
    "NETWORK_DENIED",
    "PATH_DENIED",
    "PERMISSION_NET",
    "PERMISSION_PUBLISH",
    "PERMISSION_READ",
    "PERMISSION_SHELL",
    "PERMISSION_WRITE",
    "RISK_IRREVERSIBLE",
    "RISK_READ",
    "RISK_WRITE",
    "STATUS_DUPLICATE",
    "STATUS_FAILED",
    "STATUS_FORBIDDEN",
    "STATUS_INVALID",
    "STATUS_NOT_VISIBLE",
    "STATUS_PENDING",
    "STATUS_REJECTED",
    "STATUS_UNVERIFIED",
    "STATUS_VERIFIED",
    "Action",
    "ActionResult",
    "ActionRuntime",
    "ApprovalDecision",
    "AuditLog",
    "Check",
    "CommandResult",
    "NumericCapRule",
    "PermissionDecision",
    "PermissionGuard",
    "PermissionPolicy",
    "RecordedToolTransport",
    "ReplayMissError",
    "ReplayStats",
    "ResultTruncator",
    "SandboxDecision",
    "SandboxLimits",
    "SandboxPolicy",
    "SkillExecutor",
    "SkillRegistry",
    "SkillSpec",
    "SkillSpecError",
    "SkillValidator",
    "TaskStage",
    "ToolAudit",
    "ToolExecution",
    "ToolOutcome",
    "Verification",
    "VerifyOutcome",
    "VerifySpec",
    "VisibilityDecision",
    "VisibilityPolicy",
    "args_digest",
    "builtin_skills",
    "canonical_args",
    "fixture_key",
    "get_builtin",
    "registry_snapshot",
    "risk_of",
    "validate_arguments",
]
