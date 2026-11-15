"""agent-framework —— 可复用的 Agent 框架。

## 顶层导出的是「起步面」，不是全集索引

`from agent_framework import ...` 只承诺一件事：**从这里能起一个服务、发一条 chat**。
其余能力按子包访问（`agent_framework.retrieval` / `memory` / `skills` / `planner` /
`runtime` / `multi_agent`）。

为什么不做成全集索引：顶层 `__all__` 是承诺面，每加一个名字就多一份"不能随便改"
的负担。把 179 个符号全塞进来，等于承诺一个谁也不会去读的契约，真正的兼容性
风险反而被稀释——面试官问"你的公共 API 是什么"，答案不该是"全部"。

## 这份名单是扫描出来的

`evals/scripts/api_audit.py` 扫 `apps/` 的真实 import，找出"产品在用但顶层没导出"
的符号。上一轮它报了 5 处（`build_gateway` / `Settings` / `ConfigError` /
`configure_logging` / `current_request_id`），这份 `__init__.py` 是对它的回应。

改完跑：

    uv run python evals/scripts/api_audit.py --check

它守住三件事：无私有符号泄漏、无空壳模块、产品在用的符号都能从子包浅路径导入。

## 为什么重的东西走 `__getattr__`

`bootstrap` 会连锁拉起 config → yaml → registry，`llm.client` 会拉起 httpx。
把它们写成本模块的顶层 import，等于「`import agent_framework.anything`
都要付这笔钱」——包括只想读一行配置的场景。

更要命的是它会造环：`config.py` 需要 `llm.base.Capability`，而 `llm.client`
需要 `config.Catalog`。哪个先被顶层 import 都会把另一个卡在半路。

PEP 562 的模块级 `__getattr__` 让这两件事一起解决：属性第一次被访问才加载。
`from agent_framework import ChatCall` 照常可用，但
`import agent_framework.config` 不会再顺手拉起 httpx。

## 依赖方向

`apps/*` -> `agent_framework`，框架永不 import 产品（ADR-004 单向依赖）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# 轻量：types 是纯 dataclass，observability 只依赖 logging。直接导入更直观。
from .llm.types import Message, MessageRole, ToolCall, Usage
from .observability import configure_logging, current_request_id

if TYPE_CHECKING:  # 只为 mypy/IDE：运行时由 __getattr__ 提供
    from .bootstrap import build_gateway
    from .config import ConfigError, Settings
    from .llm.client import ChatCall, ChatOutcome, LLMGateway

# 符号 -> 所在模块。值是相对本包的模块名。
_LAZY: dict[str, str] = {
    "build_gateway": "bootstrap",
    "ConfigError": "config",
    "Settings": "config",
    "ChatCall": "llm.client",
    "ChatOutcome": "llm.client",
    "LLMGateway": "llm.client",
}

__all__ = [
    "__version__",
    # 造链路（读 models.yaml + 装配 router/breaker/cache/limiter）
    "build_gateway",
    # 配置
    "ConfigError",
    "Settings",
    # 发一条 chat 需要的四个类型
    "ChatCall",
    "ChatOutcome",
    "LLMGateway",
    "Message",
    "MessageRole",
    "ToolCall",
    "Usage",
    # 可观测性：任何要上线的服务第一件事就是配日志 + 拿 request id
    "configure_logging",
    "current_request_id",
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(f".{target}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
