"""可观测的最小实现（architecture §5，但它是全项目的根）。

三件事：

1. `request_id` 用 contextvar 贯穿，**不层层传参**；
2. 每个请求产出**一条**固定字段的 JSON 日志（字段名一旦定下，W11 的成本看板直接读它）；
3. 成本计算与密钥打码这两个"错了很难发现"的逻辑放这里，只写一份。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from .config import Pricing
from .llm.types import Usage

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

_LOG = logging.getLogger("agent_framework.request")
_ROUTER_LOG = logging.getLogger("agent_framework.router")
_CANDIDATE_LOG = logging.getLogger("agent_framework.llm.candidate")


def new_request_id() -> str:
    return "req_" + uuid4().hex[:12]


def current_request_id() -> str:
    """读不到时返回空串。

    刻意**不抛异常**：日志少一个字段可以接受，但"打日志时抛异常"会把一个
    正常请求变成 500。空串由 `RequestLog.as_dict()` 统一替换为 "-"。
    """

    return request_id_var.get()


@contextmanager
def request_scope(request_id: str | None = None) -> Iterator[str]:
    """设置并复位 token。

    必须用 try/finally 复位：ASGI 的 task 会被复用，漏一次复位就会串号，
    表现为"两条不相干的请求共享同一个 request_id"——排查起来极其痛苦。
    """

    resolved = request_id or new_request_id()
    token = request_id_var.set(resolved)

    try:
        yield resolved
    finally:
        request_id_var.reset(token)


@dataclass(slots=True)
class RequestLog:
    """一次请求的完整记录。字段与 architecture §5 对齐。

    `usage_estimated` 是第 18 个字段：字段从 17 变 18 是允许的（信息更全），
    漏掉一个才不允许——没有它就分不清"成本低"和"厂商没给 usage"。
    """

    request_id: str = ""
    task: str = ""
    tier: str = ""
    provider: str = ""
    model: str = ""
    fallback_from: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    currency: str = "USD"
    latency_ms: float = 0.0
    first_token_ms: float | None = None
    retry_count: int = 0
    finish_reason: str = "unknown"
    error_code: str | None = None
    stream: bool = False
    usage_estimated: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request_id"] = self.request_id or "-"

        return payload


def emit(record: RequestLog, *, logger: logging.Logger | None = None) -> None:
    """输出一条结构化日志。

    W1 直接打 JSON 到 logging；W3 接 Langfuse 时只换 exporter，字段不动。
    """

    target = logger or _LOG
    target.info(
        json.dumps(record.as_dict(), ensure_ascii=False, default=str, sort_keys=True)
    )


def emit_candidate_failure(alias: str, attempt: int, code: str, message: str) -> None:
    """候选失败单独打 WARNING，**不**计入 RequestLog。

    一个请求可能试 3 个候选，但只允许产出一条 RequestLog（architecture §5），
    否则 W11 统计请求量与成本时会翻倍。
    """

    _CANDIDATE_LOG.warning(
        "candidate failed alias=%s attempt=%d code=%s message=%s",
        alias,
        attempt,
        code,
        message,
    )


def router_log() -> logging.Logger:
    """router 的回落提示走这里（未知 task 静默回落比报错更危险）。"""

    return _ROUTER_LOG


def get_request_logger() -> logging.Logger:
    """`RequestLog` 的输出通道。

    暴露它是为了让**读取方**（Day5 的基线脚本、W11 的成本看板 exporter）能挂
    handler 消费同一条日志，而不是各自 `time.monotonic()` 另造一套指标口径。
    写入口仍然是 `emit()`，这里只给读。
    """

    return _LOG


_FRAMEWORK_LOGGER = "agent_framework"


def configure_logging(level: int = logging.INFO) -> None:
    """让框架自己的结构化日志真正可见（幂等）。

    不配会怎样：root logger 默认 `WARNING`，而 `RequestLog` 是 `INFO`，
    于是"代码里 emit 了"与"日志里看得到"完全是两件事。这个缺陷很阴——
    Gate 验收项「`first_token_ms` 有数值入日志」看起来通过了（代码确实调了 emit），
    但 `uvicorn` 起来后一行都没有，直到 W11 做成本看板才发现历史数据是空的。
    Day5 真机联调时才暴露，所以修在这里。

    两条克制：

    - 只调 `agent_framework` 这一棵树的级别，不动 root、不动别人的 logger；
    - 宿主已经配了输出通道（pytest 的 caplog、生产的 JSON 日志采集）时**只调级别**，
      不抢输出，避免同一条日志被打两遍。
    """

    logger = logging.getLogger(_FRAMEWORK_LOGGER)
    logger.setLevel(level)

    if any(getattr(handler, "_agent_framework_owned", False) for handler in logger.handlers):
        return

    if logging.getLogger().handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    setattr(handler, "_agent_framework_owned", True)  # noqa: B010 - 打标记用于幂等判断
    logger.addHandler(handler)


def compute_cost(pricing: Pricing, usage: Usage) -> float:
    """成本 = 输入价 × 输入 token + 输出价 × 输出 token。

    - 用 `per`（每百万）除，不假设单位；
    - 保留 6 位小数：单次成本常在 0.0001 量级，按 2 位圆整会抹成 0，
      而"成本恒为 0"会让 W11 的熔断器彻底失效。
    """

    raw = (
        usage.input_tokens * pricing.input + usage.output_tokens * pricing.output
    ) / pricing.per

    return round(raw, 6)


def mask_secret(value: str | None) -> str:
    """sk-abcdef...wxyz → `sk-***wxyz`；过短一律 `***`。

    日志里只允许出现这个形态：前缀用于判断"配错的是哪家"，尾 4 位用于对账。
    """

    if not value:
        return "***"

    if len(value) < 8:
        return "***"

    return f"{value[:3]}***{value[-4:]}"


@dataclass(slots=True)
class CostSummary:
    """便于测试与后续看板复用的成本摘要。"""

    cost: float
    currency: str
    usage: Usage = field(default_factory=Usage)
