"""错误码归类：**唯一一份**码表。W4 收敛在这里，W11 D3 的执行模型复用它。

## 为什么从 `evals/scripts/classify_failures.py` 搬进框架包

W11 D3 的 `Attempt.error_code` 要回答"这次失败是不是厂商的锅"，而 `runtime/` 是
**生产代码**——它不能 import 一个评测脚本。搬过来之后分类器改为 re-export，
历史导入路径（`from classify_failures import RELIABILITY_CODES`）不变。

W4 D5 的教训值得写在这里：之前 `run_eval` 与 `classify_failures` 各有一份码表，
10 个已知错误码里有 6 个在两处判定相反（`SERVER_BUSY` 在一边算"质量失败"、
在另一边算"上游问题"）。结果是一批被限流的请求把 success_rate 拉低，
而报告说"模型答错了"。**两份码表必然漂移，唯一解是只留一份。**

## 三档，不是两档

| 档 | 含义 | 算到谁头上 |
|---|---|---|
| `RELIABILITY_CODES` | 厂商侧故障（5xx/超时/限流/上游鉴权/熔断） | 不算模型答错 |
| `CALLER_CODES` | 我方请求或配置写错了（400/能力不支持/任务不存在/内容拦截） | 不算模型答错 |
| 其余 | 未知码（厂商新加的） | **按可靠性处理** |

未知码按可靠性处理是刻意的：宁可少算一次"模型答错"，也不要把一个新出现的厂商故障
记成质量退化——质量退化会引发"改 prompt"这种方向的修复，而方向错了比不修更贵。
"""

from __future__ import annotations

__all__ = [
    "CALLER_CODES",
    "KNOWN_CODES",
    "RELIABILITY_CODES",
    "TIMEOUT_CODES",
    "is_caller_code",
    "is_reliability_code",
    "is_timeout_code",
]

# 厂商侧。`CIRCUIT_OPEN`（W11 D2）也在这里：它不是某个厂商吐的码，而是
# "我们已经停止调用上游"——语义上属于可靠性，且绝不能算成模型答错。
RELIABILITY_CODES = frozenset(
    {
        "UPSTREAM_AUTH",
        "UPSTREAM_ERROR",
        "UPSTREAM_TIMEOUT",
        "RATE_LIMITED",
        "SERVER_BUSY",
        "CIRCUIT_OPEN",
    }
)

# 慢到该单独看的一档：`SERVER_BUSY` 是我们自己的限流器在排队，
# 与"上游超时"在处置上不同（扩并发 vs 看上游），但在"慢"这件事上同类。
# `CIRCUIT_OPEN` **不在这里**：它是快速失败（毫秒返回），归 UPSTREAM——
# 把它算成 TIMEOUT，会让"延迟"看板以为有一批请求很慢，实际是我们主动拒绝的。
TIMEOUT_CODES = frozenset({"UPSTREAM_TIMEOUT", "SERVER_BUSY"})

# 我方问题：请求本身不合法、能力不匹配、配置写错。既不是厂商故障，也不是模型答错。
CALLER_CODES = frozenset(
    {
        "BAD_REQUEST",
        "CAPABILITY_UNSUPPORTED",
        "CONFIG_ERROR",
        "TASK_NOT_FOUND",
        "CONTENT_FILTERED",
    }
)

# 已知码全集。测试用它做不变量：`ERROR_HTTP` 里出现的码必须都在这里，
# 否则新增一个异常类就悄悄多了一个"未被归类"的码。
KNOWN_CODES = RELIABILITY_CODES | CALLER_CODES


def is_reliability_code(code: str | None) -> bool:
    """这个错误码是否属于"厂商侧故障"（→ `interrupted`，不进成功率分母）。

    未知码返回 `True`（按可靠性处理），理由见模块文档。
    """

    if not code:
        return False

    return code not in CALLER_CODES


def is_caller_code(code: str | None) -> bool:
    """是否是我方问题。**显式判断**，不用 `not is_reliability_code`——

    那个取反在 `code is None` 上会得到"是我方问题"，而"没有错误码"恰恰不是。
    """

    return bool(code) and code in CALLER_CODES


def is_timeout_code(code: str | None) -> bool:
    return bool(code) and code in TIMEOUT_CODES
