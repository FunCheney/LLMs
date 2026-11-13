"""Trajectory（过程）层指标（W9 D5）。

W3–W4 的评测只覆盖了两层：**Outcome**（对不对）与 **System**（快不快、贵不贵）。
第三层 **Trajectory**（怎么做到的）是空的。后果很具体：本周实验只能得出
"多 Agent 成功率 +8%"，然后被追问"为什么高？"——答不上来。可能是分工对了，
可能是工具选对了，也可能只是多花了 3 倍 token 多试了几次。三种归因对应
三种完全不同的产品决策，而只看得分是分不出来的。

四个指标各自回答一个不同的问题：

| 指标 | 回答 | 变差说明什么 |
|---|---|---|
| `tool_selection_acc` | 该调的调了、不该调的没调 | 工具面/prompt 有问题，不是模型笨 |
| `invalid_call_rate` | 有多少次调用被自己的门挡住 | 契约（schema/可见面）没讲清 |
| `steps_used` | 花了多少步 | 编排在绕路 |
| `path_repeat_rate` | 有多少步在原地打转 | 规划器没状态，或上下文窗口不够 |

**口径声明（读数字前必须看）：**

1. `tool_selection_acc` 用**最长公共子序列**（LCS）除以期望长度。它奖励
   "工具选对且顺序对"，对**换序**只保留一半（`[a,b]` vs `[b,a]` 得 0.5）——
   这是 LCS 的性质，不是 bug：多数任务的工具顺序有语义（先 search 再 fetch）。
   需要"只看集合、不管顺序"的场景应该换集合覆盖率，而不是把这条改成逐位比对。
   另外它对**多调了**不敏感（多调一次不算选错），那由 `invalid_call_rate` 抓。
2. 期望工具**没声明**时返回 `None`，不是 `1.0`。"没声明"和"完全正确"混在一起，
   会让准确率虚高，而虚高的准确率比没有准确率更糟。
3. `invalid_call_rate` 的分子只算**我们自己的门挡住的**（工具不存在/不可见/
   参数非法/没权限/沙箱拒绝），不含对方系统 5xx。前者归因到工具层，后者归因到
   外部依赖；混在一起就分不清该改哪边。
4. `steps_used` 是**工具执行次数**，不是角色轮数。多 Agent 的一次交接可能包含多次
   工具调用，两者不是一个量。
5. 每个指标都随样本数一起输出。均值不带样本数，就等于让读者自己猜它有多可信。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "TrajectoryMetrics",
    "aggregate_trajectories",
    "invalid_call_rate",
    "path_repeat_rate",
    "steps_used",
    "tool_selection_acc",
]

# 我们自己的门挡住的调用（与 `planner.base.StepRecord.is_invalid` 同一口径）。
# 刻意与"对方系统报错"分开：前者该改契约，后者该等对方修。
_INVALID_CODES = frozenset(
    {
        "tool_not_visible",
        "tool_not_found",
        "invalid_arguments",
        "permission_denied",
        "sandbox_denied",
    }
)


# --------------------------------------------------------------------------- #
# 单条 trace 的指标
# --------------------------------------------------------------------------- #


def _field(record: Any, name: str, default: Any = "") -> Any:
    """`StepRecord` 或它的 dict 形态都吃。离线产出的 jsonl 不必先转回对象。"""

    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _tools(records: Sequence[Any]) -> list[str]:
    return [str(_field(record, "tool")) for record in records if _field(record, "tool")]


def _is_invalid(record: Any) -> bool:
    return str(_field(record, "status")) == "denied" or str(
        _field(record, "code")
    ) in _INVALID_CODES


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """最长公共子序列长度。O(n·m)，轨迹长度在 20 以内，不值得上更快的东西。"""

    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            if item == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def tool_selection_acc(
    records: Sequence[Any], expected: Sequence[str] | None
) -> float | None:
    """期望工具序列与实际序列的 LCS / 期望长度。

    `expected` 为 None 或空 = 这条用例没声明期望工具，返回 `None`。
    **不要把它折成 1.0**：那会让所有"没声明"的用例白送满分。
    """

    if not expected:
        return None
    actual = _tools(records)
    return lcs_length(actual, list(expected)) / len(expected)


def invalid_call_rate(records: Sequence[Any]) -> float | None:
    """无效调用占比。没有调用时返回 `None`（"没调"不等于"没有无效调用"）。"""

    total = len(records)
    if total == 0:
        return None
    return sum(1 for record in records if _is_invalid(record)) / total


def steps_used(records: Sequence[Any]) -> int:
    """工具执行次数。0 是一个真实观测（纯对话任务本来就不该调工具）。"""

    return len(records)


def path_repeat_rate(records: Sequence[Any]) -> float | None:
    """重复路径占比：`(总步数 - 不同(工具,参数摘要) 数) / 总步数`。

    同工具同参数出现两次 = 在原地打转。它与"步数多"是两件事：
    步数多是绕远路，路径重复是**同一步走了两遍**。
    """

    total = len(records)
    if total == 0:
        return None
    distinct = {
        (str(_field(record, "tool")), str(_field(record, "args_digest")))
        for record in records
    }
    return (total - len(distinct)) / total


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TrajectoryMetrics:
    """一条或一批 trace 的过程指标。

    `*_samples` 与均值一起给：`tool_selection_acc=0.8` 由 2 条算出来还是由 50 条
    算出来，可信度完全不同，而读者无从判断——除非我们写出来。
    """

    tool_selection_acc: float | None = None
    invalid_call_rate: float | None = None
    steps_used: float | None = None
    path_repeat_rate: float | None = None

    tool_selection_samples: int = 0
    invalid_call_samples: int = 0
    steps_samples: int = 0
    path_repeat_samples: int = 0

    # 参与聚合的用例数。与上面各个 samples 的差额，就是"该指标算不出来"的条数。
    cases: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_selection_acc": self.tool_selection_acc,
            "invalid_call_rate": self.invalid_call_rate,
            "steps_used": self.steps_used,
            "path_repeat_rate": self.path_repeat_rate,
            "tool_selection_samples": self.tool_selection_samples,
            "invalid_call_samples": self.invalid_call_samples,
            "steps_samples": self.steps_samples,
            "path_repeat_samples": self.path_repeat_samples,
            "cases": self.cases,
        }


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def aggregate_trajectories(
    rows: Iterable[tuple[Sequence[Any], Sequence[str] | None]],
) -> TrajectoryMetrics:
    """把一批 `(records, expected_tools)` 折成均值。

    幂等、不看时间：同一批输入调两次必然同值，否则"可复现"这条 Gate 是空的。
    """

    acc: list[float] = []
    invalid: list[float] = []
    steps: list[float] = []
    repeat: list[float] = []
    cases = 0

    for records, expected in rows:
        cases += 1
        value = tool_selection_acc(records, expected)
        if value is not None:
            acc.append(value)
        value = invalid_call_rate(records)
        if value is not None:
            invalid.append(value)
        steps.append(float(steps_used(records)))
        value = path_repeat_rate(records)
        if value is not None:
            repeat.append(value)

    return TrajectoryMetrics(
        tool_selection_acc=_mean(acc),
        invalid_call_rate=_mean(invalid),
        steps_used=_mean(steps),
        path_repeat_rate=_mean(repeat),
        tool_selection_samples=len(acc),
        invalid_call_samples=len(invalid),
        steps_samples=len(steps),
        path_repeat_samples=len(repeat),
        cases=cases,
    )


# --------------------------------------------------------------------------- #
# CLI：从 jsonl 算
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Case:
    id: str
    records: list[Any] = field(default_factory=list)
    expected: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def load_cases(path: Path) -> list[_Case]:
    """读一行一条的轨迹 dump。

    期望格式（`agent_paradigm_ab.py` 产出的就是这个）：

    ```
    {"id": "...", "expected_tools": ["web_search"], "records": [{"tool": ..., ...}]}
    ```

    用 `.jsonl` 而不是 `.json`：跑一次就要看一行，用大 JSON 会逼着人先解析再人眼找。
    """

    cases: list[_Case] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        item = json.loads(line)
        cases.append(
            _Case(
                id=str(item.get("id", "")),
                records=list(item.get("records") or []),
                expected=item.get("expected_tools"),
                meta={
                    key: value
                    for key, value in item.items()
                    if key not in {"id", "records", "expected_tools"}
                },
            )
        )
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trajectory 层指标（W9 D5）")
    parser.add_argument("--input", required=True, help="轨迹 jsonl 路径")
    parser.add_argument(
        "--group-by",
        default="",
        help="按 meta 里的某个键分组（例如 difficulty / config）",
    )
    parser.add_argument("--json", action="store_true", help="输出 json 而不是表格")
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.input))
    if not cases:
        print(f"没有读到任何 case：{args.input}", file=sys.stderr)
        return 1

    groups: dict[str, list[_Case]] = {}
    for case in cases:
        key = str(case.meta.get(args.group_by, "")) if args.group_by else "all"
        groups.setdefault(key or "(空)", []).append(case)

    out: dict[str, Any] = {}
    for key, group in sorted(groups.items()):
        out[key] = aggregate_trajectories(
            (case.records, case.expected) for case in group
        ).as_dict()

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    header = (
        f"{'分组':<14}{'cases':>6}{'工具选择':>10}{'无效调用':>10}"
        f"{'步数':>8}{'路径重复':>10}"
    )
    print(header)
    print("-" * len(header))
    for key, row in out.items():
        print(
            f"{key:<14}{row['cases']:>6}"
            f"{_fmt(row['tool_selection_acc'], row['tool_selection_samples']):>10}"
            f"{_fmt(row['invalid_call_rate'], row['invalid_call_samples']):>10}"
            f"{_fmt(row['steps_used'], row['steps_samples']):>8}"
            f"{_fmt(row['path_repeat_rate'], row['path_repeat_samples']):>10}"
        )
    print()
    print("括号里是参与该指标计算的样本数；`-` = 全部算不出来（不是 0）。")
    return 0


def _fmt(value: float | None, samples: int) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}({samples})"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
