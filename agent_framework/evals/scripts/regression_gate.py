"""CI 回归门禁（W4 D3）：指标劣化就让流水线变红。

它防的不是"代码写错"（那是 pytest 的事），而是**"指标悄悄变差"**：
改了 prompt、换了模型快照、调了并发，单测全绿、评测也跑通，只是成功率掉了 4 个点。
没有这道门禁，这种退化只有等到某天有人翻报告才会被发现。

五条口径，每条都对应一次"用绝对阈值会翻车"的场景
------------------------------------------------
1. **相对劣化，不是绝对值**：不同机器 p95 天然差 20%，绝对阈值（"p95 必须 <50ms"）
   会天天误报，最后所有人学会忽略红叉——比没有门禁更糟。
2. **延迟类加绝对下限**：基线 p95=2ms 时，"允许劣化 20%" 等于允许涨到 2.4ms，
   抖动一次就触发。所以还要求绝对增量超过 `min_absolute`（默认 5ms）才算劣化。
3. **UPSTREAM 激增单独告警，不判 fail**：它涨了说明厂商在抖，该看状态页而不是
   改 prompt（W3 已定口径）。把它算进 success_rate 的失败里，等于让可靠性污染质量结论。
4. **基线缺失必须报错，不许静默放行**：没有基线就"通过"，等于门禁默认永远绿。
5. **基线要带 commit / 记录时间 / 模型 alias 快照**：模型快照换了，旧基线就失效
   （ADR-009），拿它比出来的"劣化"其实是换了模型。

用法
----
    uv run python evals/scripts/regression_gate.py                       # 比 baseline 与最新一期
    uv run python evals/scripts/regression_gate.py --current evals/history/2026-W47-eval.json
    uv run python evals/scripts/regression_gate.py --write-baseline      # 用当前期重建基线
    uv run python evals/scripts/regression_gate.py --strict              # 把告警也当失败

退出码：0 通过 / 1 有劣化 / 2 环境不满足（基线缺失、读不到指标）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

HISTORY_DIR = REPO_ROOT / "evals" / "history"
BASELINE_PATH = HISTORY_DIR / "baseline.json"
CATALOG_PATH = (
    REPO_ROOT / "packages" / "agent_framework" / "src" / "agent_framework" / "models.yaml"
)

__all__ = [
    "DEFAULT_RULES",
    "GateRule",
    "GateVerdict",
    "Violation",
    "evaluate",
    "load_metrics",
]

HIGHER = "higher_is_better"
LOWER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class GateRule:
    """一条门禁规则。字段含义见下，阈值怎么来的见 `DEFAULT_RULES` 上方注释。"""

    # 被比较的指标名，必须与 `TaskMetrics.as_dict()` 的键一致（拼错不会报错，
    # 只会静默变成"基线里没有这个指标 → 跳过"）。
    metric: str

    # `higher_is_better` / `lower_is_better`。方向写反会让"变好"被判成劣化，
    # 这是这门禁最安静的一种失效。
    direction: str

    # **相对**劣化上限：0.05 = 不许比基线差 5%。刻意不用绝对值——
    # 不同机器跑 p95 天然有波动，绝对阈值会天天误报，最后大家学会忽略红叉。
    max_regression: float

    # 绝对劣化下限，只对"越低越好"的指标有意义（延迟、成本）：
    # 基线本身很小时（如 2ms），5% 只有 0.1ms，纯噪声就会触发。默认 0 表示不设。
    min_absolute: float = 0.0

    # False = 基线里没有这个指标就不判。离线档没有 `judge_score`，
    # 强制要求它存在等于让门禁在离线档永远红。
    required: bool = True

    @property
    def higher_is_better(self) -> bool:
        return self.direction == HIGHER


# 阈值从 W45/W46 两期实际波动倒推：离线档 success_rate 两期都是 100%，无波动；
# 延迟 p95 在同一台机器上两次跑相差约 8%~25%（夹具回放，抖动来自进程调度）。
# 所以延迟给 20% + 5ms 绝对下限，其余给 5%~10%。
DEFAULT_RULES: tuple[GateRule, ...] = (
    GateRule("success_rate", HIGHER, 0.05),
    GateRule("latency_p95_ms", LOWER, 0.20, min_absolute=5.0, required=False),
    GateRule("cost_per_task", LOWER, 0.15, required=False),
    GateRule("judge_score", HIGHER, 0.05, required=False),
    GateRule("faithfulness", HIGHER, 0.10, required=False),
    GateRule("tool_error_rate", LOWER, 0.50, required=False),
)

# 这些指标涨了要告警，但不判 fail：可靠性问题不该阻断合并，
# 但也不能当没发生（W3 metric-spec §1 的口径）。
WARNING_RULES: tuple[GateRule, ...] = (
    GateRule("interrupted_rate", LOWER, 0.02, required=False),
)


@dataclass(slots=True)
class Violation:
    """一次越线。字段全部保留原始数字，让读 CI 日志的人能自己复算判定过程。"""

    # 越线的指标名。
    metric: str

    # 基线值与当前值（原始单位：秒数、比例或毫秒）。
    baseline: float
    current: float

    # 相对变化比例，带符号：正数 = 变大。方向解释取决于规则，
    # 所以它单独存而不是只存"劣化了多少"——后者会让人看不出是涨是跌。
    change_ratio: float

    # 当时生效的阈值。与 `change_ratio` 一起写进报告，才说得清"差多少被判红"。
    max_regression: float

    # 人读解释（"延迟 p95 从 2.29ms 涨到 30.81ms，超出 20%"）。CI 日志直接打这个。
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "current": self.current,
            "change_ratio": self.change_ratio,
            "max_regression": self.max_regression,
            "detail": self.detail,
        }


@dataclass(slots=True)
class GateVerdict:
    """门禁结论。落盘进 CI 日志与 PR 评论，所以每个字段都要能独立解释。"""

    # 是否有 `violations`。只看这一个字段决定 CI 退出码。
    passed: bool = True

    # 越线项。非空即 `passed=False`。
    violations: list[Violation] = field(default_factory=list)

    # 只告警不阻断的项（如 `interrupted_rate` 激增）。可靠性问题该被看见，
    # 但不该拦住一次内容改进的合并。
    warnings: list[str] = field(default_factory=list)

    # 实际比较过的指标名。**必须记**：`passed=True` 但 `compared=[]`
    # 意味着"一个都没比"，那和"全部通过"是两件完全不同的事。
    compared: list[str] = field(default_factory=list)

    # 跳过的指标及原因（基线里没有 / `required=False`）。同样是防"假绿"。
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [item.as_dict() for item in self.violations],
            "warnings": self.warnings,
            "compared": self.compared,
            "skipped": self.skipped,
        }


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return float(value)

    return None


def evaluate(
    current: dict[str, Any],
    baseline: dict[str, Any],
    rules: tuple[GateRule, ...] = DEFAULT_RULES,
    *,
    warning_rules: tuple[GateRule, ...] = WARNING_RULES,
) -> GateVerdict:
    """对比当前指标与基线。

    返回 `GateVerdict` 而不是抛异常：门禁的结论要能落盘进 CI 日志与 PR 评论，
    异常栈没有这个能力。
    """

    verdict = GateVerdict()

    for rule in rules:
        base = _as_float(baseline.get(rule.metric))
        now = _as_float(current.get(rule.metric))

        if now is None:
            # 当前期压根没有这个指标（例如没开 judge）。`required` 的缺失说明
            # 评测跑残了，必须红；非 required 的只是不判。
            if rule.required:
                verdict.passed = False
                verdict.violations.append(
                    Violation(
                        metric=rule.metric,
                        baseline=base if base is not None else 0.0,
                        current=0.0,
                        change_ratio=0.0,
                        max_regression=rule.max_regression,
                        detail="当前期缺少该指标，无法判定（required=True）",
                    )
                )
            else:
                verdict.skipped.append(rule.metric)

            continue

        if base is None:
            verdict.skipped.append(rule.metric)
            continue

        if base == 0:
            # 基线为 0 意味着"当时没测到这个值"（离线夹具零成本、p95 无样本）。
            # 任何"从 0 到有"都不是劣化，也不是改进——一律不判，记进 skipped 让读者知道。
            verdict.skipped.append(rule.metric)
            continue

        verdict.compared.append(rule.metric)
        detail = _check(rule, base, now)

        if detail is not None:
            verdict.violations.append(detail)

    for rule in warning_rules:
        base = _as_float(baseline.get(rule.metric))
        now = _as_float(current.get(rule.metric))

        if base is None or now is None:
            continue

        if now > base + rule.max_regression:
            verdict.warnings.append(
                f"{rule.metric} 从 {base:.2%} 升到 {now:.2%}（+{now - base:.2%}）："
                "可靠性退化，先看厂商状态页，别改 prompt"
            )

    verdict.passed = verdict.passed and not verdict.violations

    return verdict


def _check(rule: GateRule, base: float, now: float) -> Violation | None:
    if rule.higher_is_better:
        delta = base - now

        ratio = delta / base

        if ratio > rule.max_regression:
            return Violation(
                metric=rule.metric,
                baseline=base,
                current=now,
                change_ratio=round(ratio, 4),
                max_regression=rule.max_regression,
                detail=(
                    f"下降 {ratio:.2%}（允许 {rule.max_regression:.0%}）："
                    f"{base:.4f} → {now:.4f}"
                ),
            )

        return None

    # lower_is_better
    delta = now - base
    absolute_ok = delta > rule.min_absolute

    ratio = delta / base

    if ratio > rule.max_regression and absolute_ok:
        return Violation(
            metric=rule.metric,
            baseline=base,
            current=now,
            change_ratio=round(ratio, 4),
            max_regression=rule.max_regression,
            detail=(
                f"上升 {ratio:.2%}（允许 {rule.max_regression:.0%}，"
                f"且需超过 {rule.min_absolute:g} 绝对增量）：{base:.4f} → {now:.4f}"
            ),
        )

    return None


# --------------------------------------------------------------------------- #
# 读盘
# --------------------------------------------------------------------------- #


def load_metrics(path: Path) -> dict[str, Any]:
    """从一期 eval 快照里取指标。

    离线档的指标在 `pipeline`，在线档在 `quality`——**按 mode 取，不按字段是否存在猜**：
    如果两处都读，一次"离线跑了个高分"就能把门禁骗成绿的。
    """

    payload = json.loads(path.read_text("utf-8"))
    mode = payload.get("mode", "offline")
    block = payload.get("pipeline") if mode == "offline" else payload.get("quality")

    if not isinstance(block, dict):
        raise SystemExit(f"{path} 里没有可比的指标块（mode={mode}）")

    return block


def latest_eval() -> Path:
    candidates = sorted(
        path
        for path in HISTORY_DIR.glob("*-eval.json")
        if not path.name.startswith("baseline")
    )

    if not candidates:
        raise SystemExit("evals/history 里没有 *-eval.json，先跑一次 make run-eval")

    return candidates[-1]


def model_snapshot() -> dict[str, Any]:
    """基线必须带模型快照：换了 strong/cheap 的 primary，旧基线就不可比。"""

    import yaml

    catalog = yaml.safe_load(CATALOG_PATH.read_text("utf-8"))
    routes = catalog.get("routes", {})

    return {
        "cheap": routes.get("cheap", {}).get("primary"),
        "strong": routes.get("strong", {}).get("primary"),
        "aliases": sorted(catalog.get("models", {})),
        "catalog_sha256": _sha(CATALOG_PATH.read_bytes())[:12],
    }


def build_baseline(current_path: Path, *, week: str | None = None) -> dict[str, Any]:
    payload = json.loads(current_path.read_text("utf-8"))

    return {
        "schema": 1,
        "week": week or payload.get("week"),
        "mode": payload.get("mode", "offline"),
        "recorded_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "golden": payload.get("golden"),
        "model_snapshot": model_snapshot(),
        "metrics": load_metrics(current_path),
        "note": (
            "基线是**离线档**指标：success_rate 只代表执行管道。"
            "在线档接入后需要另建一份 baseline-online.json，两者的 success_rate 不可互比。"
        ),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="指标回归门禁：劣化即 fail")
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--current", default=None, help="默认取最新一期 *-eval.json")
    parser.add_argument("--write-baseline", action="store_true", help="用当前期重建基线")
    parser.add_argument("--strict", action="store_true", help="把告警也当失败")
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    current_path = Path(args.current) if args.current else latest_eval()

    if args.write_baseline:
        baseline = build_baseline(current_path)
        baseline_path.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", "utf-8"
        )
        print(f"已写入基线 {baseline_path.relative_to(REPO_ROOT)}（commit {baseline['commit']}）")

        return 0

    if not baseline_path.exists():
        print(
            f"基线不存在：{baseline_path.relative_to(REPO_ROOT)}\n"
            "门禁不会静默放行。先跑一次 `make run-eval` 再 "
            "`uv run python evals/scripts/regression_gate.py --write-baseline`",
            file=sys.stderr,
        )

        return 2

    baseline = json.loads(baseline_path.read_text("utf-8"))
    current = load_metrics(current_path)
    verdict = evaluate(current, baseline.get("metrics", {}))

    if args.json:
        print(json.dumps(verdict.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"基线：{baseline.get('week')}（commit {baseline.get('commit')}）")
        print(f"当前：{current_path.name}")
        print(f"对比了 {len(verdict.compared)} 个指标：{', '.join(verdict.compared) or '-'}")

        if verdict.skipped:
            print(f"跳过（基线或当前期缺该指标）：{', '.join(verdict.skipped)}")

        for violation in verdict.violations:
            print(f"[FAIL] {violation.metric}: {violation.detail}")

        for warning in verdict.warnings:
            print(f"[WARN] {warning}")

        print("门禁通过" if verdict.passed else "门禁失败：指标劣化")

    if args.strict and verdict.warnings:
        return 1

    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
