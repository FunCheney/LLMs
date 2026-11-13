#!/usr/bin/env python
"""三张图的数据查询（W11 D1）：从已有的落盘数据算 SLI / 误差预算 / 告警。

    python ops/slo_report.py                          # 全部 history 的趋势
    python ops/slo_report.py --week 2026-W53          # 单周分档明细
    python ops/slo_report.py --log app.jsonl          # 真实 RequestLog（算 p95）
    python ops/slo_report.py --alert                  # 只输出"会不会叫醒人"

## 为什么不另造埋点

数据源只有两个，都是 W1–W10 已经在写的：

1. `evals/history/*-eval.json` 的 `by_difficulty`（周级聚合：success_rate /
   first_token_p95_ms / cost_per_task / usage_estimated_ratio）；
2. `RequestLog` 打出的 JSONL（请求级原始记录，`--log`）。

自己再埋一套"业务指标"的后果是两套数对不上，而**对不上的时候没人知道该信哪个**。
W11 的成本看板与 W3 的评测报告必须是同一个数，否则"成本下降了"这句话永远说不清。

## 一个刻意的限制

history 里的 `first_token_p95_ms` 在离线档（fixture 回放）大量是 null——
夹具不产生流式首 token。所以本脚本在缺值时**跳过**这条 SLO 并把覆盖率打出来，
不用 `latency_p95_ms` 顶替：两者量级差 10 倍，混算等于把长尾抹平。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.slo import (  # noqa: E402 - 路径注入必须先于导入
    ALERT_RULES,
    SLO_BY_NAME,
    SLOS,
    alert_fires,
    compliance,
    error_budget,
    percentile,
    series_from_eval,
)

HISTORY = Path(__file__).resolve().parents[1] / "evals" / "history"


def load_history() -> list[tuple[str, dict[str, object]]]:
    """按期号升序读全部 `*-eval.json`。文件名里的周次是唯一的排序依据。"""

    rows: list[tuple[str, dict[str, object]]] = []
    for path in sorted(HISTORY.glob("*-eval.json")):
        week = path.name.removesuffix("-eval.json")
        try:
            rows.append((week, json.loads(path.read_text("utf-8"))))
        except json.JSONDecodeError:
            # 坏文件不静默跳过：它会让趋势图少一个点，而"少一个点"在图上
            # 看起来和"那周没数据"一样。
            print(f"!! {path.name} 解析失败，趋势会缺这一段", file=sys.stderr)
    return rows


def load_request_log(path: Path) -> list[dict[str, object]]:
    """读 RequestLog JSONL。

    兼容两种形态：纯 JSONL（`json.dumps` 直接落盘）与 `logging` 输出
    （`2026-11-13 09:26:28,281 INFO {json}`）——后者是当前 `configure_logging()`
    的实际格式。只认"行里第一个 `{` 开始能解析成 dict 且有 `latency_ms`"，
    其余行（uvicorn access log、启动日志）一律跳过：混进来会让 p95 变成噪声。
    """

    records: list[dict[str, object]] = []
    for line in path.read_text("utf-8").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            payload = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "latency_ms" in payload:
            records.append(payload)
    return records


def is_our_fault(record: dict[str, object]) -> bool:
    """可用性 SLO 的分母口径：调用方的错不算我们不可用。

    401（我方鉴权）与 400/422（护栏）都是**调用方输入的问题**，
    含进分母的后果是"被扫的时候可用性看起来下降"，真故障被攻击噪声淹没。
    """

    code = str(record.get("error_code") or "")
    return code not in {"AUTH_REQUIRED", "AUTH_INVALID", "CONTENT_BLOCKED", "BAD_REQUEST"}


def report_log(records: Sequence[dict[str, object]]) -> None:
    print(f"== RequestLog：{len(records)} 条 ==")

    availability = SLO_BY_NAME["availability"]
    graded = [compliance(availability, 1 if not r.get("error_code") else 0) for r in records
              if is_our_fault(r)]
    bad = sum(1 for item in graded if item == 0)
    observed = (len(graded) - bad) / len(graded) if graded else 0.0

    print(f"{'可用性':<12} {observed:.4f}  (分母 {len(graded)}，其中失败 {bad})")

    first_tokens = [
        float(r["first_token_ms"])
        for r in records
        if isinstance(r.get("first_token_ms"), (int, float))
    ]
    p95 = percentile(first_tokens, 0.95)
    latency = SLO_BY_NAME["latency_p95"]
    share_ok = (
        sum(1 for value in first_tokens if value <= 1500.0) / len(first_tokens)
        if first_tokens
        else None
    )
    shown_p95 = "-" if p95 is None else f"{p95:.1f}ms"
    shown_ok = "-" if share_ok is None else f"{share_ok:.3f}"
    print(
        f"{'首 token':<12} p95={shown_p95} 达标率={shown_ok} "
        f"(样本 {len(first_tokens)}/{len(records)}，非流式不参与)"
    )

    estimated = sum(1 for r in records if r.get("usage_estimated"))
    cost_total = sum(float(r.get("cost") or 0.0) for r in records)
    currency = records[0].get("currency", "CNY") if records else "CNY"
    print(
        f"{'成本':<12} 合计 {cost_total:.4f} {currency} "
        f"估算占比 {estimated / len(records):.1%}"
        if records
        else "成本         无记录"
    )

    status = error_budget(availability, observed, len(graded))
    print(f"\n{status.line()}")
    if estimated / max(1, len(records)) > 0.2:
        print("!! usage_estimated 占比 > 20%：成本结论不可信（W1 Day5）")
    if share_ok is not None:
        print(f"   latency 预算：{latency.name} 达标率 {share_ok:.3f} / 目标 {latency.target}")


def report_history(rows: Sequence[tuple[str, dict[str, object]]]) -> None:
    print("== 周级趋势（by_difficulty 聚合，直接读落盘，不重算）==")
    header = (
        f"{'week':<10}{'档':<4}{'成功率':>8}{'首token p95':>13}"
        f"{'整体 p95':>10}{'成本/任务':>10}{'估算占比':>9}"
    )
    print(header)
    print("-" * len(header))

    for week, payload in rows:
        for row in series_from_eval(payload):
            success = row["success_rate"]
            ft = row["first_token_p95_ms"]
            p95 = row["latency_p95_ms"]
            cost = row["cost_per_task"]
            est = row["usage_estimated_ratio"]
            print(
                f"{week:<10}{str(row['difficulty']):<4}"
                f"{'-' if success is None else f'{success:.3f}':>8}"
                f"{'-' if ft is None else f'{ft:.1f}ms':>13}"
                f"{'-' if p95 is None else f'{p95:.1f}ms':>10}"
                f"{'-' if cost is None else f'{cost:.4f}':>10}"
                f"{'-' if est is None else f'{est:.1%}':>9}"
            )


def report_alerts(rows: Sequence[tuple[str, dict[str, object]]]) -> int:
    """按最新一期的分档数据构造序列，判断哪些规则会触发。

    注意这是**离线判定**，不是线上告警：数据点是"档位"而不是"时间"。
    它能回答"如果线上是这个数，规则会不会响"，不能回答"响得及时不及时"——
    后者只能靠线上连续采样的 `(timestamp, value)`，本仓库还没有那个数据源。
    """

    if not rows:
        print("没有历史数据，告警无法判定")
        return 0

    week, payload = rows[-1]
    fired = 0
    print(f"== 告警判定（基于 {week}）==")
    for rule in ALERT_RULES:
        slo = SLO_BY_NAME[rule.slo]
        samples: list[tuple[float, float]] = []
        for index, row in enumerate(series_from_eval(payload)):
            if rule.slo == "availability":
                value = row["success_rate"]
            elif rule.slo == "latency_p95":
                value = row["first_token_p95_ms"]
            else:
                value = row["cost_per_task"]
            if value is None:
                continue
            status = error_budget(slo, 1.0 if compliance(slo, value) else 0.0, 1, elapsed=1.0)
            samples.append((float(index) * 600.0, status.burn_rate))

        ok = alert_fires(rule, samples)
        fired += int(ok)
        print(f"  [{'FIRES' if ok else ' ok  '}] {rule.name:<26} {rule.severity:<5} {rule.summary}")
    return fired


def main() -> int:
    parser = argparse.ArgumentParser(description="SLO 报告（复用已有埋点）")
    parser.add_argument("--week", help="只看某一周的档位明细")
    parser.add_argument("--log", type=Path, help="RequestLog 的 JSONL 路径")
    parser.add_argument("--alert", action="store_true", help="只输出告警判定")
    args = parser.parse_args()

    if args.log is not None:
        if not args.log.is_file():
            print(f"找不到 {args.log}", file=sys.stderr)
            return 2
        report_log(load_request_log(args.log))
        return 0

    rows = load_history()
    if args.week:
        rows = [item for item in rows if item[0] == args.week]
        if not rows:
            print(f"没有 {args.week} 的评测数据", file=sys.stderr)
            return 2

    if args.alert:
        report_alerts(rows)
        return 0

    report_history(rows)
    print()
    fired = report_alerts(rows)

    print()
    print("== 口径提醒 ==")
    for slo in SLOS:
        print(f"  {slo.name}: target={slo.target} window={slo.window} ({slo.comparison})")
        print(f"    {slo.discipline}")
    return 0 if fired == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
