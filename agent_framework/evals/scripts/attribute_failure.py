"""四层归因（W4 D4）：把"失败"变成"该改哪一层"。

W3 的 `classify_failures` 回答"这条失败像什么"（工具错/格式错/疑似编造）；
这一层回答"**下一步动哪个文件**"（检索/提示词/工具/模型）。

顺序不能反
----------
先规则分类定 `FailureCategory`，再用 span 信号细化到层。
反过来（先猜层再看分类）会让每个失败都变成"我觉得是 prompt 问题"——
不可复核，也不能被测试覆盖。

三条纪律
--------
1. **规则优先**：能从 `error_code` / span 数量判的，绝不交给模型猜。
2. **每个归因必须带证据字段**：哪个 span、哪个日志值。没有证据就报"证据不足"。
3. **证据不足不许硬猜**：分类说"工具错"但 trace 里没有 tool span，答案不是"工具层"，
   而是"采集层缺了 span"。把后者说成前者，团队会去改工具 schema——改完仍然红。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.metrics import EvalResult  # noqa: E402
from evals.scripts.classify_failures import FailureCategory  # noqa: E402

__all__ = [
    "Attribution",
    "Layer",
    "Signals",
    "attribute",
    "attribute_result",
    "from_case_payload",
]

# 上游错误码：出现即归模型/可靠性层，不必再看别的信号。
_UPSTREAM_CODES = {"UPSTREAM_AUTH", "UPSTREAM_ERROR", "UPSTREAM_TIMEOUT", "RATE_LIMITED"}
_BUSY_CODES = {"SERVER_BUSY"}


class Layer(StrEnum):
    """四层归因。`UNKNOWN` 是"证据不足"，不是第五层。"""

    RETRIEVAL = "retrieval"
    PROMPT = "prompt"
    TOOL = "tool"
    MODEL = "model"
    UNKNOWN = "unknown"

    @property
    def fix_hint(self) -> str:
        return {
            Layer.RETRIEVAL: "查 chunk / 召回 / 重排（W5–W6）",
            Layer.PROMPT: "改指令分层、rubric、few-shot（W7/W10）",
            Layer.TOOL: "查工具 schema 描述、参数校验、权限（W8）",
            Layer.MODEL: "查 fallback / 重试 / 降级，先看厂商状态页（W11）",
            Layer.UNKNOWN: "先补埋点：缺哪个 span 就补哪个，别急着改代码",
        }[self]


@dataclass(frozen=True, slots=True)
class Signals:
    """从 trace / `RequestLog` 里能读到的东西。

    刻意全是"可观测的量"，没有 `quality` 这类主观判断——
    归因必须是可复核的，否则它只是换了个说法的猜测。
    """

    # 归一化错误码。命中 `_UPSTREAM_CODES` / `_BUSY_CODES` 即直接归模型层（置信 0.9），
    # 不必再看别的信号——这是唯一"看一眼就能定性"的入口。
    error_code: str | None = None

    # 结束原因。`length` 指向预算/截断，`content_filter` 指向上游策略，
    # 都不是"模型不会答"。
    finish_reason: str | None = None

    # 检索层：看到几个检索 span、召回了哪些 chunk、答案实际引了哪些。
    # 三个一起看才能区分"没检索"（spans=0）、"召回了但没用"（有 ids 无 cited）、
    # 和"引了别的段"（cited 与 retrieved 不一致）。
    retrieval_spans: int = 0
    retrieved_ids: tuple[str, ...] = ()
    cited_ids: tuple[str, ...] = ()

    # 工具层：工具 span 数量、其中参数错误次数、HTTP 错误次数。
    # 参数错误指向 schema 描述（工具层），HTTP 错误指向工具服务本身。
    tool_spans: int = 0
    tool_arguments_error: int = 0
    tool_http_error: int = 0

    # 可靠性层：重试次数与从哪个 alias fallback 过来。非 0 说明这次"答对了"
    # 也可能是备用模型的功劳，不能把结论记到主模型头上。
    retry_count: int = 0
    fallback_from: str | None = None

    # 答案长度（字符）。极短（空答案/一句拒答）与 `finish_reason=length`
    # 一起构成"没答出来"的最强组合信号。
    answer_chars: int = 0

    # 逃生口：新增信号先放这里，等某一层真的靠它判定时再提升成正式字段。
    # 加正式字段的门槛是"至少有一个归因分支在用"。
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Attribution:
    """一次归因的结论。

    `confidence` 是**证据强度**，不是"我觉得有多可能"：
    - 0.9：直接信号（`arguments_error>0`、`error_code` 命中表）
    - 0.6：间接推断（有 retrieval span，但引用对不上）
    - 0.0：证据不足（此时 `layer=UNKNOWN`）
    """

    # 归到哪一层。`UNKNOWN` 是"证据不足"的显式表达，不是第五层：
    # 它的存在是为了让"没查清"不被写成"查出来是模型的问题"。
    layer: Layer

    # 判定依据：每条都是一个具体字段名 + 实际值（"tool_arguments_error=2"）。
    # 不接受"看起来像"这类描述——归因必须能被别人拿着同一条 trace 复核。
    evidence: tuple[str, ...]

    # 证据强度（0.0–1.0），不是主观概率。0.9=直接信号，0.6=间接推断，0.0=证据不足。
    confidence: float

    # 该层的典型修法（取自 `Layer.fix_hint`）。写进报告，避免归完因没人知道下一步。
    fix: str

    @property
    def actionable(self) -> bool:
        return self.layer is not Layer.UNKNOWN

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "fix": self.fix,
        }


def attribute(
    categories: tuple[FailureCategory, ...] | list[FailureCategory],
    signals: Signals,
) -> Attribution:
    """`FailureCategory` + 信号 → 一层归因。"""

    cats = set(categories)

    # 0) 上游故障：这是可靠性问题，与"模型笨"无关。
    if signals.error_code:
        if signals.error_code in _UPSTREAM_CODES | _BUSY_CODES:
            return Attribution(
                layer=Layer.MODEL,
                evidence=(
                    f"error_code={signals.error_code}",
                    f"retry_count={signals.retry_count}",
                    f"fallback_from={signals.fallback_from or '-'}",
                ),
                confidence=0.9,
                fix=Layer.MODEL.fix_hint,
            )

        # 其它错误码（例如 CONTENT_FILTERED）也先归模型层，但要标明是"策略拒绝"。
        return Attribution(
            layer=Layer.MODEL,
            evidence=(f"error_code={signals.error_code}",),
            confidence=0.7,
            fix=Layer.MODEL.fix_hint,
        )

    # 1) 工具层：必须有 tool span 才能判，否则是采集缺失。
    if FailureCategory.TOOL_ERROR in cats:
        return _attribute_tool(signals)

    # 2) 疑似编造：先看检索信号在不在。
    if FailureCategory.HALLUCINATION in cats:
        return _attribute_hallucination(signals)

    # 3) 格式/结构错：输出结构稳定错，一般是提示词里没说清，而不是模型能力问题。
    if FailureCategory.FORMAT in cats:
        return Attribution(
            layer=Layer.PROMPT,
            evidence=(
                "分类=format（帧序列或必填字段不符）",
                f"finish_reason={signals.finish_reason or '-'}",
                f"answer_chars={signals.answer_chars}",
            ),
            confidence=0.7,
            fix=Layer.PROMPT.fix_hint,
        )

    # 4) 预算/截断：先看 max_tokens 与篇幅要求，属提示词层。
    if FailureCategory.BUDGET in cats:
        return Attribution(
            layer=Layer.PROMPT,
            evidence=(
                "分类=budget",
                f"finish_reason={signals.finish_reason or '-'}",
                f"answer_chars={signals.answer_chars}",
            ),
            confidence=0.6,
            fix=Layer.PROMPT.fix_hint,
        )

    # 5) 纯断言不成立：语义层，没有信号就只能说"要人看"。
    if FailureCategory.ASSERTION in cats or cats:
        return Attribution(
            layer=Layer.PROMPT,
            evidence=(
                f"分类={', '.join(sorted(item.value for item in cats))}",
                "无工具/检索/上游异常信号，判定为答案本身不合判据",
            ),
            confidence=0.4,
            fix=Layer.PROMPT.fix_hint,
        )

    return Attribution(
        layer=Layer.UNKNOWN,
        evidence=("没有失败分类，无法归因",),
        confidence=0.0,
        fix=Layer.UNKNOWN.fix_hint,
    )


def _attribute_tool(signals: Signals) -> Attribution:
    if signals.tool_spans == 0:
        # 反向验证的那一条：分类说工具错，但 trace 里一个 tool span 都没有。
        return Attribution(
            layer=Layer.UNKNOWN,
            evidence=(
                "分类=tool_error",
                "tool_spans=0：trace 里没有工具调用记录，说明是**埋点缺失**而不是工具坏了",
            ),
            confidence=0.0,
            fix=Layer.UNKNOWN.fix_hint,
        )

    if signals.tool_arguments_error > 0:
        return Attribution(
            layer=Layer.TOOL,
            evidence=(
                f"tool_arguments_error={signals.tool_arguments_error}",
                f"tool_spans={signals.tool_spans}",
                "参数 JSON 不合法/不符 schema",
            ),
            confidence=0.9,
            fix=Layer.TOOL.fix_hint,
        )

    if signals.tool_http_error > 0:
        return Attribution(
            layer=Layer.TOOL,
            evidence=(
                f"tool_http_error={signals.tool_http_error}",
                f"tool_spans={signals.tool_spans}",
            ),
            confidence=0.8,
            fix=Layer.TOOL.fix_hint,
        )

    return Attribution(
        layer=Layer.TOOL,
        evidence=(
            "分类=tool_error",
            f"tool_spans={signals.tool_spans}，未记录到参数错误或 HTTP 错误",
        ),
        confidence=0.5,
        fix=Layer.TOOL.fix_hint,
    )


def _attribute_hallucination(signals: Signals) -> Attribution:
    if signals.retrieval_spans == 0:
        # W5 之前没有检索：这时"疑似无引用支撑"不能升级为"模型编造"。
        return Attribution(
            layer=Layer.UNKNOWN,
            evidence=(
                "分类=hallucination（疑似无引用支撑）",
                "retrieval_spans=0：没有检索信号（W5 前属正常），无法区分"
                "「没检索到」与「检索到但没用」",
            ),
            confidence=0.0,
            fix=Layer.UNKNOWN.fix_hint,
        )

    if signals.retrieved_ids and not (set(signals.retrieved_ids) & set(signals.cited_ids)):
        return Attribution(
            layer=Layer.RETRIEVAL,
            evidence=(
                f"retrieved_ids={len(signals.retrieved_ids)} 条，引用命中 0 条",
                f"retrieval_spans={signals.retrieval_spans}",
            ),
            confidence=0.8,
            fix=Layer.RETRIEVAL.fix_hint,
        )

    if signals.cited_ids:
        # 引用了检索结果，但结论仍无支撑 → 是"用了错的上下文"，仍在检索层。
        return Attribution(
            layer=Layer.RETRIEVAL,
            evidence=(
                f"引用 {len(signals.cited_ids)} 条全部来自检索结果，但结论无支撑",
                "检索到的内容本身不含答案",
            ),
            confidence=0.6,
            fix=Layer.RETRIEVAL.fix_hint,
        )

    # 跑了检索却没记录任何 doc id：这不是"模型编造"，是埋点不全。
    return Attribution(
        layer=Layer.UNKNOWN,
        evidence=(
            f"retrieval_spans={signals.retrieval_spans} 但 retrieved_ids 为空",
            "无法判断是「没召回」还是「召回了但没记录 id」",
        ),
        confidence=0.0,
        fix=Layer.UNKNOWN.fix_hint,
    )


def attribute_result(
    result: EvalResult, signals: Signals | None = None
) -> Attribution:
    """对一条 `EvalResult` 归因。信号缺失时用 metrics 里的字段尽量补。"""

    base = signals or Signals()
    merged = Signals(
        error_code=base.error_code or result.metrics.error_code,
        finish_reason=base.finish_reason,
        retrieval_spans=base.retrieval_spans,
        retrieved_ids=base.retrieved_ids,
        cited_ids=base.cited_ids,
        tool_spans=base.tool_spans or result.metrics.tool_calls,
        tool_arguments_error=base.tool_arguments_error or result.metrics.tool_errors,
        tool_http_error=base.tool_http_error,
        retry_count=base.retry_count or result.metrics.retry_count,
        fallback_from=base.fallback_from,
        answer_chars=base.answer_chars,
        extra=base.extra,
    )

    return attribute(
        [FailureCategory(value) for value in result.categories], merged
    )


def from_case_payload(payload: dict[str, Any]) -> tuple[tuple[FailureCategory, ...], Signals]:
    """从评测快照的 `cases[i]`（`snapshot()` 的落盘结构）还原输入。

    只能还原落盘里有的字段——没有检索 span 就没有，**不编**。
    """

    categories = tuple(FailureCategory(value) for value in payload.get("categories", ()))
    metrics = payload.get("metrics") or {}

    signals = Signals(
        error_code=metrics.get("error_code"),
        tool_spans=int(metrics.get("tool_calls") or 0),
        tool_arguments_error=int(metrics.get("tool_errors") or 0),
        retry_count=int(metrics.get("retry_count") or 0),
    )

    return categories, signals


# --------------------------------------------------------------------------- #
# CLI：对一期评测快照里的失败逐条归因
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """读一期 `*-eval.json`，把失败/中断的 case 逐条归因。

    它不能替代人：证据不足的会明确报 `unknown`，让读者去补埋点而不是猜代码。
    """

    import argparse
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="对评测快照里的失败做四层归因")
    parser.add_argument(
        "snapshot",
        nargs="?",
        default=None,
        help="评测快照路径，默认取 evals/history 里最新一期 *-eval.json",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    if args.snapshot:
        path = Path(args.snapshot)
    else:
        candidates = sorted((root / "evals" / "history").glob("*-eval.json"))

        if not candidates:
            print("没有可读的评测快照，先跑 make run-eval", file=sys.stderr)

            return 2

        path = candidates[-1]

    payload = json.loads(path.read_text("utf-8"))
    rows = []

    for case in payload.get("cases", ()):
        if case.get("status") not in ("failed", "interrupted"):
            continue

        categories, signals = from_case_payload(case)
        attribution = attribute(categories, signals)
        rows.append(
            {
                "case_id": case.get("id"),
                "status": case.get("status"),
                "categories": sorted(item.value for item in categories),
                "failure": (case.get("failures") or [""])[0],
                **attribution.as_dict(),
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))

        return 0

    print(f"快照：{path.name}（{payload.get('week')} / {payload.get('mode')}）")

    if not rows:
        print("本期没有失败，无需归因。")

        return 0

    header = f"{'case':<12}{'层':<12}{'置信':>6}  证据 / 修法"
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['case_id']:<12}{row['layer']:<12}{row['confidence']:>6.1f}  "
            f"{'；'.join(row['evidence'][:2])}"
        )

        if row["layer"] == "unknown":
            print(f"{'':<31}→ 证据不足：{row['fix']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
